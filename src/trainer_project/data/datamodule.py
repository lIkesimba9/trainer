"""Lightning data lifecycle with explicit distributed ownership."""

from __future__ import annotations

import copy
import random
from collections import defaultdict

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from .manifests import (
    ConcatManifestDataset,
    CsvDataset,
    _make_transform,
    common_collate,
    confined_path,
    load_mapping,
    normalize_class_map,
    read_csv,
)
from .samplers import DistributedEpochSampler, PKBatchSampler


def seed_worker(worker_id):
    value = torch.initial_seed() % (2**32)
    random.seed(value)
    np.random.seed(value)


class EpochDataLoader(DataLoader):
    """Set epoch before iterator creation, including worker-process snapshots.

    Lightning's on_train_epoch_start can follow DataLoader iterator creation;
    setting only the main-process Dataset there is too late for worker copies.
    """

    def __init__(self, dataset, *args, epoch_owner=None, training=False, **kwargs):
        self.epoch_owner = epoch_owner
        self.training = bool(training)
        super().__init__(dataset, *args, **kwargs)

    def __iter__(self):
        if self.training and self.epoch_owner is not None:
            trainer = getattr(self.epoch_owner, "trainer", None)
            epoch = int(trainer.current_epoch) if trainer is not None else self.epoch_owner.epoch
            pending = getattr(self.epoch_owner, "_resume_epoch", None)
            if pending is not None:
                epoch = max(epoch, pending)
                self.epoch_owner._resume_epoch = None
            self.epoch_owner.set_epoch(epoch)
            if self.generator is not None:
                self.generator.manual_seed(self.epoch_owner.seed + epoch * 100003)
        return super().__iter__()


class ManifestDataModule(LightningDataModule):
    def __init__(
        self,
        data_root=".",
        train=None,
        val=None,
        test=None,
        schema=None,
        class_maps=None,
        class_map=None,
        batch_size_per_device=16,
        batch_size=None,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        seed=42,
        transforms=None,
        sampler=None,
        reproducible=True,
        dataset_factory=None,
        identity_policy=None,
        train_mode="concat",
        combined_mode="max_size_cycle",
        drop_last=False,
        missing_modality_policy="error",
        validate=True,
        test_during_fit=True,
        **kwargs,
    ):
        super().__init__()
        self.data_root = confined_path(data_root)
        self.specs = {"train": list(train or []), "val": list(val or []), "test": list(test or [])}
        self.schema = load_mapping(schema) or {
            "inputs": {"image": {"column": "image", "type": "image"}},
            "targets": {},
        }
        self.class_maps = load_mapping(class_maps or class_map)
        if self.class_maps and all(isinstance(value, int) for value in self.class_maps.values()):
            key = next(iter(self.schema.get("targets", {"label": {}})))
            self.class_maps = {key: self.class_maps}
        self.class_maps = {key: normalize_class_map(value) for key, value in self.class_maps.items()}
        self.batch_size = int(batch_size or batch_size_per_device)
        self.batch_size_per_device = self.batch_size
        self.num_workers = int(num_workers)
        if self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError("batch size must be positive; num_workers nonnegative")
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.reproducible = bool(reproducible)
        if self.reproducible and persistent_workers:
            raise ValueError("reproducible epoch resume requires persistent_workers=False")
        self.seed, self.epoch = int(seed), 0
        self.transforms = load_mapping(transforms)
        self.sampler_config = load_mapping(sampler)
        self.dataset_factory = dataset_factory
        self.identity_policy = load_mapping(identity_policy)
        self.train_mode, self.combined_mode = train_mode, combined_mode
        if train_mode not in {"concat", "combined"}:
            raise ValueError("train_mode must be concat or combined")
        self.drop_last, self.missing_modality_policy = bool(drop_last), missing_modality_policy
        self.validate = bool(validate)
        self.test_during_fit = bool(test_during_fit)
        self.datasets = {"train": [], "val": [], "test": []}
        self.eval_plan = []
        self.train_dataset = None
        self._samplers = []
        self._samplers_by_key = {}
        self._pending_sampler_states = {}
        self._generator = torch.Generator().manual_seed(self.seed)
        self._ready = False

    def _derive_class_maps(self):
        for key, declaration in self.schema.get("targets", {}).items():
            spec = CsvDataset._spec(declaration)
            if key in self.class_maps:
                continue
            if "classes" in spec or "class_to_idx" in spec:
                self.class_maps[key] = normalize_class_map(spec.get("classes", spec.get("class_to_idx")))
                continue
            if spec.get("type", "categorical") not in {"categorical", "class"}:
                continue
            classes = set()
            for manifest in self.specs["train"]:
                root = confined_path(manifest.get("root", self.data_root))
                allowed = manifest.get("supervision_allowed", True)
                allowed = allowed if isinstance(allowed, bool) else allowed.get(key, False)
                if not allowed:
                    continue
                mapping = manifest.get("mapping", {})
                column = mapping.get(key, spec.get("column", key))
                for row in read_csv(confined_path(manifest["path"], root)):
                    value = row.get(column, "")
                    if value not in spec.get("missing_tokens", [""]):
                        classes.add(value)
            if not classes:
                raise ValueError(
                    f"Cannot derive {key} class order from permitted train labels; supply classes/class_maps"
                )
            self.class_maps[key] = normalize_class_map(sorted(classes))

    def setup(self, stage=None):
        if self._ready:
            return
        from hydra.utils import instantiate

        self._derive_class_maps()
        for role, manifests in self.specs.items():
            names = [item.get("name") for item in manifests]
            if any(not name or "/" in name for name in names) or len(names) != len(set(names)):
                raise ValueError(f"{role} manifest names must be nonempty, unique, and contain no slash")
            for specification in manifests:
                spec = dict(specification)
                target = spec.pop("_target_", None)
                transform = self.transforms.get("train" if role == "train" else "eval")
                kwargs = dict(
                    data_root=str(self.data_root),
                    schema=self.schema,
                    class_maps=self.class_maps,
                    role=role,
                    seed=self.seed,
                    transform=transform,
                    ssl=bool(self.transforms.get("ssl", False)),
                    missing_modality_policy=self.missing_modality_policy,
                    validate=self.validate,
                )
                kwargs.update(spec)
                if target:
                    dataset = instantiate({"_target_": target, "_recursive_": False}, **kwargs)
                elif self.dataset_factory:
                    factory = (
                        instantiate(self.dataset_factory)
                        if not callable(self.dataset_factory)
                        else self.dataset_factory
                    )
                    dataset = factory(**kwargs)
                else:
                    dataset = CsvDataset(**kwargs)
                self.datasets[role].append(dataset)
        self.train_dataset = ConcatManifestDataset(self.datasets["train"]) if self.datasets["train"] else None
        self.eval_plan = [
            {"role": role, "name": dataset.name}
            for role in ("val", "test")
            for dataset in self.datasets[role]
            if role == "val" or self.test_during_fit
        ]
        self._check_identity()
        self._ready = True
        self.set_epoch(self.epoch)

    def _check_identity(self):
        fields = self.identity_policy.get("fields", [])
        policy = self.identity_policy.get("overlap", "error")
        if not fields:
            return
        members = defaultdict(list)
        for role, datasets in self.datasets.items():
            for dataset in datasets:
                for row in dataset.records:
                    if any(row.get(field, "") == "" for field in fields):
                        raise ValueError(f"Identity field missing: {fields}")
                    members[tuple(row[field] for field in fields)].append((role, dataset, row))
        for identity, rows in members.items():
            roles = {role for role, _, _ in rows}
            if len(roles) <= 1:
                continue
            if policy == "error":
                raise ValueError(f"Cross-split overlap for identity {identity}: {sorted(roles)}")
            if policy == "allow_ssl":
                for role, dataset, row in rows:
                    if role == "train" and any(dataset._allowed(key) for key in dataset.target_schema):
                        raise ValueError(
                            "allow_ssl overlap requires supervision_allowed=false for overlapping train manifest"
                        )
            elif policy != "allow":
                raise ValueError("overlap policy must be error, allow_ssl or allow")

    def _loader(self, dataset, train=False):
        from hydra.utils import instantiate

        config = self.sampler_config if train else {}
        sampler_key = (
            ("train" if train else getattr(dataset, "role", "eval"))
            + ":"
            + getattr(dataset, "name", "concat")
        )
        if sampler_key in self._samplers_by_key:
            sampler = self._samplers_by_key[sampler_key]
        elif config and "_target_" in config:
            sampler = instantiate(config, dataset=dataset, seed=self.seed)
            if callable(sampler) and not hasattr(sampler, "__iter__"):
                sampler = sampler(dataset=dataset)
        else:
            options = {key: value for key, value in config.items() if not key.startswith("_")}
            sampler = DistributedEpochSampler(dataset, shuffle=train, seed=self.seed, **options)
        sampler.set_epoch(self.epoch)
        if sampler_key not in self._samplers_by_key:
            self._samplers.append(sampler)
            self._samplers_by_key[sampler_key] = sampler
        if sampler_key in self._pending_sampler_states:
            sampler.load_state_dict(self._pending_sampler_states.pop(sampler_key))
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 100003 + (0 if train else 500009))
        options = dict(
            epoch_owner=self,
            training=train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            worker_init_fn=seed_worker,
            generator=generator,
            collate_fn=common_collate,
        )
        if isinstance(sampler, PKBatchSampler) or getattr(sampler, "is_batch_sampler", False):
            return EpochDataLoader(dataset, batch_sampler=sampler, **options)
        return EpochDataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            drop_last=self.drop_last if train else False,
            **options,
        )

    def train_dataloader(self):
        self.setup("fit")
        if not self.train_dataset:
            raise ValueError("No train manifests configured")
        if self.train_mode == "combined":
            from lightning.pytorch.utilities.combined_loader import CombinedLoader

            return CombinedLoader(
                {dataset.name: self._loader(dataset, True) for dataset in self.datasets["train"]},
                mode=self.combined_mode,
            )
        return self._loader(self.train_dataset, True)

    def val_dataloader(self):
        self.setup("validate")
        return [
            self._loader(dataset)
            for role in ("val", "test")
            for dataset in self.datasets[role]
            if role == "val" or self.test_during_fit
        ]

    def test_dataloader(self):
        self.setup("test")
        return [self._loader(dataset) for dataset in self.datasets["test"]]

    def predict_dataloader(self):
        self.setup("predict")
        datasets = self.datasets["test"] or self.datasets["val"] or self.datasets["train"]
        return [self._loader(dataset) for dataset in datasets]

    def reference_dataloader(self, names=None, split="train"):
        self.setup()
        selected = [dataset for dataset in self.datasets[split] if names is None or dataset.name in names]
        if not selected:
            raise ValueError("No selected reference manifests")
        references = []
        for dataset in selected:
            reference = copy.copy(dataset)
            reference.role = "reference"
            reference.ssl = False
            reference.epoch = 0
            reference.transform = _make_transform(self.transforms.get("eval"))
            references.append(self._loader(reference))
        return references

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        for datasets in self.datasets.values():
            for dataset in datasets:
                dataset.set_epoch(epoch)
        for sampler in self._samplers:
            sampler.set_epoch(epoch)
        # Loader worker seeds are deterministic per epoch. Dataset augmentation
        # itself uses sample ID + epoch and is independent of worker scheduling.

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "class_maps": self.class_maps,
            "samplers": {key: sampler.state_dict() for key, sampler in self._samplers_by_key.items()},
            "generator_state": self._generator.get_state(),
        }

    def load_state_dict(self, state_dict):
        if state_dict.get("class_maps") != self.class_maps:
            raise ValueError("Checkpoint and dataset class order differ")
        self.epoch = int(state_dict.get("epoch", 0))
        # Full checkpoints are written at a completed epoch boundary. Lightning
        # creates the first iterator before advancing its restored epoch counter.
        self._resume_epoch = self.epoch + 1
        self._generator.set_state(state_dict["generator_state"])
        states = state_dict.get("samplers", {})
        for key, sampler in self._samplers_by_key.items():
            if key in states:
                sampler.load_state_dict(states[key])
        self._pending_sampler_states = {
            key: value for key, value in states.items() if key not in self._samplers_by_key
        }
        self.set_epoch(self.epoch)

    def manifest_info(self, content_hash=True):
        self.setup()
        return [
            dataset.provenance(content_hash=content_hash)
            for datasets in self.datasets.values()
            for dataset in datasets
        ]

    def validate_data(self):
        self.setup()
        return [dataset.validate_data() for datasets in self.datasets.values() for dataset in datasets]
