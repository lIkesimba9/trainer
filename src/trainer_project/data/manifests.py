"""CSV adapter and lossless common sample contract."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from trainer_project.transforms import ByteTokenizer, SampleTransform

WORKSPACE = Path(__file__).resolve().parents[3]


def confined_path(path, root=None, must_exist=True):
    result = Path(path).expanduser()
    if not result.is_absolute():
        result = Path(root or WORKSPACE) / result
    result = result.resolve()
    if not result.is_relative_to(WORKSPACE):
        raise ValueError(f"Path must remain inside project workspace: {result}")
    if must_exist and not result.exists():
        raise FileNotFoundError(result)
    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mapping(value):
    if value is None:
        return {}
    if isinstance(value, (str, Path)):
        from omegaconf import OmegaConf

        return OmegaConf.to_container(OmegaConf.load(confined_path(value)), resolve=True)
    from omegaconf import OmegaConf

    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else copy.deepcopy(value)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"CSV has duplicate columns: {path}")
        rows = list(reader)
        if any(None in row for row in rows):
            raise ValueError(f"CSV row contains surplus cells: {path}")
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def normalize_class_map(value):
    if isinstance(value, (list, tuple)):
        result = {str(name): index for index, name in enumerate(value)}
    else:
        result = {str(name): int(index) for name, index in dict(value).items()}
    if sorted(result.values()) != list(range(len(result))) or not result:
        raise ValueError("class_to_idx must have unique contiguous values starting at zero")
    return result


def _make_transform(config, default_size=32):
    if config is None:
        return SampleTransform(image_size=default_size)
    if callable(config):
        return config
    from hydra.utils import instantiate

    config = load_mapping(config)
    return instantiate(config) if "_target_" in config else SampleTransform(**config)


class CsvDataset(Dataset):
    def __init__(
        self,
        path,
        name="manifest",
        data_root=".",
        schema=None,
        class_maps=None,
        role="train",
        transform=None,
        transforms=None,
        seed=42,
        ssl=False,
        supervision_allowed=None,
        mapping=None,
        root=None,
        missing_modality_policy="error",
        records=None,
        validate=True,
        **kwargs,
    ):
        self.name, self.role, self.seed, self.epoch = str(name), str(role), int(seed), 0
        self.data_root = confined_path(root or data_root)
        self.manifest_path = confined_path(path, self.data_root) if path else None
        self.schema = load_mapping(schema) or {
            "inputs": {"image": {"column": "image", "type": "image"}},
            "targets": {},
        }
        self.class_maps = {key: normalize_class_map(value) for key, value in load_mapping(class_maps).items()}
        self.input_schema = self.schema.get("inputs", {"image": {"column": "image", "type": "image"}})
        self.target_schema = self.schema.get("targets", {})
        self.supervision = supervision_allowed
        self.missing_modality_policy = missing_modality_policy
        if missing_modality_policy not in {"error", "presence"}:
            raise ValueError("missing_modality_policy must be error or presence")
        self.mapping = dict(mapping or {})
        self.transform = _make_transform(transform or transforms)
        self.ssl = bool(ssl)
        self.records = []
        raw_rows = records if records is not None else read_csv(self.manifest_path)
        seen = set()
        for index, raw in enumerate(raw_rows):
            raw = {key: "" if value is None else value for key, value in raw.items()}
            identity_column = self.schema.get("sample_id", "sample_id")
            sample_id = str(raw.get(identity_column, ""))
            if not sample_id:
                identity = {
                    key: raw.get(self._column(key, spec), "") for key, spec in self.input_schema.items()
                }
                sample_id = hashlib.sha256(
                    json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest()[:24]
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id {sample_id!r} in {self.name}")
            seen.add(sample_id)
            row = dict(raw, sample_id=sample_id, manifest_name=self.name, row_index=index)
            row["_raw"] = dict(raw)
            assets = {}
            for section, default_type in ((self.input_schema, "image"), (self.target_schema, "categorical")):
                for key, spec in section.items():
                    spec = self._spec(spec)
                    if spec.get("type", default_type) in {
                        "image",
                        "mask",
                        "segmentation",
                        "indexed_mask",
                        "binary_mask",
                    }:
                        value = raw.get(self._column(key, spec), "")
                        if not self._missing(value, spec):
                            assets[key] = str(confined_path(value, self.data_root))
            row["_assets"] = assets
            for key, spec in self.target_schema.items():
                spec = self._spec(spec)
                value = raw.get(self._column(key, spec), "")
                if spec.get("type", "categorical") in {"categorical", "class"}:
                    if self._missing(value, spec):
                        if not spec.get("allow_missing", True):
                            raise ValueError(f"Missing required target {key} in {self.name}/{sample_id}")
                        row[key] = -1
                    else:
                        if key not in self.class_maps:
                            raise ValueError(f"No explicit training class map for categorical target {key}")
                        if str(value) not in self.class_maps[key]:
                            raise ValueError(f"Unknown label {value!r} for {key} in {self.name}")
                        row[key] = self.class_maps[key][str(value)]
            row["_target_allowed"] = {key: self._allowed(key) for key in self.target_schema}
            self.records.append(row)
        if not self.records:
            raise ValueError(f"Empty dataset: {self.name}")
        self._tokenizers = {}
        for key, spec in self.input_schema.items():
            spec = self._spec(spec)
            if spec.get("type") == "text":
                tokenizer = spec.get("tokenizer")
                if tokenizer:
                    from hydra.utils import instantiate

                    self._tokenizers[key] = instantiate(tokenizer)
                else:
                    self._tokenizers[key] = ByteTokenizer(spec.get("max_length", 64))
        if validate:
            self.validate_data()

    @staticmethod
    def _spec(spec):
        return {"column": spec} if isinstance(spec, str) else spec

    def _column(self, key, spec):
        spec = self._spec(spec)
        return self.mapping.get(key, spec.get("column", key))

    @staticmethod
    def _missing(value, spec):
        return value in spec.get("missing_tokens", [""])

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _allowed(self, key):
        if isinstance(self.supervision, bool):
            return self.supervision
        if self.supervision is not None:
            return bool(self.supervision.get(key, False))
        return self.role == "train"

    @staticmethod
    def _decode_image(path):
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            array = np.array(image, copy=True)
        return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0

    @staticmethod
    def _decode_mask(path, spec):
        with Image.open(path) as source:
            if source.format != "PNG":
                raise ValueError(f"Categorical masks require lossless PNG: {path}")
            array = np.array(source, copy=True)
        if array.ndim != 2:
            raise ValueError(f"Indexed masks must be single channel class IDs: {path}")
        if array.dtype.kind not in {"b", "u", "i"}:
            raise ValueError(f"Mask must contain integer IDs: {path}")
        value = torch.from_numpy(array.astype(np.int64))
        remap = spec.get("remap")
        if remap:
            original = value.clone()
            mapped = torch.full_like(value, int(spec.get("ignore_index", 255)))
            for source, target in remap.items():
                mapped[original == int(source)] = int(target)
            unknown = set(original.unique().tolist()) - {int(key) for key in remap}
            if unknown and spec.get("unknown_policy", "error") == "error":
                raise ValueError(f"Mask IDs missing from remap: {sorted(unknown)}")
            value = mapped
        return value

    def _raw_sample(self, index):
        row = self.records[index]
        sample = {
            "sample_id": row["sample_id"],
            "manifest_name": self.name,
            "inputs": {},
            "input_present": {},
            "views": {},
            "targets": {},
            "target_valid": {},
            "supervision_allowed": {},
            "metadata": {},
            "_mask_parents": {},
        }
        primary_size = None
        for key, declaration in self.input_schema.items():
            spec = self._spec(declaration)
            kind = spec.get("type", "image")
            value = row["_raw"].get(self._column(key, spec), "")
            present = not self._missing(value, spec)
            sample["input_present"][key] = torch.tensor(present)
            if not present and not spec.get("allow_missing", False):
                raise ValueError(f"Missing required input {key} in {self.name}/{row['sample_id']}")
            if not present and self.missing_modality_policy != "presence":
                raise ValueError(f"Missing input {key} requires missing_modality_policy=presence")
            if kind == "image":
                tensor = (
                    self._decode_image(row["_assets"][key])
                    if present
                    else torch.zeros(3, *(primary_size or (32, 32)))
                )
                if primary_size is None and present:
                    primary_size = tuple(tensor.shape[-2:])
                sample["inputs"][key] = tensor
            elif kind in {"mask", "binary_mask"}:
                parent = spec.get("image_key", spec.get("parent", next(iter(self.input_schema))))
                sample["_mask_parents"][key] = parent
                tensor = (
                    self._decode_mask(row["_assets"][key], spec)
                    if present
                    else torch.zeros(*(primary_size or (32, 32)), dtype=torch.long)
                )
                values = set(tensor.unique().tolist())
                allowed = set(map(int, spec.get("values", [0, 1])))
                if not values <= allowed:
                    raise ValueError(f"Input mask {key} has values {values}; configure explicit remap/values")
                sample["inputs"][key] = tensor.unsqueeze(0).float()
            elif kind == "text":
                sample["inputs"][key] = self._tokenizers[key](str(value) if present else "")
            elif kind in {"float", "numeric", "tabular"}:
                sample["inputs"][key] = torch.tensor(float(value) if present else 0.0)
            else:
                raise ValueError(f"Unsupported input type {kind}")
        if primary_size is None:
            raise ValueError("At least one present image input is required")
        for key, parent in sample["_mask_parents"].items():
            if sample["inputs"][key].shape[-2:] != sample["inputs"][parent].shape[-2:]:
                raise ValueError(f"Input mask {key} does not match image {parent} spatial size")
        for key, declaration in self.target_schema.items():
            spec = self._spec(declaration)
            kind = spec.get("type", "categorical")
            value = row["_raw"].get(self._column(key, spec), "")
            present = not self._missing(value, spec)
            if not present and not spec.get("allow_missing", True):
                raise ValueError(f"Missing required target {key}")
            if kind in {"categorical", "class"}:
                target = torch.tensor(row[key], dtype=torch.long)
                valid = torch.tensor(present)
            elif kind in {"segmentation", "indexed_mask", "mask", "binary_mask"}:
                ignore = int(spec.get("ignore_index", 255))
                target = (
                    self._decode_mask(row["_assets"][key], spec)
                    if present
                    else torch.full(primary_size, ignore, dtype=torch.long)
                )
                if tuple(target.shape[-2:]) != primary_size:
                    raise ValueError(f"Segmentation target {key} has different spatial size from image")
                valid = target != ignore
                count = int(spec.get("num_classes", len(self.class_maps.get(key, {})) or 2))
                if ((target[valid] < 0) | (target[valid] >= count)).any():
                    raise ValueError(
                        f"Segmentation target {key} contains IDs outside 0..{count - 1} and ignore={ignore}"
                    )
                if spec.get("validity_column"):
                    validity_path = row.get(spec["validity_column"], "")
                    if not validity_path:
                        valid = torch.zeros_like(valid)
                    else:
                        validity = self._decode_mask(confined_path(validity_path, self.data_root), {})
                        if validity.shape != target.shape:
                            raise ValueError("Spatial validity shape differs from segmentation mask")
                        if not set(validity.unique().tolist()) <= {0, 1}:
                            raise ValueError("Spatial validity must contain 0/1")
                        valid &= validity.bool()
            elif kind in {"float", "regression"}:
                target = torch.tensor(float(value) if present else 0.0)
                valid = torch.tensor(present)
            else:
                raise ValueError(f"Unsupported target type {kind}; multilabel requires an explicit adapter")
            sample["targets"][key] = target
            sample["target_valid"][key] = valid
            sample["supervision_allowed"][key] = torch.tensor(self._allowed(key))
        metadata = self.schema.get("metadata", [])
        if isinstance(metadata, dict):
            sample["metadata"].update(
                {key: row.get(self._column(key, spec), "") for key, spec in metadata.items()}
            )
        else:
            sample["metadata"].update({key: row.get(key, "") for key in metadata})
        sample["metadata"]["original_size"] = torch.tensor(primary_size)
        sample["metadata"]["row_index"] = torch.tensor(index)
        sample["metadata"]["image_path"] = row["_assets"].get(next(iter(self.input_schema)), "")
        return sample

    def __getitem__(self, index):
        sample = self._raw_sample(index)
        if self.role != "train":
            dense = {key: value.clone() for key, value in sample["targets"].items() if value.ndim >= 2}
            if dense:
                sample["metadata"]["original_targets"] = dense
                sample["metadata"]["original_target_valid"] = {
                    key: sample["target_valid"][key].clone() for key in dense
                }
        seed = int.from_bytes(
            hashlib.sha256(
                f"{self.seed}:{self.epoch if self.role == 'train' else 0}:{self.name}:{sample['sample_id']}".encode()
            ).digest()[:8],
            "little",
        ) % (2**63 - 1)
        if self.ssl:
            first = self.transform(
                copy.deepcopy(sample), generator=torch.Generator().manual_seed(seed), seed=seed
            )
            second_seed = (seed + 104729) % (2**63 - 1)
            second = self.transform(
                copy.deepcopy(sample), generator=torch.Generator().manual_seed(second_seed), seed=second_seed
            )
            primary = next(
                key
                for key, spec in self.input_schema.items()
                if self._spec(spec).get("type", "image") == "image"
            )
            first["views"] = {"view1": first["inputs"][primary].clone(), "view2": second["inputs"][primary]}
            sample = first
        else:
            sample = self.transform(sample, generator=torch.Generator().manual_seed(seed), seed=seed)
        sample.pop("_mask_parents", None)
        return sample

    def validate_data(self):
        for index in range(len(self)):
            self._raw_sample(index)
        return {"name": self.name, "role": self.role, "rows": len(self)}

    def provenance(self, content_hash=True):
        assets = {}
        for row in self.records:
            for path in row["_assets"].values():
                if path not in assets:
                    assets[path] = sha256_file(path) if content_hash else None
            for declaration in self.target_schema.values():
                spec = self._spec(declaration)
                if spec.get("validity_column") and row.get(spec["validity_column"]):
                    path = str(confined_path(row[spec["validity_column"]], self.data_root))
                    assets[path] = sha256_file(path) if content_hash else None
        return {
            "name": self.name,
            "role": self.role,
            "path": str(self.manifest_path) if self.manifest_path else None,
            "sha256": sha256_file(self.manifest_path) if self.manifest_path else None,
            "data_root": str(self.data_root),
            "rows": len(self),
            "assets": assets,
            "sample_ids": [row["sample_id"] for row in self.records],
            "supervision_allowed": {key: self._allowed(key) for key in self.target_schema},
        }


class ConcatManifestDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = list(datasets)
        self.records = [row for dataset in self.datasets for row in dataset.records]
        self.locations = [(dataset, index) for dataset in self.datasets for index in range(len(dataset))]

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, index):
        dataset, position = self.locations[index]
        return dataset[position]

    def set_epoch(self, epoch):
        for dataset in self.datasets:
            dataset.set_epoch(epoch)


def common_collate(samples):
    from torch.utils.data._utils.collate import default_collate

    if not samples:
        raise ValueError("Cannot collate empty batch")
    originals = {}
    prepared = []
    for sample in samples:
        item = dict(sample)
        item["metadata"] = dict(sample.get("metadata", {}))
        for section in ("original_targets", "original_target_valid"):
            if section in item["metadata"]:
                originals.setdefault(section, []).append(item["metadata"].pop(section))
        prepared.append(item)
    try:
        result = default_collate(prepared)
        for section, rows in originals.items():
            if len(rows) != len(samples):
                raise ValueError("Original dense targets must share a declared schema")
            result["metadata"][section] = {key: [row[key] for row in rows] for key in rows[0]}
        return result
    except (RuntimeError, KeyError) as exc:
        raise ValueError(
            "Samples must share declared schema and transformed image sizes; configure resize/padding"
        ) from exc
