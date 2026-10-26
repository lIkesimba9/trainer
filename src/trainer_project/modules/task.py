from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from copy import deepcopy

import torch
from hydra.utils import instantiate
from lightning.pytorch import LightningModule
from torch import distributed as dist

from trainer_project.experiment.artifacts import environment
from trainer_project.experiment.runtime import capture_rng, config_hash, isolated_rng, local_path, restore_rng
from trainer_project.schedulers import ScheduleController


class _InferenceAdapter(torch.nn.Module):
    def __init__(self, model, objective, target):
        super().__init__()
        self.model, self.objective, self.target = model, objective, target
        self.training = False

    def forward(self, inputs, views=None):
        from trainer_project.objectives import objective_inference_outputs

        return objective_inference_outputs(
            self.objective, self.model(inputs, views=views), primary_target=self.target
        )


class ResearchTask(LightningModule):
    """One Lightning lifecycle; replaceable models, objectives and evaluators."""

    def __init__(self, config, model=None, objective=None, run_dir=None):
        super().__init__()
        self.experiment_config = deepcopy(config)
        self.model = model if model is not None else instantiate(config["model"])
        self.objective = objective if objective is not None else instantiate(config["objective"])
        if config.get("freeze_schedule"):
            from trainer_project.callbacks.freeze import enable_scheduled_parameters

            enable_scheduled_parameters(self, config["freeze_schedule"])
        self.run_dir = str(run_dir) if run_dir else None
        gpu = config.get("gpu_transforms")
        self.gpu_transforms = instantiate(gpu) if gpu else None
        self.successful_updates = 0
        self.microsteps = 0
        self.validation_events = 0
        self.controller = None
        self._pending = None
        self._rank_states = []
        self.data_identity = {}
        self.last_metrics = {}
        self.last_records = []
        self._losses = defaultdict(lambda: [0.0, 0.0])

    def forward(self, inputs, views=None):
        tta = self.experiment_config.get("tta", {})
        if not self.training and tta.get("enabled", False):
            from trainer_project.models import tta_forward

            adapter = _InferenceAdapter(
                self.model,
                self.objective,
                self.experiment_config.get("evaluation", {}).get("target_key", "label"),
            )
            return tta_forward(adapter, inputs, tta)
        outputs = self.model(inputs, views=views)
        if not self.training:
            from trainer_project.objectives import objective_inference_outputs

            if self.experiment_config["task"] == "ssl" and hasattr(self.model, "extract_features"):
                outputs["embeddings"] = self.model.extract_features(inputs)
            outputs = objective_inference_outputs(
                self.objective,
                outputs,
                primary_target=self.experiment_config.get("evaluation", {}).get("target_key", "label"),
            )
        return outputs

    def _outputs(self, batch, training=False):
        if training and self.gpu_transforms is not None:
            batch = self.gpu_transforms(batch)
        views = batch.get("views") or None
        if self.experiment_config["task"] == "ssl" and views is None and hasattr(self.model, "update_target"):
            key = getattr(self.model, "input_key", "image")
            views = {"view1": batch["inputs"][key], "view2": batch["inputs"][key]}
        return self(batch["inputs"], views=views)

    def _evaluator(self):
        from trainer_project.evaluation import EvaluationAccumulator

        options = dict(self.experiment_config.get("evaluation", {}))
        options.pop("test_during_fit", None)
        options.pop("per_manifest", None)
        options.pop("pooled", None)
        options.pop("enabled", None)
        target = options.pop("_target_", None)
        if target:
            return instantiate(
                {"_target_": target, **options},
                task=self.experiment_config["task"],
                class_maps=self.experiment_config["data"]["class_maps"],
                _convert_="all",
            )
        return EvaluationAccumulator(
            task=self.experiment_config["task"],
            class_maps=self.experiment_config["data"]["class_maps"],
            **options,
        )

    def on_train_epoch_start(self):
        self.trainer.datamodule.set_epoch(self.current_epoch)
        self.train_evaluator = self._evaluator()
        self.epoch_started = time.perf_counter()
        self._epoch_microsteps = 0

    def training_step(self, batch, batch_idx):
        for group in self.optimizers().param_groups:
            self.log(f"train/lr/{group.get('name', 'group')}", group["lr"], on_step=True, on_epoch=False)
        # CombinedLoader keeps named sources; merge batches without losing any field.
        if "inputs" not in batch:

            def concatenate(values):
                if isinstance(values[0], dict):
                    return {key: concatenate([value[key] for value in values]) for key in values[0]}
                if isinstance(values[0], torch.Tensor):
                    return torch.cat(values, dim=0)
                return [item for value in values for item in value]

            batch = concatenate([value for value in batch.values() if value is not None])
        outputs = self._outputs(batch, training=True)
        result = self.objective(
            outputs,
            batch,
            {"stage": "train", "epoch": self.current_epoch, "successful_updates": self.successful_updates},
        )
        if result.loss.ndim != 0 or not torch.isfinite(result.loss):
            raise ValueError("Objective must return a finite scalar loss")
        self.microsteps += 1
        self._epoch_microsteps += 1
        self.log(
            "train/loss",
            result.loss.detach(),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            batch_size=len(batch["sample_id"]),
        )
        for key, value in result.components.items():
            self.log(
                f"train/components/{key}",
                value,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
                batch_size=len(batch["sample_id"]),
            )
        for key, value in result.counts.items():
            self.log(
                f"train/counts/{key}",
                float(value),
                on_step=True,
                on_epoch=False,
                batch_size=len(batch["sample_id"]),
            )
        if not batch.get("mixing"):
            from trainer_project.objectives import objective_inference_outputs

            predictions = objective_inference_outputs(
                self.objective,
                outputs,
                primary_target=self.experiment_config.get("evaluation", {}).get("target_key", "label"),
                apply_posthoc=False,
            )
            self.train_evaluator.update(predictions, batch, "train", "stream")
        return result.loss

    def _post_optimizer_step(self, optimizer, args, kwargs):
        self.successful_updates += 1
        self.controller.update()
        if hasattr(self.model, "update_target"):
            self.model.update_target(self.successful_updates, self.controller.total_updates)

    def configure_optimizers(self):
        options = dict(self.experiment_config.get("parameter_groups", {}))
        groups = []
        partition = {}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            component = (
                "objective"
                if name.startswith("objective.")
                else "backbone"
                if any(part in name for part in ("backbone", "encoder"))
                else "head"
            )
            override = options.get(component, {})
            no_decay = options.get("exclude_bias_norm_decay", True) and (
                parameter.ndim <= 1 or name.endswith("bias")
            )
            key = (component, no_decay)
            if key not in partition:
                spec = self.experiment_config["optimizer"]
                partition[key] = {
                    "params": [],
                    "lr": override.get("lr", spec.get("lr", 0.001)),
                    "weight_decay": 0.0
                    if no_decay
                    else override.get("weight_decay", spec.get("weight_decay", 0.0)),
                    "name": component + ("_no_decay" if no_decay else ""),
                }
            partition[key]["params"].append(parameter)
        groups = list(partition.values())
        spec = dict(self.experiment_config["optimizer"])
        spec.pop("_partial_", None)
        optimizer = instantiate(spec, _partial_=True)(groups)
        horizon = self.experiment_config["runtime"]["total_optimizer_updates"]
        self.controller = ScheduleController(
            optimizer,
            self.experiment_config.get("scheduler"),
            horizon,
            self.experiment_config["runtime"]["updates_per_epoch"],
        )
        optimizer.register_step_post_hook(self._post_optimizer_step)
        if self._pending and self._pending.get("scheduler"):
            self.controller.load_state_dict(self._pending["scheduler"])
        return optimizer

    def on_train_start(self):
        if hasattr(self, "_initial_rng"):
            restore_rng(self._initial_rng)
        if self._pending:
            states = self._pending["rank_states"]
            if len(states) != self.trainer.world_size:
                raise ValueError("Checkpoint world size differs; branch explicitly")
            restore_rng(states[self.global_rank]["rng"])

    def on_train_epoch_end(self):
        metrics = self.train_evaluator.compute()
        if self.controller:
            self.controller.epoch()
        metrics["train/epoch_seconds"] = time.perf_counter() - self.epoch_started
        metrics["train/successful_updates"] = self.successful_updates
        metrics["train/microsteps"] = self.microsteps
        metrics["train/trainable_parameters"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if self.device.type == "cuda":
            metrics["train/gpu_memory_bytes"] = torch.cuda.max_memory_allocated(self.device)
        self._publish(metrics)
        state = {"rng": capture_rng(), "datamodule": self.trainer.datamodule.state_dict()}
        self._rank_states = [state]
        if dist.is_available() and dist.is_initialized():
            self._rank_states = [None] * dist.get_world_size()
            dist.all_gather_object(self._rank_states, state)

    def on_validation_epoch_start(self):
        self._eval_rng = capture_rng()
        self.evaluator = self._evaluator()
        self._losses = defaultdict(lambda: [0.0, 0.0])
        cadence = self.experiment_config.get("evaluation", {}).get("test_during_fit", {})
        frequency = int(cadence.get("every_n_validation_events", 1))
        self._test_active = (
            not self.trainer.sanity_checking
            and cadence.get("enabled", True)
            and (self.validation_events + 1) % frequency == 0
        )

    def _evaluation_step(self, batch, role, name):
        outputs = self._outputs(batch)
        evaluation_batch = dict(batch)
        # Supervision permissions govern updates, while evaluation may inspect all valid held-out labels.
        evaluation_batch["supervision_allowed"] = {
            k: torch.ones_like(v, dtype=torch.bool) for k, v in batch.get("target_valid", {}).items()
        }
        loss_outputs = outputs
        indices = batch.get("metadata", {}).get("row_index")
        if indices is not None and self.trainer.world_size > 1:
            # Built-in evaluation samplers enumerate rows then append padding and
            # shard by rank. Canonical ownership removes padded copies while all
            # ranks still execute the same objective collectives.
            owner = (
                torch.as_tensor(indices, device=self.device).remainder(self.trainer.world_size)
                == self.global_rank
            )
            evaluation_batch["target_valid"] = {}
            for key, valid in batch.get("target_valid", {}).items():
                shape = (len(owner),) + (1,) * (valid.ndim - 1)
                evaluation_batch["target_valid"][key] = valid & owner.reshape(shape)
            evaluation_batch["input_present"] = {
                key: present & owner for key, present in batch.get("input_present", {}).items()
            }
            if "ssl_patch_mask" in outputs:
                loss_outputs = dict(outputs)
                loss_outputs["ssl_patch_mask"] = outputs["ssl_patch_mask"] & owner[:, None]
        result = self.objective(
            loss_outputs,
            evaluation_batch,
            {"stage": "eval", "epoch": self.current_epoch, "successful_updates": self.successful_updates},
        )
        count = float(next(iter(result.counts.values()), len(batch["sample_id"])))
        for key in (f"{role}/{name}/loss", f"{role}/pooled/loss"):
            self._losses[key][0] += float(result.loss.detach()) * count
            self._losses[key][1] += count
        self.evaluator.update(outputs, batch, role, name)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        plan = self.trainer.datamodule.eval_plan[dataloader_idx]
        if plan["role"] == "test" and not self._test_active:
            return
        self._evaluation_step(batch, plan["role"], plan["name"])

    def _finish_evaluation(self, stage):
        metrics = self.evaluator.compute()
        for key in sorted(self._losses):
            pair = torch.tensor(self._losses[key], device=self.device, dtype=torch.float64)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(pair)
            if pair[1] > 0:
                metrics[key] = float(pair[0] / pair[1])
        if stage == "val" and not self.trainer.sanity_checking:
            self.validation_events += 1
            with isolated_rng():
                metrics.update(self._ssl_metrics())
            if self.controller:
                self.controller.validation(metrics)
        if not self.trainer.sanity_checking:
            self.last_metrics, self.last_records = metrics, self.evaluator.records
            self._publish(metrics)
            if self.run_dir and self.trainer.is_global_zero:
                folder = local_path(self.run_dir) / "predictions"
                folder.mkdir(parents=True, exist_ok=True)
                path = (
                    folder
                    / f"{stage}-epoch-{self.current_epoch:04d}-event-{self.validation_events:04d}.jsonl"
                )
                self.evaluator.write_predictions(path)
        restore_rng(self._eval_rng)

    def on_validation_epoch_end(self):
        self._finish_evaluation("val")

    def on_test_epoch_start(self):
        self._eval_rng = capture_rng()
        self.evaluator = self._evaluator()
        self._losses = defaultdict(lambda: [0.0, 0.0])

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self._evaluation_step(batch, "test", self.trainer.datamodule.datasets["test"][dataloader_idx].name)

    def on_test_epoch_end(self):
        self._finish_evaluation("test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch["inputs"])

    def _ssl_metrics(self):
        settings = self.experiment_config.get("ssl_evaluation", {})
        if self.experiment_config["task"] != "ssl" or not settings.get("enabled", False):
            return {}
        if self.validation_events % settings.get("every_n_validation_events", 1):
            return {}
        from lightning.fabric.utilities.apply_func import move_data_to_device

        from trainer_project.evaluation import EvaluationAccumulator, KnnEvaluator, LinearProbeEvaluator
        from trainer_project.evaluation.metrics import class_names

        target = settings.get("target_key", "label")
        maps = self.experiment_config["data"]["class_maps"]
        if target not in maps:
            raise ValueError("SSL evaluation target requires an explicit class mapping")
        num_classes = len(class_names(maps[target]))
        queries = [row for row in self.evaluator.records if row["role"] == "val"]
        if not queries:
            raise ValueError("SSL evaluation requires validation queries")
        identity_fields = settings.get("identity_fields", [settings.get("identity_key", "sample_id")])
        group_field = settings.get("group_field")

        def identity(row):
            values = tuple(row.get(field, row.get("metadata", {}).get(field)) for field in identity_fields)
            if any(value is None for value in values):
                raise ValueError("SSL evaluation identity field is missing")
            return values

        active = []
        for name, cls in (("knn", KnnEvaluator), ("linear_probe", LinearProbeEvaluator)):
            spec = settings.get(name, {})
            if (
                spec.get("enabled", False)
                and self.validation_events % int(spec.get("every_n_validation_events", 1)) == 0
            ):
                active.append((name, cls, spec))
        if not active:
            return {}
        needs_reference = any(
            name == "linear_probe" or spec.get("protocol", "reference_query") != "val_leave_one_out"
            for name, _, spec in active
        )
        refs = []
        if needs_reference:
            reference_config = settings.get("reference")
            if not reference_config:
                raise ValueError("SSL evaluation requires an explicit labeled reference")
            if reference_config.get("split", "train") in {"val", "test"}:
                raise ValueError("Held-out probe reference cannot use validation/test labels")
            reference = EvaluationAccumulator("ssl", maps, target_key=target)
            datamodule = self.trainer.datamodule
            declarations = reference_config.get("manifests")
            external = reference_config.get("path") is not None or (
                declarations and isinstance(declarations[0], dict)
            )
            if external:
                declarations = declarations or [
                    {
                        key: value
                        for key, value in reference_config.items()
                        if key not in {"split", "labeled_only", "data_root"}
                    }
                ]
                loaders = []
                for declaration in declarations:
                    spec = dict(declaration)
                    dataset_target = spec.pop("_target_", "trainer_project.data.CsvDataset")
                    dataset = instantiate(
                        {"_target_": dataset_target, "_recursive_": False},
                        data_root=reference_config.get("data_root", str(datamodule.data_root)),
                        schema=datamodule.schema,
                        class_maps=maps,
                        role="reference",
                        transform=datamodule.transforms.get("eval"),
                        ssl=False,
                        **spec,
                    )
                    loaders.append(datamodule._loader(dataset))
            else:
                loaders = datamodule.reference_dataloader(
                    declarations, reference_config.get("split", "train")
                )
            with torch.no_grad():
                for loader in loaders:
                    for batch in loader:
                        batch = move_data_to_device(batch, self.device)
                        outputs = {"embeddings": self.model.extract_features(batch["inputs"])}
                        reference.update(outputs, batch, "reference", batch["manifest_name"][0])
            reference.gather()
            refs = [row for row in reference.records if row["target_valid"].get(target, False)]
            if not refs:
                raise ValueError("Labeled reference is empty; validation labels are never a fallback")
        query_features = torch.tensor([row["embeddings"] for row in queries])
        result = {}
        for name, cls, spec in active:
            options = {
                key: value
                for key, value in spec.items()
                if key not in {"enabled", "_target_", "every_n_validation_events"}
            }
            options.setdefault("num_classes", num_classes)
            evaluator = (
                instantiate({"_target_": spec["_target_"], **options})
                if "_target_" in spec
                else cls(**options)
            )
            bank = refs
            if name == "knn" and options.get("protocol") == "val_leave_one_out":
                if len({row["manifest_name"] for row in queries}) != 1:
                    raise ValueError("val_leave_one_out requires a single validation manifest")
                bank = [row for row in queries if row["target_valid"].get(target, False)]
                if not bank:
                    raise ValueError("Leave-one-out validation has no labeled neighbors")
            features = torch.tensor([row["embeddings"] for row in bank])
            labels = torch.tensor([row["targets"][target] for row in bank])
            kwargs = {}
            if group_field:
                kwargs = {
                    "reference_groups": [
                        row.get(group_field, row.get("metadata", {}).get(group_field)) for row in bank
                    ],
                    "query_groups": [
                        row.get(group_field, row.get("metadata", {}).get(group_field)) for row in queries
                    ],
                }
                if any(value is None for values in kwargs.values() for value in values):
                    raise ValueError("SSL evaluation group field is missing")
            probabilities = evaluator.predict(
                features,
                labels,
                query_features,
                reference_ids=[identity(row) for row in bank],
                query_ids=[identity(row) for row in queries],
                task="ssl",
                query_role="val",
                **kwargs,
            )
            accumulator = EvaluationAccumulator("classification", maps, target_key=target)
            rows = deepcopy(queries)
            for row, probability in zip(rows, probabilities):
                row["probabilities"] = {target: probability.tolist()}
                row["logits"] = {target: probability.clamp_min(1e-20).log().tolist()}
                row.setdefault("provenance", {})["representation_evaluator"] = {
                    "name": name,
                    "options": options,
                    "reference": settings.get("reference"),
                    "protocol": options.get("protocol", "reference_query"),
                    "identity_fields": identity_fields,
                }
            accumulator.add_records(rows)
            result.update(
                {
                    key.replace("val/", f"val/{name}/", 1): value
                    for key, value in accumulator.compute().items()
                }
            )
            if self.run_dir and self.trainer.is_global_zero:
                folder = local_path(self.run_dir) / "predictions"
                folder.mkdir(parents=True, exist_ok=True)
                accumulator.write_predictions(
                    folder
                    / f"val-{name}-epoch-{self.current_epoch:04d}-event-{self.validation_events:04d}.jsonl"
                )
        return result

    def _publish(self, metrics):
        for key, value in metrics.items():
            if value is not None and isinstance(value, (float, int)) and math.isfinite(float(value)):
                self.log(
                    key, float(value), on_step=False, on_epoch=True, sync_dist=self.trainer.world_size > 1
                )
        if self.run_dir and self.trainer.is_global_zero:
            with (local_path(self.run_dir) / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                json.dump(
                    {"epoch": self.current_epoch, "step": self.successful_updates, **metrics},
                    stream,
                    default=str,
                )
                stream.write("\n")

    def on_save_checkpoint(self, checkpoint):
        checkpoint["experiment"] = {
            "config": self.experiment_config,
            "config_hash": config_hash(self.experiment_config),
            "class_maps": self.experiment_config["data"]["class_maps"],
            "successful_updates": self.successful_updates,
            "microsteps": self.microsteps,
            "validation_events": self.validation_events,
            "rank_states": self._rank_states,
            "scheduler": self.controller.state_dict() if self.controller else None,
            "data_identity": self.data_identity,
            "environment": environment(),
            "boundary": "epoch"
            if self._epoch_microsteps >= self.trainer.num_training_batches
            else "partial_epoch",
            "code_manifest": getattr(self, "code_manifest", {}),
            "weights_provenance": getattr(self, "weights_provenance", {}),
            "world_size": self.trainer.world_size,
        }

    def on_load_checkpoint(self, checkpoint):
        self._pending = checkpoint["experiment"]
        self.successful_updates = self._pending["successful_updates"]
        self.microsteps = self._pending["microsteps"]
        self.validation_events = self._pending["validation_events"]
