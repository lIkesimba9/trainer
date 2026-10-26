"""Prediction artifacts and distributed-safe evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from .metrics import class_names, classification_metrics, segmentation_metrics


def project_path(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    root = Path(__file__).resolve().parents[3]
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path must remain inside project: {root}")
    return resolved


def json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _row(value: Any, index: int, batch_size: int) -> Any:
    if isinstance(value, Mapping):
        return {key: _row(item, index, batch_size) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return json_value(value[index] if value.ndim and len(value) == batch_size else value)
    if isinstance(value, np.ndarray):
        return json_value(value[index] if value.ndim and len(value) == batch_size else value)
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        return json_value(value[index])
    return json_value(value)


def read_predictions(path: str | Path) -> list[dict[str, Any]]:
    path = project_path(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("records", []) if isinstance(payload, dict) else payload
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_predictions(
    records: Iterable[dict[str, Any]], path: str | Path, provenance: dict | None = None
) -> Path:
    """Create an artifact without replacing any existing file."""
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            row = dict(record)
            if provenance:
                row["provenance"] = {**row.get("provenance", {}), **provenance}
            stream.write(json.dumps(json_value(row), ensure_ascii=False, allow_nan=False) + "\n")
    return path


class EvaluationAccumulator:
    def __init__(
        self,
        task: str,
        class_maps: dict[str, Any] | None = None,
        target_key: str = "label",
        pooled_policy: str = "observations",
        identity_key: str = "sample_id",
        ignore_index: int = 255,
        zero_division: float = 0.0,
        manifest_macro: bool = False,
        slice_fields: Iterable[str] = (),
        provenance: dict | None = None,
        empty_mask_policy: str = "exclude",
        **_: Any,
    ):
        if pooled_policy not in {"observations", "deduplicate"}:
            raise ValueError("pooled_policy must be observations or deduplicate")
        self.task, self.class_maps, self.target_key = str(task), class_maps or {}, target_key
        self.pooled_policy, self.identity_key, self.ignore_index = pooled_policy, identity_key, ignore_index
        self.zero_division, self.manifest_macro = zero_division, manifest_macro
        self.slice_fields, self.provenance = list(slice_fields), provenance or {}
        self.empty_mask_policy = empty_mask_policy
        self.records: list[dict[str, Any]] = []
        self.details: dict[str, Any] = {}
        self._gathered = False

    def reset(self) -> None:
        self.records.clear()
        self.details.clear()
        self._gathered = False

    def add_records(self, rows: Iterable[dict[str, Any]]) -> None:
        self.records.extend(copy.deepcopy(list(rows)))
        self._gathered = False

    def update(self, outputs: dict[str, Any], batch: dict[str, Any], role: str, name: str) -> None:
        ids = batch["sample_id"]
        batch_size = len(ids)
        logits = outputs.get("logits")
        if isinstance(logits, torch.Tensor):
            logits = {self.target_key: logits}
        probabilities = outputs.get("probabilities", {})
        if isinstance(probabilities, torch.Tensor):
            probabilities = {self.target_key: probabilities}
        dense = outputs.get("dense_logits")
        for index in range(batch_size):
            manifest_names = batch.get("manifest_name")
            source_name = manifest_names[index] if manifest_names is not None else name
            record: dict[str, Any] = {
                "sample_id": str(ids[index]),
                "manifest_name": str(source_name),
                "role": role,
                "task": self.task,
                "targets": _row(batch.get("targets", {}), index, batch_size),
                "target_valid": _row(batch.get("target_valid", {}), index, batch_size),
                "supervision_allowed": _row(batch.get("supervision_allowed", {}), index, batch_size),
                "metadata": _row(batch.get("metadata", {}), index, batch_size),
                "input_present": _row(batch.get("input_present", {}), index, batch_size),
                "class_maps": json_value(self.class_maps),
                "logits": {},
                "probabilities": {},
                "provenance": copy.deepcopy(self.provenance),
                "evaluation_protocol": self.protocol(),
                "protocol_hash": self.protocol_hash(),
            }
            record["row_index"] = record["metadata"].get("row_index", index)
            for key, tensor in (logits or {}).items():
                item = tensor[index].detach().float().cpu()
                record["logits"][key] = item.tolist()
                record["probabilities"][key] = item.softmax(dim=-1).tolist()
            for key, tensor in probabilities.items():
                record["probabilities"][key] = json_value(tensor[index])
            for key in ("features", "embeddings"):
                if isinstance(outputs.get(key), torch.Tensor):
                    record[key] = json_value(outputs[key][index])
            if dense is not None:
                self._dense_record(record, dense[index].detach().float().cpu())
            self.records.append(record)
        self._gathered = False

    def _dense_record(self, record: dict, logits: torch.Tensor) -> None:
        targets = record["targets"]
        key = (
            self.target_key
            if self.target_key in targets
            else next(
                (
                    candidate
                    for candidate in ("segmentation_target", "segmentation_mask", "mask")
                    if candidate in targets
                ),
                self.target_key,
            )
        )
        target = torch.as_tensor(targets[key]) if key in targets else None
        original_size = record["metadata"].get("original_size")
        # Dataset targets can be resized. Evaluation compares in original coordinates
        # when the adapter retained its original target and validity.
        if "original_targets" in record["metadata"] and key in record["metadata"]["original_targets"]:
            target = torch.as_tensor(record["metadata"]["original_targets"][key])
        size = (
            tuple(target.shape[-2:])
            if target is not None and target.ndim >= 2
            else (tuple(original_size) if original_size else tuple(logits.shape[-2:]))
        )
        logits = F.interpolate(logits.unsqueeze(0), size=size, mode="bilinear", align_corners=False)[0]
        if logits.shape[0] == 1:
            positive = logits[0].sigmoid()
            probs = torch.stack((1 - positive, positive))
        else:
            probs = logits.softmax(dim=0)
        prediction = probs.argmax(dim=0)
        record["dense_probabilities"] = probs.tolist()
        record["predicted_mask"] = prediction.tolist()
        record["dense_target_key"] = key
        record["original_size"] = list(size)
        if target is None:
            return
        if target.ndim == 3:
            if target.shape[0] != 1:
                raise ValueError(
                    "Overlapping multilabel targets need a multilabel evaluator; conversion to exclusive IDs is forbidden"
                )
            target = target[0]
        valid_value = record["target_valid"].get(key, True)
        if "original_target_valid" in record["metadata"]:
            valid_value = record["metadata"]["original_target_valid"].get(key, valid_value)
        valid = torch.as_tensor(valid_value, dtype=torch.bool)
        if valid.ndim == 0:
            valid = valid.expand_as(target)
        if tuple(valid.shape[-2:]) != tuple(target.shape[-2:]):
            valid = F.interpolate(valid.float()[None, None], size=target.shape[-2:], mode="nearest")[
                0, 0
            ].bool()
        valid = valid & (target != self.ignore_index)
        record["targets"][key] = target.tolist()
        record["target_valid"][key] = valid.tolist()
        count = probs.shape[0]
        if ((target[valid] < 0) | (target[valid] >= count)).any():
            raise ValueError("Segmentation class ID outside output classes")
        confusion = torch.bincount(
            target[valid].long() * count + prediction[valid], minlength=count * count
        ).reshape(count, count)
        record["segmentation_confusion"] = confusion.tolist()
        record["valid_pixels"] = int(valid.sum())
        record["annotated"] = bool(valid.any())

    def gather(self) -> list[dict[str, Any]]:
        if self._gathered:
            return self.records
        if dist.is_available() and dist.is_initialized():
            shards: list[Any] = [None] * dist.get_world_size()
            dist.all_gather_object(shards, self.records)
            rows = [row for shard in shards for row in shard]
        else:
            rows = self.records
        unique: dict[tuple[Any, ...], dict] = {}
        retained: list[dict] = []
        for row in rows:
            if row["role"] == "train":
                retained.append(row)  # repeated sampled observations are real training observations
                continue
            key = (row["role"], row["manifest_name"], row["sample_id"], row.get("row_index"))
            if key not in unique:
                unique[key] = row
                retained.append(row)
            elif unique[key].get("targets") != row.get("targets"):
                raise ValueError(f"Conflicting labels for duplicate evaluation identity {key}")
        self.records = retained
        self._gathered = True
        return self.records

    def _pooled(self, rows: list[dict]) -> list[dict]:
        if self.pooled_policy == "observations":
            return rows
        groups: dict[Any, list[dict]] = defaultdict(list)
        for row in rows:
            identity = row.get(self.identity_key, row.get("metadata", {}).get(self.identity_key))
            if identity is None:
                raise ValueError(f"Missing pooled identity field {self.identity_key}")
            groups[identity].append(row)
        result = []
        for identity, repeats in groups.items():
            item = copy.deepcopy(repeats[0])
            for other in repeats[1:]:
                if item["targets"] != other["targets"] or item["target_valid"] != other["target_valid"]:
                    raise ValueError(f"Conflicting targets during pooled deduplication: {identity}")
                if "segmentation_confusion" in item:
                    if item["predicted_mask"] != other["predicted_mask"]:
                        raise ValueError(
                            "Pooled segmentation duplicates require identical coordinate-aligned predictions"
                        )
            for key in item.get("probabilities", {}):
                item["probabilities"][key] = np.mean(
                    [row["probabilities"][key] for row in repeats], axis=0
                ).tolist()
            result.append(item)
        return result

    def _metrics(self, rows: list[dict]) -> dict[str, Any]:
        details: dict[str, Any] = {"observations": len(rows), "pooled_policy": self.pooled_policy}
        fields = set(self.class_maps)
        fields.update(key for row in rows for key in row.get("probabilities", {}))
        for field in sorted(fields):
            candidates = [row for row in rows if field in row.get("probabilities", {})]
            if not candidates:
                continue
            valid = [
                row
                for row in candidates
                if bool(np.asarray(row.get("target_valid", {}).get(field, False)).all())
                and field in row.get("targets", {})
                and np.asarray(row["targets"][field]).ndim == 0
                and int(row["targets"][field]) >= 0
            ]
            mapping = self.class_maps.get(field) or candidates[0].get("class_maps", {}).get(field)
            names = (
                class_names(mapping)
                if mapping
                else [str(i) for i in range(len(candidates[0]["probabilities"][field]))]
            )
            for row in candidates:
                row_map = row.get("class_maps", {}).get(field)
                if row_map is not None and class_names(row_map) != names:
                    raise ValueError(f"Incompatible class order during pooled evaluation for {field}")
            metric = classification_metrics(
                [int(row["targets"][field]) for row in valid],
                [row["probabilities"][field] for row in valid],
                names,
                zero_division=self.zero_division,
            )
            metric["coverage"] = len(valid) / len(rows) if rows else 0.0
            metric["observations"] = len(rows)
            details[field] = metric
        dense = [row for row in rows if "segmentation_confusion" in row]
        if dense:
            key = dense[0].get("dense_target_key", self.target_key)
            count = len(dense[0]["segmentation_confusion"])
            mapping = self.class_maps.get(key)
            names = class_names(mapping) if mapping else [str(i) for i in range(count)]
            metric = segmentation_metrics(
                np.sum([row["segmentation_confusion"] for row in dense], axis=0),
                names,
                empty_policy=self.empty_mask_policy,
            )
            metric["annotated_samples"] = sum(bool(row.get("annotated")) for row in dense)
            metric["coverage"] = metric["annotated_samples"] / len(rows) if rows else 0.0
            details[key] = metric
        return details

    def compute(self) -> dict[str, float]:
        self.gather()
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in self.records:
            groups[(row["role"], row["manifest_name"])].append(row)
        for role in sorted({row["role"] for row in self.records}):
            groups[(role, "pooled")] = self._pooled([row for row in self.records if row["role"] == role])
        self.details = {}
        flat: dict[str, float] = {}
        for (role, name), rows in groups.items():
            details = self._metrics(rows)
            prefix = f"{role}/{name}"
            self.details[prefix] = details
            flat[f"{prefix}/observations"] = float(len(rows))
            for field, metrics in details.items():
                if not isinstance(metrics, dict):
                    continue
                field_prefix = prefix if field == self.target_key else f"{prefix}/{field}"
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        flat[f"{field_prefix}/{key}"] = float(value)
                for cls, values in metrics.get("per_class", {}).items():
                    for key, value in values.items():
                        if isinstance(value, (int, float)):
                            flat[f"{field_prefix}/class/{cls}/{key}"] = float(value)
            for field in self.slice_fields:
                values = sorted({str(row.get("metadata", {}).get(field, "<missing>")) for row in rows})
                for value in values:
                    sliced = [
                        row for row in rows if str(row.get("metadata", {}).get(field, "<missing>")) == value
                    ]
                    self.details[f"{prefix}/slice/{field}/{value}"] = self._metrics(sliced)
        if self.manifest_macro:
            by_role: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            for key, value in flat.items():
                role, name, suffix = key.split("/", 2)
                if name != "pooled":
                    by_role[role][suffix].append(value)
            for role, suffixes in by_role.items():
                for suffix, values in suffixes.items():
                    flat[f"{role}/manifest_macro/{suffix}"] = float(np.mean(values))
        return flat

    def write_predictions(self, path: str | Path, provenance: dict | None = None) -> Path:
        self.gather()
        for row in self.records:
            row.setdefault("evaluation_protocol", self.protocol())
            row.setdefault("protocol_hash", self.protocol_hash())
        return write_predictions(self.records, path, provenance)

    def protocol(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "class_maps": self.class_maps,
            "target_key": self.target_key,
            "pooled_policy": self.pooled_policy,
            "identity_key": self.identity_key,
            "ignore_index": self.ignore_index,
            "zero_division": self.zero_division,
            "empty_mask_policy": self.empty_mask_policy,
            "manifest_macro": self.manifest_macro,
            "slice_fields": self.slice_fields,
        }

    def protocol_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.protocol(), sort_keys=True).encode()).hexdigest()
