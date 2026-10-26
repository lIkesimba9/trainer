"""Identity-aligned prediction ensembles and explicit OOF provenance checks."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np

from .metrics import class_names


def _identity(row: dict) -> tuple[str, str, str]:
    return str(row["role"]), str(row["manifest_name"]), str(row["sample_id"])


def _ordered(row: dict, canonical: dict[str, list[str]]) -> dict:
    result = copy.deepcopy(row)
    for field, names in canonical.items():
        source = class_names(row.get("class_maps", {}).get(field, names))
        if set(source) != set(names) or len(source) != len(names):
            raise ValueError(f"Ensemble class schemas mismatch for {field}")
        order = [source.index(name) for name in names]
        for output in ("probabilities", "logits"):
            if field in result.get(output, {}):
                value = np.asarray(result[output][field])
                if value.shape[-1] != len(names):
                    raise ValueError(f"Prediction dimension does not match class map for {field}")
                result[output][field] = value[..., order].tolist()
        target = result.get("targets", {}).get(field)
        if target is not None:
            array = np.asarray(target)
            known = (array >= 0) & (array < len(source))
            translated = array.copy()
            for index, name in enumerate(source):
                translated[(array == index) & known] = names.index(name)
            result["targets"][field] = translated.tolist()
        if result.get("dense_target_key") == field and "dense_probabilities" in result:
            value = np.asarray(result["dense_probabilities"])
            result["dense_probabilities"] = value[order].tolist()
    result["class_maps"] = {
        key: {name: i for i, name in enumerate(names)} for key, names in canonical.items()
    }
    return result


def ensemble_predictions(
    members: Sequence[Iterable[dict[str, Any]]],
    weights: Sequence[float] | None = None,
    method: str = "mean_probabilities",
) -> list[dict[str, Any]]:
    if not members:
        raise ValueError("Ensemble requires at least one prediction artifact")
    if method not in {"mean_probabilities", "mean_logits"}:
        raise ValueError("Ensemble method must be mean_probabilities or mean_logits")
    weight = np.asarray(weights if weights is not None else np.ones(len(members)), dtype=float)
    if (
        weight.shape != (len(members),)
        or not np.isfinite(weight).all()
        or (weight < 0).any()
        or weight.sum() <= 0
    ):
        raise ValueError("Ensemble weights must be finite, nonnegative, with a positive sum")
    weight /= weight.sum()
    indexed = []
    for rows in members:
        index = {}
        for row in rows:
            identity = _identity(row)
            if identity in index:
                raise ValueError(f"Duplicate ensemble sample identity {identity}")
            index[identity] = row
        indexed.append(index)
    expected = set(indexed[0])
    if not expected:
        raise ValueError("Ensemble predictions are empty")
    if any(set(index) != expected for index in indexed[1:]):
        raise ValueError("Ensemble members have missing or additional sample identities")
    first = next(iter(indexed[0].values()))
    canonical = {field: class_names(mapping) for field, mapping in first.get("class_maps", {}).items()}
    result = []
    for identity in sorted(expected):
        rows = [_ordered(index[identity], canonical) for index in indexed]
        row = copy.deepcopy(rows[0])
        fields = set(row.get("probabilities", {}))
        for other in rows[1:]:
            if other.get("task") != row.get("task"):
                raise ValueError("Ensemble task mismatch")
            if set(other.get("class_maps", {})) != set(row.get("class_maps", {})):
                raise ValueError("Ensemble target schemas mismatch")
            if other.get("targets") != row.get("targets") or other.get("target_valid") != row.get(
                "target_valid"
            ):
                raise ValueError(f"Ensemble targets conflict for {identity}")
            if set(other.get("probabilities", {})) != fields:
                raise ValueError("Ensemble prediction fields mismatch")
        for field in fields:
            output = "probabilities" if method == "mean_probabilities" else "logits"
            if any(field not in item.get(output, {}) for item in rows):
                raise ValueError(f"Ensemble {method} requires {output} in every member")
            values = [np.asarray(item[output][field], dtype=float) for item in rows]
            if len({value.shape for value in values}) != 1:
                raise ValueError("Ensemble predictions have incompatible dimensions")
            combined = np.tensordot(weight, np.stack(values), axes=1)
            if method == "mean_logits":
                row.setdefault("logits", {})[field] = combined.tolist()
                combined = np.exp(combined - combined.max(axis=-1, keepdims=True))
                combined /= combined.sum(axis=-1, keepdims=True)
            row.setdefault("probabilities", {})[field] = combined.tolist()
        if "dense_probabilities" in row:
            if method != "mean_probabilities":
                raise ValueError("Dense mean_logits requires exported dense logits; use mean_probabilities")
            arrays = [np.asarray(item.get("dense_probabilities"), dtype=float) for item in rows]
            if len({array.shape for array in arrays}) != 1:
                raise ValueError(
                    "Dense ensemble coordinate systems differ; invert geometry before aggregation"
                )
            combined = np.tensordot(weight, np.stack(arrays), axes=1)
            prediction = combined.argmax(axis=0)
            row["dense_probabilities"], row["predicted_mask"] = combined.tolist(), prediction.tolist()
            field = row.get("dense_target_key")
            if field in row.get("targets", {}):
                target = np.asarray(row["targets"][field])
                if target.shape != prediction.shape:
                    raise ValueError("Dense target coordinates differ from prediction coordinates")
                valid = np.asarray(row.get("target_valid", {}).get(field, True), dtype=bool)
                valid = np.broadcast_to(valid, target.shape) & (target >= 0) & (target < combined.shape[0])
                count = combined.shape[0]
                row["segmentation_confusion"] = (
                    np.bincount(target[valid] * count + prediction[valid], minlength=count * count)
                    .reshape(count, count)
                    .tolist()
                )
        row["provenance"] = {
            "ensemble": {
                "method": method,
                "weights": weight.tolist(),
                "members": [item.get("provenance", {}) for item in rows],
            }
        }
        result.append(row)
    return result


def validate_oof(
    records: Iterable[dict],
    training_membership: dict[str, dict],
    *,
    identity_key: str = "sample_id",
    require_image_independence: bool = False,
) -> dict[str, Any]:
    """Membership indexed by checkpoint hash/fold with supervised_targets and ssl IDs."""
    checked, transductive = 0, []
    for row in records:
        identity = row.get(identity_key, row.get("metadata", {}).get(identity_key))
        if identity is None:
            raise ValueError(f"OOF identity field is missing: {identity_key}")
        provenance = row.get("provenance", {})
        model = str(provenance.get("checkpoint_hash", row.get("checkpoint_hash", row.get("fold", ""))))
        if model not in training_membership:
            raise ValueError(f"No training provenance available for OOF model {model}")
        membership = training_membership[model]
        generic_supervised = set(membership.get("supervised", []))
        supervised_fields = membership.get("supervised_targets", {})
        for field, valid in row.get("target_valid", {}).items():
            if np.asarray(valid).any() and identity in generic_supervised | set(
                supervised_fields.get(field, [])
            ):
                raise ValueError(f"OOF label leakage for identity {identity}, target {field}, model {model}")
        image_training = (
            set(membership.get("ssl", [])) | set(membership.get("images", [])) | generic_supervised
        )
        image_training.update(item for values in supervised_fields.values() for item in values)
        if identity in image_training:
            if require_image_independence:
                raise ValueError(f"OOF image leakage for identity {identity}, model {model}")
            transductive.append(identity)
        checked += 1
    return {
        "checked_rows": checked,
        "identity_key": identity_key,
        "protocol": "label-held-out OOF with transductive representation learning"
        if transductive
        else "independent OOF",
        "ssl_exposed_identities": sorted(set(transductive)),
        "require_image_independence": require_image_independence,
    }


def cross_validation_summary(fold_metrics: Sequence[dict[str, float]]) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for fold in fold_metrics:
        for key, value in fold.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                values[key].append(value)
    return {
        key: {"mean": float(np.mean(items)), "std": float(np.std(items)), "folds": len(items)}
        for key, items in values.items()
    }
