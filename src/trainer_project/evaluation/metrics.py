"""Metrics computed from complete predictions, never from batch-wise F1 means."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def class_names(mapping: Any) -> list[str]:
    if isinstance(mapping, dict):
        ordered = sorted(mapping.items(), key=lambda pair: int(pair[1]))
        if [int(value) for _, value in ordered] != list(range(len(ordered))):
            raise ValueError("Class IDs must be contiguous from zero")
        return [str(key) for key, _ in ordered]
    return [str(item) for item in mapping]


def classification_metrics(
    targets: Sequence[int], probabilities: Any, classes: Sequence[str], *, zero_division: float = 0.0
) -> dict[str, Any]:
    """Use a fixed class universe; missing AUC classes explicitly return N/A."""
    names = list(classes)
    count = len(names)
    if count < 1 or len(set(names)) != count:
        raise ValueError("Class names must be nonempty and unique")
    target = np.asarray(targets, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64).reshape((-1, count))
    if len(target) != len(probs):
        raise ValueError("Targets and predictions have different lengths")
    if len(target) and ((target < 0).any() or (target >= count).any()):
        raise ValueError("Target outside class mapping")
    if not np.isfinite(probs).all():
        raise ValueError("Non-finite probabilities")
    if len(probs) and ((probs < 0).any() or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)):
        raise ValueError("Probabilities must be nonnegative and sum to one")
    prediction = probs.argmax(axis=-1) if len(target) else np.empty(0, dtype=np.int64)
    confusion = np.bincount(target * count + prediction, minlength=count * count).reshape(count, count)
    true_positive = confusion.diagonal().astype(float)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)

    def divide(numerator: Any, denominator: Any) -> np.ndarray:
        return np.divide(
            numerator, denominator, out=np.full(count, zero_division, dtype=float), where=denominator != 0
        )

    precision = divide(true_positive, predicted)
    recall = divide(true_positive, support)
    f1 = divide(2 * true_positive, predicted + support)
    result: dict[str, Any] = {
        "count": int(len(target)),
        "accuracy": float(true_positive.sum() / len(target)) if len(target) else None,
        "f1_macro": float(f1.mean()) if len(target) else None,
        "precision_macro": float(precision.mean()) if len(target) else None,
        "recall_macro": float(recall.mean()) if len(target) else None,
        "confusion_matrix": confusion.tolist(),
        "class_names": names,
        "zero_division": zero_division,
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "predicted": int(predicted[i]),
            }
            for i, name in enumerate(names)
        },
    }
    if len(target):
        confidence = probs.max(axis=1)
        correct = prediction == target
        ece = 0.0
        for low in np.linspace(0.0, 0.9, 10):
            selected = (confidence >= low) & ((confidence < low + 0.1) if low < 0.9 else (confidence <= 1.0))
            if selected.any():
                ece += selected.mean() * abs(float(correct[selected].mean() - confidence[selected].mean()))
        result["ece"] = float(ece)
        result["brier"] = float(((probs - np.eye(count)[target]) ** 2).sum(axis=1).mean())
    else:
        result.update(ece=None, brier=None, unavailable_reason="no valid targets")
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError:
        result["auc_unavailable_reason"] = "Install scikit-learn for AUROC and PR-AUC"
    else:
        auc, ap = [], []
        for i, name in enumerate(names):
            binary = target == i
            item = result["per_class"][name]
            if binary.any() and (~binary).any():
                item["auroc"] = float(roc_auc_score(binary, probs[:, i]))
                item["pr_auc"] = float(average_precision_score(binary, probs[:, i]))
                auc.append(item["auroc"])
                ap.append(item["pr_auc"])
            else:
                item.update(
                    auroc=None, pr_auc=None, auc_unavailable_reason="requires positive and negative targets"
                )
        result["auroc_macro"] = float(np.mean(auc)) if len(auc) == count else None
        result["pr_auc_macro"] = float(np.mean(ap)) if len(ap) == count else None
    return result


def segmentation_metrics(
    confusion: Any, classes: Sequence[str], *, empty_policy: str = "exclude"
) -> dict[str, Any]:
    """Aggregate pixel counts globally before dividing; absent classes are N/A."""
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.shape != (len(classes), len(classes)):
        raise ValueError("Confusion matrix and class mapping mismatch")
    tp = matrix.diagonal().astype(float)
    union = matrix.sum(0) + matrix.sum(1) - tp
    denominator = matrix.sum(0) + matrix.sum(1)
    default = np.nan if empty_policy == "exclude" else (1.0 if empty_policy == "one" else 0.0)
    iou = np.divide(tp, union, out=np.full(len(classes), default), where=union != 0)
    dice = np.divide(2 * tp, denominator, out=np.full(len(classes), default), where=denominator != 0)
    finite_iou, finite_dice = iou[np.isfinite(iou)], dice[np.isfinite(dice)]
    return {
        "count": int(matrix.sum()),
        "valid_pixels": int(matrix.sum()),
        "iou_macro": float(finite_iou.mean()) if len(finite_iou) else None,
        "dice_macro": float(finite_dice.mean()) if len(finite_dice) else None,
        "confusion_matrix": matrix.tolist(),
        "class_names": list(classes),
        "empty_policy": empty_policy,
        "per_class": {
            name: {
                "iou": float(iou[i]) if np.isfinite(iou[i]) else None,
                "dice": float(dice[i]) if np.isfinite(dice[i]) else None,
                "support": int(matrix[i].sum()),
                "intersection": int(tp[i]),
                "union": int(union[i]),
            }
            for i, name in enumerate(classes)
        },
    }


class ClassificationMetrics:
    def __init__(self, classes: Sequence[str], zero_division: float = 0.0):
        self.classes, self.zero_division = list(classes), zero_division
        self.targets: list[int] = []
        self.probabilities: list[list[float]] = []

    def update(self, targets: Any, probabilities: Any) -> None:
        self.targets.extend(np.asarray(targets).tolist())
        self.probabilities.extend(np.asarray(probabilities).tolist())

    def compute(self) -> dict[str, Any]:
        return classification_metrics(
            self.targets, self.probabilities, self.classes, zero_division=self.zero_division
        )
