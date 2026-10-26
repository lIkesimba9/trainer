"""Label-independent predictions from registered, optionally nested objectives."""

import torch

from .classification import ArcFaceObjective, LogitAdjustedLoss
from .composite import CompositeObjective


def _leaves(objective):
    if isinstance(objective, CompositeObjective):
        for child in objective.components.values():
            yield from _leaves(child)
    else:
        yield objective


def objective_inference_outputs(objective, outputs, *, primary_target=None, apply_posthoc=True):
    """Overlay center-based logits and apply each post-hoc correction at most once.

    A single target retains tensor outputs['logits']; multiple targets produce
    outputs['logits']={target_name: tensor}, understood by EvaluationAccumulator.
    No target labels or batch data are read. apply_posthoc=False is required while
    training, so a post-hoc correction never becomes an unintended training loss.
    """
    result = dict(outputs)
    original = outputs.get("logits")
    logits = (
        dict(original)
        if isinstance(original, dict)
        else ({primary_target or "label": original} if isinstance(original, torch.Tensor) else {})
    )
    centers, adjustments = {}, {}
    seen = set()
    for leaf in _leaves(objective):
        if id(leaf) in seen:
            continue
        seen.add(id(leaf))
        if isinstance(leaf, ArcFaceObjective):
            if leaf.target_key in centers:
                raise ValueError(f"Ambiguous inference: multiple center objectives for {leaf.target_key}")
            centers[leaf.target_key] = leaf.inference_logits(outputs[leaf.output_key])
        elif isinstance(leaf, LogitAdjustedLoss) and leaf.mode == "posthoc" and apply_posthoc:
            if leaf.target_key in adjustments:
                raise ValueError(f"Duplicate post-hoc correction for {leaf.target_key}")
            adjustments[leaf.target_key] = leaf
    logits.update(centers)
    for target, adjustment in adjustments.items():
        if target in logits:
            logits[target] = adjustment.inference_logits(logits[target])
        else:
            raw = outputs.get(adjustment.output_key)
            if not isinstance(raw, torch.Tensor):
                raise ValueError(f"No class logits available for post-hoc target {target}")
            logits[target] = adjustment.inference_logits(raw)
    if logits:
        result["logits"] = (
            next(iter(logits.values()))
            if len(logits) == 1 and next(iter(logits)) == (primary_target or "label")
            else logits
        )
        if centers or adjustments:
            probabilities = {target: value.softmax(-1) for target, value in logits.items()}
            result["probabilities"] = (
                next(iter(probabilities.values()))
                if isinstance(result["logits"], torch.Tensor)
                else probabilities
            )
    return result
