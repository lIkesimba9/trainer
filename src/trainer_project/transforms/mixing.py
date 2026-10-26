"""Checkpointed batch mixing for fully annotated categorical classification."""

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class _BatchMix(nn.Module):
    """Preserve identity and hard labels while replacing targets by soft labels.

    All declared targets must be included in target_classes, and every example
    must be valid and permitted for every target. Mixed observations have no
    single hard ground truth; online hard-label F1 should be omitted for them.
    """

    def __init__(
        self,
        alpha=0.4,
        target_key="label",
        num_classes=2,
        target_classes=None,
        image_keys=None,
        seed=42,
        probability=1.0,
    ):
        super().__init__()
        if alpha <= 0 or not 0 <= probability <= 1:
            raise ValueError("Mixing alpha must be positive and probability in [0,1]")
        self.alpha, self.probability = float(alpha), float(probability)
        self.target_classes = {
            str(key): int(count) for key, count in (target_classes or {target_key: num_classes}).items()
        }
        if not self.target_classes or min(self.target_classes.values()) < 2:
            raise ValueError("Categorical mixing needs at least two classes for every target")
        self.image_keys = list(image_keys or ["image"])
        self.rng = np.random.default_rng(int(seed))

    def get_extra_state(self):
        return {"rng": copy.deepcopy(self.rng.bit_generator.state)}

    def set_extra_state(self, state):
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])

    def _validate(self, batch, extra_inputs=()):
        if batch.get("views"):
            raise ValueError("Batch mixing is a supervised classification transform and rejects SSL views")
        if "mixing" in batch:
            raise ValueError("Apply exactly one batch-mixing transform to a batch")
        if set(batch.get("targets", {})) != set(self.target_classes):
            raise ValueError(
                "target_classes must declare every target field; labels cannot be discarded during mixing"
            )
        if set(batch.get("inputs", {})) != set(self.image_keys) | set(extra_inputs):
            raise ValueError(
                "Batch mixing requires only its declared image/mask inputs; text/tabular mixing needs a custom adapter"
            )
        batch_size = len(batch["sample_id"])
        for key, count in self.target_classes.items():
            target = batch["targets"][key]
            if not isinstance(target, torch.Tensor) or target.ndim != 1 or len(target) != batch_size:
                raise ValueError("Batch mixing requires hard categorical [B] targets")
            if torch.is_floating_point(target) or ((target < 0) | (target >= count)).any():
                raise ValueError("Batch-mixing targets must be valid integer class IDs")
            for section in ("target_valid", "supervision_allowed"):
                validity = batch.get(section, {}).get(key)
                if validity is None or not torch.as_tensor(validity).bool().all():
                    raise ValueError(
                        "MixUp/CutMix require fully valid labels and explicit supervision permission"
                    )
        for key in (*self.image_keys, *extra_inputs):
            value = batch["inputs"][key]
            if not isinstance(value, torch.Tensor) or value.ndim != 4 or len(value) != batch_size:
                raise ValueError("Mixing inputs must be tensor [B,C,H,W]")
            presence = batch.get("input_present", {}).get(key)
            if presence is not None and not torch.as_tensor(presence).bool().all():
                raise ValueError("Batch mixing rejects missing modalities")
        return batch_size

    def _targets(self, batch, permutation, coefficient, method, **details):
        originals = {key: value.clone() for key, value in batch["targets"].items()}
        for key, count in self.target_classes.items():
            value = F.one_hot(originals[key].long(), count).float()
            index = permutation.to(value.device)
            batch["targets"][key] = coefficient * value + (1 - coefficient) * value[index]
        batch["mixing"] = {
            "method": method,
            "coefficient": float(coefficient),
            "permutation": permutation.clone(),
            "original_targets": originals,
            "paired_sample_ids": [batch["sample_id"][int(index)] for index in permutation],
            "hard_label_metrics": "not_defined_for_mixed_observations",
            **details,
        }
        return batch


class MixUp(_BatchMix):
    """Blend images and every categorical target with a shared permutation."""

    def forward(self, batch):
        batch_size = self._validate(batch)
        if self.rng.random() >= self.probability:
            return batch
        coefficient = float(self.rng.beta(self.alpha, self.alpha))
        permutation = torch.from_numpy(self.rng.permutation(batch_size).copy()).long()
        for key in self.image_keys:
            value = batch["inputs"][key]
            batch["inputs"][key] = (
                coefficient * value + (1 - coefficient) * value[permutation.to(value.device)]
            )
        return self._targets(batch, permutation, coefficient, "mixup")


class CutMix(_BatchMix):
    """Replace the same spatial rectangle in images and declared input masks.

    The target coefficient uses the actual clipped rectangle area. All image/mask
    inputs must have the same spatial size; categorical masks retain their IDs.
    """

    def __init__(self, mask_input_keys=None, **kwargs):
        super().__init__(**kwargs)
        self.mask_input_keys = list(mask_input_keys or [])
        if set(self.mask_input_keys) & set(self.image_keys):
            raise ValueError("Image and input-mask keys must be distinct")

    def forward(self, batch):
        batch_size = self._validate(batch, self.mask_input_keys)
        spatial = {
            tuple(batch["inputs"][key].shape[-2:]) for key in (*self.image_keys, *self.mask_input_keys)
        }
        if len(spatial) != 1:
            raise ValueError("CutMix images and input masks need the same spatial size")
        if self.rng.random() >= self.probability:
            return batch
        height, width = next(iter(spatial))
        coefficient = float(self.rng.beta(self.alpha, self.alpha))
        permutation = torch.from_numpy(self.rng.permutation(batch_size).copy()).long()
        ratio = math.sqrt(1 - coefficient)
        box_width, box_height = round(width * ratio), round(height * ratio)
        center_x, center_y = int(self.rng.integers(width)), int(self.rng.integers(height))
        x1, x2 = max(0, center_x - box_width // 2), min(width, center_x + (box_width + 1) // 2)
        y1, y2 = max(0, center_y - box_height // 2), min(height, center_y + (box_height + 1) // 2)
        for key in (*self.image_keys, *self.mask_input_keys):
            original = batch["inputs"][key]
            mixed = original.clone()
            mixed[..., y1:y2, x1:x2] = original[permutation.to(original.device), ..., y1:y2, x1:x2]
            batch["inputs"][key] = mixed
        coefficient = 1 - ((y2 - y1) * (x2 - x1) / (height * width))
        return self._targets(batch, permutation, coefficient, "cutmix", rectangle=[x1, y1, x2, y2])
