import torch
from torch import distributed as dist
from torch.nn import functional as F

from .base import Objective, ObjectiveResult, distributed, global_masked_mean, target_mask


def _global_stats(tensor):
    if distributed():
        if torch.is_grad_enabled() and tensor.requires_grad:
            from torch.distributed.nn.functional import all_reduce

            return all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor.clone()
        dist.all_reduce(tensor)
    return tensor


class SegmentationObjective(Objective):
    """Masked CE/BCE and global soft Dice for exclusive, binary, multilabel profiles.

    Unknown pixels never enter either loss. Dice averages over classes containing
    any valid positive annotations; the all-empty case has a graph-connected zero.
    """

    def __init__(
        self,
        target_key="segmentation_target",
        output_key="dense_logits",
        target_mode="exclusive",
        ignore_index=255,
        class_weights=None,
        dice_weight=0.0,
        ce_weight=1.0,
        include_background=True,
        smooth=1e-6,
    ):
        super().__init__()
        if target_mode not in ("exclusive", "binary", "multilabel"):
            raise ValueError("Unknown segmentation target_mode")
        self.target_key, self.output_key = target_key, output_key
        self.target_mode, self.ignore_index = target_mode, int(ignore_index)
        self.dice_weight, self.ce_weight = float(dice_weight), float(ce_weight)
        self.include_background, self.smooth = bool(include_background), float(smooth)
        self.register_buffer(
            "class_weights",
            None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32),
        )

    def forward(self, outputs, batch, context=None):
        logits = outputs[self.output_key]
        targets = batch["targets"][self.target_key].to(logits.device)
        if self.target_mode == "exclusive":
            if targets.ndim != 3 or logits.ndim != 4:
                raise ValueError("Exclusive segmentation expects integer [B,H,W] and logits [B,C,H,W]")
            valid = target_mask(batch, self.target_key, targets) & (targets != self.ignore_index)
            if (valid & ((targets < 0) | (targets >= logits.shape[1]))).any():
                raise ValueError("Valid segmentation class ID is outside the class mapping")
            safe = torch.where(valid, targets, torch.zeros_like(targets)).long()
            values = F.cross_entropy(logits, safe, weight=self.class_weights, reduction="none")
            probabilities = logits.softmax(1)
            truth = F.one_hot(safe, logits.shape[1]).permute(0, 3, 1, 2).to(logits)
            dice_mask = valid[:, None].expand_as(probabilities)
        else:
            if targets.ndim == 3:
                targets = targets[:, None]
            if targets.shape != logits.shape:
                raise ValueError(
                    "Binary/multilabel targets must match dense logits; lossy exclusive conversion is forbidden"
                )
            valid = target_mask(batch, self.target_key, targets) & (targets != self.ignore_index)
            safe = torch.where(valid, targets, torch.zeros_like(targets)).to(logits)
            if (valid & ((safe < 0) | (safe > 1))).any():
                raise ValueError("BCE segmentation targets must be in [0,1] on valid pixels")
            values = F.binary_cross_entropy_with_logits(logits, safe, reduction="none")
            if self.class_weights is not None:
                values = values * self.class_weights[None, :, None, None]
            probabilities, truth, dice_mask = logits.sigmoid(), safe, valid
        ce, count = global_masked_mean(values, valid)
        total = self.ce_weight * ce
        components = {"segmentation_ce": ce.detach()}
        counts = {"pixels": count}
        if self.dice_weight:
            dimensions = (0, 2, 3)
            weighted_prediction = probabilities * dice_mask
            weighted_truth = truth * dice_mask
            stats = _global_stats(
                torch.stack(
                    (
                        (weighted_prediction * weighted_truth).sum(dimensions),
                        weighted_prediction.sum(dimensions),
                        weighted_truth.sum(dimensions),
                    )
                )
            )
            annotated_class = stats[2] > 0
            if not self.include_background and len(annotated_class) > 1:
                annotated_class = annotated_class.clone()
                annotated_class[0] = False
            dice = 1 - (2 * stats[0] + self.smooth) / (stats[1] + stats[2] + self.smooth)
            dice_loss = (dice * annotated_class).sum() / annotated_class.sum().clamp_min(1)
            total = total + self.dice_weight * dice_loss
            components["segmentation_dice"] = dice_loss.detach()
            counts["annotated_classes"] = annotated_class.sum().detach()
        return ObjectiveResult(total, components, counts)
