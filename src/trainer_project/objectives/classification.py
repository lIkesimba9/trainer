"""Classification and metric-learning objectives with explicit missingness."""

import math

import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F

from .base import Objective, distributed, global_masked_mean, result, target_mask


def _labels(batch, key, logits):
    if key not in batch.get("targets", {}):
        raise ValueError(f"Objective target {key!r} is not declared in the batch")
    targets = batch["targets"][key].to(logits.device)
    if targets.ndim != 1:
        raise ValueError(
            "This objective requires integer categorical [B] targets; soft targets are incompatible"
        )
    valid = target_mask(batch, key, targets)
    invalid_values = valid & ((targets < 0) | (targets >= logits.shape[-1]))
    if invalid_values.any():
        raise ValueError("A valid target is outside the configured class mapping")
    safe = torch.where(valid, targets, torch.zeros_like(targets)).long()
    return safe, valid


class MaskedCrossEntropy(Objective):
    supports_soft_targets = True

    def __init__(self, target_key="label", output_key="logits", class_weights=None, label_smoothing=0.0):
        super().__init__()
        self.target_key, self.output_key = target_key, output_key
        self.label_smoothing = float(label_smoothing)
        self.register_buffer(
            "class_weights",
            None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32),
        )

    def forward(self, outputs, batch, context=None):
        logits = outputs[self.output_key]
        target = batch["targets"][self.target_key].to(logits.device)
        if target.ndim == 2:
            if target.shape != logits.shape:
                raise ValueError("Soft categorical targets must have the same shape as class logits")
            valid = target_mask(batch, self.target_key, target).all(-1)
            safe = torch.where(valid[:, None], target, torch.zeros_like(target))
            losses = F.cross_entropy(
                logits,
                safe,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
                reduction="none",
            )
        else:
            target, valid = _labels(batch, self.target_key, logits)
            losses = F.cross_entropy(
                logits,
                target,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
                reduction="none",
            )
        loss, count = global_masked_mean(losses, valid)
        return result(loss, "cross_entropy", count)


class FocalLoss(MaskedCrossEntropy):
    supports_soft_targets = False

    def __init__(self, gamma=2.0, alpha=None, **kwargs):
        super().__init__(**kwargs)
        self.gamma = float(gamma)
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, outputs, batch, context=None):
        logits = outputs[self.output_key]
        targets, valid = _labels(batch, self.target_key, logits)
        ce = F.cross_entropy(
            logits, targets, reduction="none", weight=self.class_weights, label_smoothing=self.label_smoothing
        )
        probabilities = logits.softmax(-1).gather(1, targets[:, None]).squeeze(1)
        losses = ce * (1.0 - probabilities).pow(self.gamma)
        if self.alpha is not None:
            losses = losses * (self.alpha if self.alpha.ndim == 0 else self.alpha[targets])
        loss, count = global_masked_mean(losses, valid)
        return result(loss, "focal", count)


class LDAMLoss(MaskedCrossEntropy):
    supports_soft_targets = False

    def __init__(
        self, class_counts, max_margin=0.5, scale=30.0, deferred_reweight_epoch=None, beta=0.9999, **kwargs
    ):
        super().__init__(**kwargs)
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        if (counts <= 0).any():
            raise ValueError("LDAM class counts must be positive and computed from allowed train labels")
        margins = counts.pow(-0.25)
        self.register_buffer("margins", margins * float(max_margin) / margins.max())
        self.register_buffer("training_counts", counts)
        self.scale = float(scale)
        self.deferred_reweight_epoch, self.beta = deferred_reweight_epoch, float(beta)

    def forward(self, outputs, batch, context=None):
        logits = outputs[self.output_key]
        targets, valid = _labels(batch, self.target_key, logits)
        adjusted = logits - F.one_hot(targets, logits.shape[-1]) * self.margins[targets, None]
        weight = self.class_weights
        if (
            self.deferred_reweight_epoch is not None
            and (context or {}).get("epoch", 0) >= self.deferred_reweight_epoch
        ):
            weight = (1 - self.beta) / (1 - self.beta**self.training_counts)
            weight = weight / weight.mean()
        losses = F.cross_entropy(adjusted * self.scale, targets, weight=weight, reduction="none")
        loss, count = global_masked_mean(losses, valid)
        return result(loss, "ldam", count)


class LogitAdjustedLoss(MaskedCrossEntropy):
    supports_soft_targets = False

    def __init__(self, class_counts, tau=1.0, mode="training", **kwargs):
        super().__init__(**kwargs)
        if mode not in ("training", "posthoc"):
            raise ValueError("Logit adjustment mode must be training or posthoc")
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        if (counts <= 0).any():
            raise ValueError("Logit adjustment requires positive training-only class counts")
        self.register_buffer("log_prior", (counts / counts.sum()).log())
        self.tau, self.mode = float(tau), mode

    def inference_logits(self, logits):
        return logits - self.tau * self.log_prior if self.mode == "posthoc" else logits

    def forward(self, outputs, batch, context=None):
        logits = outputs[self.output_key]
        targets, valid = _labels(batch, self.target_key, logits)
        adjusted = logits + self.tau * self.log_prior if self.mode == "training" else logits
        loss, count = global_masked_mean(
            F.cross_entropy(adjusted, targets, reduction="none", weight=self.class_weights), valid
        )
        return result(loss, "logit_adjusted", count)


class SeesawLoss(MaskedCrossEntropy):
    """Image-classification adaptation: every class is foreground, no background term."""

    supports_soft_targets = False

    def __init__(self, num_classes, mitigation=0.8, compensation=2.0, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("class_counts", torch.zeros(int(num_classes)))
        self.mitigation, self.compensation = float(mitigation), float(compensation)

    def forward(self, outputs, batch, context=None):
        logits = outputs[self.output_key]
        targets, valid = _labels(batch, self.target_key, logits)
        if (context or {}).get("stage", "train") == "train":
            increments = torch.bincount(targets[valid], minlength=logits.shape[-1]).to(self.class_counts)
            if distributed():
                dist.all_reduce(increments)
            self.class_counts.add_(increments)
        counts = self.class_counts.clamp_min(1)
        ratio = counts[None, :] / counts[targets, None]
        mitigation = torch.where(ratio < 1, ratio.pow(self.mitigation), torch.ones_like(ratio))
        scores = logits.detach().softmax(-1).clamp_min(1e-7)
        ratio_scores = scores / scores.gather(1, targets[:, None])
        compensation = torch.where(
            ratio_scores > 1, ratio_scores.pow(self.compensation), torch.ones_like(ratio_scores)
        )
        one_hot = F.one_hot(targets, logits.shape[-1]).bool()
        seesaw = torch.where(one_hot, torch.ones_like(logits), mitigation * compensation)
        losses = F.cross_entropy(logits + seesaw.clamp_min(1e-7).log(), targets, reduction="none")
        loss, count = global_masked_mean(losses, valid)
        return result(loss, "seesaw", count)


class ArcFaceObjective(Objective):
    """Trainable class centers live in the registered objective, including subcenters.

    inference_logits(embeddings) must be used for predictions with these centers.
    This objective accepts hard targets only and never places labels in forward.
    """

    def __init__(
        self,
        embedding_dim,
        num_classes,
        target_key="label",
        output_key="embeddings",
        scale=32.0,
        margin=0.5,
        subcenters=1,
    ):
        super().__init__()
        self.num_classes, self.embedding_dim = int(num_classes), int(embedding_dim)
        self.target_key, self.output_key = target_key, output_key
        self.scale, self.margin = float(scale), float(margin)
        self.centers = nn.Parameter(torch.empty(num_classes, subcenters, embedding_dim))
        nn.init.xavier_uniform_(self.centers.reshape(-1, embedding_dim))

    def cosine_logits(self, embeddings):
        return torch.einsum(
            "bd,ckd->bck", F.normalize(embeddings, dim=-1), F.normalize(self.centers, dim=-1)
        ).amax(-1)

    def inference_logits(self, embeddings):
        return self.cosine_logits(embeddings) * self.scale

    def forward(self, outputs, batch, context=None):
        cosine = self.cosine_logits(outputs[self.output_key])
        targets, valid = _labels(batch, self.target_key, cosine)
        # Piecewise monotonic ArcFace margin avoids angular wrap beyond pi-margin.
        safe_cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        phi = safe_cosine * math.cos(self.margin) - (1 - safe_cosine.square()).sqrt() * math.sin(self.margin)
        phi = torch.where(
            safe_cosine > math.cos(math.pi - self.margin),
            phi,
            safe_cosine - math.sin(math.pi - self.margin) * self.margin,
        )
        adjusted = torch.where(F.one_hot(targets, self.num_classes).bool(), phi, cosine)
        loss, count = global_masked_mean(
            F.cross_entropy(adjusted * self.scale, targets, reduction="none"), valid
        )
        return result(loss, "arcface", count)


class TripletLoss(Objective):
    """Local batch-hard mining; DDP normalizes across globally valid anchors."""

    def __init__(
        self,
        target_key="label",
        output_key="embeddings",
        margin=0.2,
        distance="euclidean",
        identity_field=None,
    ):
        super().__init__()
        if distance not in ("euclidean", "cosine"):
            raise ValueError("Triplet distance must be euclidean or cosine")
        self.target_key, self.output_key = target_key, output_key
        self.margin, self.distance, self.identity_field = float(margin), distance, identity_field

    def forward(self, outputs, batch, context=None):
        embeddings = outputs[self.output_key]
        targets = batch["targets"][self.target_key].to(embeddings.device)
        if targets.ndim != 1:
            raise ValueError("Triplet mining requires categorical hard targets")
        valid = target_mask(batch, self.target_key, targets)
        distances = (
            torch.cdist(embeddings, embeddings)
            if self.distance == "euclidean"
            else (1 - F.normalize(embeddings, dim=-1) @ F.normalize(embeddings, dim=-1).T)
        )
        available = valid[:, None] & valid[None, :]
        same = targets[:, None] == targets[None, :]
        positive = available & same & ~torch.eye(len(targets), device=targets.device, dtype=torch.bool)
        negative = available & ~same
        if self.identity_field:
            identities = batch["metadata"][self.identity_field]
            equal_identity = torch.tensor(
                [[a == b for b in identities] for a in identities], device=targets.device
            )
            negative &= ~equal_identity
        anchors = positive.any(1) & negative.any(1)
        hardest_positive = distances.masked_fill(~positive, -1e6).amax(1)
        hardest_negative = distances.masked_fill(~negative, 1e6).amin(1)
        values = F.relu(hardest_positive - hardest_negative + self.margin)
        loss, count = global_masked_mean(values, anchors)
        return result(loss, "triplet", count)
