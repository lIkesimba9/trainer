"""Objective contract and globally normalized masked losses."""

from dataclasses import dataclass, field

import torch
from torch import distributed as dist
from torch import nn


@dataclass
class ObjectiveResult:
    loss: torch.Tensor
    components: dict[str, torch.Tensor] = field(default_factory=dict)
    counts: dict[str, int | float | torch.Tensor] = field(default_factory=dict)


def distributed():
    return dist.is_available() and dist.is_initialized()


def global_masked_mean(values, mask=None, weights=None):
    """Global mean with gradients corrected for DDP's subsequent rank averaging.

    Every rank participates, including ranks with zero valid elements. The
    returned value is the global mean; its local derivative is world_size times
    local numerator/global denominator. Counts represent valid observations,
    while optional sample weights modify the effective denominator.
    """
    mask = (
        torch.ones_like(values, dtype=torch.bool)
        if mask is None
        else mask.to(device=values.device, dtype=torch.bool)
    )
    if mask.shape != values.shape:
        mask = torch.broadcast_to(mask, values.shape)
    numeric_weights = mask.to(values.dtype)
    if weights is not None:
        numeric_weights = numeric_weights * torch.broadcast_to(weights.to(values), values.shape)
    accumulation_dtype = torch.float32 if values.dtype in (torch.float16, torch.bfloat16) else values.dtype
    numerator = (torch.where(mask, values, torch.zeros_like(values)) * numeric_weights).sum(
        dtype=accumulation_dtype
    )
    denominator = numeric_weights.sum(dtype=torch.float64).detach()
    count = mask.sum().detach()
    totals = torch.stack((numerator.detach().double(), denominator, count.double()))
    world = 1
    if distributed():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        world = dist.get_world_size()
    gradient_value = numerator * world / totals[1].clamp_min(1).to(numerator)
    global_value = (totals[0] / totals[1].clamp_min(1)).to(numerator)
    return gradient_value + (global_value - gradient_value.detach()), totals[2]


def target_mask(batch, key, target):
    valid = batch.get("target_valid", {}).get(key)
    allowed = batch.get("supervision_allowed", {}).get(key)
    result = torch.ones_like(target, dtype=torch.bool)
    for value in (valid, allowed):
        if value is not None:
            value = torch.as_tensor(value, device=target.device, dtype=torch.bool)
            if target.ndim == 4 and value.ndim == 3 and value.shape == (target.shape[0], *target.shape[-2:]):
                value = value.unsqueeze(1)
            while value.ndim < target.ndim:
                value = value.unsqueeze(-1)
            result = result & torch.broadcast_to(value, target.shape)
    return result


def result(loss, name, count):
    return ObjectiveResult(loss, {name: loss.detach()}, {name: count})


class Objective(nn.Module):
    """Subclasses implement forward(outputs, full_batch, context)."""

    supports_soft_targets = False

    def forward(self, outputs, batch, context=None):
        raise NotImplementedError
