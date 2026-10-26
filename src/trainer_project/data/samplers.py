"""Epoch-addressable samplers; the global draw is sharded across ranks."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import torch
from torch.utils.data import Sampler


def distributed_context(rank=None, world_size=None):
    active = torch.distributed.is_available() and torch.distributed.is_initialized()
    return (
        int(torch.distributed.get_rank() if active and rank is None else rank or 0),
        int(torch.distributed.get_world_size() if active and world_size is None else world_size or 1),
    )


class DistributedEpochSampler(Sampler[int]):
    """Shuffle globally then shard. Padding is explicit and keeps train steps equal."""

    def __init__(
        self,
        dataset,
        shuffle=True,
        seed=42,
        rank=None,
        world_size=None,
        samples_per_epoch=None,
        tail_policy="pad",
        **kwargs,
    ):
        self.dataset = dataset
        self.shuffle, self.seed, self.epoch = bool(shuffle), int(seed), 0
        self.rank, self.world_size = distributed_context(rank, world_size)
        if not 0 <= self.rank < self.world_size:
            raise ValueError("Invalid distributed rank/world_size")
        self.samples_per_epoch = int(samples_per_epoch or len(dataset))
        if tail_policy not in {"pad", "drop"}:
            raise ValueError("tail_policy must be pad or drop")
        self.tail_policy = tail_policy
        if not len(dataset) or self.samples_per_epoch <= 0:
            raise ValueError("Sampler requires a nonempty dataset and positive epoch length")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def state_dict(self):
        return {"epoch": self.epoch, "seed": self.seed, "world_size": self.world_size}

    def load_state_dict(self, state):
        if state.get("world_size", self.world_size) != self.world_size:
            raise ValueError("Sampler world_size changed")
        if state.get("seed", self.seed) != self.seed:
            raise ValueError("Sampler seed changed")
        self.epoch = int(state["epoch"])

    def __len__(self):
        op = math.ceil if self.tail_policy == "pad" else math.floor
        return int(op(self.samples_per_epoch / self.world_size))

    def global_indices(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        result = []
        while len(result) < self.samples_per_epoch:
            result.extend(
                torch.randperm(len(self.dataset), generator=generator).tolist()
                if self.shuffle
                else range(len(self.dataset))
            )
        return result[: self.samples_per_epoch]

    def __iter__(self):
        indices = self.global_indices()
        total = len(self) * self.world_size
        if total > len(indices):
            indices += (indices * math.ceil((total - len(indices)) / len(indices)))[: total - len(indices)]
        return iter(indices[:total][self.rank : total : self.world_size])


class DistributedFieldWeightedSampler(DistributedEpochSampler):
    """Uniform groups by default, or explicit relative group mass; no batch quota."""

    def __init__(
        self,
        dataset,
        field="manifest_name",
        group_weights=None,
        sample_weight_field=None,
        replacement=True,
        missing_policy="error",
        **kwargs,
    ):
        super().__init__(dataset, **kwargs)
        if missing_policy not in {"error", "exclude", "group"}:
            raise ValueError("Unknown missing_policy")
        self.replacement = bool(replacement)
        records = getattr(dataset, "records", None)
        if records is None:
            raise ValueError("Field sampling requires dataset.records")
        fields = [field] if isinstance(field, str) else list(field)
        groups, keep = [], []
        for index, row in enumerate(records):
            values = tuple(
                row.get(key) if row.get("_target_allowed", {}).get(key, True) else None for key in fields
            )
            missing = any(value is None or value == "" or value == -1 for value in values)
            if missing and missing_policy == "error":
                raise ValueError(f"Missing sampler field {fields} at row {index}")
            if missing and missing_policy == "exclude":
                continue
            groups.append(values[0] if len(values) == 1 else values)
            keep.append(index)
        counts = Counter(groups)
        weights = []
        for index, group in zip(keep, groups):
            mass = float((group_weights or {}).get(str(group), 1.0))
            value = mass / counts[group]
            if sample_weight_field:
                value *= float(records[index].get(sample_weight_field, 1.0))
            weights.append(value)
        self.indices = keep
        self.weights = torch.tensor(weights, dtype=torch.double)
        if (
            not len(weights)
            or not torch.isfinite(self.weights).all()
            or (self.weights < 0).any()
            or self.weights.sum() <= 0
        ):
            raise ValueError("Sampler weights must be finite, nonnegative and nonempty")
        if not self.replacement and self.samples_per_epoch > int((self.weights > 0).sum()):
            raise ValueError("Without replacement samples_per_epoch exceeds available positive-weight rows")

    def global_indices(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        positions = torch.multinomial(
            self.weights, self.samples_per_epoch, replacement=self.replacement, generator=generator
        ).tolist()
        return [self.indices[i] for i in positions]


class DistributedSampleWeightedSampler(DistributedFieldWeightedSampler):
    def __init__(self, dataset, field="sample_weight", **kwargs):
        super().__init__(dataset, field="manifest_name", sample_weight_field=field, **kwargs)
        self.weights = torch.tensor(
            [float(dataset.records[i][field]) for i in self.indices], dtype=torch.double
        )
        if not torch.isfinite(self.weights).all() or (self.weights < 0).any() or self.weights.sum() <= 0:
            raise ValueError("Invalid sample weights")


class PKBatchSampler(Sampler[list[int]]):
    """P distinct groups, K observations per group on each rank independently."""

    is_batch_sampler = True

    def __init__(
        self,
        dataset,
        field="label",
        p=2,
        k=2,
        batches_per_epoch=None,
        samples_per_epoch=None,
        seed=42,
        rank=None,
        world_size=None,
        missing_policy="exclude",
        insufficient_policy="replacement",
        scope="local",
        **kwargs,
    ):
        if scope != "local":
            raise ValueError("PKBatchSampler supports scope=local to preserve positives on every rank")
        if missing_policy not in {"error", "exclude"}:
            raise ValueError("PK missing_policy must be error or exclude")
        if insufficient_policy not in {"error", "replacement"}:
            raise ValueError("PK insufficient_policy must be error or replacement")
        self.rank, self.world_size = distributed_context(rank, world_size)
        self.p, self.k, self.seed, self.epoch = int(p), int(k), int(seed), 0
        if self.p <= 0 or self.k <= 0:
            raise ValueError("P and K must be positive")
        self.groups = defaultdict(list)
        for index, row in enumerate(dataset.records):
            value = row.get(field) if row.get("_target_allowed", {}).get(field, True) else None
            if value in (None, "", -1):
                if missing_policy == "error":
                    raise ValueError(f"Missing PK field {field}")
                continue
            self.groups[value].append(index)
        if len(self.groups) < self.p:
            raise ValueError("PK requires at least P distinct valid groups")
        self.insufficient_policy = insufficient_policy
        if insufficient_policy == "error" and any(len(items) < self.k for items in self.groups.values()):
            raise ValueError("PK group has fewer than K unique samples")
        count = samples_per_epoch or len(dataset)
        self.batches_per_epoch = int(
            batches_per_epoch or math.ceil(count / (self.p * self.k * self.world_size))
        )

    def __len__(self):
        return self.batches_per_epoch

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def state_dict(self):
        return {"epoch": self.epoch, "seed": self.seed, "world_size": self.world_size}

    def load_state_dict(self, state):
        if (
            state.get("seed", self.seed) != self.seed
            or state.get("world_size", self.world_size) != self.world_size
        ):
            raise ValueError("PK sampler execution context changed")
        self.epoch = int(state["epoch"])

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        keys = list(self.groups)
        for _ in range(self.batches_per_epoch):
            for rank in range(self.world_size):
                selected = torch.randperm(len(keys), generator=generator)[: self.p].tolist()
                batch = []
                for group_index in selected:
                    candidates = self.groups[keys[group_index]]
                    if len(candidates) >= self.k:
                        positions = torch.randperm(len(candidates), generator=generator)[: self.k]
                    else:
                        positions = torch.randint(len(candidates), (self.k,), generator=generator)
                    batch.extend(candidates[i] for i in positions.tolist())
                if rank == self.rank:
                    yield batch
