from collections import Counter

import pytest

from trainer_project.data import (
    DistributedEpochSampler,
    DistributedFieldWeightedSampler,
    PKBatchSampler,
)


class Records:
    def __init__(self, rows):
        self.records = rows

    def __len__(self):
        return len(self.records)


def test_global_shuffle_is_sharded_and_repeatable():
    data = Records([{}] * 9)
    left = DistributedEpochSampler(data, rank=0, world_size=2, seed=11)
    right = DistributedEpochSampler(data, rank=1, world_size=2, seed=11)
    first = list(left)
    assert first == list(left)
    combined = [item for pair in zip(left, right) for item in pair]
    assert set(combined) == set(range(9))
    assert len(combined) == 10
    left.set_epoch(1)
    assert list(left) != first
    state = left.state_dict()
    restored = DistributedEpochSampler(data, rank=0, world_size=2, seed=11)
    restored.load_state_dict(state)
    assert list(restored) == list(left)


def test_field_sampling_balances_arbitrary_groups_and_splits_global_draw():
    data = Records([{"source": "small"}] * 2 + [{"source": "large"}] * 18)
    options = {"field": "source", "samples_per_epoch": 4000, "seed": 3}
    left = DistributedFieldWeightedSampler(data, rank=0, world_size=2, **options)
    right = DistributedFieldWeightedSampler(data, rank=1, world_size=2, **options)
    draws = list(left) + list(right)
    small = sum(data.records[index]["source"] == "small" for index in draws)
    assert 1700 < small < 2300
    assert list(left) != list(right)


def test_field_missing_and_without_replacement_policies():
    data = Records([{"source": ""}, {"source": "a"}])
    with pytest.raises(ValueError, match="Missing"):
        DistributedFieldWeightedSampler(data, field="source")
    with pytest.raises(ValueError, match="Without replacement"):
        DistributedFieldWeightedSampler(data, field="source", missing_policy="exclude", replacement=False)


def test_pk_preserves_local_positive_groups_on_every_rank():
    data = Records([{"species": group} for group in "abcd" for _ in range(5)])
    for rank in range(2):
        sampler = PKBatchSampler(
            data, field="species", p=3, k=2, rank=rank, world_size=2, batches_per_epoch=3
        )
        for batch in sampler:
            counts = Counter(data.records[index]["species"] for index in batch)
            assert sorted(counts.values()) == [2, 2, 2]
        state = sampler.state_dict()
        sampler.set_epoch(2)
        sampler.load_state_dict(state)
        assert sampler.epoch == 0


def test_pk_insufficient_group_fails_explicitly():
    data = Records([{"label": "a"}, {"label": "b"}])
    with pytest.raises(ValueError, match="fewer than K"):
        PKBatchSampler(data, p=2, k=2, insufficient_policy="error")
