import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from trainer_project.callbacks import ScheduledFreeze, enable_scheduled_parameters
from trainer_project.models import TestTimeAugmentation as TTA
from trainer_project.models import tta_forward
from trainer_project.objectives import (
    ArcFaceObjective,
    CompositeObjective,
    LogitAdjustedLoss,
    MaskedCrossEntropy,
    objective_inference_outputs,
)
from trainer_project.transforms import CutMix, MixUp


def example(aux=False, mask=False):
    image = torch.arange(4, dtype=torch.float32)[:, None, None, None].expand(4, 3, 8, 8).clone()
    targets = {"label": torch.tensor([0, 1, 0, 1])}
    if aux:
        targets["aux"] = torch.tensor([0, 1, 2, 0])
    inputs = {"image": image}
    if mask:
        inputs["input_mask"] = (image[:, :1] % 2).clone()
    return {
        "sample_id": ["a", "b", "c", "d"],
        "manifest_name": ["train"] * 4,
        "inputs": inputs,
        "input_present": {key: torch.ones(4, dtype=torch.bool) for key in inputs},
        "views": {},
        "targets": targets,
        "target_valid": {key: torch.ones(4, dtype=torch.bool) for key in targets},
        "supervision_allowed": {key: torch.ones(4, dtype=torch.bool) for key in targets},
        "metadata": {"source": ["x", "y", "x", "y"]},
    }


def test_mixup_preserves_all_labels_permissions_and_ids_and_uses_soft_ce():
    transform = MixUp(target_classes={"label": 2, "aux": 3}, seed=4)
    batch = example(aux=True)
    original = copy.deepcopy(batch)
    assert transform(batch) is batch
    coefficient, permutation = batch["mixing"]["coefficient"], batch["mixing"]["permutation"]
    torch.testing.assert_close(
        batch["inputs"]["image"],
        coefficient * original["inputs"]["image"]
        + (1 - coefficient) * original["inputs"]["image"][permutation],
    )
    for key, count in {"label": 2, "aux": 3}.items():
        expected = F.one_hot(original["targets"][key], count).float()
        torch.testing.assert_close(
            batch["targets"][key], coefficient * expected + (1 - coefficient) * expected[permutation]
        )
        assert torch.equal(batch["mixing"]["original_targets"][key], original["targets"][key])
        assert torch.equal(batch["target_valid"][key], original["target_valid"][key])
        assert torch.equal(batch["supervision_allowed"][key], original["supervision_allowed"][key])
    assert batch["sample_id"] == original["sample_id"]
    assert batch["metadata"] == original["metadata"]
    logits = torch.randn(4, 2, requires_grad=True)
    loss = MaskedCrossEntropy()({"logits": logits}, batch, {}).loss
    loss.backward()
    assert torch.isfinite(loss) and logits.grad.abs().sum() > 0


@pytest.mark.parametrize("kind", [MixUp, CutMix])
def test_batch_mixing_rng_resumes_exactly(kind):
    first = kind(seed=7)
    first(example())
    state = copy.deepcopy(first.state_dict())
    expected = first(example())
    resumed = kind(seed=999)
    resumed.load_state_dict(state)
    actual = resumed(example())
    assert actual["mixing"]["coefficient"] == expected["mixing"]["coefficient"]
    assert torch.equal(actual["mixing"]["permutation"], expected["mixing"]["permutation"])
    torch.testing.assert_close(actual["inputs"]["image"], expected["inputs"]["image"], rtol=0, atol=0)
    torch.testing.assert_close(actual["targets"]["label"], expected["targets"]["label"], rtol=0, atol=0)


@pytest.mark.parametrize("section", ["target_valid", "supervision_allowed", "input_present"])
def test_mixing_rejects_missing_or_forbidden_examples(section):
    batch = example()
    key = "image" if section == "input_present" else "label"
    batch[section][key][0] = False
    with pytest.raises(ValueError):
        MixUp()(batch)


def test_mixing_rejects_undeclared_auxiliary_labels_and_ssl_views():
    with pytest.raises(ValueError, match="every target"):
        MixUp()(example(aux=True))
    batch = example()
    batch["views"] = {"view1": batch["inputs"]["image"]}
    with pytest.raises(ValueError, match="SSL views"):
        MixUp()(batch)


def test_cutmix_masks_follow_rectangle_and_target_coefficient_is_actual_area():
    transform = CutMix(mask_input_keys=["input_mask"], seed=11)
    batch = example(mask=True)
    original = copy.deepcopy(batch)
    transform(batch)
    x1, y1, x2, y2 = batch["mixing"]["rectangle"]
    coefficient, permutation = batch["mixing"]["coefficient"], batch["mixing"]["permutation"]
    assert coefficient == 1 - ((x2 - x1) * (y2 - y1) / 64)
    for key in ("image", "input_mask"):
        expected = original["inputs"][key].clone()
        expected[..., y1:y2, x1:x2] = original["inputs"][key][permutation, ..., y1:y2, x1:x2]
        torch.testing.assert_close(batch["inputs"][key], expected, rtol=0, atol=0)
    assert set(batch["inputs"]["input_mask"].unique().tolist()) <= {0, 1}


def test_zero_probability_mixing_preserves_batch():
    batch = example()
    original = copy.deepcopy(batch)
    MixUp(probability=0)(batch)
    assert "mixing" not in batch
    torch.testing.assert_close(batch["inputs"]["image"], original["inputs"]["image"], rtol=0, atol=0)
    assert batch["targets"]["label"].ndim == 1


class PointwiseSegmentation(nn.Module):
    def __init__(self, binary=False):
        super().__init__()
        self.binary = binary

    def forward(self, inputs, views=None):
        prediction = inputs["image"][:, :1] + inputs.get("input_mask", 0)
        return {"dense_logits": prediction if self.binary else torch.cat((prediction, -prediction), dim=1)}


@pytest.mark.parametrize("aggregation", ["mean_logits", "mean_probabilities"])
@pytest.mark.parametrize("binary", [False, True])
def test_tta_inverse_geometry_and_probability_aggregation(aggregation, binary):
    model = PointwiseSegmentation(binary).eval()
    inputs = {"image": torch.randn(2, 3, 5, 7), "input_mask": torch.rand(2, 1, 5, 7)}
    original = copy.deepcopy(inputs)
    state = torch.get_rng_state().clone()
    output = tta_forward(
        model,
        inputs,
        {
            "transforms": ["identity", "hflip", "vflip", "hvflip"],
            "aggregation": aggregation,
            "target_mode": "binary" if binary else "exclusive",
        },
    )
    expected = model(inputs)["dense_logits"]
    if aggregation == "mean_logits":
        torch.testing.assert_close(output["dense_logits"], expected)
    else:
        convert = torch.sigmoid if binary else lambda value: value.softmax(1)
        torch.testing.assert_close(convert(output["dense_logits"]), convert(expected))
    assert torch.equal(state, torch.get_rng_state())
    for key in inputs:
        torch.testing.assert_close(inputs[key], original[key], rtol=0, atol=0)


def test_tta_classification_aggregates_actual_different_view_predictions():
    class SpatialClassifier(nn.Module):
        def forward(self, inputs, views=None):
            image = inputs["image"]
            return {"logits": torch.stack((image[:, 0, 0, 0], image[:, 0, 0, -1]), dim=1)}

    model = SpatialClassifier().eval()
    image = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
    first = model({"image": image})["logits"]
    second = model({"image": image.flip(-1)})["logits"]
    output = TTA(model, aggregation="mean_probabilities")({"image": image})
    expected = (first.softmax(1) + second.softmax(1)) / 2
    torch.testing.assert_close(output["probabilities"], expected)
    torch.testing.assert_close(output["logits"].softmax(1), expected)
    with pytest.raises(ValueError, match="evaluation-mode"):
        TTA(SpatialClassifier())({"image": image})


def test_nested_arcface_inference_keeps_each_target_and_ignores_labels():
    main, auxiliary = ArcFaceObjective(4, 2), ArcFaceObjective(4, 3, target_key="aux")
    objective = CompositeObjective({"nested": CompositeObjective({"main": main}), "aux": auxiliary})
    embeddings = torch.randn(3, 4)
    raw = {"embeddings": embeddings, "logits": torch.randn(3, 2)}
    output = objective_inference_outputs(objective, raw, primary_target="label")
    assert set(output["logits"]) == {"label", "aux"}
    torch.testing.assert_close(output["logits"]["label"], main.inference_logits(embeddings))
    torch.testing.assert_close(output["logits"]["aux"], auxiliary.inference_logits(embeddings))
    assert isinstance(raw["logits"], torch.Tensor)


def test_posthoc_correction_is_never_used_for_training_and_not_duplicated():
    adjustment = LogitAdjustedLoss([1, 4], mode="posthoc", tau=0.5)
    raw = {"logits": torch.randn(3, 2)}
    training = objective_inference_outputs(adjustment, raw, apply_posthoc=False)
    torch.testing.assert_close(training["logits"], raw["logits"], rtol=0, atol=0)
    evaluation = objective_inference_outputs(adjustment, raw)
    torch.testing.assert_close(evaluation["logits"], raw["logits"] - 0.5 * adjustment.log_prior)
    duplicate = CompositeObjective({"one": adjustment, "two": LogitAdjustedLoss([1, 4], mode="posthoc")})
    with pytest.raises(ValueError, match="Duplicate post-hoc"):
        objective_inference_outputs(duplicate, raw)


def test_tta_supports_named_target_logits_from_composite_objectives():
    class NamedClassifier(nn.Module):
        def forward(self, inputs, views=None):
            image = inputs["image"]
            logits = torch.stack((image[:, 0, 0, 0], image[:, 0, 0, -1]), dim=1)
            return {
                "logits": {"label": logits, "aux": torch.cat((logits, -logits[:, :1]), 1)},
                "probabilities": {"label": torch.zeros_like(logits)},
            }

    model = NamedClassifier().eval()
    outputs = TTA(model, transforms=["identity", "hflip"], aggregation="mean_logits")(
        {"image": torch.randn(3, 3, 4, 4)}
    )
    assert set(outputs["logits"]) == {"label", "aux"}
    for key, value in outputs["logits"].items():
        torch.testing.assert_close(outputs["probabilities"][key], value.softmax(1))


def test_scheduled_freeze_holds_weights_bn_and_optimizer_state_then_unfreezes():
    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(0.1))
            self.head = nn.Linear(4, 2)
            self.backbone.requires_grad_(False)

        def forward(self, inputs):
            return self.head(self.backbone(inputs))

    torch.manual_seed(5)
    model = Network()
    schedule = {"backbone": 2}
    names = enable_scheduled_parameters(model, schedule)
    assert names and all(parameter.requires_grad for parameter in model.parameters())
    callback = ScheduledFreeze(schedule)
    trainer = SimpleNamespace(current_epoch=0)
    callback.setup(trainer, model, "fit")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.3)
    original = {name: value.detach().clone() for name, value in model.backbone.state_dict().items()}
    groups = len(optimizer.param_groups)
    for epoch in range(3):
        trainer.current_epoch = epoch
        model.train()
        callback.on_train_epoch_start(trainer, model)
        optimizer.zero_grad()
        model(torch.randn(5, 4)).square().mean().backward()
        callback.on_before_optimizer_step(trainer, model, optimizer)
        optimizer.step()
        assert len(optimizer.param_groups) == groups
        if epoch < 2:
            for name, value in model.backbone.state_dict().items():
                torch.testing.assert_close(value, original[name], rtol=0, atol=0)
            assert all(parameter not in optimizer.state for parameter in model.backbone.parameters())
        else:
            assert not torch.equal(model.backbone[0].weight, original["0.weight"])
            assert all(parameter in optimizer.state for parameter in model.backbone.parameters())
    state = callback.state_dict()
    resumed = ScheduledFreeze(schedule)
    resumed.load_state_dict(state)
    assert resumed.last_epoch == 2
    with pytest.raises(ValueError, match="differs"):
        ScheduledFreeze({"backbone": 3}).load_state_dict(state)
