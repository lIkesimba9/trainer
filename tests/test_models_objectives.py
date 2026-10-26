import copy
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from trainer_project.models import (
    ArcFaceHead,
    BYOLModel,
    ImageClassifier,
    MAEModel,
    MultiModalModel,
    TinySegmenter,
)
from trainer_project.objectives import (
    ArcFaceObjective,
    BYOLObjective,
    CompositeObjective,
    FocalLoss,
    LDAMLoss,
    LogitAdjustedLoss,
    MAEObjective,
    MaskedCrossEntropy,
    SeesawLoss,
    SegmentationObjective,
    TripletLoss,
)


def batch(labels=(0, 1, -1), valid=(True, True, False)):
    labels = torch.tensor(labels)
    return {
        "targets": {"label": labels, "aux": torch.tensor([2] * len(labels))},
        "target_valid": {"label": torch.tensor(valid), "aux": torch.ones(len(labels), dtype=torch.bool)},
        "supervision_allowed": {"label": torch.ones(len(labels), dtype=torch.bool)},
    }


def test_classifier_predictions_do_not_depend_on_labels():
    torch.manual_seed(4)
    model = ImageClassifier(num_classes=3, embedding_dim=8).eval()
    inputs = {"image": torch.rand(3, 3, 12, 12)}
    original = model(inputs)["logits"]
    altered_batch = batch((2, 2, 2), (True, True, True))
    assert altered_batch["targets"]["label"].tolist() == [2, 2, 2]
    torch.testing.assert_close(model(inputs)["logits"], original, rtol=0, atol=0)


@pytest.mark.parametrize(
    "objective",
    [
        MaskedCrossEntropy(),
        FocalLoss(),
        LDAMLoss([4, 2]),
        LogitAdjustedLoss([4, 2]),
        SeesawLoss(2),
    ],
)
def test_classification_missing_labels_have_no_gradient(objective):
    logits = torch.tensor([[0.4, -0.1], [0.1, 0.7], [-2.0, 4.0]], requires_grad=True)
    result = objective({"logits": logits}, batch(), {"stage": "train", "epoch": 0})
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert logits.grad[-1].abs().sum() == 0
    assert sum(float(x) for x in result.counts.values()) == 2


def test_all_unlabeled_cross_entropy_graph_zero():
    logits = torch.randn(3, 2, requires_grad=True)
    result = MaskedCrossEntropy()({"logits": logits}, batch((-1, -1, -1), (False, False, False)), {})
    result.loss.backward()
    assert result.loss.item() == 0
    assert logits.grad.abs().sum() == 0


def test_supervision_permission_is_distinct_from_label_validity():
    logits = torch.randn(3, 2, requires_grad=True)
    example = batch((0, 1, 0), (True, True, True))
    example["supervision_allowed"]["label"] = torch.tensor([False, True, False])
    MaskedCrossEntropy()({"logits": logits}, example, {}).loss.backward()
    assert logits.grad[[0, 2]].abs().sum() == 0
    assert logits.grad[1].abs().sum() > 0


def test_arcface_centers_registered_optimized_restored_and_inference_independent():
    torch.manual_seed(1)
    objective = ArcFaceObjective(8, 2, subcenters=3)
    embeddings = torch.randn(3, 8, requires_grad=True)
    optimizer = torch.optim.SGD(objective.parameters(), lr=0.01)
    before = objective.centers.detach().clone()
    predictions = objective.inference_logits(embeddings).detach().clone()
    objective({"embeddings": embeddings}, batch(), {}).loss.backward()
    optimizer.step()
    assert not torch.equal(before, objective.centers)
    restored = ArcFaceObjective(8, 2, subcenters=3)
    restored.load_state_dict(objective.state_dict())
    torch.testing.assert_close(restored.inference_logits(embeddings), objective.inference_logits(embeddings))
    assert predictions.shape == (3, 2)
    assert ArcFaceHead(8, 2, subcenters=3)(embeddings).shape == (3, 2)


def test_seesaw_evaluation_does_not_mutate_counts():
    objective = SeesawLoss(2)
    logits = torch.randn(3, 2)
    objective({"logits": logits}, batch(), {"stage": "train"})
    counts = objective.class_counts.clone()
    objective({"logits": logits}, batch(), {"stage": "val"})
    torch.testing.assert_close(objective.class_counts, counts, rtol=0, atol=0)
    assert counts.tolist() == [1, 1]


def test_triplet_empty_miner_graph_zero():
    embeddings = torch.randn(3, 8, requires_grad=True)
    result = TripletLoss()({"embeddings": embeddings}, batch((0, 1, -1)), {})
    result.loss.backward()
    assert result.loss.item() == 0
    assert embeddings.grad.abs().sum() == 0


def test_triplet_has_positives_and_negatives():
    embeddings = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.1, 0.1]], requires_grad=True)
    example = batch((0, 0, 1), (True, True, True))
    result = TripletLoss()({"embeddings": embeddings}, example, {})
    assert result.loss.item() > 0
    assert float(result.counts["triplet"]) == 2


def test_byol_teacher_update_is_successful_step_only_and_checkpointed():
    torch.manual_seed(2)
    model = BYOLModel(feature_dim=8, projection_dim=4, hidden_dim=8)
    objective = BYOLObjective()
    image = torch.randn(3, 3, 8, 8)
    outputs = model({"image": image}, {"view1": image, "view2": image.flip(-1)})
    result = objective(outputs, {}, {})
    result.loss.backward()
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step()
    teacher_before = next(model.target_encoder.parameters()).clone()
    model.update_target(0, 10)
    torch.testing.assert_close(next(model.target_encoder.parameters()), teacher_before)
    model.update_target(1, 10)
    assert not torch.equal(next(model.target_encoder.parameters()), teacher_before)
    teacher_after = next(model.target_encoder.parameters()).clone()
    model.update_target(1, 10)
    torch.testing.assert_close(next(model.target_encoder.parameters()), teacher_after)
    model.eval()
    assert set(model({"image": image})) == {"features", "embeddings"}
    assert not model.target_encoder.training
    restored = BYOLModel(feature_dim=8, projection_dim=4, hidden_dim=8)
    restored.load_state_dict(model.state_dict())
    assert restored.ema_updates.item() == 1


def test_mae_masks_resume_and_feature_extraction_does_not_mask():
    torch.manual_seed(2)
    model = MAEModel(
        image_size=8,
        patch_size=2,
        embedding_dim=8,
        decoder_dim=8,
        encoder_depth=1,
        decoder_depth=1,
        num_heads=2,
        mask_ratio=0.5,
    )
    image = torch.randn(2, 3, 8, 8)
    outputs = model({"image": image})
    assert outputs["ssl_patch_mask"].sum().item() == 16
    assert outputs["reconstruction"].shape == (2, 16, 12)
    loss = MAEObjective()(outputs, {}, {}).loss
    loss.backward()
    assert model.reconstruction_head.weight.grad.abs().sum() > 0
    state = copy.deepcopy(model.state_dict())
    expected = model({"image": image})
    restored = MAEModel(
        image_size=8,
        patch_size=2,
        embedding_dim=8,
        decoder_dim=8,
        encoder_depth=1,
        decoder_depth=1,
        num_heads=2,
        mask_ratio=0.5,
    )
    restored.load_state_dict(state)
    actual = restored({"image": image})
    assert torch.equal(actual["ssl_patch_mask"], expected["ssl_patch_mask"])
    torch.testing.assert_close(actual["reconstruction"], expected["reconstruction"], rtol=0, atol=0)
    generator_state = restored.mask_generator.get_state().clone()
    first = restored.extract_features({"image": image})
    second = restored.extract_features({"image": image})
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(generator_state, restored.mask_generator.get_state())
    restored.eval()
    eval_first = restored({"image": image})
    eval_second = restored({"image": image})
    assert torch.equal(eval_first["ssl_patch_mask"], eval_second["ssl_patch_mask"])


@pytest.mark.parametrize("mode", ["exclusive", "binary", "multilabel"])
def test_segmentation_invalid_area_does_not_change_loss_or_gradient(mode):
    torch.manual_seed(8)
    channels = 1 if mode == "binary" else 2
    logits = torch.randn(2, channels, 4, 4, requires_grad=True)
    target = torch.randint(0, 2, (2, 4, 4) if mode == "exclusive" else (2, channels, 4, 4))
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[..., :2, :2] = False
    data = {"targets": {"segmentation_target": target}, "target_valid": {"segmentation_target": valid}}
    objective = SegmentationObjective(target_mode=mode, dice_weight=0.5)
    first = objective({"dense_logits": logits}, data, {}).loss
    first.backward()
    gradient = logits.grad.detach().clone()
    assert gradient[..., :2, :2].abs().sum() == 0
    logits.grad.zero_()
    data["targets"]["segmentation_target"] = target.masked_fill(~valid, 991)
    second = objective({"dense_logits": logits}, data, {}).loss
    second.backward()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(logits.grad, gradient, rtol=0, atol=0)


def test_multilabel_allows_overlapping_positive_channels():
    objective = SegmentationObjective(target_mode="multilabel", dice_weight=0.5)
    logits = torch.randn(2, 2, 4, 4, requires_grad=True)
    target = torch.ones_like(logits)
    result = objective({"dense_logits": logits}, {"targets": {"segmentation_target": target}}, {})
    result.loss.backward()
    assert torch.isfinite(result.loss)


@pytest.mark.parametrize("profile", ["image", "image_image", "image_image_text", "image_mask"])
def test_required_multimodal_profiles_forward(profile):
    model = MultiModalModel(profile=profile, feature_dim=8, embedding_dim=8, num_classes=3)
    image = torch.randn(2, 3, 8, 8)
    inputs = {
        "image": image,
        "image_a": image,
        "image_b": image.flip(-1),
        "input_mask": torch.ones(2, 1, 8, 8),
        "text": {
            "input_ids": torch.tensor([[1, 2, 0], [2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        },
    }
    assert model(inputs)["logits"].shape == (2, 3)
    assert model.extract_features(inputs).shape == (2, 8)


def test_composite_registers_components_and_preserves_fields():
    class Custom(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.coefficient = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, outputs, example, context):
            from trainer_project.objectives import ObjectiveResult

            assert "aux" in example["targets"]
            return ObjectiveResult(outputs["logits"].square().mean() * self.coefficient)

    objective = CompositeObjective({"main": MaskedCrossEntropy(), "custom": Custom()}, {"custom": 0.5})
    logits = torch.randn(3, 2, requires_grad=True)
    objective({"logits": logits}, batch(), {}).loss.backward()
    assert objective.components["custom"].coefficient.grad is not None
    assert "components.custom.coefficient" in objective.state_dict()


def test_tiny_segmenter_preserves_image_coordinates():
    model = TinySegmenter(num_classes=3)
    assert model({"image": torch.randn(2, 3, 11, 13)})["dense_logits"].shape == (2, 3, 11, 13)


def test_ddp_masked_gradient_matches_global_batch_with_unlabeled_rank():
    """Both ranks join collectives, although one has no supervised observations."""
    worker = r"""
import json, sys, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from trainer_project.objectives import MaskedCrossEntropy
rank, port = map(int, sys.argv[1:])
torch.set_num_threads(1)
dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
torch.manual_seed(19)
model = DistributedDataParallel(torch.nn.Linear(3, 2, bias=False))
all_inputs = torch.tensor([[1.,2.,3.], [3.,2.,1.], [2.,4.,1.], [5.,2.,1.]])
labels = torch.tensor([0,1,-1,-1])[rank*2:(rank+1)*2]
valid = labels >= 0
data = {"targets":{"label":labels},"target_valid":{"label":valid}}
loss = MaskedCrossEntropy()({"logits":model(all_inputs[rank*2:(rank+1)*2])}, data, {}).loss
loss.backward()
print("RESULT=" + json.dumps({"gradient": model.module.weight.grad.tolist(), "loss":loss.item()}), flush=True)
dist.destroy_process_group()
"""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    project = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": str(project / "src")}
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", worker, str(rank), str(port)],
            cwd=project,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for rank in range(2)
    ]
    results = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=60)
        assert process.returncode == 0, stderr
        payload = next(
            line.removeprefix("RESULT=") for line in stdout.splitlines() if line.startswith("RESULT=")
        )
        results.append(json.loads(payload))
    torch.manual_seed(19)
    reference = torch.nn.Linear(3, 2, bias=False)
    target = torch.tensor([0, 1])
    expected_loss = torch.nn.functional.cross_entropy(
        reference(torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])), target
    )
    expected_loss.backward()
    for observed in results:
        torch.testing.assert_close(torch.tensor(observed["gradient"]), reference.weight.grad)
        assert observed["loss"] == pytest.approx(expected_loss.item())
