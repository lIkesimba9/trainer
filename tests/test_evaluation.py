import copy
import json
import random
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from trainer_project.evaluation import (
    EvaluationAccumulator,
    KnnEvaluator,
    LinearProbeEvaluator,
    TemperatureCalibrator,
    classification_metrics,
    ensemble_predictions,
    read_predictions,
    validate_oof,
)
from trainer_project.reporting import build_report


def batch(ids, targets, *, row_indices=None):
    return {
        "sample_id": ids,
        "targets": {"label": torch.tensor(targets)},
        "target_valid": {"label": torch.tensor([label >= 0 for label in targets])},
        "supervision_allowed": {"label": torch.ones(len(ids), dtype=torch.bool)},
        "metadata": {"row_index": torch.tensor(row_indices or list(range(len(ids))))},
    }


def test_pooled_is_global_confusion_and_padding_is_removed():
    evaluator = EvaluationAccumulator("classification", {"label": {"a": 0, "b": 1}})
    evaluator.update(
        {"logits": torch.tensor([[5.0, 0.0], [5.0, 0.0]])}, batch(["a0", "a1"], [0, 1]), "val", "A"
    )
    evaluator.update({"logits": torch.tensor([[0.0, 5.0]])}, batch(["b0"], [1]), "val", "B")
    evaluator.update({"logits": torch.tensor([[5.0, 0.0]])}, batch(["a0"], [0]), "val", "A")
    metrics = evaluator.compute()
    assert metrics["val/pooled/observations"] == 3
    assert metrics["val/pooled/f1_macro"] == pytest.approx(2 / 3)
    average = (metrics["val/A/f1_macro"] + metrics["val/B/f1_macro"]) / 2
    assert metrics["val/pooled/f1_macro"] != pytest.approx(average)
    assert metrics["val/pooled/count"] == 3


def test_repeated_training_observations_remain_and_ssl_keeps_all_labels():
    evaluator = EvaluationAccumulator("ssl", {"label": ["a", "b"]})
    data = batch(["same"], [0])
    data["targets"]["aux"] = torch.tensor([-1])
    data["target_valid"]["aux"] = torch.tensor([False])
    for _ in range(2):
        evaluator.update({"embeddings": torch.tensor([[1.0, 2.0]])}, data, "train", "A")
    assert evaluator.compute()["train/pooled/observations"] == 2
    assert evaluator.records[0]["targets"] == {"label": 0, "aux": -1}
    assert evaluator.records[0]["target_valid"]["aux"] is False
    assert evaluator.records[0]["embeddings"] == [1.0, 2.0]


def test_distributed_shards_count_uneven_evaluation_rows_once(monkeypatch):
    import trainer_project.evaluation.accumulator as module

    first = EvaluationAccumulator("classification", {"label": ["a", "b"]})
    first.update(
        {"logits": torch.tensor([[5.0, 0.0], [0.0, 5.0]])},
        batch(["a0", "a2"], [0, 1], row_indices=[0, 2]),
        "val",
        "A",
    )
    second = EvaluationAccumulator("classification", {"label": ["a", "b"]})
    second.update(
        {"logits": torch.tensor([[0.0, 5.0], [5.0, 0.0]])},
        batch(["a1", "a0"], [1, 0], row_indices=[1, 0]),
        "val",
        "A",
    )
    monkeypatch.setattr(module.dist, "is_available", lambda: True)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        module.dist,
        "all_gather_object",
        lambda result, local: result.__setitem__(slice(None), [local, second.records]),
    )
    metrics = first.compute()
    assert metrics["val/pooled/observations"] == 3
    assert metrics["val/pooled/count"] == 3
    assert metrics["val/pooled/accuracy"] == 1


def test_class_universe_and_unknown_targets():
    result = classification_metrics([0], [[1.0, 0.0, 0.0]], ["a", "b", "c"])
    assert result["f1_macro"] == pytest.approx(1 / 3)
    assert result["per_class"]["b"]["auroc"] is None
    assert result["auroc_macro"] is None


def test_segmentation_invalid_target_values_do_not_affect_metrics():
    def evaluate(value):
        evaluator = EvaluationAccumulator(
            "segmentation", {"mask": ["background", "object"]}, target_key="mask"
        )
        data = {
            "sample_id": ["s"],
            "targets": {"mask": torch.tensor([[[0, value], [1, 1]]])},
            "target_valid": {"mask": torch.tensor([[[True, False], [True, True]]])},
            "metadata": {"original_size": torch.tensor([[2, 2]]), "row_index": torch.tensor([0])},
        }
        evaluator.update(
            {"dense_logits": torch.tensor([[[[8.0, -8.0], [-8.0, -8.0]], [[-8.0, 8.0], [8.0, 8.0]]]])},
            data,
            "val",
            "A",
        )
        return evaluator.compute(), evaluator

    first, evaluator = evaluate(255)
    second, _ = evaluate(19)
    assert first == second
    assert first["val/pooled/valid_pixels"] == 3
    assert first["val/pooled/dice_macro"] == 1
    assert evaluator.records[0]["predicted_mask"] == [[0, 1], [1, 1]]


def test_multilabel_is_not_silently_converted():
    evaluator = EvaluationAccumulator("segmentation", {"mask": ["a", "b"]}, target_key="mask")
    data = {
        "sample_id": ["s"],
        "targets": {"mask": torch.ones((1, 2, 2, 2))},
        "target_valid": {"mask": torch.ones((1, 2, 2, 2), dtype=torch.bool)},
    }
    with pytest.raises(ValueError, match="multilabel"):
        evaluator.update({"dense_logits": torch.zeros((1, 2, 2, 2))}, data, "val", "A")


def test_segmentation_restores_each_original_coordinate_system(tmp_path):
    original_a = torch.zeros((3, 5), dtype=torch.long)
    original_a[0, 0] = 255
    original_b = torch.ones((4, 6), dtype=torch.long)
    evaluator = EvaluationAccumulator("segmentation", {"mask": ["background", "object"]}, target_key="mask")
    data = {
        "sample_id": ["a", "b"],
        "targets": {"mask": torch.zeros((2, 8, 8), dtype=torch.long)},
        "target_valid": {"mask": torch.ones((2, 8, 8), dtype=torch.bool)},
        "metadata": {
            "original_size": torch.tensor([[3, 5], [4, 6]]),
            "row_index": torch.tensor([0, 1]),
            "original_targets": {"mask": [original_a, original_b]},
            "original_target_valid": {
                "mask": [original_a != 255, torch.ones_like(original_b, dtype=torch.bool)]
            },
        },
    }
    logits = torch.zeros((2, 2, 8, 8))
    logits[0, 0], logits[1, 1] = 8.0, 8.0
    evaluator.update({"dense_logits": logits}, data, "val", "A")
    metrics = evaluator.compute()
    assert metrics["val/pooled/valid_pixels"] == 38
    assert metrics["val/pooled/dice_macro"] == 1
    assert np.asarray(evaluator.records[0]["predicted_mask"]).shape == (3, 5)
    assert np.asarray(evaluator.records[1]["predicted_mask"]).shape == (4, 6)
    # Stored protocol supplies the non-default segmentation target to the reporter.
    build_report(evaluator.records, tmp_path / "original_coordinates")
    report = json.loads((tmp_path / "original_coordinates" / "report.json").read_text())
    assert report["metrics"]["val/pooled/dice_macro"] == 1


def test_knn_heldout_reference_and_leave_one_out():
    ref = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    query = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    evaluator = KnnEvaluator(k=1, num_classes=2)
    result = evaluator.predict(ref, [0, 1], query, reference_ids=["r0", "r1"], query_ids=["q0", "q1"])
    assert result.argmax(1).tolist() == [0, 1]
    with pytest.raises(ValueError, match="overlap"):
        evaluator.predict(ref, [0, 1], query, reference_ids=["r0", "r1"], query_ids=["r0", "q1"])
    with pytest.raises(ValueError, match="only for SSL"):
        evaluator.predict(
            ref, [0, 1], query, reference_ids=["r0", "r1"], query_ids=["q0", "q1"], task="classification"
        )
    with pytest.raises(ValueError, match="validation"):
        evaluator.predict(
            ref, [0, 1], query, reference_ids=["r0", "r1"], query_ids=["q0", "q1"], query_role="test"
        )
    loo = KnnEvaluator(k=1, protocol="val_leave_one_out", num_classes=2)
    prediction = loo.predict(ref, [0, 1], ref, reference_ids=["v0", "v1"], query_ids=["v0", "v1"])
    assert prediction.argmax(1).tolist() == [1, 0]


def test_probe_uses_separate_head_preserves_rng_and_features():
    ref = torch.tensor([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.1, 1.0]])
    query = ref[[0, 2]].clone()
    original = ref.clone()
    torch.manual_seed(53)
    random.seed(12)
    np.random.seed(9)
    state = torch.get_rng_state().clone()
    python = random.getstate()
    numpy = np.random.get_state()
    probe = LinearProbeEvaluator(num_classes=2, epochs=40, lr=0.3, seed=2)
    result = probe.predict(
        ref, [0, 0, 1, 1], query, reference_ids=["r0", "r1", "r2", "r3"], query_ids=["q0", "q1"]
    )
    assert result.argmax(1).tolist() == [0, 1]
    assert torch.equal(torch.get_rng_state(), state)
    assert random.getstate() == python
    assert np.array_equal(np.random.get_state()[1], numpy[1])
    assert torch.equal(ref, original)
    assert probe.head_state is not None


def test_representation_evaluators_keep_declared_missing_reference_classes():
    reference = torch.tensor([[1.0, 0.0], [0.9, 0.1]])
    query = torch.tensor([[1.0, 0.0]])
    kwargs = {"reference_ids": ["r0", "r1"], "query_ids": ["q0"]}
    assert KnnEvaluator(k=1, num_classes=3).predict(reference, [0, 0], query, **kwargs).shape == (1, 3)
    with torch.inference_mode():
        assert LinearProbeEvaluator(num_classes=3, epochs=2).predict(
            reference, [0, 0], query, **kwargs
        ).shape == (1, 3)


def test_task_excludes_padding_from_eval_loss_and_keeps_prediction_targets():
    from trainer_project.modules.task import ResearchTask
    from trainer_project.objectives import MaskedCrossEntropy

    data = batch(["real", "padding"], [1, 0], row_indices=[1, 0])
    data["input_present"] = {"image": torch.tensor([True, True])}
    evaluator = EvaluationAccumulator("classification", {"label": ["a", "b"]})
    task = SimpleNamespace(
        experiment_config={"task": "classification"},
        _outputs=lambda value: {"logits": torch.tensor([[0.0, 8.0], [0.0, 8.0]])},
        objective=MaskedCrossEntropy(),
        trainer=SimpleNamespace(world_size=2),
        global_rank=1,
        device=torch.device("cpu"),
        current_epoch=0,
        successful_updates=0,
        _losses=defaultdict(lambda: [0.0, 0.0]),
        evaluator=evaluator,
    )
    ResearchTask._evaluation_step(task, data, "val", "A")
    numerator, denominator = task._losses["val/A/loss"]
    assert denominator == 1
    assert numerator / denominator < 0.001
    assert evaluator.records[1]["target_valid"]["label"] is True
    assert data["target_valid"]["label"].tolist() == [True, True]


def test_task_ssl_query_features_do_not_use_mae_masking():
    from trainer_project.modules.task import ResearchTask
    from trainer_project.objectives.base import ObjectiveResult

    data = batch(["query"], [-1])
    data["inputs"] = {"image": torch.zeros((1, 3, 2, 2))}
    evaluator = EvaluationAccumulator("ssl", {"label": ["a", "b"]})

    class MaskedModel(torch.nn.Module):
        def forward(self, inputs, views=None):
            return {"embeddings": torch.tensor([[0.0, 0.0]])}

        def extract_features(self, inputs):
            return torch.tensor([[7.0, 9.0]])

    class ReconstructionObjective(torch.nn.Module):
        def forward(self, *args):
            return ObjectiveResult(torch.tensor(0.0), counts={"masked_patches": 1})

    task = ResearchTask({"task": "ssl"}, model=MaskedModel(), objective=ReconstructionObjective())
    task._trainer = SimpleNamespace(world_size=1, current_epoch=0)
    task.evaluator = evaluator
    task.eval()
    ResearchTask._evaluation_step(task, data, "val", "A")
    assert evaluator.records[0]["embeddings"] == [7.0, 9.0]


def prediction_rows():
    return [
        {
            "sample_id": str(i),
            "manifest_name": "A",
            "role": "val",
            "task": "classification",
            "row_index": i,
            "targets": {"label": label},
            "target_valid": {"label": True},
            "metadata": {},
            "class_maps": {"label": {"a": 0, "b": 1}},
            "probabilities": {"label": probs},
            "provenance": {"checkpoint_hash": "model"},
        }
        for i, (label, probs) in enumerate([(0, [0.8, 0.2]), (1, [0.1, 0.9])])
    ]


def test_ensemble_aligns_rows_and_class_names():
    first = prediction_rows()
    second = copy.deepcopy(first[::-1])
    for row in second:
        row["class_maps"] = {"label": {"b": 0, "a": 1}}
        row["targets"]["label"] = 1 - row["targets"]["label"]
        row["probabilities"]["label"] = row["probabilities"]["label"][::-1]
    combined = ensemble_predictions([first, second], weights=[1, 3])
    assert [row["sample_id"] for row in combined] == ["0", "1"]
    assert combined[0]["probabilities"]["label"] == pytest.approx([0.8, 0.2])
    with pytest.raises(ValueError, match="missing"):
        ensemble_predictions([first, second[:1]])
    second[0]["targets"]["label"] = 1 - second[0]["targets"]["label"]
    with pytest.raises(ValueError, match="targets conflict"):
        ensemble_predictions([first, second])


def test_oof_distinguishes_label_and_image_leakage():
    rows = prediction_rows()
    membership = {"model": {"supervised_targets": {"label": ["other"]}, "ssl": ["0", "1"]}}
    result = validate_oof(rows, membership)
    assert "transductive" in result["protocol"]
    with pytest.raises(ValueError, match="image leakage"):
        validate_oof(rows, membership, require_image_independence=True)
    membership["model"]["supervised_targets"]["label"].append("0")
    with pytest.raises(ValueError, match="label leakage"):
        validate_oof(rows, membership)


def test_report_rebuilds_from_predictions_and_preserves_existing_files(tmp_path):
    rows = prediction_rows()
    path = tmp_path / "predictions.jsonl"
    evaluator = EvaluationAccumulator("classification", {"label": ["a", "b"]})
    evaluator.add_records(rows)
    evaluator.write_predictions(path)
    restored = read_predictions(path)
    assert [row["targets"] for row in restored] == [row["targets"] for row in rows]
    assert restored[0]["evaluation_protocol"]["target_key"] == "label"
    report = build_report(path, tmp_path / "report")
    content = json.loads((tmp_path / "report" / "report.json").read_text())
    assert content["metrics"]["val/pooled/f1_macro"] == 1
    assert "Trainer evaluation report" in (tmp_path / "report" / "report.html").read_text()
    assert report["csv"].endswith("predictions.csv")
    with pytest.raises(FileExistsError):
        evaluator.write_predictions(path)


def test_calibration_requires_allowed_explicit_source():
    calibrator = TemperatureCalibrator()
    with pytest.raises(ValueError, match="validation/OOF"):
        calibrator.fit([[1.0, 0.0]], [0], data_role="test", source="test.csv")
    calibrator.fit(
        [[8.0, 0.0], [8.0, 0.0], [0.0, 8.0], [0.0, 8.0]],
        [0, 1, 1, 1],
        data_role="val",
        source="val.csv",
        max_iter=10,
    )
    assert calibrator.temperature > 1
    assert calibrator.state_dict()["provenance"]["tuning_result"] is True


def test_segmentation_report_exports_lossless_indexed_png(tmp_path):
    from PIL import Image

    evaluator = EvaluationAccumulator("segmentation", {"mask": ["background", "object"]}, target_key="mask")
    data = {
        "sample_id": ["s"],
        "targets": {"mask": torch.tensor([[[0, 1], [1, 255]]])},
        "target_valid": {"mask": torch.tensor([[[True, True], [True, False]]])},
        "metadata": {"original_size": torch.tensor([[2, 2]]), "row_index": torch.tensor([0])},
    }
    evaluator.update(
        {"dense_logits": torch.tensor([[[[8.0, -8.0], [-8.0, 8.0]], [[-8.0, 8.0], [8.0, -8.0]]]])},
        data,
        "val",
        "A",
    )
    build_report(evaluator.records, tmp_path / "segmentation_report", target_key="mask")
    index = json.loads((tmp_path / "segmentation_report" / "masks.json").read_text())
    with Image.open(tmp_path / "segmentation_report" / index[0]["mask"]) as image:
        assert image.mode == "L"
        assert np.asarray(image).tolist() == [[0, 1], [1, 0]]


def test_ensemble_cli_writes_predictions_and_report(tmp_path, capsys):
    from trainer_project.cli import main
    from trainer_project.evaluation import write_predictions

    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_predictions(prediction_rows(), first)
    write_predictions(prediction_rows()[::-1], second)
    output = tmp_path / "ensemble"
    main(["ensemble", "--predictions", str(first), str(second), "--output", str(output)])
    rows = read_predictions(output / "predictions.jsonl")
    assert len(rows) == 2
    assert rows[0]["probabilities"]["label"] == pytest.approx([0.8, 0.2])
    assert (output / "reports" / "report.html").exists()
