import json
import subprocess
from copy import deepcopy

import pytest
import torch
from omegaconf import OmegaConf

from trainer_project.engine import train
from trainer_project.evaluation import read_predictions
from trainer_project.experiment.artifacts import load_checkpoint
from trainer_project.experiment.config import load_config
from trainer_project.experiment.runtime import WORKSPACE


def demo_recipe(tmp_path, method):
    overrides = ["task=ssl", f"model={method}", f"objective={method}"]
    if method == "byol":
        overrides.append("transforms.ssl=true")
    config = load_config(WORKSPACE / "examples" / "classification.yaml", overrides=overrides)
    config["output_dir"] = str(tmp_path / "runs")
    config["trainer"].update(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    config["logger"] = None
    config["checkpoint"].update(monitor="val/pooled/loss", mode="min")
    config["ssl_evaluation"] = {
        "enabled": True,
        "target_key": "label",
        "reference": {"split": "train", "manifests": ["train_a", "train_b"], "labeled_only": True},
        "knn": {"enabled": True, "k": 3},
        "linear_probe": {"enabled": True, "epochs": 3, "lr": 0.1, "seed": 15},
    }
    return config


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("method", ["byol", "mae"])
def test_periodic_knn_probe_are_val_only_and_preserve_training_trajectory(tmp_path, method):
    config = demo_recipe(tmp_path, method)
    baseline_config = deepcopy(config)
    baseline_config["ssl_evaluation"]["enabled"] = False
    baseline = train(baseline_config)
    evaluated = train(config)
    original, original_payload = load_checkpoint(baseline / "checkpoints" / "last.ckpt")
    actual, actual_payload = load_checkpoint(evaluated / "checkpoints" / "last.ckpt")
    assert_tree_equal(original["state_dict"], actual["state_dict"])
    assert_tree_equal(original["optimizer_states"], actual["optimizer_states"])
    assert original_payload["scheduler"] == actual_payload["scheduler"]
    metrics = json.loads((evaluated / "result.json").read_text())["metrics"]
    for evaluator in ("knn", "linear_probe"):
        for manifest in ("val_a", "val_b", "pooled"):
            assert f"val/{evaluator}/{manifest}/f1_macro" in metrics
        assert metrics[f"val/{evaluator}/pooled/observations"] == 8
        assert not any(key.startswith(f"test/{evaluator}/") for key in metrics)
        artifacts = sorted((evaluated / "predictions").glob(f"val-{evaluator}-*.jsonl"))
        assert len(artifacts) == 2
        records = read_predictions(artifacts[-1])
        assert {row["role"] for row in records} == {"val"}
        assert {row["manifest_name"] for row in records} == {"val_a", "val_b"}
        assert all(row["probabilities"]["label"] for row in records)
    # Compare future CPU random draws from saved boundary states, after probe fit.
    assert_tree_equal(
        original_payload["rank_states"][0]["rng"]["torch"], actual_payload["rank_states"][0]["rng"]["torch"]
    )


@pytest.mark.parametrize("command", ["knn", "linear_probe"])
def test_standalone_representation_cli_and_report(tmp_path, command):
    config = demo_recipe(tmp_path, "byol")
    config["trainer"]["max_epochs"] = 1
    config["ssl_evaluation"]["enabled"] = False
    run = train(config)
    patch = {"ssl_evaluation": deepcopy(config["ssl_evaluation"])}
    patch["ssl_evaluation"]["enabled"] = True
    eval_config = tmp_path / "evaluation.yaml"
    OmegaConf.save(OmegaConf.create(patch), eval_config)
    output = tmp_path / command
    completed = subprocess.run(
        [
            str(WORKSPACE / ".venv" / "bin" / "trainer"),
            command,
            "--checkpoint",
            str(run / "checkpoints" / "last.ckpt"),
            "--eval-config",
            str(eval_config),
            "--output",
            str(output),
        ],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    metrics = json.loads((output / "metrics.json").read_text())
    assert f"val/{command}/pooled/f1_macro" in metrics
    predictions = next((output / "predictions").glob(f"val-{command}-*.jsonl"))
    report = tmp_path / f"{command}_report"
    completed = subprocess.run(
        [
            str(WORKSPACE / ".venv" / "bin" / "trainer"),
            "report",
            "--predictions",
            str(predictions),
            "--output",
            str(report),
        ],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert (report / "report.html").is_file()


def test_leave_one_out_periodic_needs_no_reference_or_folds(tmp_path):
    config = demo_recipe(tmp_path, "mae")
    config["trainer"]["max_epochs"] = 1
    config["data"]["val"] = config["data"]["val"][:1]
    config["ssl_evaluation"].pop("reference")
    config["ssl_evaluation"]["knn"]["protocol"] = "val_leave_one_out"
    config["ssl_evaluation"]["linear_probe"]["enabled"] = False
    run = train(config)
    metrics = json.loads((run / "result.json").read_text())["metrics"]
    assert metrics["val/knn/val_a/f1_macro"] == metrics["val/knn/pooled/f1_macro"]
    assert metrics["val/knn/pooled/observations"] == 5
