"""Acceptance checks that exercise worker lifecycle and observational evaluation."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from test_lifecycle import assert_state_equal, recipe

from trainer_project.data import CsvDataset
from trainer_project.engine import train
from trainer_project.experiment.artifacts import load_checkpoint
from trainer_project.experiment.runtime import local_path


class TracingCsvDataset(CsvDataset):
    def __init__(self, *args, trace_dir, **kwargs):
        self.trace_dir = local_path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if self.role == "train":
            worker = torch.utils.data.get_worker_info()
            worker_id = worker.id if worker else "main"
            image = next(
                value
                for value in sample["inputs"].values()
                if isinstance(value, torch.Tensor) and value.ndim == 3
            )
            record = {
                "epoch": self.epoch,
                "sample_id": sample["sample_id"],
                "image_sha256": hashlib.sha256(image.numpy().tobytes()).hexdigest(),
            }
            with (self.trace_dir / f"epoch{self.epoch}_worker{worker_id}.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        return sample


def traces(path):
    return sorted(
        (row["epoch"], row["sample_id"], row["image_sha256"])
        for filename in path.glob("*.jsonl")
        for line in filename.read_text().splitlines()
        for row in [json.loads(line)]
    )


def test_strict_resume_with_two_workers_keeps_augmented_samples_and_state(tmp_path):
    cfg = recipe(tmp_path)
    cfg["data"]["num_workers"] = 2
    # Spatially asymmetric images make augmentation differences observable.
    yy, xx = np.indices((16, 16))
    for index in range(8):
        image = np.stack([xx * 8 + index, yy * 8 + index, (xx + yy) * 4 + index], axis=-1).astype(np.uint8)
        Image.fromarray(image).save(tmp_path / f"train{index}.png")
    cfg["data"]["dataset_factory"] = {
        "_target_": "test_acceptance_data.TracingCsvDataset",
        "_partial_": True,
        "trace_dir": str(tmp_path / "continuous_trace"),
    }
    continuous = train(cfg)
    interrupted_cfg = deepcopy(cfg)
    interrupted_cfg["data"]["dataset_factory"]["trace_dir"] = str(tmp_path / "resumed_trace")
    stopped = train(interrupted_cfg, stop_after_epoch=1)
    path = stopped / "checkpoints" / "last.ckpt"
    resumed = train(load_checkpoint(path)[1]["config"], checkpoint_path=path, strict=True)
    uninterrupted, expected = load_checkpoint(continuous / "checkpoints" / "last.ckpt")
    restarted, actual = load_checkpoint(resumed / "checkpoints" / "last.ckpt")
    assert traces(tmp_path / "continuous_trace") == traces(tmp_path / "resumed_trace")
    assert len(traces(tmp_path / "continuous_trace")) == 16
    assert_state_equal(uninterrupted["state_dict"], restarted["state_dict"])
    assert_state_equal(uninterrupted["optimizer_states"], restarted["optimizer_states"])
    assert expected["scheduler"] == actual["scheduler"]
    assert expected["successful_updates"] == actual["successful_updates"] == 8


def test_test_observers_preserve_training_model_optimizer_and_rng(tmp_path):
    cfg = recipe(tmp_path, "byol")
    with_observers = train(cfg)
    disabled = deepcopy(cfg)
    disabled["evaluation"]["test_during_fit"]["enabled"] = False
    without_observers = train(disabled)
    observed, expected = load_checkpoint(with_observers / "checkpoints" / "last.ckpt")
    silent, actual = load_checkpoint(without_observers / "checkpoints" / "last.ckpt")
    assert_state_equal(observed["state_dict"], silent["state_dict"])
    assert_state_equal(observed["optimizer_states"], silent["optimizer_states"])
    assert expected["scheduler"] == actual["scheduler"]
    assert expected["successful_updates"] == actual["successful_updates"]
    for rank_expected, rank_actual in zip(expected["rank_states"], actual["rank_states"]):
        assert_state_equal(rank_expected["rng"]["torch"], rank_actual["rng"]["torch"])
        assert rank_expected["rng"]["python"] == rank_actual["rng"]["python"]
        np.testing.assert_equal(rank_expected["rng"]["numpy"], rank_actual["rng"]["numpy"])
    assert any(
        key.startswith("test/") for key in json.loads((with_observers / "result.json").read_text())["metrics"]
    )
    assert not any(
        key.startswith("test/")
        for key in json.loads((without_observers / "result.json").read_text())["metrics"]
    )


def test_initialization_replay_and_reproduce_cli_include_trainable_objective(tmp_path, capsys):
    import pytest

    from trainer_project.cli import main
    from trainer_project.experiment.artifacts import load_initialization

    cfg = recipe(tmp_path)
    cfg["model"]["dropout"] = 0.3
    cfg["objective"] = {
        "_target_": "trainer_project.objectives.CompositeObjective",
        "components": {
            "ce": {"_target_": "trainer_project.objectives.MaskedCrossEntropy", "target_key": "label"},
            "arc": {
                "_target_": "trainer_project.objectives.ArcFaceObjective",
                "embedding_dim": 32,
                "num_classes": 2,
                "target_key": "label",
                "scale": 8.0,
                "margin": 0.1,
            },
        },
        "weights": {"ce": 1.0, "arc": 0.1},
    }
    original = train(cfg)
    weights_lock = json.loads((original / "weights.lock.json").read_text())
    initialization = load_initialization(
        original / "initialization.pt", weights_lock["initialization"]["sha256"]
    )
    assert initialization["rng"]
    assert "objective.components.arc.centers" in initialization["task_state"]
    expected, metadata = load_checkpoint(original / "checkpoints" / "last.ckpt")
    assert not torch.equal(
        initialization["task_state"]["objective.components.arc.centers"],
        expected["state_dict"]["objective.components.arc.centers"],
    )
    replayed = train(metadata["config"], initialization_path=original / "initialization.pt", prefix="replay")
    actual, actual_metadata = load_checkpoint(replayed / "checkpoints" / "last.ckpt")
    assert_state_equal(expected["state_dict"], actual["state_dict"])
    assert_state_equal(expected["optimizer_states"], actual["optimizer_states"])
    assert metadata["scheduler"] == actual_metadata["scheduler"]
    capsys.readouterr()
    main(["reproduce", "--config", str(original / "checkpoints" / "last.yaml")])
    reproduced = Path(json.loads(capsys.readouterr().out))
    from_cli, _ = load_checkpoint(reproduced / "checkpoints" / "last.ckpt")
    assert_state_equal(expected["state_dict"], from_cli["state_dict"])
    with pytest.raises(ValueError, match="SHA-256"):
        load_initialization(original / "initialization.pt", "0" * 64)
    corrupted = deepcopy(initialization)
    first = next(
        key
        for key, value in corrupted["model_state"].items()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    )
    corrupted["model_state"][first].add_(1)
    with (tmp_path / "corrupted_initialization.pt").open("xb") as stream:
        torch.save(corrupted, stream)
    with pytest.raises(ValueError, match="tensor identity"):
        load_initialization(tmp_path / "corrupted_initialization.pt")


def test_resume_inherits_original_initialization_reproduce_and_checkpoint_registry(tmp_path, capsys):
    import shutil

    from trainer_project.cli import main
    from trainer_project.experiment.runtime import digest

    cfg = recipe(tmp_path)
    cfg["model"]["dropout"] = 0.25
    continuous = train(cfg)
    stopped = train(cfg, stop_after_epoch=1)
    source_checkpoint = stopped / "checkpoints" / "last.ckpt"
    original_registry = json.loads((stopped / "checkpoints" / "registry.json").read_text())
    resumed = train(
        load_checkpoint(source_checkpoint)[1]["config"], checkpoint_path=source_checkpoint, strict=True
    )
    assert digest(stopped / "initialization.pt") == digest(resumed / "initialization.pt")
    inherited_lock = json.loads((resumed / "weights.lock.json").read_text())
    original_lock = json.loads((stopped / "weights.lock.json").read_text())
    assert inherited_lock["sources"] == original_lock["sources"]
    assert inherited_lock["initialization"]["sha256"] == original_lock["initialization"]["sha256"]
    assert inherited_lock["reproduce"]["available"]
    manifest = json.loads((resumed / "manifest.json").read_text())
    assert manifest["parent"] == {
        "mode": "resume",
        "checkpoint": str(source_checkpoint),
        "sha256": digest(source_checkpoint),
    }
    registry = json.loads((resumed / "checkpoints" / "registry.json").read_text())
    assert set(original_registry["top_k"]) <= set(registry["top_k"])
    assert all(Path(path).exists() for path in registry["top_k"])
    expected, _ = load_checkpoint(continuous / "checkpoints" / "last.ckpt")
    actual, _ = load_checkpoint(resumed / "checkpoints" / "last.ckpt")
    assert_state_equal(expected["state_dict"], actual["state_dict"])
    capsys.readouterr()
    main(["reproduce", "--config", str(resumed / "checkpoints" / "last.yaml")])
    reproduced = Path(json.loads(capsys.readouterr().out))
    from_cli, _ = load_checkpoint(reproduced / "checkpoints" / "last.ckpt")
    assert_state_equal(expected["state_dict"], from_cli["state_dict"])
    assert_state_equal(expected["optimizer_states"], from_cli["optimizer_states"])
    # A standalone copied checkpoint remains resumable, but cannot advertise a
    # reconstruction of initial pretrained weights it no longer possesses.
    moved = tmp_path / "moved" / "checkpoints" / "last.ckpt"
    moved.parent.mkdir(parents=True)
    shutil.copyfile(source_checkpoint, moved)
    moved_resume = train(load_checkpoint(moved)[1]["config"], checkpoint_path=moved, strict=True)
    unavailable = json.loads((moved_resume / "weights.lock.json").read_text())
    assert unavailable["reproduce"]["available"] is False
    assert unavailable["initialization"] is None
    assert not (moved_resume / "initialization.pt").exists()
