import csv
import json

import torch
from test_lifecycle import recipe

from trainer_project.engine import train
from trainer_project.experiment.artifacts import load_checkpoint
from trainer_project.schedulers import ScheduleController


def test_combined_loader_horizon_and_sampled_source_counts(tmp_path):
    cfg = recipe(tmp_path)
    with (tmp_path / "train.csv").open() as stream:
        rows = list(csv.reader(stream))
    with (tmp_path / "short.csv").open("x", newline="") as stream:
        csv.writer(stream).writerows(rows[:5])
    cfg["data"]["train"].append({"name": "short", "path": "short.csv"})
    cfg["data"].update(train_mode="combined", combined_mode="max_size_cycle")
    cfg["trainer"]["max_epochs"] = 1
    run = train(cfg)
    _, payload = load_checkpoint(run / "checkpoints" / "last.ckpt")
    assert payload["successful_updates"] == 4
    assert payload["config"]["runtime"]["total_optimizer_updates"] == 4
    assert payload["config"]["runtime"]["effective_batch_size"] == 4
    metrics = json.loads((run / "metrics.jsonl").read_text().splitlines()[-1])
    assert metrics["train/train/count"] == 8
    assert metrics["train/short/count"] == 8


def test_skipped_test_cadence_does_not_advance_checkpoint_or_early_stopping(tmp_path):
    cfg = recipe(tmp_path)
    cfg["optimizer"]["lr"] = 0.0
    cfg["trainer"]["max_epochs"] = 3
    cfg["evaluation"]["test_during_fit"]["every_n_validation_events"] = 2
    cfg["checkpoint"].update(monitor="test/pooled/f1_macro", save_top_k=1)
    cfg["early_stopping"] = {
        "monitor": "test/pooled/f1_macro",
        "mode": "max",
        "patience": 0,
        "check_on_train_epoch_end": False,
    }
    run = train(cfg)
    checkpoint, _ = load_checkpoint(run / "checkpoints" / "last.ckpt")
    stopping = next(
        value for key, value in checkpoint["callbacks"].items() if key.startswith("FreshEarlyStopping")
    )
    assert stopping["wait_count"] == 0
    result = json.loads((run / "result.json").read_text())
    assert "epoch_0001" in result["best"]
    assert not any(key.startswith("test/") for key in result["metrics"])


def test_custom_scheduler_keeps_live_optimizer_and_respects_external_warmup():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    controller = ScheduleController(
        optimizer,
        {"_target_": "torch.optim.lr_scheduler.StepLR", "step_size": 1, "gamma": 0.5, "warmup_steps": 2},
        10,
        5,
    )
    assert controller.custom.optimizer is optimizer
    for _ in range(2):
        optimizer.step()
        controller.update()
    assert controller.custom.last_epoch == 0
    assert optimizer.param_groups[0]["lr"] == 0.1
    optimizer.step()
    controller.update()
    assert optimizer.param_groups[0]["lr"] == 0.05
    plateau = ScheduleController(
        optimizer,
        {
            "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
            "mode": "max",
            "patience": 0,
            "factor": 0.1,
            "monitor": "val/f1",
        },
        10,
        5,
    )
    plateau.validation({"val/f1": 0.8})
    plateau.validation({})
    assert optimizer.param_groups[0]["lr"] == 0.05
    plateau.validation({"val/f1": 0.7})
    assert abs(optimizer.param_groups[0]["lr"] - 0.005) < 1e-10
