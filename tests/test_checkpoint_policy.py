from types import SimpleNamespace

import pytest
import torch
from test_lifecycle import recipe

from trainer_project.callbacks import RetainingModelCheckpoint
from trainer_project.engine import train
from trainer_project.experiment.artifacts import load_checkpoint


@pytest.mark.parametrize("mode", ["min", "max"])
@pytest.mark.parametrize("tie_policy", ["keep_earlier", "keep_later"])
def test_checkpoint_ties_nan_and_periodic_snapshots_retain_files(tmp_path, mode, tie_policy):
    cfg = recipe(tmp_path)
    cfg["trainer"]["max_epochs"] = 3
    cfg["optimizer"]["lr"] = 0.0
    cfg["checkpoint"].update(save_top_k=1, mode=mode, tie_policy=tie_policy, snapshot_every_n_epochs=1)
    run = train(cfg)
    checkpoint, _ = load_checkpoint(run / "checkpoints" / "last.ckpt")
    state = next(v for k, v in checkpoint["callbacks"].items() if k.startswith("RetainingModelCheckpoint"))
    assert len(state["best_k_models"]) == 1
    best = state["best_model_path"]
    assert ("epoch_0000" if tie_policy == "keep_earlier" else "epoch_0002") in best
    assert len(list((run / "checkpoints").glob("snapshot_epoch_*.ckpt"))) == 3
    if tie_policy == "keep_later":
        assert len(list((run / "checkpoints").glob("epoch_*.ckpt"))) == 3
    callback = RetainingModelCheckpoint(tmp_path / "unused", monitor="val/score", save_top_k=1)
    trainer = SimpleNamespace(strategy=SimpleNamespace(reduce_boolean_decision=lambda value: value))
    assert callback.check_monitor_top_k(trainer, torch.tensor(float("nan"))) is False
    assert callback.check_monitor_top_k(trainer, torch.tensor(float("inf"))) is False
    callback.nan_policy = "error"
    with pytest.raises(ValueError, match="NaN"):
        callback.check_monitor_top_k(trainer, torch.tensor(float("nan")))


def test_strict_resume_rejects_changed_batch_recipe(tmp_path):
    cfg = recipe(tmp_path)
    run = train(cfg, stop_after_epoch=1)
    path = run / "checkpoints" / "last.ckpt"
    _, payload = load_checkpoint(path)
    changed = payload["config"]
    changed["data"]["batch_size_per_device"] = 4
    with pytest.raises(ValueError, match="recipe/execution"):
        train(changed, checkpoint_path=path, strict=True)
