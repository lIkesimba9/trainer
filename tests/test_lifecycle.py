import csv

import numpy as np
import pytest
import torch
from PIL import Image

from trainer_project.engine import inference, restore_model, train
from trainer_project.experiment.artifacts import load_checkpoint, verify_identity
from trainer_project.experiment.config import load_config
from trainer_project.experiment.runtime import local_path
from trainer_project.schedulers import ScheduleController


def recipe(tmp_path, task="classification"):
    for split in ("train", "val", "test"):
        with (tmp_path / f"{split}.csv").open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sample_id", "image", "label", "mask"])
            for i in range(5 if split != "train" else 8):
                image = np.full((16, 16, 3), 30 + i * 20, dtype=np.uint8)
                Image.fromarray(image).save(tmp_path / f"{split}{i}.png")
                mask = np.full((16, 16), i % 2, dtype=np.uint8)
                mask[:2] = 255
                Image.fromarray(mask).save(tmp_path / f"{split}{i}_mask.png")
                writer.writerow(
                    [f"{split}{i}", f"{split}{i}.png", ["class_a", "class_b"][i % 2], f"{split}{i}_mask.png"]
                )
    overrides = []
    if task == "byol":
        overrides = ["task=ssl", "model=byol", "objective=byol", "transforms.ssl=true"]
    elif task == "mae":
        overrides = ["task=ssl", "model=mae", "objective=mae"]
    elif task == "segmentation":
        overrides = ["task=segmentation", "model=segmentation", "objective=segmentation"]
    cfg = load_config(overrides=overrides)
    cfg["output_dir"] = str(tmp_path / "runs")
    cfg["data"].update(data_root=str(tmp_path), batch_size_per_device=2)
    for split in ("train", "val", "test"):
        cfg["data"][split] = [{"name": split, "path": f"{split}.csv"}]
    cfg["trainer"].update(
        accelerator="cpu",
        max_epochs=2,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    cfg["logger"] = None
    if task in ("byol", "mae"):
        cfg["checkpoint"].update(monitor="val/pooled/loss", mode="min")
    if task == "segmentation":
        cfg["data"]["schema"]["targets"] = {
            "segmentation_target": {
                "column": "mask",
                "type": "segmentation",
                "classes": ["background", "foreground"],
                "num_classes": 2,
                "ignore_index": 255,
            }
        }
        cfg["evaluation"]["target_key"] = "segmentation_target"
        cfg["checkpoint"]["monitor"] = "val/pooled/dice_macro"
    return cfg


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            assert_state_equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_state_equal(x, y)
    else:
        assert a == b


def test_epoch_resume_and_snapshot_reconstruction(tmp_path):
    cfg = recipe(tmp_path)
    continuous = train(cfg)
    stopped = train(cfg, stop_after_epoch=1)
    path = stopped / "checkpoints" / "last.ckpt"
    resumed = train(load_checkpoint(path)[1]["config"], checkpoint_path=path, strict=True)
    expected, payload = load_checkpoint(continuous / "checkpoints" / "last.ckpt")
    actual, other = load_checkpoint(resumed / "checkpoints" / "last.ckpt")
    assert_state_equal(expected["state_dict"], actual["state_dict"])
    assert_state_equal(expected["optimizer_states"], actual["optimizer_states"])
    assert payload["scheduler"] == other["scheduler"]
    assert (stopped / "checkpoints" / "last.yaml").exists()
    assert payload["config"]["runtime"]["total_optimizer_updates"] == 8
    module, data, _ = restore_model(continuous / "checkpoints" / "last.ckpt")
    batch = next(iter(data.val_dataloader()[0]))
    outputs = module(batch["inputs"])["logits"]
    batch["targets"]["label"] = 1 - batch["targets"]["label"]
    torch.testing.assert_close(outputs, module(batch["inputs"])["logits"], rtol=0, atol=0)
    result = inference(continuous / "checkpoints" / "last.ckpt")
    assert (result / "reports" / "report.html").exists()
    # Check asset identity, not only CSV bytes.
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(tmp_path / "train0.png")
    with pytest.raises(ValueError, match="content|changed|identity"):
        verify_identity(payload["config"], payload["data_identity"])


@pytest.mark.parametrize("task", ["byol", "mae", "segmentation"])
def test_full_task_fit_restore(task, tmp_path):
    cfg = recipe(tmp_path, task)
    cfg["trainer"]["max_epochs"] = 1
    run = train(cfg)
    module, data, payload = restore_model(run / "checkpoints" / "last.ckpt")
    assert payload["successful_updates"] == 4
    result = inference(run / "checkpoints" / "last.ckpt")
    assert (result / "predictions.jsonl").exists()


def test_schedule_state_and_plateau_skip():
    p = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([p], lr=0.1)
    controller = ScheduleController(
        optimizer, {"name": "plateau", "monitor": "val/f1", "warmup_steps": 2, "patience": 0}, 10, 5
    )
    controller.validation({"val/f1": 0.9})
    assert controller.plateau.best == -float("inf")
    controller.update()
    controller.update()
    controller.validation({"val/f1": 0.9})
    controller.validation({})
    assert optimizer.param_groups[0]["lr"] == 0.1
    state = controller.state_dict()
    other = ScheduleController(
        optimizer, {"name": "plateau", "monitor": "val/f1", "warmup_steps": 2, "patience": 0}, 10, 5
    )
    other.load_state_dict(state)
    assert other.state_dict() == state


def test_workspace_boundary():
    with pytest.raises(ValueError, match="inside"):
        local_path("../outside")
