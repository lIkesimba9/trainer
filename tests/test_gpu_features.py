"""Hardware acceptance: AMP overflow must not advance update-driven state."""

import pytest
import torch
from lightning.pytorch.callbacks import Callback
from test_lifecycle import recipe

from trainer_project.engine import train
from trainer_project.experiment.artifacts import load_checkpoint


class InjectOverflow(Callback):
    def on_after_backward(self, trainer, pl_module):
        if pl_module.microsteps == 1:
            for parameter in pl_module.parameters():
                if parameter.grad is not None:
                    parameter.grad.fill_(float("inf"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP hardware test")
def test_amp_skipped_step_preserves_scheduler_and_ema(tmp_path):
    cfg = recipe(tmp_path, "byol")
    cfg["trainer"].update(accelerator="gpu", devices=1, precision="16-mixed", max_epochs=1)
    cfg["callbacks"] = [{"_target_": "test_gpu_features.InjectOverflow"}]
    run = train(cfg)
    checkpoint, payload = load_checkpoint(run / "checkpoints" / "last.ckpt")
    assert payload["microsteps"] == 4
    assert payload["successful_updates"] == 3
    assert payload["scheduler"]["updates"] == 3
    assert int(checkpoint["state_dict"]["model.ema_updates"]) == 3
    assert checkpoint["MixedPrecision"]["scale"] < 65536.0
