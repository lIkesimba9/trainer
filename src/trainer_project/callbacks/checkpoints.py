from __future__ import annotations

import math
from pathlib import Path

from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from omegaconf import OmegaConf

from trainer_project.experiment.runtime import digest, local_path, write_json


class RetainingModelCheckpoint(ModelCheckpoint):
    """Lightning top-K policy with immutable epoch files and no checkpoint removal."""

    def __init__(
        self,
        dirpath,
        monitor=None,
        mode="max",
        save_top_k=3,
        save_last=True,
        every_n_epochs=1,
        tie_policy="keep_earlier",
        nan_policy="skip",
        snapshot_every_n_epochs=0,
        filename="epoch_{epoch:04d}-step_{step:08d}",
        **kwargs,
    ):
        if tie_policy not in {"keep_earlier", "keep_later"} or nan_policy not in {"skip", "error"}:
            raise ValueError("Unsupported checkpoint tie/NaN policy")
        self.tie_policy, self.nan_policy = tie_policy, nan_policy
        self.snapshot_every_n_epochs = int(snapshot_every_n_epochs)
        super().__init__(
            dirpath=str(local_path(dirpath)),
            filename=filename,
            auto_insert_metric_name=False,
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=save_last,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=True,
            **kwargs,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        if (
            self.snapshot_every_n_epochs > 0
            and (trainer.current_epoch + 1) % self.snapshot_every_n_epochs == 0
        ):
            path = str(
                Path(self.dirpath)
                / f"snapshot_epoch_{trainer.current_epoch:04d}-step_{trainer.global_step:08d}.ckpt"
            )
            self._save_checkpoint(trainer, path)

    def check_monitor_top_k(self, trainer, current=None):
        if current is not None and not math.isfinite(float(current)):
            if self.nan_policy == "error":
                raise ValueError("Checkpoint monitor is NaN or Inf")
            return False
        if (
            self.tie_policy == "keep_later"
            and current is not None
            and self.best_k_models
            and self.save_top_k > 0
        ):
            if len(self.best_k_models) == self.save_top_k and current == self.kth_value:
                return True
        return super().check_monitor_top_k(trainer, current)

    def _remove_checkpoint(self, trainer, filepath):
        return None

    def _save_topk_checkpoint(self, trainer, monitor_candidates):
        if (
            self.monitor
            and self.monitor.startswith(("val/", "test/"))
            and self.monitor not in trainer.lightning_module.last_metrics
        ):
            return
        super()._save_topk_checkpoint(trainer, monitor_candidates)

    def load_state_dict(self, state_dict):
        # A resumed run has a new artifact directory. Preserve the complete
        # registry, whose old paths remain valid because retention never removes files.
        for name in (
            "best_model_score",
            "best_model_path",
            "kth_best_model_path",
            "kth_value",
            "best_k_models",
            "last_model_path",
        ):
            if name in state_dict:
                setattr(self, name, state_dict[name])

    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        if trainer.is_global_zero:
            path = local_path(filepath)
            OmegaConf.save(
                OmegaConf.create(trainer.lightning_module.experiment_config), path.with_suffix(".yaml")
            )
            manifest_path = Path(self.dirpath).parent / "manifest.json"
            if manifest_path.is_file():
                import json

                manifest = json.loads(manifest_path.read_text())
                artifacts = manifest.setdefault("artifacts", {})
                artifacts[str(path.relative_to(manifest_path.parent))] = digest(path)
                artifacts[str(path.with_suffix(".yaml").relative_to(manifest_path.parent))] = digest(
                    path.with_suffix(".yaml")
                )
                write_json(manifest_path, manifest)
            write_json(
                Path(self.dirpath) / "registry.json",
                {
                    "best": self.best_model_path,
                    "last": self.last_model_path,
                    "top_k": {p: float(v) for p, v in self.best_k_models.items()},
                    "retention": "all files retained by explicit user instruction",
                },
            )


class StopAfterEpoch(Callback):
    def __init__(self, count):
        self.count = count

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch + 1 >= self.count:
            trainer.should_stop = True


class FreshEarlyStopping(EarlyStopping):
    def _run_early_stopping_check(self, trainer):
        if (
            self.monitor.startswith(("val/", "test/"))
            and self.monitor not in trainer.lightning_module.last_metrics
        ):
            return
        super()._run_early_stopping_check(trainer)
