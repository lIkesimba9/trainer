"""Scheduled freezing with fixed optimizer groups and DDP parameter membership."""

from lightning.pytorch import Callback


def _schedule_rows(schedule):
    if isinstance(schedule, dict):
        rows = [{"module": path, "unfreeze_epoch": epoch} for path, epoch in schedule.items()]
    else:
        rows = [dict(row) for row in schedule]
    paths = []
    for row in rows:
        path, epoch = row.get("module"), row.get("unfreeze_epoch")
        if not isinstance(path, str) or not path or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("Freeze schedule requires module paths and nonnegative integer unfreeze_epoch")
        if any(
            path == previous or path.startswith(previous + ".") or previous.startswith(path + ".")
            for previous in paths
        ):
            raise ValueError("Freeze schedule module paths must be unique and nonoverlapping")
        paths.append(path)
    return rows


def enable_scheduled_parameters(module, schedule):
    """Call in LightningModule.__init__, before DDP and optimizer construction.

    Scheduled parameters remain registered with DDP for the entire run. The
    callback suppresses optimizer updates during the frozen phase by clearing
    their gradients; this also prevents decay and momentum-state mutation.
    """
    names = []
    for row in _schedule_rows(schedule):
        try:
            child = module.get_submodule(row["module"])
        except AttributeError as error:
            raise ValueError(f"Unknown scheduled module {row['module']}") from error
        for name, parameter in child.named_parameters():
            parameter.requires_grad_(True)
            parameter._scheduled_unfreeze = True
            names.append(f"{row['module']}.{name}")
    return names


class ScheduledFreeze(Callback):
    """Freeze weights/statistics until each epoch while keeping optimizer groups.

    Parameters still take part in autograd/DDP collectives while frozen. This is
    deliberately a state-safe policy, not a compute-saving frozen-backbone pass.
    Callback state is checkpointed and the original schedule is verified on load.
    """

    def __init__(self, schedule):
        self.schedule = _schedule_rows(schedule)
        self.last_epoch = -1

    @property
    def state_key(self):
        return "ScheduledFreeze"

    def setup(self, trainer, pl_module, stage):
        if stage != "fit":
            return
        for row in self.schedule:
            child = pl_module.get_submodule(row["module"])
            if any(not parameter.requires_grad for parameter in child.parameters()):
                raise ValueError(
                    "Call enable_scheduled_parameters before DDP/optimizer setup for scheduled modules"
                )

    def _set_modes(self, trainer, pl_module):
        for row in self.schedule:
            child = pl_module.get_submodule(row["module"])
            child.train(int(trainer.current_epoch) >= row["unfreeze_epoch"])

    def on_train_epoch_start(self, trainer, pl_module):
        self.last_epoch = int(trainer.current_epoch)
        self._set_modes(trainer, pl_module)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._set_modes(trainer, pl_module)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        for row in self.schedule:
            if int(trainer.current_epoch) < row["unfreeze_epoch"]:
                for parameter in pl_module.get_submodule(row["module"]).parameters():
                    parameter.grad = None

    def state_dict(self):
        return {"schedule": self.schedule, "last_epoch": self.last_epoch}

    def load_state_dict(self, state_dict):
        if state_dict["schedule"] != self.schedule:
            raise ValueError("Freeze schedule differs from checkpoint; create an explicit branch")
        self.last_epoch = int(state_dict["last_epoch"])
