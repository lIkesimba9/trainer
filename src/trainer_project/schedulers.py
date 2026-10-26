"""Schedulers advance only on successful optimizer updates or fresh evaluation events."""

from __future__ import annotations

import math

import torch


class ScheduleController:
    def __init__(self, optimizer, config, total_updates: int, updates_per_epoch: int):
        self.optimizer = optimizer
        self.config = dict(config or {"name": "constant"})
        self.interval = self.config.get("interval", "optimizer_step")
        self.name = self.config.get("name", "constant")
        self.total_updates = max(int(total_updates), 1)
        self.updates_per_epoch = max(int(updates_per_epoch), 1)
        if self.config.get("warmup_steps") and self.config.get("warmup_epochs"):
            raise ValueError("Specify warmup_steps OR warmup_epochs")
        self.warmup = int(
            self.config.get("warmup_steps", 0) or self.config.get("warmup_epochs", 0) * self.updates_per_epoch
        )
        if self.warmup >= self.total_updates:
            raise ValueError("Warmup must be shorter than the training horizon")
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.updates = 0
        self.epochs = 0
        self.events = 0
        self.plateau = None
        self.custom = None
        if "_target_" in self.config:
            from hydra.utils import instantiate

            spec = {
                k: v
                for k, v in self.config.items()
                if k not in {"interval", "monitor", "frequency", "warmup_steps", "warmup_epochs", "name"}
            }
            spec.pop("_partial_", None)
            self.custom = instantiate(spec, _partial_=True)(optimizer=optimizer)
            if isinstance(self.custom, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if not self.config.get("monitor"):
                    raise ValueError("Plateau requires a monitor")
                self.plateau, self.custom = self.custom, None
        elif self.name == "plateau":
            if not self.config.get("monitor"):
                raise ValueError("Plateau requires a monitor")
            self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.config.get("mode", "max"),
                factor=self.config.get("factor", 0.1),
                patience=self.config.get("patience", 5),
                min_lr=self.config.get("min_lr", 0.0),
            )
        elif self.name not in {"constant", "cosine", "step", "multistep", "exponential", "onecycle"}:
            raise ValueError(f"Unsupported scheduler {self.name}")
        if self.name == "onecycle" and self.warmup:
            raise ValueError("OneCycle has its own warmup")
        self._apply()

    def _apply(self):
        if self.custom or self.plateau:
            if self.warmup and self.updates < self.warmup:
                factor = (self.updates + 1) / self.warmup
            else:
                return
        elif self.warmup and self.updates < self.warmup:
            factor = (self.updates + 1) / self.warmup
        else:
            progress = min(max((self.updates - self.warmup) / max(self.total_updates - self.warmup, 1), 0), 1)
            tick = self.epochs if self.interval == "epoch" else self.updates
            gamma = self.config.get("gamma", 0.1)
            if self.name == "cosine":
                factor = (1 + math.cos(math.pi * progress)) / 2
            elif self.name == "step":
                factor = gamma ** (tick // self.config.get("step_size", 10))
            elif self.name == "multistep":
                factor = gamma ** sum(tick >= m for m in self.config.get("milestones", [30, 60]))
            elif self.name == "exponential":
                factor = gamma**tick
            elif self.name == "onecycle":
                pct = self.config.get("pct_start", 0.3)
                factor = (
                    (0.04 + 0.96 * progress / pct)
                    if progress <= pct
                    else 0.0001 + 0.9999 * (1 + math.cos(math.pi * (progress - pct) / (1 - pct))) / 2
                )
            else:
                factor = 1.0
        minimum = self.config.get("min_lr", 0.0)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = max(minimum, base * factor)

    def update(self):
        self.updates += 1
        if self.custom and self.updates <= self.warmup:
            for group, base in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = base * min(1.0, (self.updates + 1) / self.warmup)
            return
        if self.custom and self.interval == "optimizer_step":
            self.custom.step()
        elif self.interval == "optimizer_step" or self.updates <= self.warmup:
            self._apply()

    def epoch(self):
        self.epochs += 1
        if self.custom and self.interval == "epoch" and self.updates >= self.warmup:
            self.custom.step()
        elif self.interval == "epoch":
            self._apply()

    def validation(self, metrics):
        self.events += 1
        if (
            self.plateau
            and self.updates >= self.warmup
            and self.events % self.config.get("frequency", 1) == 0
        ):
            value = metrics.get(self.config["monitor"])
            if value is not None and math.isfinite(float(value)):
                self.plateau.step(float(value))

    def state_dict(self):
        return {
            "updates": self.updates,
            "epochs": self.epochs,
            "events": self.events,
            "base_lrs": self.base_lrs,
            "total_updates": self.total_updates,
            "lrs": [g["lr"] for g in self.optimizer.param_groups],
            "plateau": self.plateau.state_dict() if self.plateau else None,
            "custom": self.custom.state_dict() if self.custom else None,
        }

    def load_state_dict(self, state):
        if state["total_updates"] != self.total_updates:
            raise ValueError("Scheduler horizon changed; use a branch rather than strict resume")
        for key in ("updates", "epochs", "events", "base_lrs"):
            setattr(self, key, state[key])
        if self.plateau and state["plateau"]:
            self.plateau.load_state_dict(state["plateau"])
        if self.custom and state["custom"]:
            self.custom.load_state_dict(state["custom"])
        for group, lr in zip(self.optimizer.param_groups, state["lrs"]):
            group["lr"] = lr
