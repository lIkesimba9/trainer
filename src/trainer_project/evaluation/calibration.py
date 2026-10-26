"""Temperature calibration fitted only on an explicitly named tuning set."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .representation import preserve_rng


class TemperatureCalibrator:
    def __init__(self, temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.temperature = float(temperature)
        self.provenance: dict[str, Any] = {}

    def fit(
        self, logits: Any, targets: Any, *, data_role: str, source: str, max_iter: int = 50
    ) -> "TemperatureCalibrator":
        if data_role not in {"val", "validation", "calibration", "oof"} or not source:
            raise ValueError("Calibration requires an explicit validation/OOF source")
        with preserve_rng(), torch.inference_mode(False), torch.enable_grad():
            x = torch.as_tensor(logits).detach().float().cpu().clone()
            y = torch.as_tensor(targets).detach().long().cpu().clone()
            valid = y >= 0
            x, y = x[valid], y[valid]
            if x.ndim != 2 or not len(y) or x.shape[0] != len(y):
                raise ValueError("Calibration requires labeled [N,C] logits")
            log_temperature = torch.tensor(self.temperature).log().requires_grad_()
            optimizer = torch.optim.LBFGS(
                [log_temperature], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe"
            )

            def closure():
                optimizer.zero_grad()
                loss = F.cross_entropy(x / log_temperature.exp().clamp(0.01, 100.0), y)
                loss.backward()
                return loss

            optimizer.step(closure)
            self.temperature = float(log_temperature.detach().exp().clamp(0.01, 100.0))
        self.provenance = {
            "data_role": data_role,
            "source": source,
            "valid_targets": int(len(y)),
            "tuning_result": True,
        }
        return self

    def __call__(self, logits: Any) -> torch.Tensor:
        return (torch.as_tensor(logits) / self.temperature).softmax(dim=-1)

    def state_dict(self) -> dict:
        return {"temperature": self.temperature, "provenance": self.provenance}

    def load_state_dict(self, state: dict) -> None:
        if float(state["temperature"]) <= 0:
            raise ValueError("Temperature must be positive")
        self.temperature, self.provenance = float(state["temperature"]), dict(state.get("provenance", {}))
