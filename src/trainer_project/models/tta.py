"""Deterministic flip TTA with inverse dense-output geometry."""

import torch
from torch import nn

_FLIPS = {"identity": (), "hflip": (-1,), "vflip": (-2,), "hvflip": (-2, -1)}


class TestTimeAugmentation(nn.Module):
    """Evaluate each view and aggregate logits or probabilities in input coordinates.

    Spatial inputs (images and input masks) receive the same declared flip; text
    and other nonspatial inputs pass through unchanged. No training or RNG state
    is modified. The supplied model must be in evaluation mode.
    """

    def __init__(
        self,
        model,
        transforms=None,
        aggregation="mean_logits",
        output_key=None,
        input_keys=None,
        target_mode="exclusive",
    ):
        super().__init__()
        self.model = model
        self.transforms = tuple(transforms or ("identity", "hflip"))
        if (
            not self.transforms
            or len(set(self.transforms)) != len(self.transforms)
            or any(transform not in _FLIPS for transform in self.transforms)
        ):
            raise ValueError("TTA supports distinct identity/hflip/vflip/hvflip views")
        if aggregation not in {"mean_logits", "mean_probabilities"}:
            raise ValueError("TTA aggregation must be mean_logits or mean_probabilities")
        if target_mode not in {"exclusive", "binary", "multilabel"}:
            raise ValueError("Unknown TTA target_mode")
        self.aggregation, self.output_key, self.input_keys, self.target_mode = (
            aggregation,
            output_key,
            input_keys,
            target_mode,
        )

    def _probabilities(self, logits):
        if self.target_mode in {"binary", "multilabel"}:
            return logits.sigmoid()
        return logits.softmax(1)

    def _prediction(self, prediction, dimensions):
        if isinstance(prediction, dict):
            return {key: self._prediction(value, dimensions) for key, value in prediction.items()}
        if not isinstance(prediction, torch.Tensor) or prediction.ndim not in {2, 4}:
            raise ValueError("TTA requires categorical [B,C] or dense [B,C,H,W] logits tensors")
        if prediction.ndim == 4 and dimensions:
            prediction = prediction.flip(dimensions)
        return self._probabilities(prediction) if self.aggregation == "mean_probabilities" else prediction

    def _average(self, predictions):
        if isinstance(predictions[0], dict):
            if any(set(value) != set(predictions[0]) for value in predictions):
                raise ValueError("TTA outputs must preserve the same target fields across views")
            return {key: self._average([value[key] for value in predictions]) for key in predictions[0]}
        return torch.stack(predictions).mean(0)

    def _as_logits(self, averaged):
        if isinstance(averaged, dict):
            return {key: self._as_logits(value) for key, value in averaged.items()}
        if self.target_mode in {"binary", "multilabel"}:
            return torch.logit(averaged.clamp(1e-7, 1 - 1e-7))
        return averaged.clamp_min(1e-30).log()

    def _as_probabilities(self, logits):
        if isinstance(logits, dict):
            return {key: self._as_probabilities(value) for key, value in logits.items()}
        return self._probabilities(logits)

    @torch.no_grad()
    def forward(self, inputs, views=None):
        if self.model.training:
            raise ValueError("TTA requires an evaluation-mode model")
        outputs, collected = None, []
        chosen_key = self.output_key
        for transform in self.transforms:
            dimensions = _FLIPS[transform]
            transformed = dict(inputs)
            for key, value in inputs.items():
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim == 4
                    and (self.input_keys is None or key in self.input_keys)
                ):
                    transformed[key] = value.flip(dimensions) if dimensions else value
            current = self.model(transformed, views=None)
            if chosen_key is None:
                chosen_key = "dense_logits" if "dense_logits" in current else "logits"
            collected.append(self._prediction(current[chosen_key], dimensions))
            if outputs is None:
                outputs = dict(current)
        averaged = self._average(collected)
        if self.aggregation == "mean_probabilities":
            outputs[chosen_key] = self._as_logits(averaged)
            if chosen_key == "logits":
                outputs["probabilities"] = averaged
        else:
            outputs[chosen_key] = averaged
            if chosen_key == "logits":
                outputs["probabilities"] = self._as_probabilities(averaged)
        outputs["tta"] = {
            "transforms": list(self.transforms),
            "aggregation": self.aggregation,
            "target_mode": self.target_mode,
        }
        return outputs


def tta_forward(model, inputs, config):
    """Convenience entry point; config includes the wrapper keyword arguments."""
    options = dict(config)
    options.pop("enabled", None)
    options.pop("_target_", None)
    return TestTimeAugmentation(model=model, **options)(inputs)
