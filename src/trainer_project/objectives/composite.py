from collections.abc import Mapping

from torch import nn

from .base import Objective, ObjectiveResult


class CompositeObjective(Objective):
    """Registered custom loss composition preserving the complete batch contract.

    Each component is a module or {_target_: ...}; weights may be scalar or
    {value, start_epoch, warmup_epochs}. Schedule state follows checkpointed epoch.
    """

    def __init__(self, components, weights=None):
        super().__init__()
        from hydra.utils import instantiate

        self.components = nn.ModuleDict(
            {
                name: instantiate(component) if not isinstance(component, nn.Module) else component
                for name, component in components.items()
            }
        )
        if not self.components:
            raise ValueError("Composite objective needs at least one component")
        self.weights = dict(weights or {})

    def _weight(self, name, context):
        weight = self.weights.get(name, 1.0)
        if isinstance(weight, Mapping) or weight.__class__.__name__ == "DictConfig":
            epoch = (context or {}).get("epoch", 0)
            start = float(weight.get("start_epoch", 0))
            duration = float(weight.get("warmup_epochs", 0))
            factor = 0.0 if epoch < start else (min(1.0, (epoch - start + 1) / duration) if duration else 1.0)
            return float(weight.get("value", 1.0)) * factor
        return float(weight)

    def forward(self, outputs, batch, context=None):
        total, values, counts = None, {}, {}
        for name, objective in self.components.items():
            component = objective(outputs, batch, context)
            term = component.loss * self._weight(name, context)
            total = term if total is None else total + term
            values[name] = component.loss.detach()
            values.update({f"{name}/{key}": value for key, value in component.components.items()})
            counts.update({f"{name}/{key}": value for key, value in component.counts.items()})
        return ObjectiveResult(total, values, counts)
