from torch.nn import functional as F

from .base import Objective, global_masked_mean, result


class BYOLObjective(Objective):
    def __init__(self, input_key="image"):
        super().__init__()
        self.input_key = input_key

    def forward(self, outputs, batch, context=None):
        keys = ("prediction1", "prediction2", "target1", "target2")
        if not all(key in outputs for key in keys):
            raise ValueError("BYOL objective needs BYOLModel and two augmented views")
        first = 2 - 2 * F.cosine_similarity(outputs["prediction1"], outputs["target2"].detach(), dim=-1)
        second = 2 - 2 * F.cosine_similarity(outputs["prediction2"], outputs["target1"].detach(), dim=-1)
        valid = None
        present = batch.get("input_present", {})
        if self.input_key in present:
            valid = present[self.input_key]
        loss, count = global_masked_mean((first + second) * 0.5, valid)
        return result(loss, "byol", count)


class MAEObjective(Objective):
    def forward(self, outputs, batch, context=None):
        values = (outputs["reconstruction"] - outputs["reconstruction_target"].detach()).square().mean(-1)
        loss, count = global_masked_mean(values, outputs["ssl_patch_mask"])
        return result(loss, "mae_reconstruction", count)
