import csv

import numpy as np
import pytest
import torch
from PIL import Image

from trainer_project.engine import prepare
from trainer_project.experiment.config import load_config, validate_config
from trainer_project.modules import ResearchTask
from trainer_project.objectives import MaskedCrossEntropy, ObjectiveResult


def fixture_config(tmp_path, experiment=None):
    overrides = [f"+experiment={experiment}"] if experiment else []
    configuration = load_config(overrides=overrides)
    for index in range(4):
        Image.fromarray(np.full((8, 8, 3), 50 + index * 20, dtype=np.uint8)).save(tmp_path / f"{index}.png")
        Image.fromarray(np.full((8, 8), index % 2, dtype=np.uint8)).save(tmp_path / f"mask{index}.png")
    for split, indices in (("train", (0, 1)), ("val", (2, 3))):
        with (tmp_path / f"{split}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "image", "label", "segmentation_mask"])
            writer.writeheader()
            for index in indices:
                writer.writerow(
                    {
                        "sample_id": str(index),
                        "image": f"{index}.png",
                        "label": "class_a" if index % 2 == 0 else "class_b",
                        "segmentation_mask": f"mask{index}.png",
                    }
                )
    configuration["data"]["data_root"] = str(tmp_path)
    configuration["data"]["train"] = [{"name": "train", "path": "train.csv"}]
    configuration["data"]["val"] = [{"name": "val", "path": "val.csv"}]
    return configuration


@pytest.mark.parametrize(
    "experiment,task",
    [(None, "classification"), ("byol", "ssl"), ("mae", "ssl"), ("segmentation", "segmentation")],
)
def test_hydra_groups_compose_and_prepare_real_batch(tmp_path, experiment, task):
    configuration, data, model = prepare(fixture_config(tmp_path, experiment))
    assert configuration["task"] == task
    sample = next(iter(data.train_dataloader()))
    outputs = model(sample["inputs"], sample["views"] or None)
    assert outputs
    assert configuration["runtime"]["total_optimizer_updates"] == 10
    assert configuration["data"]["class_maps"]


def test_num_classes_inferred_when_omitted_and_head_dimensions_checked(tmp_path):
    config = fixture_config(tmp_path)
    config["model"].pop("num_classes")
    config["data"]["schema"]["targets"]["label"]["classes"] = ["class_a", "class_b", "class_c"]
    resolved, _, model = prepare(config)
    assert model.num_classes == resolved["model"]["num_classes"] == 3
    config["model"]["head"] = {
        "_target_": "trainer_project.models.LinearHead",
        "in_features": 31,
        "num_classes": 3,
    }
    with pytest.raises(ValueError, match="in_features"):
        prepare(config)


@pytest.mark.parametrize("name", ["LDAMLoss", "LogitAdjustedLoss"])
def test_frequency_objectives_use_only_permitted_training_rows(tmp_path, name):
    config = fixture_config(tmp_path)
    config["data"]["train"].append({"name": "forbidden", "path": "val.csv", "supervision_allowed": False})
    config["objective"] = {"_target_": f"trainer_project.objectives.{name}", "target_key": "label"}
    resolved, _, _ = prepare(config)
    assert resolved["objective"]["class_counts"] == [1, 1]
    assert resolved["runtime"]["class_counts"]["label"] == {"class_a": 1, "class_b": 1}


@pytest.mark.parametrize(
    "model,objective,task",
    [
        ("BYOLModel", "MaskedCrossEntropy", "ssl"),
        ("ImageClassifier", "BYOLObjective", "ssl"),
        ("TinySegmenter", "MaskedCrossEntropy", "segmentation"),
    ],
)
def test_incompatible_output_contract_rejected(model, objective, task):
    config = load_config()
    config["task"] = task
    config["model"] = {"_target_": f"trainer_project.models.{model}"}
    config["objective"] = {"_target_": f"trainer_project.objectives.{objective}"}
    with pytest.raises(ValueError, match="requires"):
        validate_config(config)


def test_nested_objective_compatibility_and_soft_target_validation():
    config = load_config()
    config["objective"] = {
        "_target_": "trainer_project.objectives.CompositeObjective",
        "components": {
            "ce": {"_target_": "trainer_project.objectives.MaskedCrossEntropy"},
            "triplet": {"_target_": "trainer_project.objectives.TripletLoss"},
        },
    }
    validate_config(config)
    config["gpu_transforms"] = {"_target_": "custom.MixUp", "alpha": 0.2}
    with pytest.raises(ValueError, match="MixUp/CutMix"):
        validate_config(config)


def test_custom_objective_parameter_joins_optimizer_and_state(tmp_path):
    class CustomLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.ce = MaskedCrossEntropy()

        def forward(self, outputs, batch, context):
            result = self.ce(outputs, batch, context)
            return ObjectiveResult(result.loss * self.scale, result.components, result.counts)

    config, data, model = prepare(fixture_config(tmp_path))
    module = ResearchTask(config, model=model, objective=CustomLoss())
    optimizer = module.configure_optimizers()
    parameter_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert id(module.objective.scale) in parameter_ids
    sample = next(iter(data.train_dataloader()))
    before = module.objective.scale.detach().clone()
    module.objective(module(sample["inputs"]), sample, {"stage": "train"}).loss.backward()
    optimizer.step()
    assert not torch.equal(before, module.objective.scale)
    restored = ResearchTask(config, objective=CustomLoss())
    restored.load_state_dict(module.state_dict())
    torch.testing.assert_close(restored.objective.scale, module.objective.scale, rtol=0, atol=0)
