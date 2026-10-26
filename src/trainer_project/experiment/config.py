from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from .runtime import local_path

SCHEMA_VERSION = 2
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

# Output contracts are intentionally finite. Custom implementations opt into their
# own validated contract instead of inheriting compatibility from a class name.
MODEL_OUTPUTS = {
    "ImageClassifier": {"features", "embeddings", "logits"},
    "MultiModalModel": {"features", "embeddings", "logits", "modality_embeddings"},
    "BYOLModel": {"features", "embeddings", "prediction1", "prediction2", "target1", "target2"},
    "MAEModel": {"features", "embeddings", "reconstruction", "reconstruction_target", "ssl_patch_mask"},
    "TinySegmenter": {"features", "dense_logits"},
    "SegFormerAdapter": {"dense_logits"},
}
OBJECTIVE_OUTPUTS = {
    "MaskedCrossEntropy": "logits",
    "FocalLoss": "logits",
    "LDAMLoss": "logits",
    "LogitAdjustedLoss": "logits",
    "SeesawLoss": "logits",
    "TripletLoss": "embeddings",
    "ArcFaceObjective": "embeddings",
    "SubcenterArcFaceObjective": "embeddings",
    "AAMLoss": "embeddings",
    "BYOLObjective": "prediction1",
    "MAEObjective": "reconstruction",
    "SegmentationObjective": "dense_logits",
    "MaskedSegmentationLoss": "dense_logits",
}


def objective_specs(spec: dict):
    """Yield all leaf objectives, preserving their Hydra configuration objects."""
    name = spec.get("_target_", "").rsplit(".", 1)[-1]
    if name == "CompositeObjective":
        components = spec.get("components", {})
        if not isinstance(components, dict) or not components:
            raise ValueError("CompositeObjective requires named components")
        for component in components.values():
            if not isinstance(component, dict) or "_target_" not in component:
                raise ValueError("Each composite objective component requires _target_")
            yield from objective_specs(component)
    else:
        yield spec


def _target_name(spec):
    return spec.get("_target_", "").rsplit(".", 1)[-1]


def _has_mixing(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if (key.lower() in {"mixup", "cutmix", "mixup_alpha", "cutmix_alpha"} and child) or (
                key == "_target_" and any(token in str(child).lower() for token in ("mixup", "cutmix"))
            ):
                return True
            if _has_mixing(child):
                return True
    elif isinstance(value, list):
        return any(_has_mixing(child) for child in value)
    return False


def validate_compatibility(config):
    model = config["model"]
    model_name = _target_name(model)
    known_model = model_name in MODEL_OUTPUTS
    custom = config.get("compatibility", {}).get("custom_validated", False)
    task = config["task"]
    if task == "ssl" and config.get("tta", {}).get("enabled", False):
        raise ValueError(
            "Built-in TTA requires classification/segmentation logits; SSL uses its feature extraction protocol"
        )
    allowed_models = {
        "classification": {"ImageClassifier", "MultiModalModel"},
        "ssl": {"BYOLModel", "MAEModel", "MultiModalModel", "ImageClassifier"},
        "segmentation": {"TinySegmenter", "SegFormerAdapter"},
    }
    if known_model and model_name not in allowed_models[task]:
        raise ValueError(f"{model_name} does not support task={task}")
    if not known_model and not custom:
        raise ValueError("Custom model requires compatibility.custom_validated=true")
    target_schema = config["data"].get("schema", {})
    target_schema = target_schema.get("targets", {}) if isinstance(target_schema, dict) else {}
    mixing = _has_mixing(config.get("transforms", {})) or _has_mixing(config.get("gpu_transforms", {}))
    for objective in objective_specs(config["objective"]):
        name = _target_name(objective)
        if name not in OBJECTIVE_OUTPUTS:
            if not custom:
                raise ValueError("Custom objective requires compatibility.custom_validated=true")
            continue
        required = objective.get("output_key", OBJECTIVE_OUTPUTS[name])
        if known_model and required not in MODEL_OUTPUTS[model_name]:
            raise ValueError(f"{name} requires output {required!r}, unavailable from {model_name}")
        if name in {"BYOLObjective", "MAEObjective"} and task != "ssl":
            raise ValueError(f"{name} is supported only for task=ssl")
        if name == "BYOLObjective" and known_model and model_name != "BYOLModel":
            raise ValueError("BYOL objective requires a compatible BYOL model")
        if name == "MAEObjective" and known_model and model_name != "MAEModel":
            raise ValueError("MAE objective requires a compatible MAE model")
        if name in {"SegmentationObjective", "MaskedSegmentationLoss"} and task != "segmentation":
            raise ValueError("Dense segmentation objective requires task=segmentation")
        if target_schema and name not in {"BYOLObjective", "MAEObjective"}:
            target = objective.get("target_key", "segmentation_target" if "Segmentation" in name else "label")
            if not target_schema.get(target):
                raise ValueError(f"Objective target {target!r} is absent from data.schema.targets")
        if mixing and name != "MaskedCrossEntropy":
            raise ValueError(f"MixUp/CutMix soft targets are incompatible with {name}")
    schema = config["data"].get("schema", {})
    inputs = schema.get("inputs", {}) if isinstance(schema, dict) else {}
    if model_name in {"ImageClassifier", "BYOLModel", "MAEModel", "TinySegmenter", "SegFormerAdapter"}:
        input_key = model.get("input_key", "image")
        if inputs and input_key not in inputs:
            raise ValueError(f"Model requires input {input_key!r}, absent from data.schema.inputs")
        required_inputs = {input_key}
    elif model_name == "MultiModalModel":
        profile_inputs = {
            "image": {"image"},
            "image_image": {"image_a", "image_b"},
            "image_image_text": {"image_a", "image_b", "text"},
            "image_mask": {"image", "input_mask"},
        }
        profile = model.get("profile", "image_image")
        if profile not in profile_inputs:
            raise ValueError(f"Unsupported multimodal profile {profile}")
        required_inputs = profile_inputs[profile]
        if inputs and not required_inputs.issubset(inputs):
            raise ValueError(f"Profile {profile} requires inputs {sorted(required_inputs)}")
    else:
        required_inputs = set()
    if known_model:
        for key in required_inputs:
            declaration = inputs.get(key, {})
            if isinstance(declaration, dict) and declaration.get("allow_missing", False):
                raise ValueError(
                    f"Built-in {model_name} requires present input {key!r}; use a presence-aware custom model"
                )
    if model_name == "MAEModel":
        size, patch = model.get("image_size", 32), model.get("patch_size", 4)
        if not isinstance(size, int) or not isinstance(patch, int) or patch <= 0 or size % patch:
            raise ValueError("MAE image_size must be divisible by positive patch_size")
        evaluation = config.get("transforms", {}).get("eval", {})
        if (
            isinstance(evaluation, dict)
            and isinstance(evaluation.get("image_size"), int)
            and evaluation["image_size"] != size
        ):
            raise ValueError("MAE image_size differs from evaluation preprocessing")


def load_config(path: str | None = None, overrides: list[str] | None = None) -> dict:
    overrides = overrides or []
    if path:
        config_path = local_path(path)
        with initialize_config_dir(version_base="1.3", config_dir=str(config_path.parent)):
            cfg = compose(config_name=config_path.stem, overrides=overrides)
    else:
        with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
            cfg = compose(config_name="config", overrides=overrides)
    return resolved(cfg)


def resolved(config: dict | DictConfig) -> dict:
    cfg = OmegaConf.create(config)
    OmegaConf.resolve(cfg)
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


def without_pretrained(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: False if k == "pretrained" else without_pretrained(v) for k, v in value.items()}
    if isinstance(value, list):
        return [without_pretrained(v) for v in value]
    return value


def validate_config(config: dict, *, strict: bool = False) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version {config.get('schema_version')}; expected {SCHEMA_VERSION}"
        )
    task = config["task"]
    if isinstance(task, dict):
        task = task["name"]
        config["task"] = task
    if task not in {"classification", "ssl", "segmentation"}:
        raise ValueError(f"Unsupported task: {task}")
    for key in ("model", "objective", "optimizer", "data"):
        if not isinstance(config.get(key), dict) or "_target_" not in config[key]:
            raise ValueError(f"{key} requires a Hydra _target_")
    if task != "ssl" and config.get("ssl_evaluation", {}).get("enabled", False):
        raise ValueError("kNN and linear probe are supported only for task=ssl")
    validate_compatibility(config)
    trainer = config["trainer"]
    if trainer.get("num_nodes", 1) != 1:
        raise ValueError("Multi-node is an extension; this implementation validates one-server DDP")
    if trainer.get("precision", "32-true") not in ("32-true", "16-mixed", "bf16-mixed"):
        raise ValueError("Unsupported precision")
    if trainer.get("use_distributed_sampler", False):
        raise ValueError("DataModule owns distributed sharding; use_distributed_sampler must be false")
    for key in ("max_epochs", "accumulate_grad_batches"):
        if not isinstance(trainer.get(key, 1), int) or trainer.get(key, 1) < 1:
            raise ValueError(f"trainer.{key} must be a positive integer")
    if config.get("reproducibility", {}).get("mode", "reproducible") not in {"reproducible", "fast"}:
        raise ValueError("reproducibility.mode must be reproducible or fast")
    warmup = config.get("scheduler", {})
    if warmup.get("warmup_steps", 0) and warmup.get("warmup_epochs", 0):
        raise ValueError("Specify warmup_steps or warmup_epochs, not both")
    repro = config.get("reproducibility", {}).get("mode", "reproducible") == "reproducible"
    if (strict or repro) and config["data"].get("persistent_workers", False):
        raise ValueError("Strict epoch resume requires persistent_workers=false")
    if strict and not repro:
        raise ValueError("Strict resume requires reproducibility.mode=reproducible")
    if config.get("scheduler", {}).get("name") == "onecycle" and any(
        config["scheduler"].get(k, 0) for k in ("warmup_steps", "warmup_epochs")
    ):
        raise ValueError("OneCycle cannot be combined with external warmup")
    for role in ("train", "val", "test"):
        names = [m["name"] for m in config["data"].get(role, [])]
        if len(names) != len(set(names)) or any(n in {"pooled", "manifest_macro"} or "/" in n for n in names):
            raise ValueError(f"Invalid or duplicate {role} manifest names")
    local_path(config.get("output_dir", "runs"))


def remap_paths(config: dict, mapping: dict[str, str]) -> dict:
    config = deepcopy(config)

    def visit(value):
        if isinstance(value, dict):
            return {k: visit(v) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v) for v in value]
        if isinstance(value, str):
            for old, new in mapping.items():
                if value == old or value.startswith(old.rstrip("/") + "/"):
                    return str(local_path(new + value[len(old) :]))
        return value

    return visit(config)
