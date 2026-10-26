from __future__ import annotations

import json
import math
import os
from copy import deepcopy

import torch
from hydra.utils import instantiate
from lightning.pytorch import Trainer, seed_everything
from omegaconf import OmegaConf

from .callbacks import RetainingModelCheckpoint, StopAfterEpoch
from .callbacks.checkpoints import FreshEarlyStopping
from .experiment.artifacts import environment, load_checkpoint, snapshot, verify_identity
from .experiment.config import validate_config, without_pretrained
from .experiment.runtime import WorkspaceCheckpointIO, local_path, new_run, write_json
from .modules import ResearchTask


def prepare(config, *, restore=False):
    from .experiment.config import objective_specs

    config = deepcopy(config)
    if isinstance(config.get("data", {}).get("schema"), str):
        config["data"]["schema"] = OmegaConf.to_container(
            OmegaConf.load(local_path(config["data"]["schema"])), resolve=True
        )
    validate_config(config)
    seed_everything(config["seed"], workers=True)
    if config.get("reproducibility", {}).get("mode") == "reproducible":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(config.get("cpu_threads", 2)))
    for key in list(config["data"].get("schema", {}).get("targets", {})):
        if config["data"]["schema"]["targets"][key] is None:
            config["data"]["schema"]["targets"].pop(key)
    config["data"]["transforms"] = deepcopy(config.get("transforms", {}))
    config["data"]["sampler"] = deepcopy(config.get("sampler", {}))
    config["data"]["seed"] = config["seed"]
    config["data"]["reproducible"] = config.get("reproducibility", {}).get("mode") == "reproducible"
    config["data"]["test_during_fit"] = (
        config.get("evaluation", {}).get("test_during_fit", {}).get("enabled", True)
    )
    data = instantiate(config["data"], _recursive_=False, _convert_="all")
    data.setup()
    config["data"]["class_maps"] = data.class_maps
    target = config.get("evaluation", {}).get("target_key", "label")
    nclasses = len(data.class_maps.get(target, {}))
    counts = {}
    records = getattr(data.train_dataset, "records", [])
    for key, mapping in data.class_maps.items():
        counts[key] = {
            name: sum(
                row.get(key, -1) == index
                and bool(row.get(f"{key}__valid", True))
                and bool(row.get("_target_allowed", {}).get(key, True))
                for row in records
            )
            for name, index in mapping.items()
        }
    for specification in objective_specs(config["objective"]):
        name = specification["_target_"].rsplit(".", 1)[-1]
        objective_target = specification.get("target_key", "label")
        mapping = data.class_maps.get(objective_target, {})
        if name == "BYOLObjective":
            specification.setdefault("input_key", config["model"].get("input_key", "image"))
        if name in {"LDAMLoss", "LogitAdjustedLoss"} and specification.get("class_counts") is None:
            if not records:
                raise ValueError(f"{name} needs saved class_counts or permitted training rows")
            ordered = sorted(mapping, key=mapping.get)
            specification["class_counts"] = [counts[objective_target][label] for label in ordered]
            if not ordered or min(specification["class_counts"]) <= 0:
                raise ValueError(
                    f"{name} requires a positive permitted train count for every class of {objective_target}"
                )
        if name in {"ArcFaceObjective", "SubcenterArcFaceObjective", "AAMLoss", "SeesawLoss"}:
            classes = len(mapping)
            if classes:
                if specification.get("num_classes", classes) != classes:
                    raise ValueError(f"{name} class count differs from target {objective_target} mapping")
                specification.setdefault("num_classes", classes)
            if name != "SeesawLoss":
                dimension = config["model"].get("embedding_dim", config["model"].get("feature_dim", 32))
                specification.setdefault("embedding_dim", dimension)
    model_name = config["model"]["_target_"].rsplit(".", 1)[-1]
    has_head = model_name in {"ImageClassifier", "MultiModalModel", "TinySegmenter", "SegFormerAdapter"}
    segmentation_spec = next(
        (
            specification
            for specification in objective_specs(config["objective"])
            if specification["_target_"].rsplit(".", 1)[-1]
            in {"SegmentationObjective", "MaskedSegmentationLoss"}
        ),
        None,
    )
    binary = segmentation_spec is not None and segmentation_spec.get("target_mode", "exclusive") == "binary"
    expected_outputs = 1 if binary else nclasses
    if has_head and expected_outputs:
        config["model"].setdefault("num_classes", expected_outputs)
        if config["model"]["num_classes"] != expected_outputs:
            raise ValueError(
                f"Model head has {config['model']['num_classes']} outputs; target schema requires {expected_outputs}"
            )
        head = config["model"].get("head")
        if isinstance(head, dict):
            if "num_classes" in head and head["num_classes"] != expected_outputs:
                raise ValueError("Configured head num_classes differs from target class mapping")
            if "out_features" in head and head["out_features"] != expected_outputs:
                raise ValueError("Configured head out_features differs from target class mapping")
            embedding_dim = config["model"].get("embedding_dim", 32)
            if "in_features" in head and head["in_features"] != embedding_dim:
                raise ValueError("Head in_features differs from model embedding_dim")
    if config["task"] in {"classification", "segmentation"} and not nclasses:
        raise ValueError(f"Evaluation target {target!r} has no saved class mapping")
    model = instantiate(without_pretrained(config["model"]) if restore else config["model"], _convert_="all")
    if has_head and expected_outputs and getattr(model, "num_classes", expected_outputs) != expected_outputs:
        raise ValueError("Instantiated model head dimension differs from the data schema")

    # Persist actual HF architecture so reconstruction can stay offline.
    def capture_architecture(module, specification):
        if not isinstance(specification, dict):
            return
        provenance = getattr(module, "weights_provenance", {})
        if specification.get("_target_", "").rsplit(".", 1)[-1] in {
            "HuggingFaceBackbone",
            "SegFormerAdapter",
        }:
            if provenance.get("config"):
                specification["config"] = provenance["config"]
        for child_name, child in module.named_children():
            if isinstance(specification.get(child_name), dict):
                capture_architecture(child, specification[child_name])

    capture_architecture(model, config["model"])
    if not restore or "runtime" not in config:
        devices = config["trainer"].get("devices", 1)
        world = len(devices) if isinstance(devices, list) else int(devices) if isinstance(devices, int) else 1
        sampler_spec = deepcopy(config.get("sampler", {}))
        sampler_spec.pop("_partial_", None)
        if data.train_dataset and sampler_spec.get("_target_"):
            sampler = instantiate(
                sampler_spec, dataset=data.train_dataset, rank=0, world_size=world, seed=config["seed"]
            )
            count = len(sampler)
            batches = (
                count
                if getattr(sampler, "is_batch_sampler", False)
                else (count // data.batch_size if data.drop_last else math.ceil(count / data.batch_size))
            )
        else:
            batches = math.ceil(math.ceil(len(data.train_dataset or []) / world) / data.batch_size)
        combined = data.train_mode == "combined"
        if combined:
            lengths = []
            for dataset in data.datasets["train"]:
                source_sampler = instantiate(
                    sampler_spec, dataset=dataset, rank=0, world_size=world, seed=config["seed"]
                )
                lengths.append(
                    len(source_sampler)
                    if getattr(source_sampler, "is_batch_sampler", False)
                    else (
                        len(source_sampler) // data.batch_size
                        if data.drop_last
                        else math.ceil(len(source_sampler) / data.batch_size)
                    )
                )
            if not lengths:
                raise ValueError("CombinedLoader requires train sources")
            batches = (
                min(lengths)
                if data.combined_mode == "min_size"
                else sum(lengths)
                if data.combined_mode == "sequential"
                else max(lengths)
            )
        limit = config["trainer"].get("limit_train_batches", 1.0)
        batches = min(batches, limit) if isinstance(limit, int) else int(batches * limit)
        accumulation = int(config["trainer"].get("accumulate_grad_batches", 1))
        if accumulation > 1 and batches % accumulation:
            raise ValueError(
                "Incomplete accumulation window: choose batch count divisible by accumulate_grad_batches"
            )
        updates = math.ceil(batches / accumulation)
        max_steps = config["trainer"].get("max_steps", -1)
        total_updates = updates * config["trainer"]["max_epochs"]
        if max_steps > 0:
            total_updates = min(total_updates, max_steps)
        sample_batch = (
            sampler_spec.get("p", 1) * sampler_spec.get("k", 1)
            if getattr(locals().get("sampler"), "is_batch_sampler", False)
            else data.batch_size
        )
        if combined and data.combined_mode != "sequential":
            sample_batch *= len(data.datasets["train"])
        config["runtime"] = {
            "world_size": world,
            "effective_batch_size": sample_batch * world * accumulation,
            "updates_per_epoch": updates,
            "total_optimizer_updates": total_updates,
            "class_counts": counts,
            "embedding_dim": getattr(model, "embedding_dim", getattr(model, "feature_dim", None)),
        }
    return config, data, model


def make_trainer(config, run_dir, *, training=True, stop_after_epoch=None):
    callbacks = []
    callbacks.extend(instantiate(spec) for spec in config.get("callbacks", []))
    if training:
        if config.get("freeze_schedule"):
            from .callbacks.freeze import ScheduledFreeze

            callbacks.append(ScheduledFreeze(config["freeze_schedule"]))
        if stop_after_epoch:
            callbacks.append(StopAfterEpoch(stop_after_epoch))
        early = config.get("early_stopping")
        if early:
            callbacks.append(FreshEarlyStopping(**early))
        cp = dict(config.get("checkpoint", {}))
        cp.pop("_target_", None)
        callbacks.append(RetainingModelCheckpoint(local_path(run_dir) / "checkpoints", **cp))
    logger = False
    if training and config.get("logger"):
        spec = dict(config["logger"])
        spec.pop("save_dir", None)
        logger = instantiate(spec, save_dir=str(run_dir), version="")
    options = dict(config["trainer"])
    options.pop("_target_", None)
    options.update(
        default_root_dir=str(run_dir),
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=training,
        inference_mode=False,
        plugins=[WorkspaceCheckpointIO.create()],
    )
    if not training:
        options.update(devices=1, strategy="auto", num_sanity_val_steps=0)
    return Trainer(**options)


def train(
    config,
    *,
    checkpoint_path=None,
    strict=False,
    overrides=None,
    stop_after_epoch=None,
    prefix="run",
    initialization_path=None,
    parent_checkpoint_path=None,
):
    source = deepcopy(config)
    payload = None
    if checkpoint_path:
        _, payload = load_checkpoint(checkpoint_path)
        if strict:
            validate_config(config, strict=True)

            def execution_recipe(value):
                if isinstance(value, dict):
                    return {
                        key: execution_recipe(item)
                        for key, item in value.items()
                        if key not in {"path", "root", "data_root", "output_dir", "save_dir"}
                    }
                if isinstance(value, list):
                    return [execution_recipe(item) for item in value]
                return value

            if execution_recipe(config) != execution_recipe(payload["config"]):
                raise ValueError("Strict resume recipe/execution settings changed; create a branch")
            if payload.get("boundary") != "epoch":
                raise ValueError("Strict resume supports completed epoch boundaries only")
            if payload["environment"] != environment():
                raise ValueError("Strict resume environment changed")
            if not payload.get("code_manifest"):
                raise ValueError("Strict checkpoint lacks code identity")
            from .experiment.runtime import digest

            for path, expected in payload["code_manifest"].items():
                if not local_path(path).is_file() or digest(path) != expected:
                    raise ValueError(f"Strict resume code changed: {path}; use a branch")
            verify_identity(config, payload["data_identity"])
    config, data, model = prepare(
        config, restore=bool(checkpoint_path or initialization_path or parent_checkpoint_path)
    )
    if parent_checkpoint_path:
        parent_checkpoint, _ = load_checkpoint(parent_checkpoint_path)
        state = {
            key.removeprefix("model."): value
            for key, value in parent_checkpoint["state_dict"].items()
            if key.startswith("model.")
        }
        own_state = model.state_dict()
        mismatched = [
            key
            for key in state
            if key in own_state
            and isinstance(state[key], torch.Tensor)
            and state[key].shape != own_state[key].shape
        ]
        forbidden = [key for key in mismatched if not key.startswith("head.")]
        if forbidden:
            raise ValueError(f"Incompatible backbone parameter shapes: {forbidden}")
        for key in mismatched:
            state.pop(key)
        result = model.load_state_dict(state, strict=False)
        config["parent"].update(
            missing_keys=result.missing_keys,
            unexpected_keys=result.unexpected_keys,
            reinitialized_keys=mismatched,
        )
    if payload and strict and config["data"]["class_maps"] != payload["class_maps"]:
        raise ValueError("Class mappings/order differ from checkpoint")
    inherited = os.environ.get("TRAINER_ACTIVE_RUN") if os.environ.get("LOCAL_RANK") else None
    run_dir = local_path(inherited) if inherited else new_run(config.get("output_dir", "runs"), prefix)
    os.environ["TRAINER_ACTIVE_RUN"] = str(run_dir)
    module = ResearchTask(config, model=model, run_dir=run_dir)
    if initialization_path:
        from .experiment.artifacts import load_initialization

        initialization = load_initialization(initialization_path)
        if initialization.get("task_state"):
            module.load_state_dict(initialization["task_state"], strict=True)
        else:
            module.model.load_state_dict(initialization["model_state"], strict=True)
        module._initial_rng = initialization["rng"]
    snapshot_options = {}
    if checkpoint_path:
        from .experiment.artifacts import load_initialization
        from .experiment.runtime import digest

        source_checkpoint = local_path(checkpoint_path)
        source_run = (
            source_checkpoint.parent.parent
            if source_checkpoint.parent.name == "checkpoints"
            else source_checkpoint.parent
        )
        source_run = local_path(source_run)
        source_lock = source_run / "weights.lock.json"
        original_weights = json.loads(source_lock.read_text()) if source_lock.exists() else None
        source_initialization = source_run / "initialization.pt"
        if source_initialization.exists() and (
            original_weights is None or original_weights.get("initialization")
        ):
            expected_hash = (original_weights or {}).get("initialization", {}).get("sha256")
            load_initialization(source_initialization, expected_hash)
            original_initialization = source_initialization
        else:
            original_initialization = False
        snapshot_options = {
            "initialization_source": original_initialization,
            "weights_source": original_weights,
            "parent": {
                "mode": "resume",
                "checkpoint": str(source_checkpoint),
                "sha256": digest(source_checkpoint),
            },
        }
    if not os.environ.get("LOCAL_RANK") or os.environ.get("LOCAL_RANK") == "0":
        module.data_identity = snapshot(
            run_dir,
            config,
            source,
            overrides or [],
            data,
            model,
            initialization_module=module,
            **snapshot_options,
        )
    else:
        from .experiment.artifacts import data_identity

        module.data_identity = data_identity(config)
    trainer = make_trainer(config, run_dir, stop_after_epoch=stop_after_epoch)
    module.code_manifest = json.loads((run_dir / "code_manifest.json").read_text())["files"]
    module.weights_provenance = json.loads((run_dir / "weights.lock.json").read_text())
    try:
        trainer.fit(
            module, datamodule=data, ckpt_path=str(local_path(checkpoint_path)) if checkpoint_path else None
        )
        if trainer.is_global_zero:
            write_json(
                run_dir / "result.json",
                {
                    "metrics": module.last_metrics,
                    "run_dir": str(run_dir),
                    "best": trainer.checkpoint_callback.best_model_path,
                    "last": trainer.checkpoint_callback.last_model_path,
                },
            )
    finally:
        if not os.environ.get("LOCAL_RANK"):
            os.environ.pop("TRAINER_ACTIVE_RUN", None)
    return run_dir


def restore_model(checkpoint_path):
    checkpoint, payload = load_checkpoint(checkpoint_path)
    config, data, model = prepare(payload["config"], restore=True)
    module = ResearchTask(config, model=model)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    module.eval()
    return module, data, payload


def inference(checkpoint_path, *, eval_config=None, manifest=None, embeddings=False, output=None):
    from .experiment.runtime import digest, isolated_rng
    from .reporting import build_report

    checkpoint, payload = load_checkpoint(checkpoint_path)
    config = deepcopy(payload["config"])
    device = "cpu"
    if eval_config:
        cfg = OmegaConf.to_container(OmegaConf.load(local_path(eval_config)), resolve=True)
        config["data"].update(cfg.get("data", {}))
        config["evaluation"].update(cfg.get("evaluation", {}))
        config["tta"] = cfg.get("tta", config.get("tta", {}))
        device = cfg.get("device", "cpu")
    config["data"]["train"] = []
    if manifest:
        config["data"]["train"], config["data"]["val"] = [], []
        config["data"]["test"] = [{"name": "predict", "path": str(local_path(manifest))}]
        # An unlabeled CSV is valid: every declared categorical label is optional.
        for target in config["data"].get("schema", {}).get("targets", {}).values():
            target["allow_missing"] = True
    config, data, model = prepare(config, restore=True)
    if data.class_maps != payload["class_maps"]:
        raise ValueError("Evaluation class mapping/order differs from the checkpoint")
    module = ResearchTask(config, model=model)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    module.to(device).eval()
    run_dir = local_path(output) if output else new_run(config.get("output_dir", "runs"), "evaluate")
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluator = module._evaluator()
    evaluator.provenance.update(
        {
            "checkpoint": str(local_path(checkpoint_path)),
            "checkpoint_hash": digest(checkpoint_path),
            "preprocessing": config.get("transforms"),
            "evaluation": config.get("evaluation"),
        }
    )
    # Standalone evaluation explicitly uses a single process; fit evaluates with DDP gathering.
    with isolated_rng(), torch.no_grad():
        for role in ("val", "test"):
            for dataset in data.datasets[role]:
                loader = data._loader(dataset, False)
                for batch in loader:
                    from lightning.fabric.utilities.apply_func import move_data_to_device

                    batch = move_data_to_device(batch, torch.device(device))
                    outputs = (
                        {"embeddings": module.model.extract_features(batch["inputs"])}
                        if embeddings
                        else module(batch["inputs"])
                    )
                    evaluator.update(outputs, batch, role, dataset.name)
    metrics = evaluator.compute()
    evaluator.write_predictions(run_dir / "predictions.jsonl")
    write_json(run_dir / "metrics.json", metrics)
    if not embeddings:
        build_report(
            evaluator.records,
            run_dir / "reports",
            task=config["task"],
            class_maps=data.class_maps,
            target_key=config.get("evaluation", {}).get("target_key", "label"),
        )
    return run_dir
