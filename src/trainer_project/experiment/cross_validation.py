"""Prepared-fold execution with checked, model-specific OOF provenance."""

from __future__ import annotations

import json

from hydra.utils import instantiate
from omegaconf import OmegaConf

from .config import load_config, objective_specs
from .runtime import digest, local_path, new_run, write_json


def training_membership(config, identity_key="sample_id"):
    data = instantiate(config["data"], _recursive_=False, _convert_="all")
    data.setup()
    fields = set()
    for objective in objective_specs(config["objective"]):
        name = objective["_target_"].rsplit(".", 1)[-1]
        if name not in {"BYOLObjective", "MAEObjective"}:
            fields.add(objective.get("target_key", "label"))
    result = {"images": [], "ssl": [], "supervised_targets": {key: [] for key in fields}}
    for dataset in data.datasets["train"]:
        for row in dataset.records:
            identity = row.get(identity_key)
            if identity is None:
                raise ValueError(f"Missing declared OOF identity {identity_key}")
            result["images"].append(identity)
            if config["task"] == "ssl":
                result["ssl"].append(identity)
            for key in fields:
                if key not in dataset.target_schema or not dataset._allowed(key):
                    continue
                specification = dataset._spec(dataset.target_schema[key])
                value = row["_raw"].get(dataset._column(key, specification), "")
                if not dataset._missing(value, specification):
                    result["supervised_targets"][key].append(identity)
    return result


def run_cross_validation(path):
    from trainer_project.engine import inference, train
    from trainer_project.evaluation import (
        EvaluationAccumulator,
        cross_validation_summary,
        read_predictions,
        validate_oof,
        write_predictions,
    )
    from trainer_project.reporting import build_report

    specification = OmegaConf.to_container(OmegaConf.load(local_path(path)), resolve=True)
    folds = specification.get("folds", [])
    if not folds:
        raise ValueError("CV needs a nonempty list of prepared fold configs")
    output = new_run(specification.get("output_dir", "runs"), "cv")
    rows, metrics, membership, sources = [], [], {}, []
    class_maps = None
    task = None
    target = "label"
    for index, fold in enumerate(folds):
        config = (
            load_config(fold)
            if isinstance(fold, str)
            else load_config(fold["config"], fold.get("overrides", []))
        )
        config["fold"] = index
        run = train(config, prefix=f"fold_{index}")
        result = json.loads((run / "result.json").read_text())
        checkpoint = result["best"] or result["last"]
        if not checkpoint:
            raise ValueError("CV requires a retained model checkpoint")
        evaluated = inference(checkpoint)
        predictions = [
            row for row in read_predictions(evaluated / "predictions.jsonl") if row["role"] == "val"
        ]
        checkpoint_hash = digest(checkpoint)
        resolved = OmegaConf.to_container(OmegaConf.load(run / "config.resolved.yaml"), resolve=True)
        if class_maps is not None and resolved["data"]["class_maps"] != class_maps:
            raise ValueError("Prepared folds have incompatible class mappings")
        class_maps = resolved["data"]["class_maps"]
        task = resolved["task"]
        target = resolved.get("evaluation", {}).get("target_key", "label")
        membership[checkpoint_hash] = training_membership(
            resolved, specification.get("identity_key", "sample_id")
        )
        for row in predictions:
            row["fold"] = index
            row["provenance"]["checkpoint_hash"] = checkpoint_hash
            row["provenance"]["fold"] = index
        rows.extend(predictions)
        metrics.append(json.loads((evaluated / "metrics.json").read_text()))
        sources.append(
            {
                "fold": index,
                "run": str(run),
                "checkpoint": checkpoint,
                "predictions": str(evaluated / "predictions.jsonl"),
            }
        )
    validation = validate_oof(
        rows,
        membership,
        identity_key=specification.get("identity_key", "sample_id"),
        require_image_independence=specification.get("require_image_independence", False),
    )
    identities = [
        row.get(
            specification.get("identity_key", "sample_id"),
            row.get("metadata", {}).get(specification.get("identity_key", "sample_id")),
        )
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise ValueError(
            "Prepared validation folds contain repeated OOF identities; explicitly aggregate these observations first"
        )
    evaluator = EvaluationAccumulator(task, class_maps, target_key=target)
    evaluator.add_records(rows)
    summary = {
        "folds": cross_validation_summary(metrics),
        "oof": evaluator.compute(),
        "protocol": validation,
        "sources": sources,
    }
    write_predictions(rows, output / "oof_predictions.jsonl")
    write_json(output / "training_membership.json", membership)
    write_json(output / "summary.json", summary)
    build_report(
        rows,
        output / "reports",
        task=task,
        class_maps=class_maps,
        target_key=target,
        provenance={"oof_protocol": validation, "folds": sources},
    )
    return output
