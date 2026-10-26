from __future__ import annotations

import argparse
import json
from copy import deepcopy

from omegaconf import OmegaConf

from .experiment.config import load_config, remap_paths
from .experiment.runtime import local_path, new_run, write_json


def parser():
    cli = argparse.ArgumentParser(prog="trainer", description="Lightning/Hydra research trainer (uv)")
    commands = cli.add_subparsers(dest="command", required=True)
    for name in ("train", "validate-data"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--config")
        cmd.add_argument("--stop-after-epoch", type=int)
        cmd.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. trainer.max_epochs=2")
    cmd = commands.add_parser("reproduce")
    cmd.add_argument("--config", required=True)
    cmd.add_argument("--path-map")
    for name in ("resume", "branch", "finetune"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--checkpoint", required=True)
        cmd.add_argument("--strict", action="store_true")
        cmd.add_argument("--path-map")
        cmd.add_argument("overrides", nargs="*")
    for name in ("evaluate", "predict", "extract_embeddings"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--checkpoint", required=True)
        cmd.add_argument("--eval-config")
        cmd.add_argument("--manifest")
        cmd.add_argument("--output")
    for name in ("knn", "linear_probe"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--checkpoint", required=True)
        cmd.add_argument("--eval-config", required=True)
        cmd.add_argument("--output")
    cmd = commands.add_parser("ensemble")
    cmd.add_argument("--predictions", nargs="+", required=True)
    cmd.add_argument("--weights", nargs="+", type=float)
    cmd.add_argument("--method", choices=["mean_probabilities", "mean_logits"], default="mean_probabilities")
    cmd.add_argument("--output")
    cmd = commands.add_parser("report")
    cmd.add_argument("--predictions", required=True)
    cmd.add_argument("--output")
    cmd = commands.add_parser("cv")
    cmd.add_argument("--config", required=True, help="YAML with a folds list of complete config paths")
    cmd = commands.add_parser("oof")
    cmd.add_argument("--predictions", nargs="+", required=True)
    cmd.add_argument("--membership", required=True)
    cmd.add_argument("--identity-key", default="sample_id")
    cmd.add_argument("--require-image-independence", action="store_true")
    cmd.add_argument("--output")
    cmd = commands.add_parser("export")
    cmd.add_argument("--checkpoint", required=True)
    cmd.add_argument("--output", required=True)
    return cli


def _path_map(config, path):
    if path:
        mapping = OmegaConf.to_container(OmegaConf.load(local_path(path)), resolve=True)
        return remap_paths(config, mapping)
    return config


def _branch(checkpoint_path, overrides):
    from .engine import train
    from .experiment.artifacts import load_checkpoint
    from .experiment.runtime import digest

    checkpoint, payload = load_checkpoint(checkpoint_path)
    cfg = OmegaConf.merge(OmegaConf.create(payload["config"]), OmegaConf.from_dotlist(overrides))
    config = OmegaConf.to_container(cfg, resolve=True)
    config.pop("runtime", None)
    # Keep the parent checkpoint as an explicit local initialization artifact.
    config["parent"] = {
        "checkpoint": str(local_path(checkpoint_path)),
        "sha256": digest(checkpoint_path),
        "overrides": overrides,
        "mode": "branch",
    }
    return train(config, overrides=overrides, prefix="branch", parent_checkpoint_path=checkpoint_path)


def _representation(args):
    from .engine import make_trainer, prepare
    from .experiment.artifacts import load_checkpoint
    from .modules import ResearchTask

    checkpoint, payload = load_checkpoint(args.checkpoint)
    config = deepcopy(payload["config"])
    patch = OmegaConf.to_container(OmegaConf.load(local_path(args.eval_config)), resolve=True)
    for key in ("data", "evaluation", "ssl_evaluation"):
        config[key] = OmegaConf.to_container(
            OmegaConf.merge(config.get(key, {}), patch.get(key, {})), resolve=True
        )
    if config["task"] != "ssl":
        raise ValueError("kNN and linear probe require an SSL checkpoint")
    config["ssl_evaluation"]["enabled"] = True
    config["ssl_evaluation"]["every_n_validation_events"] = 1
    for name in ("knn", "linear_probe"):
        config["ssl_evaluation"].setdefault(name, {})["enabled"] = args.command == name
        config["ssl_evaluation"][name]["every_n_validation_events"] = 1
    config, data, model = prepare(config, restore=True)
    run = local_path(args.output) if args.output else new_run(config.get("output_dir", "runs"), args.command)
    run.mkdir(parents=True, exist_ok=True)
    module = ResearchTask(config, model=model, run_dir=run)
    module.load_state_dict(checkpoint["state_dict"])
    trainer = make_trainer(config, run, training=False)
    trainer.validate(module, datamodule=data)
    write_json(run / "metrics.json", module.last_metrics)
    return run


def main(argv=None):
    args = parser().parse_args(argv)
    from .engine import inference, prepare, train
    from .experiment.artifacts import load_checkpoint, verify_identity

    if args.command == "train":
        result = train(
            load_config(args.config, args.overrides),
            overrides=args.overrides,
            stop_after_epoch=args.stop_after_epoch,
        )
    elif args.command == "validate-data":
        config, data, model = prepare(load_config(args.config, args.overrides))
        result = data.manifest_info()
    elif args.command == "reproduce":
        config = load_config(args.config)
        config_path = local_path(args.config)
        original_run = (
            config_path.parent.parent if config_path.parent.name == "checkpoints" else config_path.parent
        )
        expected_path = original_run / "data_manifest.json"
        config = _path_map(config, args.path_map)
        if expected_path.exists():
            verify_identity(config, json.loads(expected_path.read_text()))
        weights_path = original_run / "weights.lock.json"
        initialization_path = None
        if weights_path.exists():
            weights = json.loads(weights_path.read_text())
            if weights.get("reproduce", {}).get("available") is False:
                raise ValueError(weights["reproduce"]["reason"])
            if weights.get("initialization"):
                from .experiment.artifacts import load_initialization

                initialization_path = original_run / "initialization.pt"
                load_initialization(initialization_path, weights["initialization"]["sha256"])
        result = train(config, prefix="reproduce", initialization_path=initialization_path)
    elif args.command == "resume":
        if args.overrides:
            raise ValueError("Resume preserves the recipe; use branch for changed settings")
        _, payload = load_checkpoint(args.checkpoint)
        config = _path_map(payload["config"], args.path_map)
        result = train(config, checkpoint_path=args.checkpoint, strict=args.strict, prefix="resume")
    elif args.command in ("branch", "finetune"):
        result = _branch(args.checkpoint, args.overrides)
    elif args.command in ("evaluate", "predict", "extract_embeddings"):
        result = inference(
            args.checkpoint,
            eval_config=args.eval_config,
            manifest=args.manifest,
            embeddings=args.command == "extract_embeddings",
            output=args.output,
        )
    elif args.command in ("knn", "linear_probe"):
        result = _representation(args)
    elif args.command == "ensemble":
        from .evaluation import ensemble_predictions, read_predictions, write_predictions
        from .reporting import build_report

        run = local_path(args.output) if args.output else new_run("runs", "ensemble")
        run.mkdir(parents=True, exist_ok=True)
        rows = ensemble_predictions(
            [read_predictions(local_path(p)) for p in args.predictions],
            weights=args.weights,
            method=args.method,
        )
        write_predictions(rows, run / "predictions.jsonl")
        result = build_report(rows, run / "reports")
    elif args.command == "report":
        from .reporting import build_report

        run = local_path(args.output) if args.output else new_run("runs", "report")
        result = build_report(local_path(args.predictions), run)
    elif args.command == "cv":
        from .experiment.cross_validation import run_cross_validation

        result = run_cross_validation(args.config)
    elif args.command == "oof":
        from .evaluation import read_predictions, validate_oof, write_predictions
        from .reporting import build_report

        rows = [
            row
            for path in args.predictions
            for row in read_predictions(local_path(path))
            if row["role"] == "val"
        ]
        membership = json.loads(local_path(args.membership).read_text())
        protocol = validate_oof(
            rows,
            membership,
            identity_key=args.identity_key,
            require_image_independence=args.require_image_independence,
        )
        run = local_path(args.output) if args.output else new_run("runs", "oof")
        run.mkdir(parents=True, exist_ok=True)
        write_predictions(rows, run / "predictions.jsonl")
        write_json(run / "protocol.json", protocol)
        result = build_report(rows, run / "reports", provenance={"oof_protocol": protocol})
    elif args.command == "export":
        import torch

        checkpoint, payload = load_checkpoint(args.checkpoint)
        output = local_path(args.output)
        if output.exists():
            raise ValueError("Export destination already exists")
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "weights_only_not_resumable",
                "state_dict": checkpoint["state_dict"],
                "config": payload["config"],
                "class_maps": payload["class_maps"],
            },
            output,
        )
        result = output
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
