from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import shutil
from copy import deepcopy
from importlib import metadata
from pathlib import Path
from urllib.parse import urlparse

import torch
from omegaconf import OmegaConf

from .runtime import WORKSPACE, capture_rng, config_hash, digest, local_path, write_json


def environment() -> dict:
    packages = {}
    for name in (
        "trainer-project",
        "torch",
        "lightning",
        "hydra-core",
        "omegaconf",
        "numpy",
        "Pillow",
        "timm",
        "transformers",
        "albumentations",
        "kornia",
        "datasets",
        "pycocotools",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }


def data_identity(config: dict, data_module=None) -> dict:
    """Hash the effective adapter output, including image/mask/validity content.

    A DataModule is used so per-manifest column mappings, optional adapters and
    implicit column names have exactly the same interpretation as training.
    This constructs data adapters only; it never constructs a pretrained model.
    """
    if data_module is None:
        from hydra.utils import instantiate

        from .runtime import isolated_rng

        with isolated_rng():
            data_module = instantiate(
                config["data"],
                transforms=config.get("transforms", config["data"].get("transforms", {})),
                seed=config.get("seed", 42),
            )
            data_module.setup()
    manifests, assets = [], {}
    for record in data_module.manifest_info(content_hash=True):
        item = {
            "role": record["role"],
            "name": record["name"],
            "path": record["path"],
            "sha256": record["sha256"],
            "root": record["data_root"],
            "rows": record["rows"],
            "supervision": record.get("supervision_allowed", {}),
        }
        if "source" in record:
            item["source"] = record["source"]
        manifests.append(item)
        assets.update(record.get("assets", {}))
    return {
        "manifests": manifests,
        "assets": assets,
        "class_maps": data_module.class_maps,
        "schema": data_module.schema,
    }


def verify_identity(config: dict, expected: dict) -> None:
    actual = data_identity(config)

    def identities(value):
        return [
            (
                item["role"],
                item["name"],
                None if item.get("source") else item["sha256"],
                item.get("rows"),
                item.get("supervision"),
                {
                    key: value
                    for key, value in item.get("source", {}).items()
                    if key not in {"path", "annotation_path"}
                },
            )
            for item in value["manifests"]
        ]

    if identities(expected) != identities(actual):
        raise ValueError("Dataset manifest content, adapter source, role or identity changed")
    # Root remapping changes filenames, so compare content identities rather than
    # spelling. Per-manifest hashes still preserve manifest membership/order.
    if sorted(expected.get("assets", {}).values()) != sorted(actual.get("assets", {}).values()):
        raise ValueError("Dataset image/mask/validity content changed")
    if expected.get("class_maps", actual["class_maps"]) != actual["class_maps"]:
        raise ValueError("Dataset class order changed")
    if expected.get("schema", actual["schema"]) != actual["schema"]:
        raise ValueError("Dataset target/input schema changed")


def tensor_state_hash(state: dict) -> str:
    """Stable identity for actual loaded tensors, independent of archive metadata."""
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        result.update(name.encode())
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            result.update(str(tensor.dtype).encode())
            result.update(json.dumps(list(tensor.shape)).encode())
            if tensor.is_quantized:
                result.update(str(tensor.qscheme()).encode())
                tensor = tensor.dequantize()
            result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            result.update(json.dumps(value, sort_keys=True, default=str).encode())
    return result.hexdigest()


def _cached_weight_files(record: dict) -> list[dict]:
    """Resolve only the configured repository/checkpoint inside local caches."""
    files = set()

    def candidate(path):
        try:
            path = local_path(path)
        except (TypeError, ValueError):
            return
        if path.is_file():
            files.add(path)

    def explicit(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {
                    "path",
                    "file",
                    "filename",
                    "checkpoint",
                    "weights_path",
                    "checkpoint_path",
                } and isinstance(value, str):
                    candidate(value)
                explicit(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                explicit(item)

    explicit(record)
    configuration = record.get("config") or {}
    repository = (
        record.get("model_name")
        if record.get("provider") == "huggingface"
        else configuration.get("hf_hub_id")
    )
    revision = record.get("revision") or configuration.get("hf_hub_revision")
    if repository:
        cache = (
            local_path(os.environ.get("HF_HOME", WORKSPACE / ".cache" / "huggingface"))
            / "hub"
            / ("models--" + repository.replace("/", "--"))
        )
        if not revision and (cache / "refs" / "main").is_file():
            revision = (cache / "refs" / "main").read_text().strip()
            record["resolved_revision"] = revision
        snapshot_path = cache / "snapshots" / str(revision or "")
        if revision and snapshot_path.is_dir():
            for path in snapshot_path.rglob("*"):
                if path.suffix in {".safetensors", ".bin", ".pt", ".pth"}:
                    candidate(path)
    url = configuration.get("url")
    if record.get("provider") == "torchvision" and record.get("weights") not in {None, "None"}:
        try:
            from torchvision.models import get_model_weights

            enum = get_model_weights(record["model_name"])
            requested = record["weights"].rsplit(".", 1)[-1]
            url = enum[requested].url
        except (ImportError, KeyError, ValueError):
            pass
    if url:
        candidate(
            local_path(os.environ.get("TORCH_HOME", WORKSPACE / ".cache" / "torch"))
            / "hub"
            / "checkpoints"
            / Path(urlparse(url).path).name
        )
    return [{"path": str(path), "filename": path.name, "sha256": digest(path)} for path in sorted(files)]


def load_initialization(path, expected_sha256=None) -> dict:
    path = local_path(path)
    if expected_sha256 and digest(path) != expected_sha256:
        raise ValueError("Initialization artifact SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("format") != "trainer_initialization_v1"
        or "model_state" not in payload
        or "rng" not in payload
    ):
        raise ValueError("Not a complete trainer initialization artifact")
    if tensor_state_hash(payload["model_state"]) != payload["model_state_sha256"]:
        raise ValueError("Initialization model tensor identity mismatch")
    if (
        payload.get("task_state") is not None
        and tensor_state_hash(payload["task_state"]) != payload["task_state_sha256"]
    ):
        raise ValueError("Initialization objective/model tensor identity mismatch")
    return payload


def snapshot(
    run_dir: Path,
    config: dict,
    source_config: dict,
    overrides: list[str],
    data_module,
    model,
    initialization_module=None,
    initialization_source=None,
    weights_source=None,
    parent=None,
) -> dict:
    run_dir = local_path(run_dir)
    OmegaConf.save(OmegaConf.create(source_config), run_dir / "config.source.yaml")
    OmegaConf.save(OmegaConf.create(config), run_dir / "config.resolved.yaml")
    OmegaConf.save(OmegaConf.create(overrides), run_dir / "overrides.yaml")
    (run_dir / ".hydra").mkdir(exist_ok=True)
    OmegaConf.save(OmegaConf.create(source_config), run_dir / ".hydra" / "config.yaml")
    OmegaConf.save(OmegaConf.create(overrides), run_dir / ".hydra" / "overrides.yaml")
    write_json(run_dir / "class_maps.json", data_module.class_maps)
    OmegaConf.save(OmegaConf.create(config["data"].get("schema", {})), run_dir / "labels.schema.yaml")
    write_json(run_dir / "preprocessing.json", config.get("transforms", config["data"].get("transforms", {})))
    identity = data_identity(config, data_module)
    for item in identity["manifests"]:
        destination = run_dir / "manifests" / item["role"] / (item["name"] + Path(item["path"]).suffix)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path(item["path"]), destination)
    write_json(run_dir / "data_manifest.json", identity)
    weights = {"seed": config["seed"], "architecture": config["model"], "sources": []}
    if weights_source is not None:
        weights["sources"] = deepcopy(weights_source.get("sources", []))
    elif initialization_source is not False:
        for name, component in model.named_modules():
            provenance = getattr(component, "weights_provenance", None)
            if provenance:
                record = deepcopy(provenance)
                record["component"] = name
                record["loaded_state_sha256"] = tensor_state_hash(component.state_dict())
                record["files"] = _cached_weight_files(record)
                weights["sources"].append(record)
    initial_path = run_dir / "initialization.pt"
    if initialization_source is False:
        # A copied checkpoint can still resume completely. Its throwaway model
        # construction must never be advertised as the original initialization.
        weights["initialization"] = None
        weights["reproduce"] = {
            "available": False,
            "reason": "Original initialization artifact is unavailable beside the source checkpoint",
        }
    else:
        if initialization_source is not None:
            source_path = local_path(initialization_source)
            expected_hash = (weights_source or {}).get("initialization", {}).get("sha256")
            initialization = load_initialization(source_path, expected_hash)
            if initialization["class_maps"] != data_module.class_maps:
                raise ValueError("Inherited initialization class order differs from the resumed experiment")
            if initial_path.exists():
                raise ValueError("Initialization destination already exists")
            shutil.copyfile(source_path, initial_path)
            weights["inherited_initialization"] = {"source": str(source_path), "sha256": digest(source_path)}
        else:
            model_state = {
                key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else deepcopy(value)
                for key, value in model.state_dict().items()
            }
            task_state = (
                {
                    key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else deepcopy(value)
                    for key, value in initialization_module.state_dict().items()
                }
                if initialization_module is not None
                else None
            )
            initialization = {
                "format": "trainer_initialization_v1",
                "model_state": model_state,
                "model_state_sha256": tensor_state_hash(model_state),
                "task_state": task_state,
                "task_state_sha256": tensor_state_hash(task_state) if task_state is not None else None,
                "rng": capture_rng(),
                "seed": config["seed"],
                "architecture": config["model"],
                "class_maps": data_module.class_maps,
            }
            with initial_path.open("xb") as stream:
                torch.save(initialization, stream)
        weights["initialization"] = {
            "path": str(initial_path),
            "filename": initial_path.name,
            "sha256": digest(initial_path),
            "model_state_sha256": initialization["model_state_sha256"],
            "task_state_sha256": initialization["task_state_sha256"],
        }
        weights["reproduce"] = {"available": True}
    write_json(run_dir / "weights.lock.json", weights)
    write_json(run_dir / "environment.json", environment())
    if (WORKSPACE / "uv.lock").exists():
        shutil.copyfile(WORKSPACE / "uv.lock", run_dir / "uv.lock")
    source_files = set((WORKSPACE / "src" / "trainer_project").rglob("*.py"))

    def custom_targets(node):
        if isinstance(node, dict):
            if "_target_" in node:
                from hydra.utils import get_object

                try:
                    source = Path(inspect.getfile(get_object(node["_target_"]))).resolve()
                    if source.is_relative_to(WORKSPACE) and ".venv" not in source.parts:
                        source_files.add(source)
                except (TypeError, ImportError, AttributeError):
                    pass
            for value in node.values():
                custom_targets(value)
        elif isinstance(node, list):
            for value in node:
                custom_targets(value)

    custom_targets(config)
    code = {}
    for path in sorted(source_files):
        relative = path.relative_to(WORKSPACE)
        destination = run_dir / "code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        code[str(relative)] = digest(path)
    write_json(run_dir / "code_manifest.json", {"files": code})
    manifest = {
        "run_id": run_dir.name,
        "config_hash": config_hash(config),
        "schema_version": config["schema_version"],
        "parent": parent if parent is not None else config.get("parent"),
        "selection_uses_test": any(
            (config.get(key) or {}).get("monitor", "").startswith("test/")
            for key in ("checkpoint", "early_stopping", "scheduler")
        ),
        "artifacts": {
            str(path.relative_to(run_dir)): digest(path) for path in run_dir.iterdir() if path.is_file()
        },
    }
    write_json(run_dir / "manifest.json", manifest)
    return identity


def load_checkpoint(path: str | Path) -> tuple[dict, dict]:
    checkpoint = torch.load(local_path(path), map_location="cpu", weights_only=False)
    if "experiment" not in checkpoint:
        raise ValueError("Checkpoint lacks experiment metadata; weights-only exports cannot resume")
    payload = checkpoint["experiment"]
    if payload["config_hash"] != config_hash(payload["config"]):
        raise ValueError("Checkpoint resolved configuration hash mismatch")
    return checkpoint, payload
