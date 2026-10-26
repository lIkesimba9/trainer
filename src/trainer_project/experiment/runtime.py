"""Workspace boundaries, hashing and explicit random-state ownership."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[3]


def local_path(path: str | Path, *, base: str | Path | None = None) -> Path:
    path = Path(path).expanduser()
    result = (Path(base or WORKSPACE) / path).resolve() if not path.is_absolute() else path.resolve()
    if not result.is_relative_to(WORKSPACE):
        raise ValueError(f"Path must remain inside {WORKSPACE}: {result}")
    return result


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with local_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def config_hash(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def write_json(path: str | Path, data: Any) -> Path:
    path = local_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, default=str, allow_nan=False)
    return path


def capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


@contextlib.contextmanager
def isolated_rng():
    state = capture_rng()
    try:
        yield
    finally:
        restore_rng(state)


def new_run(root: str | Path, prefix: str = "run") -> Path:
    from datetime import datetime, timezone

    root = local_path(root)
    path = root / f"{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:10]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


class WorkspaceCheckpointIO:
    """Factory defined lazily to keep command-line help light."""

    @staticmethod
    def create():
        from lightning.fabric.plugins.io import CheckpointIO

        class RetainingCheckpointIO(CheckpointIO):
            def save_checkpoint(self, checkpoint, path, storage_options=None):
                destination = local_path(path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                pending = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.pending")
                torch.save(checkpoint, pending)
                # Retain the staging inode; no user or generated file is removed.
                if destination.exists():
                    archive = destination.with_name(
                        f"{destination.stem}.previous-{uuid.uuid4().hex}{destination.suffix}"
                    )
                    import shutil

                    shutil.copyfile(destination, archive)
                    # Updating an existing pointer is deliberately a file write, not unlink/rename.
                    with pending.open("rb") as source, destination.open("wb") as target:
                        import shutil

                        shutil.copyfileobj(source, target)
                else:
                    os.link(pending, destination)

            def load_checkpoint(self, path, map_location=None):
                return torch.load(local_path(path), map_location=map_location or "cpu", weights_only=False)

            def remove_checkpoint(self, path):
                # Explicit user instruction: retain every checkpoint, including evicted top-K.
                return None

        return RetainingCheckpointIO()
