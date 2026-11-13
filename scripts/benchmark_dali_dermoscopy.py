#!/usr/bin/env python3
"""Measure an explicit JSONL image stream; never starts model training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset


class FixedBatches:
    """Finite epoch order, including a smaller last batch without padding."""

    def __init__(self, size: int, batch_size: int, seed: int, shuffle: bool):
        if size <= 0 or batch_size <= 0:
            raise ValueError("Manifest and batch_size must be positive")
        self.size, self.batch_size = size, batch_size
        self.seed, self.shuffle, self.epoch = seed, shuffle, 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return math.ceil(self.size / self.batch_size)

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.size, generator=generator).tolist()
        else:
            indices = list(range(self.size))
        for start in range(0, self.size, self.batch_size):
            yield indices[start : start + self.batch_size]


def read_manifest(path: Path, path_root: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            image_key = "prepared_image_path" if row.get("prepared_image_path") else "image_path"
            if not row.get(image_key):
                raise ValueError(f"Line {line_number}: expected image_path or prepared_image_path")
            image_path = Path(row[image_key]).expanduser()
            row[image_key] = str(image_path if image_path.is_absolute() else path_root / image_path)
            if not Path(row[image_key]).is_file():
                raise FileNotFoundError(f"Line {line_number}: {row[image_key]}")
            label = row.get("label")
            if isinstance(label, bool) or not isinstance(label, int) or label < 0:
                raise ValueError(f"Line {line_number}: label must be a nonnegative integer")
            rows.append(row)
    if not rows:
        raise ValueError("Manifest is empty")
    return rows


class PillowReference(Dataset):
    """Deterministic EXIF/RGB/BB reference; no training augmentations."""

    def __init__(self, rows, resolution, mean, std):
        self.rows, self.resolution = rows, resolution
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        prepared = row.get("prepared_image_path")
        with Image.open(prepared or row["image_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            if not prepared and row.get("bbox_xyxy") is not None:
                box = tuple(map(int, row["bbox_xyxy"]))
                x1, y1, x2, y2 = box
                if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
                    raise ValueError(f"Invalid upright bbox: {box}")
                image = image.crop(box)
            scale = self.resolution / max(image.size)
            width, height = (max(1, round(axis * scale)) for axis in image.size)
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            canvas = Image.new("RGB", (self.resolution, self.resolution))
            canvas.paste(image, ((self.resolution - width) // 2, (self.resolution - height) // 2))
            array = np.array(canvas, copy=True)
        tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255)
        return (tensor - self.mean) / self.std, row["label"]


def summarize(values):
    array = np.array(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this benchmark, including the Torch reference")
    device = torch.device("cuda", args.device_id)
    torch.cuda.set_device(device)
    rows = read_manifest(args.manifest, args.path_root)
    sampler = FixedBatches(len(rows), args.batch_size, args.seed, args.shuffle)
    torch.set_num_threads(args.cpu_threads)
    construction_start = time.perf_counter()
    if args.backend == "dali":
        from trainer_project.data.dali_dermoscopy import DaliDermLoader

        loader = DaliDermLoader(
            rows,
            sampler,
            resolution=args.resolution,
            mean=args.mean,
            std=args.std,
            workspace=args.path_root,
            training=args.training,
            microbatch=args.microbatch,
            num_threads=args.workers,
            seed=args.seed,
            recipe=args.recipe,
            max_long_edge=args.max_long_edge,
            device_id=args.device_id,
        )
        metadata = loader.metadata
    else:
        if args.training:
            raise ValueError("--training is available only for DALI; Torch is a deterministic reference")
        options = {"prefetch_factor": 1} if args.workers else {}
        loader = DataLoader(
            PillowReference(rows, args.resolution, args.mean, args.std),
            batch_sampler=sampler,
            num_workers=args.workers,
            pin_memory=True,
            **options,
        )
        metadata = {
            "recipe": "pillow_eval", "training": False,
            "resize": "Pillow bilinear, longest edge to resolution, centered black padding",
        }
    construction_s = time.perf_counter() - construction_start
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    iterator_started = time.perf_counter()
    iterator = iter(loader)
    iterator_startup_s = time.perf_counter() - iterator_started
    times, sizes = [], []
    labels_digest = hashlib.sha256()
    warmup_s, first_batch_s = 0.0, None
    started = time.perf_counter()
    for batch_number in range(min(len(sampler), args.warmup + args.steps)):
        batch_started = time.perf_counter()
        images, labels = next(iterator)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - batch_started
        if first_batch_s is None:
            first_batch_s = iterator_startup_s + elapsed
        if batch_number < args.warmup:
            warmup_s += elapsed
            continue
        if images.shape[0] != labels.shape[0]:
            raise RuntimeError("Image and label batch lengths differ")
        times.append(elapsed)
        sizes.append(len(labels))
        labels_digest.update(labels.cpu().numpy().astype("<i8", copy=False).tobytes())
    total_s = time.perf_counter() - started
    if not times:
        raise ValueError("No measured batches remain; lower --warmup or supply a larger manifest")
    return {
        "backend": args.backend,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "rows": len(rows),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "resolution": args.resolution,
        "batch_size": args.batch_size,
        "microbatch": args.microbatch,
        "workers": args.workers,
        "seed": args.seed,
        "shuffle": args.shuffle,
        "normalization": {"mean": args.mean, "std": args.std},
        "construction_s": construction_s,
        "iterator_startup_s": iterator_startup_s,
        "first_batch_s": first_batch_s,
        "warmup_batches": min(args.warmup, len(sampler)),
        "warmup_s": warmup_s,
        "measured_batches": len(times),
        "measured_images": sum(sizes),
        "measured_batch_sizes": sizes,
        "ready_batch_seconds": summarize(times),
        "images_per_s": sum(sizes) / sum(times),
        "loop_wall_s_including_warmup": total_s,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "labels_sha256_measured_stream": labels_digest.hexdigest(),
        "loader": metadata,
        "notes": [
            "Loader benchmark only; does not measure model forward, backward, optimizer, or validation.",
            "DALI training and Torch evaluation use different augmentation recipes; speeds are not a pure backend A/B.",
            "DALI internal allocations are excluded from Torch allocator peaks; use nvidia-smi externally.",
            "Torch/Pillow and DALI resize need not be pixel-identical.",
        ],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--path-root", type=Path, default=Path.cwd())
    parser.add_argument("--backend", choices=["dali", "torch"], default="dali")
    parser.add_argument("--resolution", type=int, choices=[448, 512], default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2, help="DALI CPU threads or Torch loader workers")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--device-id", type=int, default=0, help="Index inside CUDA_VISIBLE_DEVICES")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--recipe", choices=["gpu_native", "gpu_bounded"], default="gpu_bounded")
    parser.add_argument("--max-long-edge", type=int, default=1024)
    parser.add_argument("--mean", type=float, nargs=3, default=[0.485, 0.456, 0.406])
    parser.add_argument("--std", type=float, nargs=3, default=[0.229, 0.224, 0.225])
    parser.add_argument("--output", type=Path, help="New JSON report path; refuses to overwrite")
    args = parser.parse_args(argv)
    for key in ["batch_size", "microbatch", "cpu_threads", "steps", "max_long_edge"]:
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.workers < 0 or args.warmup < 0 or args.device_id < 0 or args.seed < 0:
        parser.error("workers, warmup, device-id, and seed must be nonnegative")
    if args.backend == "dali" and args.workers == 0:
        parser.error("DALI requires at least one CPU thread (--workers)")
    if any(not math.isfinite(value) for value in [*args.mean, *args.std]) or min(args.std) <= 0:
        parser.error("mean/std must be finite and std must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.output and args.output.exists():
        raise FileExistsError(f"Report already exists: {args.output}")
    result = benchmark(args)
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
