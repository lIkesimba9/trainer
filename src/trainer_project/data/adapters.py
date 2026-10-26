"""Explicit local HF Dataset and COCO-to-exclusive-semantic adapters.

Adapters materialize immutable, content-addressed PNGs/manifests inside the
workspace. They never modify source datasets or remove cache artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .manifests import WORKSPACE, CsvDataset, confined_path, load_mapping, sha256_file


def _cache_directory(kind, identity, cache_dir=None):
    root = confined_path(cache_dir or WORKSPACE / ".cache" / "adapters", must_exist=False)
    result = root / kind / identity
    result.mkdir(parents=True, exist_ok=True)
    return result


def _directory_identity(path):
    return {
        str(item.relative_to(path)): sha256_file(confined_path(item))
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _write_manifest(path, records):
    if path.exists():
        return
    fields = list(dict.fromkeys(key for row in records for key in row))
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _as_pil(value):
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        if value.get("bytes"):
            return Image.open(io.BytesIO(value["bytes"]))
        if value.get("path"):
            return Image.open(confined_path(value["path"]))
    if isinstance(value, (str, Path)):
        return Image.open(confined_path(value))
    return Image.fromarray(np.asarray(value))


class HFDatasetAdapter(CsvDataset):
    """Read a provided HF split without inventing a train/validation partition."""

    def __init__(
        self,
        path,
        split="train",
        revision=None,
        dataset_name=None,
        data_root=".",
        schema=None,
        cache_dir=None,
        name="hf",
        **kwargs,
    ):
        try:
            import datasets
        except ImportError as exc:
            raise ImportError("Install trainer-project[data] for HFDatasetAdapter") from exc
        schema = load_mapping(schema)
        local_path = confined_path(path, data_root, must_exist=dataset_name is None)
        if dataset_name is None:
            loaded = datasets.load_from_disk(str(local_path))
            dataset = loaded[split] if isinstance(loaded, datasets.DatasetDict) else loaded
            fingerprint = getattr(dataset, "_fingerprint", None) or sha256_file(local_path / "state.json")
            source_content = _directory_identity(local_path)
        else:
            if not revision:
                raise ValueError("Remote HF datasets require an explicit pinned revision")
            hf_cache = confined_path(cache_dir or WORKSPACE / ".cache" / "huggingface", must_exist=False)
            hf_cache.mkdir(parents=True, exist_ok=True)
            dataset = datasets.load_dataset(
                dataset_name, name=path or None, split=split, revision=revision, cache_dir=str(hf_cache)
            )
            fingerprint = getattr(dataset, "_fingerprint", revision)
            source_content = None
        identity = hashlib.sha256(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "source_content": source_content,
                    "split": split,
                    "schema": schema,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        cache = _cache_directory("hf", identity, cache_dir)
        records = []
        for index, row in enumerate(dataset):
            output = {
                key: value
                for key, value in row.items()
                if isinstance(value, (str, int, float, bool)) or value is None
            }
            output[schema.get("sample_id", "sample_id")] = str(
                row.get(schema.get("sample_id", "sample_id"), index)
            )
            for section, default_type in (
                (schema.get("inputs", {}), "image"),
                (schema.get("targets", {}), "categorical"),
            ):
                for key, declaration in section.items():
                    spec = self._spec(declaration)
                    column, kind = spec.get("column", key), spec.get("type", default_type)
                    value = row.get(column)
                    if value is None:
                        output[column] = ""
                    elif kind in {"image", "mask", "segmentation", "indexed_mask", "binary_mask"}:
                        target = cache / f"{index:09d}_{key}.png"
                        if not target.exists():
                            _as_pil(value).save(target, format="PNG")
                        output[column] = str(target)
                    elif kind in {"categorical", "class"}:
                        feature = dataset.features.get(column)
                        output[column] = (
                            feature.int2str(value)
                            if hasattr(feature, "int2str") and isinstance(value, int)
                            else value
                        )
            records.append(output)
        manifest = cache / "manifest.csv"
        _write_manifest(manifest, records)
        self.source_provenance = {
            "adapter": "huggingface",
            "split": split,
            "revision": revision,
            "dataset_name": dataset_name,
            "fingerprint": fingerprint,
            "source_content": source_content,
            "path": str(local_path),
        }
        super().__init__(path=str(manifest), name=name, data_root=data_root, schema=schema, **kwargs)

    def provenance(self, content_hash=True):
        return dict(super().provenance(content_hash), source=self.source_provenance)


class CocoSemanticDataset(CsvDataset):
    """COCO polygon/RLE or panoptic data with explicit overlap/background policy."""

    def __init__(
        self,
        path,
        category_mapping,
        variant="instances",
        overlap_policy="error",
        background_id=0,
        ignore_index=255,
        image_key="image",
        target_key="segmentation_target",
        images_root=None,
        panoptic_root=None,
        data_root=".",
        schema=None,
        cache_dir=None,
        name="coco",
        **kwargs,
    ):
        if variant not in {"instances", "stuff", "panoptic"}:
            raise ValueError("COCO variant must be instances, stuff or panoptic")
        if overlap_policy not in {"error", "first", "last", "ignore"}:
            raise ValueError("COCO overlap_policy must be error, first, last or ignore")
        mapping = {int(key): int(value) for key, value in dict(category_mapping).items()}
        if not mapping or any(value < 0 or value >= 255 for value in mapping.values()):
            raise ValueError("COCO category_mapping must map category IDs to uint8 class IDs below255")
        source = confined_path(path, data_root)
        image_root = confined_path(images_root or data_root)
        with source.open(encoding="utf-8") as handle:
            document = json.load(handle)
        identity = hashlib.sha256(
            json.dumps(
                {
                    "sha256": sha256_file(source),
                    "mapping": mapping,
                    "variant": variant,
                    "overlap": overlap_policy,
                    "background": background_id,
                    "ignore": ignore_index,
                    "decoder": "pycocotools-v1",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        cache = _cache_directory("coco", identity, cache_dir)
        annotations = {}
        for annotation in document.get("annotations", []):
            annotations.setdefault(annotation["image_id"], []).append(annotation)
        records = []
        schema = load_mapping(schema) or {
            "inputs": {image_key: {"column": image_key, "type": "image"}},
            "targets": {
                target_key: {
                    "column": target_key,
                    "type": "segmentation",
                    "ignore_index": ignore_index,
                    "num_classes": max([background_id, *mapping.values()]) + 1,
                }
            },
        }
        input_column = self._spec(schema["inputs"][image_key]).get("column", image_key)
        target_column = self._spec(schema["targets"][target_key]).get("column", target_key)
        for image in document.get("images", []):
            width, height = int(image["width"]), int(image["height"])
            target = np.full((height, width), int(background_id), dtype=np.uint8)
            assigned = np.zeros((height, width), dtype=bool)
            for annotation in annotations.get(image["id"], []):
                if variant == "panoptic":
                    if not panoptic_root:
                        raise ValueError("COCO panoptic requires panoptic_root")
                    panoptic_path = confined_path(annotation["file_name"], confined_path(panoptic_root))
                    with Image.open(panoptic_path) as source_image:
                        rgb = np.asarray(source_image).astype(np.int64)
                    if rgb.ndim != 3 or rgb.shape[2] != 3:
                        raise ValueError("COCO panoptic PNG must be RGB segment IDs")
                    segment_ids = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
                    parts = [
                        (
                            segment_ids == segment["id"],
                            segment["category_id"],
                            bool(segment.get("iscrowd", 0)),
                        )
                        for segment in annotation["segments_info"]
                    ]
                else:
                    segmentation = annotation.get("segmentation")
                    if isinstance(segmentation, list):
                        try:
                            from pycocotools import mask as mask_util
                        except ImportError as exc:
                            raise ImportError("COCO polygons require trainer-project[data]") from exc
                        for polygon in segmentation:
                            if len(polygon) < 6 or len(polygon) % 2:
                                raise ValueError("Invalid COCO polygon")
                        if not segmentation:
                            raise ValueError("Empty COCO polygon segmentation")
                        rles = mask_util.frPyObjects(segmentation, height, width)
                        mask = np.asarray(mask_util.decode(mask_util.merge(rles)), dtype=bool)
                    elif isinstance(segmentation, dict):
                        try:
                            from pycocotools import mask as mask_util
                        except ImportError as exc:
                            raise ImportError("COCO RLE requires trainer-project[data]") from exc
                        rle = segmentation
                        if isinstance(rle.get("counts"), list):
                            rle = mask_util.frPyObjects(rle, height, width)
                        mask = np.asarray(mask_util.decode(rle), dtype=bool)
                    else:
                        raise ValueError("COCO annotation requires explicit polygon or RLE segmentation")
                    parts = [(mask, annotation["category_id"], bool(annotation.get("iscrowd", 0)))]
                for mask, category, crowd in parts:
                    if mask.shape != target.shape:
                        raise ValueError("COCO mask shape differs from image dimensions")
                    if category not in mapping:
                        raise ValueError(f"COCO category {category} missing from category_mapping")
                    overlap = assigned & mask
                    if overlap.any() and overlap_policy == "error":
                        raise ValueError(
                            "Overlapping COCO instances require explicit lossy-conversion policy"
                        )
                    selected = mask & ~assigned if overlap_policy == "first" else mask
                    target[selected] = ignore_index if crowd else mapping[category]
                    if overlap_policy == "ignore":
                        target[overlap] = ignore_index
                    assigned |= mask
            mask_path = cache / f"{image['id']}.png"
            if not mask_path.exists():
                Image.fromarray(target).save(mask_path, format="PNG")
            records.append(
                {
                    "sample_id": str(image["id"]),
                    input_column: str(confined_path(image["file_name"], image_root)),
                    target_column: str(mask_path),
                }
            )
        manifest = cache / "manifest.csv"
        _write_manifest(manifest, records)
        self.source_provenance = {
            "adapter": "coco",
            "variant": variant,
            "annotation_path": str(source),
            "annotation_sha256": sha256_file(source),
            "category_mapping": mapping,
            "overlap_policy": overlap_policy,
            "background_id": background_id,
            "ignore_index": ignore_index,
        }
        super().__init__(path=str(manifest), name=name, data_root=data_root, schema=schema, **kwargs)

    def provenance(self, content_hash=True):
        return dict(super().provenance(content_hash), source=self.source_provenance)
