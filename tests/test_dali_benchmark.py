import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_dali_dermoscopy.py"
SPEC = importlib.util.spec_from_file_location("dali_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_finite_stream_keeps_tail_and_epoch_order():
    sequential = benchmark.FixedBatches(10, 4, seed=42, shuffle=False)
    assert list(sequential) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert len(sequential) == 3
    shuffled = benchmark.FixedBatches(10, 4, seed=42, shuffle=True)
    first = list(shuffled)
    assert sorted(index for batch in first for index in batch) == list(range(10))
    assert list(shuffled) == first
    shuffled.set_epoch(1)
    assert list(shuffled) != first


def test_manifest_resolves_paths_and_rejects_unlabelled_images(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(image)
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(json.dumps({"image_path": "image.png", "label": 2}) + "\n")
    rows = benchmark.read_manifest(manifest, tmp_path)
    assert rows[0] == {"image_path": str(image), "label": 2}
    manifest.write_text(json.dumps({"image_path": "image.png", "label": "MEL"}) + "\n")
    with pytest.raises(ValueError, match="nonnegative integer"):
        benchmark.read_manifest(manifest, tmp_path)


def test_pillow_reference_crops_before_resize_and_normalizes(tmp_path):
    array = np.zeros((6, 8, 3), dtype=np.uint8)
    array[1:5, 2:6] = [255, 128, 0]
    image = tmp_path / "image.png"
    Image.fromarray(array).save(image)
    rows = [{"image_path": str(image), "bbox_xyxy": [2, 1, 6, 5], "label": 7}]
    reference = benchmark.PillowReference(rows, 2, [0.5, 0.5, 0.5], [0.5, 0.25, 1])
    pixels, label = reference[0]
    expected = torch.tensor([1.0, (128 / 255 - 0.5) / 0.25, -0.5])
    assert label == 7
    assert tuple(pixels.shape) == (3, 2, 2)
    torch.testing.assert_close(pixels[:, 0, 0], expected)
    assert torch.equal(pixels[:, 0, 0], pixels[:, 1, 1])


def test_prepared_image_is_not_cropped_twice(tmp_path):
    image = tmp_path / "crop.png"
    Image.new("RGB", (4, 4), (255, 0, 0)).save(image)
    rows = [{"prepared_image_path": str(image), "bbox_xyxy": [10, 10, 20, 20], "label": 0}]
    reference = benchmark.PillowReference(rows, 2, [0, 0, 0], [1, 1, 1])
    pixels, _ = reference[0]
    assert torch.equal(pixels[0], torch.ones(2, 2))
    assert pixels[1:].count_nonzero() == 0


def test_eval_geometry_preserves_aspect_and_centers_black_padding(tmp_path):
    image = tmp_path / "wide.png"
    Image.new("RGB", (8, 4), (255, 255, 255)).save(image)
    reference = benchmark.PillowReference(
        [{"image_path": str(image), "label": 0}], 4, [0, 0, 0], [1, 1, 1]
    )
    pixels, _ = reference[0]
    assert torch.equal(pixels[:, 1:3], torch.ones(3, 2, 4))
    assert pixels[:, [0, 3]].count_nonzero() == 0


def test_cli_rejects_invalid_normalization_and_hyperparameters():
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--manifest", "samples.jsonl", "--std", "1", "0", "1"])
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--manifest", "samples.jsonl", "--batch-size", "0"])
