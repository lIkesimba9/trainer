"""CPU contracts for the optional loader; GPU smoke is an explicit remote check."""

from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

MODULE_PATH = Path(__file__).resolve().parents[1]/"src/trainer_project/data/dali_dermoscopy.py"
SPEC = importlib.util.spec_from_file_location("standalone_dali_dermoscopy", MODULE_PATH)
dali = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dali)


class Sampler:
    def __init__(self, batches):
        self.batches, self.epochs = batches, []

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


class FakeOutput:
    def __init__(self, tensor):
        self.tensor = tensor

    def as_tensor(self):
        return self.tensor


class FakePipeline:
    def __init__(self, loader):
        self.loader, self.inputs, self.observed_parameters = loader, {}, []
        self.storage = torch.empty(loader.microbatch, 3, 2, 2)

    def feed_input(self, name, value):
        self.inputs[name] = value

    def run(self):
        encoded = self.inputs["encoded"]
        for index, sample in enumerate(encoded):
            self.storage[index].fill_(int(sample[0]))
        self.observed_parameters.extend(self.loader._active_parameters)
        return (FakeOutput(self.storage[:len(encoded)]),)


def fake_loader(tmp_path, microbatch=2):
    rows = []
    for index in range(11):
        path = tmp_path/f"sample{index}.bin"
        path.write_bytes(bytes([index]))
        rows.append({"image_path": path.name, "height": 2, "width": 2, "label": index})
    sampler = Sampler([list(reversed(range(11))), [2, 2, 8]])
    loader = dali.DaliDermLoader(rows, sampler, resolution=448, mean=[.1, .2, .3],
                                std=[1, 1, 1], workspace=tmp_path, microbatch=microbatch)
    instances = []

    def build():
        pipeline = FakePipeline(loader)
        instances.append(pipeline)
        return pipeline

    loader._build_pipeline = build
    return loader, instances


def test_import_does_not_require_dali(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith(("nvidia.dali", "trainer_project.milk_finetune"))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    spec = importlib.util.spec_from_file_location("standalone_optional_dali", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.DaliDermLoader


def test_sampler_order_all_classes_duplicates_tail_and_owned_storage(tmp_path):
    loader, instances = fake_loader(tmp_path)
    batches = list(loader)
    assert len(loader) == len(batches) == 2
    assert batches[0][1].tolist() == list(reversed(range(11)))
    assert batches[1][1].tolist() == [2, 2, 8]
    assert batches[0][0][:, 0, 0, 0].tolist() == list(reversed(range(11)))
    assert batches[1][0][:, 0, 0, 0].tolist() == [2, 2, 8]
    assert len(instances[0].observed_parameters) == 14


def test_rng_is_stable_across_microbatch_and_epoch_is_forwarded(tmp_path):
    loader2, instances2 = fake_loader(tmp_path, 2)
    loader8, instances8 = fake_loader(tmp_path, 8)
    loader2.set_epoch(7)
    loader8.set_epoch(7)
    list(loader2)
    list(loader8)
    assert instances2[0].observed_parameters == instances8[0].observed_parameters
    assert loader2.batch_sampler.epochs == [7]
    previous = instances2[0].observed_parameters
    loader2.set_epoch(8)
    list(loader2)
    assert len(instances2) == 2
    assert previous != instances2[1].observed_parameters


def test_bbox_is_integer_yx_after_orientation_and_prepared_path_ignores_old_box(tmp_path):
    path = tmp_path/"oriented.jpg"
    source = Image.new("RGB", (10, 20))
    exif = Image.Exif()
    exif[274] = 6
    source.save(path, exif=exif)
    anchor, shape = dali.crop_metadata({"bbox_xyxy": [3, 2, 15, 8]}, path)
    assert anchor.tolist() == [2, 3]
    assert shape.tolist() == [6, 12]
    anchor, shape = dali.crop_metadata({"prepared_image_path": str(path), "bbox_xyxy": [99, 99, 100, 100]}, path)
    assert anchor.tolist() == [0, 0]
    assert shape.tolist() == [10, 20]
    with pytest.raises(ValueError, match="Invalid upright bbox"):
        dali.crop_metadata({"height": 10, "width": 20, "bbox_xyxy": [-1, 0, 8, 8]}, path)


def test_reflect101_coordinate_folding_and_scipy_symmetric_gaussian():
    coordinates = torch.tensor([-10., -5., -1., 0., .5, 4., 5., 10.])
    assert dali._reflect_coordinates(coordinates, 5).tolist() == [2., 3., 1., 0., .5, 4., 3., 2.]
    np.testing.assert_equal(dali._reflect_coordinates(coordinates, 1), np.zeros(8))
    scipy = pytest.importorskip("scipy.ndimage")
    source = np.random.default_rng(42).uniform(-1, 1, (13, 17)).astype(np.float32)
    expected = scipy.gaussian_filter(source, sigma=1.5, radius=6, mode="reflect")
    actual = dali._gaussian_field(torch.from_numpy(source), sigma=1.5, radius=6).numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)


def test_gpu_median_small_image_matches_reflect101_reference():
    cv2 = pytest.importorskip("cv2")
    source = np.random.default_rng(21).integers(0, 256, (7, 9, 3), dtype=np.uint8)
    padded = cv2.copyMakeBorder(source, 1, 1, 1, 1, cv2.BORDER_REFLECT_101)
    # OpenCV median internally repeats borders; explicit pad isolates comparison.
    expected = cv2.medianBlur(padded, 3)[1:-1, 1:-1]
    actual = dali._median(torch.from_numpy(source), 3).numpy()
    np.testing.assert_array_equal(actual, expected)


def test_recipe_and_normalization_are_explicit_and_validated(tmp_path):
    assert dali.recipe_metadata("gpu_native")["max_long_edge"] is None
    assert dali.recipe_metadata("gpu_bounded", 1536)["max_long_edge"] == 1536
    changes = " ".join(dali.recipe_metadata("gpu_native")["differences_from_legacy"])
    assert "Lab L" in changes and "RNG" in changes
    assert dali._planckian_gains(4500) == pytest.approx((.5623/.4457, 1., .3154/.4457))
    with pytest.raises(ValueError, match="recipe"):
        dali.recipe_metadata("legacy")
    with pytest.raises(ValueError, match="normalization"):
        dali.DaliDermLoader([], Sampler([]), resolution=512, mean=[0]*3,
                            std=[1, 0, 1], workspace=tmp_path)


@pytest.mark.parametrize("label", [None, -1, True, False, 0.2, 1., "3"])
def test_labels_must_be_nonnegative_integers_before_dali_initialization(tmp_path, label):
    with pytest.raises(ValueError, match="nonnegative integer classification label"):
        dali.DaliDermLoader([{"label": label}], Sampler([]), resolution=448,
                            mean=[0]*3, std=[1]*3, workspace=tmp_path)
    accepted = dali.DaliDermLoader([{"label": np.int64(0)}], Sampler([]), resolution=448,
                                  mean=[0]*3, std=[1]*3, workspace=tmp_path)
    assert accepted._labels == [0]


@pytest.mark.parametrize("index", [-1, 11, True, .5, "2"])
def test_invalid_sampler_index_fails_before_dali_initialization(tmp_path, index):
    loader, instances = fake_loader(tmp_path)
    loader.batch_sampler = Sampler([[index]])
    with pytest.raises(ValueError, match="outside integer range"):
        list(loader)
    assert not instances
