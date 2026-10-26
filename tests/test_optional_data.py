"""Local-only checks for optional adapter and transform integrations."""

import csv

import numpy as np
import pytest
import torch
from PIL import Image

from trainer_project.data import HFDatasetAdapter, common_collate
from trainer_project.transforms import AlbumentationsTransform, KorniaBatchTransform


def spatial_sample():
    image = torch.arange(48).reshape(3, 4, 4).float() / 48
    target = torch.zeros(4, 4, dtype=torch.long)
    target[:, :2] = 1
    valid = torch.ones(4, 4, dtype=torch.bool)
    valid[:1] = False
    return {
        "sample_id": "a",
        "manifest_name": "data",
        "inputs": {"image": image, "input_mask": target[None].float()},
        "input_present": {"image": torch.tensor(True), "input_mask": torch.tensor(True)},
        "views": {},
        "targets": {"segmentation_target": target},
        "target_valid": {"segmentation_target": valid},
        "supervision_allowed": {"segmentation_target": torch.tensor(True)},
        "metadata": {},
        "_mask_parents": {"input_mask": "image"},
    }


def test_albumentations_geometry_keeps_image_mask_and_validity_synchronous():
    A = pytest.importorskip("albumentations")
    sample = spatial_sample()
    before_image = sample["inputs"]["image"].clone()
    before_target = sample["targets"]["segmentation_target"].clone()
    before_valid = sample["target_valid"]["segmentation_target"].clone()
    transform = AlbumentationsTransform(A.Compose([A.HorizontalFlip(p=1)]))
    result = transform(sample, seed=42)
    assert torch.equal(result["inputs"]["image"], before_image.flip(-1))
    assert torch.equal(result["targets"]["segmentation_target"], before_target.flip(-1))
    assert torch.equal(result["target_valid"]["segmentation_target"], before_valid.flip(-1))
    assert torch.equal(result["inputs"]["input_mask"][0].long(), result["targets"]["segmentation_target"])


def test_kornia_geometry_keeps_image_mask_and_validity_synchronous():
    pytest.importorskip("kornia")
    sample = spatial_sample()
    sample.pop("_mask_parents")
    batch = common_collate([sample])
    before_image = batch["inputs"]["image"].clone()
    before_target = batch["targets"]["segmentation_target"].clone()
    before_valid = batch["target_valid"]["segmentation_target"].clone()
    transform = KorniaBatchTransform(
        [{"_target_": "kornia.augmentation.RandomHorizontalFlip", "p": 1.0}],
        image_key="image",
        mask_input_keys=["input_mask"],
        target_keys=["segmentation_target"],
    )
    result = transform(batch)
    assert torch.equal(result["inputs"]["image"], before_image.flip(-1))
    assert torch.equal(result["targets"]["segmentation_target"], before_target.flip(-1))
    assert torch.equal(result["target_valid"]["segmentation_target"], before_valid.flip(-1))
    assert torch.equal(result["inputs"]["input_mask"][:, 0].long(), result["targets"]["segmentation_target"])


def test_hf_local_dataset_preserves_class_names_all_labels_and_source_hashes(tmp_path):
    datasets = pytest.importorskip("datasets")
    for index in range(2):
        Image.fromarray(np.full((5, 6, 3), index * 100, dtype=np.uint8)).save(tmp_path / f"image{index}.png")
    source = datasets.Dataset.from_dict(
        {
            "sample_id": ["one", "two"],
            "image": [str(tmp_path / "image0.png"), str(tmp_path / "image1.png")],
            "label": [0, 1],
            "aux": ["x", ""],
        },
        features=datasets.Features(
            {
                "sample_id": datasets.Value("string"),
                "image": datasets.Image(),
                "label": datasets.ClassLabel(names=["alpha", "beta"]),
                "aux": datasets.Value("string"),
            }
        ),
    )
    source.save_to_disk(str(tmp_path / "source"))
    schema = {
        "inputs": {"image": {"type": "image"}},
        "targets": {"label": {"type": "categorical"}, "aux": {"type": "categorical", "allow_missing": True}},
    }
    adapter = HFDatasetAdapter(
        path=str(tmp_path / "source"),
        data_root=tmp_path,
        schema=schema,
        class_maps={"label": ["alpha", "beta"], "aux": ["x", "y"]},
        cache_dir=tmp_path / "adapter_cache",
    )
    batch = common_collate([adapter[0], adapter[1]])
    assert batch["targets"]["label"].tolist() == [0, 1]
    assert batch["target_valid"]["aux"].tolist() == [True, False]
    identity = adapter.provenance()
    assert identity["source"]["source_content"]
    assert all(len(value) == 64 for value in identity["source"]["source_content"].values())


def test_original_dense_targets_collate_with_different_image_sizes(tmp_path):
    from trainer_project.data import CsvDataset

    for index, (height, width) in enumerate([(5, 6), (7, 8)]):
        Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)).save(tmp_path / f"image{index}.png")
        Image.fromarray(np.ones((height, width), dtype=np.uint8)).save(tmp_path / f"mask{index}.png")
    with (tmp_path / "data.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "image", "mask"])
        writer.writerows([[str(index), f"image{index}.png", f"mask{index}.png"] for index in range(2)])
    dataset = CsvDataset(
        path="data.csv",
        data_root=tmp_path,
        role="val",
        schema={
            "inputs": {"image": {"type": "image"}},
            "targets": {"mask": {"type": "indexed_mask", "num_classes": 2}},
        },
        transform={"image_size": 8},
    )
    batch = common_collate([dataset[0], dataset[1]])
    assert batch["targets"]["mask"].shape == (2, 8, 8)
    assert [tuple(value.shape) for value in batch["metadata"]["original_targets"]["mask"]] == [(5, 6), (7, 8)]


def test_coco_polygon_adapter_preserves_category_and_explicit_overlap(tmp_path):
    import json

    from trainer_project.data import CocoSemanticDataset

    pytest.importorskip("pycocotools")
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / "coco.png")
    document = {
        "images": [{"id": 1, "file_name": "coco.png", "height": 8, "width": 8}],
        "categories": [{"id": 10, "name": "object"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 10,
                "iscrowd": 0,
                "segmentation": [[1, 1, 6, 1, 6, 6, 1, 6]],
                "area": 25,
                "bbox": [1, 1, 5, 5],
            }
        ],
    }
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps(document))
    dataset = CocoSemanticDataset(
        annotations,
        category_mapping={10: 1},
        data_root=tmp_path,
        images_root=tmp_path,
        variant="instances",
        overlap_policy="error",
        transform={"_target_": "trainer_project.transforms.SampleTransform", "image_size": 8},
    )
    target = dataset[0]["targets"]["segmentation_target"]
    assert set(target.unique().tolist()) == {0, 1}
    assert int((target == 1).sum()) == 25
    document["annotations"].append({**document["annotations"][0], "id": 2})
    overlap = tmp_path / "overlap.json"
    overlap.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="[Oo]verlap"):
        CocoSemanticDataset(
            overlap,
            category_mapping={10: 1},
            data_root=tmp_path,
            images_root=tmp_path,
            overlap_policy="error",
        )
