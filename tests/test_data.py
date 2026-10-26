import csv
import json

import numpy as np
import pytest
import torch
from PIL import Image

from trainer_project.data import CsvDataset, ManifestDataModule, common_collate
from trainer_project.data.adapters import CocoSemanticDataset
from trainer_project.transforms import SampleTransform


def make_image(path, index=0):
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    image[:, :5] = [240, index, 0]
    Image.fromarray(image).save(path)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def schema():
    return {
        "inputs": {"image": {"type": "image"}},
        "targets": {
            "label": {"type": "categorical", "classes": ["NA", "b"], "allow_missing": True},
            "aux": {"type": "categorical", "classes": ["x", "y"], "allow_missing": True},
        },
    }


def test_optional_labels_preserve_NA_and_validity(tmp_path):
    make_image(tmp_path / "a.png")
    make_image(tmp_path / "b.png")
    manifest = tmp_path / "data.csv"
    write_csv(
        manifest, [{"image": "a.png", "label": "NA", "aux": ""}, {"image": "b.png", "label": "", "aux": "y"}]
    )
    module = ManifestDataModule(
        data_root=str(tmp_path), train=[{"name": "train", "path": "data.csv"}], schema=schema()
    )
    module.setup()
    batch = common_collate([module.train_dataset[0], module.train_dataset[1]])
    assert batch["targets"]["label"].tolist() == [0, -1]
    assert batch["target_valid"]["label"].tolist() == [True, False]
    assert batch["target_valid"]["aux"].tolist() == [False, True]
    assert len(set(batch["sample_id"])) == 2
    assert module.class_maps["label"] == {"NA": 0, "b": 1}


def test_multimodal_and_synchronous_input_target_masks(tmp_path):
    make_image(tmp_path / "a.png")
    make_image(tmp_path / "b.png", 2)
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[:, :5] = 1
    Image.fromarray(mask).save(tmp_path / "mask.png")
    write_csv(
        tmp_path / "data.csv",
        [
            {
                "sample_id": "one",
                "image_a": "a.png",
                "image_b": "b.png",
                "text": "текст",
                "input_mask": "mask.png",
                "segmentation_target": "mask.png",
            }
        ],
    )
    declaration = {
        "inputs": {
            "image_a": {"type": "image"},
            "image_b": {"type": "image"},
            "text": {"type": "text", "max_length": 12},
            "input_mask": {"type": "mask", "image_key": "image_a"},
        },
        "targets": {"segmentation_target": {"type": "segmentation", "num_classes": 2}},
    }
    dataset = CsvDataset(
        path="data.csv",
        name="train",
        data_root=tmp_path,
        schema=declaration,
        transform=SampleTransform(size=(8, 10), hflip=1),
    )
    sample = dataset[0]
    assert torch.equal(sample["targets"]["segmentation_target"], sample["inputs"]["input_mask"][0].long())
    assert sample["inputs"]["image_a"][0, :, :5].sum() == 0
    assert sample["inputs"]["text"]["attention_mask"].sum() > 0
    assert sample["metadata"]["original_size"].tolist() == [8, 10]


def test_missing_modality_explicit_presence_policy(tmp_path):
    make_image(tmp_path / "a.png")
    write_csv(tmp_path / "data.csv", [{"image": "a.png", "image_b": ""}])
    declaration = {
        "inputs": {"image": {"type": "image"}, "image_b": {"type": "image", "allow_missing": True}}
    }
    with pytest.raises(ValueError, match="presence"):
        CsvDataset("data.csv", data_root=tmp_path, schema=declaration)
    dataset = CsvDataset(
        "data.csv", data_root=tmp_path, schema=declaration, missing_modality_policy="presence"
    )
    assert not dataset[0]["input_present"]["image_b"]


def test_corrupt_and_invalid_masks_fail_before_training(tmp_path):
    make_image(tmp_path / "a.png")
    Image.fromarray(np.full((8, 10), 3, dtype=np.uint8)).save(tmp_path / "mask.png")
    write_csv(tmp_path / "data.csv", [{"image": "a.png", "segmentation_target": "mask.png"}])
    declaration = {
        "inputs": {"image": {"type": "image"}},
        "targets": {"segmentation_target": {"type": "segmentation", "num_classes": 2}},
    }
    with pytest.raises(ValueError, match="outside"):
        CsvDataset("data.csv", data_root=tmp_path, schema=declaration)


def test_content_identity_and_class_order(tmp_path):
    make_image(tmp_path / "a.png")
    write_csv(tmp_path / "data.csv", [{"image": "a.png", "label": "b", "aux": "x"}])
    module = ManifestDataModule(
        data_root=tmp_path, schema=schema(), train=[{"name": "tr", "path": "data.csv"}]
    )
    module.setup()
    first = module.manifest_info()[0]
    make_image(tmp_path / "a.png", 6)
    second = module.manifest_info()[0]
    assert first["sha256"] == second["sha256"]
    assert first["assets"] != second["assets"]
    state = module.state_dict()
    state["class_maps"] = {"label": {"b": 0, "NA": 1}}
    with pytest.raises(ValueError, match="class order"):
        module.load_state_dict(state)


def test_reference_uses_eval_transform_and_no_ssl_views(tmp_path):
    make_image(tmp_path / "a.png")
    write_csv(tmp_path / "data.csv", [{"image": "a.png", "label": "NA", "aux": "x"}])
    module = ManifestDataModule(
        data_root=tmp_path,
        schema=schema(),
        train=[{"name": "tr", "path": "data.csv"}],
        transforms={
            "train": {"image_size": [8, 10], "hflip": 1},
            "eval": {"image_size": [8, 10], "hflip": 0},
            "ssl": True,
        },
    )
    module.setup()
    train_image = module.train_dataset[0]["inputs"]["image"]
    reference = next(iter(module.reference_dataloader()[0]))
    assert reference["views"] == {}
    assert torch.equal(train_image.flip(-1), reference["inputs"]["image"][0])
    assert module.datasets["train"][0].ssl


def test_duplicate_ids_and_paths_outside_workspace_rejected(tmp_path):
    make_image(tmp_path / "a.png")
    write_csv(
        tmp_path / "data.csv",
        [{"sample_id": "same", "image": "a.png"}, {"sample_id": "same", "image": "a.png"}],
    )
    with pytest.raises(ValueError, match="Duplicate"):
        CsvDataset("data.csv", data_root=tmp_path)
    with pytest.raises(ValueError, match="workspace"):
        ManifestDataModule(data_root="/")


def test_coco_polygon_conversion_requires_overlap_policy(tmp_path):
    make_image(tmp_path / "a.png")
    annotation = {
        "images": [{"id": 1, "width": 10, "height": 8, "file_name": "a.png"}],
        "annotations": [{"image_id": 1, "category_id": 7, "segmentation": [[0, 0, 4, 0, 4, 7, 0, 7]]}],
    }
    path = tmp_path / "coco.json"
    path.write_text(json.dumps(annotation))
    dataset = CocoSemanticDataset(
        path=path, data_root=tmp_path, category_mapping={7: 1}, cache_dir=tmp_path / "cache"
    )
    sample = dataset[0]
    assert set(sample["targets"]["segmentation_target"].unique().tolist()) == {0, 1}
    assert dataset.provenance()["source"]["variant"] == "instances"


def test_experiment_identity_catches_implicit_columns_and_validity_mutation(tmp_path):
    from trainer_project.experiment.artifacts import data_identity, verify_identity

    make_image(tmp_path / "a.png")
    Image.fromarray(np.ones((8, 10), dtype=np.uint8)).save(tmp_path / "mask.png")
    Image.fromarray(np.ones((8, 10), dtype=np.uint8)).save(tmp_path / "valid.png")
    write_csv(tmp_path / "data.csv", [{"image": "a.png", "mask": "mask.png", "valid": "valid.png"}])
    declaration = {
        "inputs": {"image": {"type": "image"}},
        "targets": {"mask": {"type": "indexed_mask", "num_classes": 2, "validity_column": "valid"}},
    }
    config = {
        "seed": 42,
        "data": {
            "_target_": "trainer_project.data.ManifestDataModule",
            "_recursive_": False,
            "data_root": str(tmp_path),
            "schema": declaration,
            "train": [{"name": "train", "path": "data.csv"}],
        },
    }
    original = data_identity(config)
    assert len(original["assets"]) == 3
    Image.fromarray(np.zeros((8, 10), dtype=np.uint8)).save(tmp_path / "valid.png")
    with pytest.raises(ValueError, match="validity content changed"):
        verify_identity(config, original)


def test_two_manifests_per_split_have_stable_eval_roles_and_permissions(tmp_path):
    make_image(tmp_path / "a.png")
    entries = {}
    for role in ("train", "val", "test"):
        entries[role] = []
        for index in range(2):
            name = f"{role}_{index}"
            write_csv(
                tmp_path / f"{name}.csv", [{"sample_id": name, "image": "a.png", "label": "NA", "aux": "x"}]
            )
            entries[role].append({"name": name, "path": f"{name}.csv"})
    module = ManifestDataModule(data_root=tmp_path, schema=schema(), **entries)
    module.setup()
    assert len(module.train_dataset) == 2
    assert [plan["role"] for plan in module.eval_plan] == ["val", "val", "test", "test"]
    assert len(module.val_dataloader()) == 4
    assert len(module.test_dataloader()) == 2
    assert not module.datasets["test"][0][0]["supervision_allowed"]["label"]
    first_state = module.state_dict()
    module.val_dataloader()
    assert len(module.state_dict()["samplers"]) == len(first_state["samplers"])


def test_fully_unlabeled_ssl_retains_empty_target_contract(tmp_path):
    make_image(tmp_path / "a.png")
    write_csv(tmp_path / "data.csv", [{"image": "a.png"}])
    module = ManifestDataModule(
        data_root=tmp_path,
        train=[{"name": "unlabeled", "path": "data.csv"}],
        schema={"inputs": {"image": {"type": "image"}}, "targets": {}},
        transforms={"ssl": True},
    )
    module.setup()
    sample = module.train_dataset[0]
    assert set(sample["views"]) == {"view1", "view2"}
    assert sample["targets"] == sample["target_valid"] == sample["supervision_allowed"] == {}


def test_indexed_palette_png_keeps_class_ids(tmp_path):
    make_image(tmp_path / "a.png")
    raw = np.zeros((8, 10), dtype=np.uint8)
    raw[:, 4:7] = 1
    raw[:, 7:] = 255
    mask = Image.fromarray(raw, mode="P")
    mask.putpalette([0, 0, 0, 255, 0, 0] + [0] * (768 - 6))
    mask.save(tmp_path / "mask.png")
    write_csv(tmp_path / "data.csv", [{"image": "a.png", "segmentation_target": "mask.png"}])
    dataset = CsvDataset(
        "data.csv",
        data_root=tmp_path,
        schema={
            "inputs": {"image": {"type": "image"}},
            "targets": {"segmentation_target": {"type": "segmentation", "num_classes": 2}},
        },
        transform=SampleTransform(size=(8, 10)),
    )
    sample = dataset[0]
    assert torch.equal(sample["targets"]["segmentation_target"], torch.from_numpy(raw.astype(np.int64)))
    assert sample["target_valid"]["segmentation_target"].sum() == 56
