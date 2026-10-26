import json

from omegaconf import OmegaConf
from test_lifecycle import recipe

from trainer_project.cli import _branch
from trainer_project.engine import train
from trainer_project.experiment.artifacts import load_checkpoint
from trainer_project.experiment.cross_validation import run_cross_validation


def test_branch_reinitializes_changed_head_records_parent(tmp_path):
    config = recipe(tmp_path)
    config["trainer"]["max_epochs"] = 1
    run = train(config)
    checkpoint = run / "checkpoints" / "last.ckpt"
    branch = _branch(
        checkpoint,
        [
            "model.num_classes=3",
            "data.class_maps.label.class_c=2",
            "data.schema.targets.label.classes=[class_a,class_b,class_c]",
        ],
    )
    _, payload = load_checkpoint(branch / "checkpoints" / "last.ckpt")
    parent = payload["config"]["parent"]
    assert parent["mode"] == "branch"
    assert set(parent["reinitialized_keys"]) == {"head.weight", "head.bias"}
    assert payload["successful_updates"] == 4


def test_prepared_cv_collects_checked_oof_predictions(tmp_path):
    config = recipe(tmp_path)
    config["trainer"]["max_epochs"] = 1
    config["data"]["test"] = []
    paths = []
    for fold in range(2):
        current = OmegaConf.create(config)
        current.data.train = [{"name": "reference", "path": "train.csv" if fold == 0 else "val.csv"}]
        current.data.val = [{"name": "query", "path": "val.csv" if fold == 0 else "train.csv"}]
        path = tmp_path / f"fold_{fold}.yaml"
        OmegaConf.save(current, path)
        paths.append(str(path))
    cv = tmp_path / "cv.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {"folds": paths, "output_dir": str(tmp_path / "cv_runs"), "require_image_independence": True}
        ),
        cv,
    )
    run = run_cross_validation(cv)
    summary = json.loads((run / "summary.json").read_text())
    assert summary["protocol"]["checked_rows"] == 13
    assert summary["protocol"]["protocol"] == "independent OOF"
    assert summary["oof"]["val/pooled/count"] == 13
    assert (run / "reports" / "report.html").exists()
