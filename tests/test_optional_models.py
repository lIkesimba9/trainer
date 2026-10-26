"""Optional vision integrations are exercised offline with tiny random models."""

import importlib.util

import pytest
import torch

from trainer_project.engine import restore_model, train
from trainer_project.experiment.config import load_config
from trainer_project.experiment.runtime import WORKSPACE


def require(name):
    if importlib.util.find_spec(name) is None:
        pytest.skip(f"Optional dependency {name} is not installed")


def vit_configuration():
    return {
        "model_type": "vit",
        "image_size": 32,
        "patch_size": 8,
        "num_channels": 3,
        "hidden_size": 24,
        "num_hidden_layers": 1,
        "num_attention_heads": 3,
        "intermediate_size": 48,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
    }


def segformer_configuration():
    return {
        "num_encoder_blocks": 4,
        "depths": [1, 1, 1, 1],
        "hidden_sizes": [8, 16, 24, 32],
        "num_attention_heads": [1, 2, 3, 4],
        "sr_ratios": [4, 2, 1, 1],
        "patch_sizes": [7, 3, 3, 3],
        "strides": [4, 2, 2, 2],
        "mlp_ratios": [2, 2, 2, 2],
        "decoder_hidden_size": 16,
        "drop_path_rate": 0.0,
        "classifier_dropout_prob": 0.0,
    }


def forbidden_download(*args, **kwargs):
    raise AssertionError("Checkpoint reconstruction attempted to download source pretrained weights")


def test_timm_backbone_offline_forward_uses_explicit_input_preprocessing(monkeypatch):
    require("timm")
    import timm

    from trainer_project.models import TimmBackbone

    original = timm.create_model
    calls = []

    def construct(*args, **kwargs):
        calls.append(kwargs.get("pretrained"))
        assert kwargs.get("pretrained") is False
        return original(*args, **kwargs)

    monkeypatch.setattr(timm, "create_model", construct)
    model = TimmBackbone("resnet10t", pretrained=False).eval()
    # Model adapters receive already preprocessed tensors; they do not resize or
    # apply pretrained_cfg normalization behind the configured transform pipeline.
    image = torch.rand((2, 3, 32, 32))
    with torch.no_grad():
        output = model(image)
        direct = model.model(image)
    assert output.shape == (2, model.num_features)
    torch.testing.assert_close(output, direct, rtol=0, atol=0)
    assert calls == [False]


def test_torchvision_resnet18_offline_forward(monkeypatch):
    require("torchvision")
    from torchvision.models import _api

    from trainer_project.models import TorchvisionBackbone

    monkeypatch.setattr(_api.WeightsEnum, "get_state_dict", forbidden_download)
    model = TorchvisionBackbone("resnet18", pretrained=False).eval()
    with torch.no_grad():
        output = model(torch.rand((2, 3, 32, 32)))
    assert output.shape == (2, 512)


def test_huggingface_tiny_vit_serialized_config_forward(monkeypatch):
    require("transformers")
    from transformers import AutoModel

    from trainer_project.models import HuggingFaceBackbone

    monkeypatch.setattr(AutoModel, "from_pretrained", forbidden_download)
    model = HuggingFaceBackbone(pretrained=False, config=vit_configuration()).eval()
    with torch.no_grad():
        output = model(torch.rand((2, 3, 32, 32)))
    assert output.shape == (2, 24)
    assert model.weights_provenance["config"]["model_type"] == "vit"


def test_segformer_tiny_serialized_config_forward(monkeypatch):
    require("transformers")
    from transformers import SegformerForSemanticSegmentation

    from trainer_project.models import SegFormerAdapter

    monkeypatch.setattr(SegformerForSemanticSegmentation, "from_pretrained", forbidden_download)
    model = SegFormerAdapter(num_classes=2, pretrained=False, config=segformer_configuration()).eval()
    with torch.no_grad():
        result = model({"image": torch.rand((2, 3, 32, 32))})
    assert result["dense_logits"].shape == (2, 2, 32, 32)


@pytest.mark.parametrize("provider", ["vit", "segformer"])
def test_full_hf_checkpoint_restore_never_loads_source_pretrained(tmp_path, monkeypatch, provider):
    require("transformers")
    from transformers import AutoConfig, AutoModel, SegformerConfig, SegformerForSemanticSegmentation

    config = load_config(WORKSPACE / "examples" / "classification.yaml")
    config["output_dir"] = str(tmp_path / "runs")
    config["trainer"].update(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    config["logger"] = None
    initialized = []
    if provider == "vit":

        def local_pretrained(*args, **kwargs):
            initialized.append("vit")
            architecture = vit_configuration()
            return AutoModel.from_config(AutoConfig.for_model(architecture.pop("model_type"), **architecture))

        monkeypatch.setattr(AutoModel, "from_pretrained", local_pretrained)
        config["model"]["backbone"] = {
            "_target_": "trainer_project.models.HuggingFaceBackbone",
            "model_name": "offline-test-vit",
            "revision": "1" * 40,
            "pretrained": True,
        }
    else:

        def local_pretrained(*args, **kwargs):
            initialized.append("segformer")
            return SegformerForSemanticSegmentation(
                SegformerConfig(**segformer_configuration(), num_labels=2)
            )

        monkeypatch.setattr(SegformerForSemanticSegmentation, "from_pretrained", local_pretrained)
        config["task"] = "segmentation"
        config["model"] = {
            "_target_": "trainer_project.models.SegFormerAdapter",
            "num_classes": 2,
            "model_name": "offline-test-segformer",
            "revision": "1" * 40,
            "pretrained": True,
        }
        config["objective"] = {
            "_target_": "trainer_project.objectives.SegmentationObjective",
            "target_key": "segmentation_target",
        }
        config["data"]["schema"]["targets"] = {
            "segmentation_target": {
                "column": "segmentation_mask",
                "type": "segmentation",
                "classes": ["background", "foreground"],
                "num_classes": 2,
            }
        }
        config["evaluation"]["target_key"] = "segmentation_target"
        config["checkpoint"].update(monitor="val/pooled/dice_macro", mode="max")
    run = train(config)
    assert initialized == [provider]
    monkeypatch.setattr(AutoModel, "from_pretrained", forbidden_download)
    monkeypatch.setattr(SegformerForSemanticSegmentation, "from_pretrained", forbidden_download)
    module, data, payload = restore_model(run / "checkpoints" / "last.ckpt")
    batch = next(iter(data.val_dataloader()[0]))
    with torch.no_grad():
        result = module(batch["inputs"])
    assert payload["config"]["model"].get("backbone", payload["config"]["model"])["pretrained"] is True
    if provider == "vit":
        assert result["logits"].shape == (4, 2)
        assert "config" in payload["config"]["model"]["backbone"]
    else:
        assert result["dense_logits"].shape == (4, 2, 32, 32)
        assert "config" in payload["config"]["model"]
