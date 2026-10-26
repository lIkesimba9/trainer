"""Small offline models and explicit adapters for external feature extractors."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def build_component(value: Any) -> Any:
    if isinstance(value, Mapping) or value.__class__.__name__ == "DictConfig":
        from hydra.utils import instantiate

        return instantiate(value)
    return value


def image_input(inputs, key="image"):
    if isinstance(inputs, torch.Tensor):
        return inputs
    if key not in inputs:
        raise ValueError(f"Required model input {key!r} is missing")
    return inputs[key]


class TinyBackbone(nn.Module):
    """Offline CNN without batch statistics; supports deterministic DDP checks."""

    def __init__(self, feature_dim=32, in_channels=3):
        super().__init__()
        self.feature_dim = self.num_features = int(feature_dim)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, self.feature_dim, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )

    def forward(self, image):
        return self.layers(image)


class GlobalAveragePool(nn.Module):
    def forward(self, features):
        if features.ndim == 4:
            return features.mean((-2, -1))
        if features.ndim == 3:
            return features.mean(1)
        if features.ndim != 2:
            raise ValueError(f"Unsupported feature shape {tuple(features.shape)}")
        return features


class LinearHead(nn.Linear):
    def __init__(self, in_features, num_classes, bias=True):
        super().__init__(int(in_features), int(num_classes), bias=bias)
        self.num_classes = int(num_classes)


class ArcFaceHead(nn.Module):
    """Label-independent cosine logits; target margins belong to the objective."""

    def __init__(self, in_features, num_classes, subcenters=1, scale=32.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.subcenters = int(subcenters)
        self.scale = float(scale)
        self.weight = nn.Parameter(torch.empty(self.num_classes, self.subcenters, int(in_features)))
        nn.init.xavier_uniform_(self.weight.reshape(-1, int(in_features)))

    def forward(self, embeddings):
        similarity = torch.einsum(
            "bd,ckd->bck", F.normalize(embeddings, dim=-1), F.normalize(self.weight, dim=-1)
        )
        return similarity.amax(-1) * self.scale


class ImageClassifier(nn.Module):
    supported_profiles = ("image",)

    def __init__(
        self,
        backbone=None,
        num_classes=2,
        embedding_dim=32,
        input_key="image",
        pooling=None,
        neck=None,
        head=None,
        feature_dim=None,
        normalize=False,
        dropout=0.0,
        freeze_backbone=False,
    ):
        super().__init__()
        self.input_key = input_key
        self.backbone = build_component(backbone) if backbone is not None else TinyBackbone(embedding_dim)
        source_dim = (
            feature_dim
            or getattr(self.backbone, "num_features", None)
            or getattr(self.backbone, "feature_dim", None)
        )
        if source_dim is None:
            raise ValueError("Backbone must declare num_features/feature_dim or config must set feature_dim")
        self.pooling = build_component(pooling) if pooling is not None else GlobalAveragePool()
        self.neck = (
            build_component(neck)
            if neck is not None
            else (nn.Identity() if source_dim == embedding_dim else nn.Linear(source_dim, embedding_dim))
        )
        self.dropout = nn.Dropout(float(dropout))
        self.normalize = bool(normalize)
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)
        self.head = build_component(head) if head is not None else LinearHead(embedding_dim, num_classes)
        if getattr(self.head, "num_classes", num_classes) != num_classes:
            raise ValueError("Head class count differs from model.num_classes")
        if freeze_backbone:
            self.backbone.requires_grad_(False)

    def extract_features(self, inputs):
        features = self.pooling(self.backbone(image_input(inputs, self.input_key)))
        embedding = self.neck(features)
        return F.normalize(embedding, dim=-1) if self.normalize else embedding

    def forward(self, inputs, views=None):
        features = self.pooling(self.backbone(image_input(inputs, self.input_key)))
        embeddings = self.neck(features)
        if self.normalize:
            embeddings = F.normalize(embeddings, dim=-1)
        return {"features": features, "embeddings": embeddings, "logits": self.head(self.dropout(embeddings))}


class TimmBackbone(nn.Module):
    def __init__(self, model_name, pretrained=False, in_channels=3, **kwargs):
        super().__init__()
        import timm

        if pretrained and "." not in model_name:
            raise ValueError("Pretrained timm model_name must include an explicit weights variant")
        self.model = timm.create_model(
            model_name, pretrained=pretrained, in_chans=in_channels, num_classes=0, **kwargs
        )
        self.num_features = self.feature_dim = self.model.num_features
        self.weights_provenance = {
            "provider": "timm",
            "model_name": model_name,
            "pretrained": bool(pretrained),
            "config": self.model.pretrained_cfg,
        }

    def forward(self, image):
        return self.model(image)


class TorchvisionBackbone(nn.Module):
    def __init__(self, model_name="resnet18", weights=None, pretrained=False):
        super().__init__()
        from torchvision.models import get_model

        if pretrained and weights is None:
            raise ValueError("Torchvision pretrained initialization requires an exact weights enum")
        self.model = get_model(model_name, weights=weights if pretrained else None)
        if hasattr(self.model, "fc"):
            self.num_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif hasattr(self.model, "heads") and hasattr(self.model, "hidden_dim"):
            self.num_features = self.model.hidden_dim
            self.model.heads = nn.Identity()
        elif hasattr(self.model, "head") and isinstance(self.model.head, nn.Linear):
            self.num_features = self.model.head.in_features
            self.model.head = nn.Identity()
        else:
            raise ValueError("This torchvision architecture needs an explicit feature adapter")
        self.feature_dim = self.num_features
        self.weights_provenance = {
            "provider": "torchvision",
            "model_name": model_name,
            "weights": str(weights),
        }

    def forward(self, image):
        return self.model(image)


class HuggingFaceBackbone(nn.Module):
    def __init__(self, model_name=None, revision=None, pretrained=False, config=None):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        if pretrained:
            if revision is None or len(revision) != 40:
                raise ValueError("Pretrained Hugging Face weights require a full commit revision")
            self.model = AutoModel.from_pretrained(model_name, revision=revision)
        else:
            if config is None:
                raise ValueError("Offline HF reconstruction requires serialized config")
            config = dict(config)
            self.model = AutoModel.from_config(AutoConfig.for_model(config.pop("model_type"), **config))
        self.num_features = self.feature_dim = self.model.config.hidden_size
        self.weights_provenance = {
            "provider": "huggingface",
            "model_name": model_name,
            "revision": revision,
            "config": self.model.config.to_dict(),
        }

    def forward(self, image):
        result = self.model(pixel_values=image)
        pooled = getattr(result, "pooler_output", None)
        return pooled if pooled is not None else result.last_hidden_state.mean(1)
