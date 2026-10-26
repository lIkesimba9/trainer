"""Explicit branch/fusion recipes for the four required input profiles."""

from collections import OrderedDict

import torch
from torch import nn

from .classification import GlobalAveragePool, LinearHead, TinyBackbone, build_component, image_input


class TokenTextEncoder(nn.Module):
    """Masked token mean; expects tokenizer output input_ids/attention_mask."""

    def __init__(self, vocab_size=32768, embedding_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.feature_dim = self.num_features = int(embedding_dim)

    def forward(self, text):
        if not isinstance(text, dict) or "input_ids" not in text or "attention_mask" not in text:
            raise ValueError("Text branch requires tokenized input_ids and attention_mask")
        embeddings = self.embedding(text["input_ids"])
        mask = text["attention_mask"].to(embeddings.dtype).unsqueeze(-1)
        return (embeddings * mask).sum(1) / mask.sum(1).clamp_min(1)


class MaskChannelAdapter(nn.Module):
    """Explicit image+mask concatenation; backbone must accept four input channels."""

    def __init__(self, encoder=None, image_key="image", mask_key="input_mask", feature_dim=32):
        super().__init__()
        self.encoder = (
            build_component(encoder) if encoder is not None else TinyBackbone(feature_dim, in_channels=4)
        )
        self.image_key, self.mask_key = image_key, mask_key
        self.feature_dim = self.num_features = getattr(self.encoder, "num_features", feature_dim)

    def forward(self, inputs):
        image = image_input(inputs, self.image_key)
        mask = image_input(inputs, self.mask_key)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[1] != 1 or mask.shape[-2:] != image.shape[-2:]:
            raise ValueError("input_mask must have one channel in image coordinates")
        return self.encoder(torch.cat((image, mask.to(image.dtype)), dim=1))


class MultiModalModel(nn.Module):
    """Concatenate named branch embeddings; missing modalities require custom fusion.

    Profiles: image, image_image, image_image_text, image_mask. Branches and
    projections can be replaced independently through Hydra. Separate image
    branches make no geometric-registration assumption.
    """

    supported_profiles = ("image", "image_image", "image_image_text", "image_mask")

    def __init__(
        self,
        profile="image_image",
        branches=None,
        feature_dim=32,
        embedding_dim=32,
        num_classes=2,
        fusion=None,
        head=None,
    ):
        super().__init__()
        if profile not in self.supported_profiles:
            raise ValueError(f"Unsupported multimodal profile {profile}")
        self.profile = profile
        if branches is None:
            if profile == "image_mask":
                branches = {"image_mask": MaskChannelAdapter(feature_dim=feature_dim)}
            else:
                keys = ["image"] if profile == "image" else ["image_a", "image_b"]
                branches = OrderedDict((key, TinyBackbone(feature_dim)) for key in keys)
                if profile == "image_image_text":
                    branches["text"] = TokenTextEncoder(embedding_dim=feature_dim)
        self.branches = nn.ModuleDict({key: build_component(value) for key, value in branches.items()})
        self.pool = GlobalAveragePool()
        dimension = sum(
            getattr(branch, "num_features", getattr(branch, "feature_dim", feature_dim))
            for branch in self.branches.values()
        )
        self.fusion = build_component(fusion) if fusion is not None else nn.Linear(dimension, embedding_dim)
        self.head = build_component(head) if head is not None else LinearHead(embedding_dim, num_classes)
        self.embedding_dim, self.num_classes = int(embedding_dim), int(num_classes)

    def _branches(self, inputs):
        result = {}
        for key, branch in self.branches.items():
            value = inputs if key == "image_mask" else image_input(inputs, key)
            result[key] = self.pool(branch(value))
        return result

    def extract_features(self, inputs):
        return self.fusion(torch.cat(list(self._branches(inputs).values()), dim=-1))

    def forward(self, inputs, views=None):
        branches = self._branches(inputs)
        embeddings = self.fusion(torch.cat(list(branches.values()), dim=-1))
        return {
            "features": embeddings,
            "embeddings": embeddings,
            "modality_embeddings": branches,
            "logits": self.head(embeddings),
        }
