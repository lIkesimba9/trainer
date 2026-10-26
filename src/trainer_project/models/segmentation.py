"""Dense prediction adapters. All logits are in input image coordinates."""

from torch import nn
from torch.nn import functional as F

from .classification import image_input


class TinySegmenter(nn.Module):
    supported_profiles = ("image",)

    def __init__(self, num_classes=2, in_channels=3, width=16, input_key="image"):
        super().__init__()
        self.input_key = input_key
        self.num_classes = int(num_classes)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
        )
        self.decoder = nn.Conv2d(width, num_classes, 1)

    def extract_features(self, inputs):
        return self.encoder(image_input(inputs, self.input_key)).mean((-2, -1))

    def forward(self, inputs, views=None):
        features = self.encoder(image_input(inputs, self.input_key))
        return {"features": features.mean((-2, -1)), "dense_logits": self.decoder(features)}


class SegFormerAdapter(nn.Module):
    supported_profiles = ("image",)

    def __init__(
        self, num_classes=2, pretrained=False, model_name=None, revision=None, config=None, input_key="image"
    ):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        self.input_key = input_key
        self.num_classes = int(num_classes)
        if pretrained:
            if not revision or len(revision) != 40:
                raise ValueError("SegFormer pretrained weights require a full commit revision")
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name, revision=revision, num_labels=num_classes, ignore_mismatched_sizes=True
            )
        else:
            self.model = SegformerForSemanticSegmentation(
                SegformerConfig(**({**dict(config or {}), "num_labels": num_classes}))
            )
        self.weights_provenance = {
            "provider": "huggingface",
            "model_name": model_name,
            "revision": revision,
            "config": self.model.config.to_dict(),
        }

    def extract_features(self, inputs):
        result = self.model.segformer(pixel_values=image_input(inputs, self.input_key))
        features = result.last_hidden_state
        return features.mean((-2, -1)) if features.ndim == 4 else features.mean(1)

    def forward(self, inputs, views=None):
        image = image_input(inputs, self.input_key)
        logits = self.model(pixel_values=image).logits
        return {
            "dense_logits": F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        }
