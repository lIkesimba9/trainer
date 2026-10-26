"""BYOL and a compact, genuine visible-token MAE implementation."""

import copy
import math

import torch
from torch import nn

from .classification import GlobalAveragePool, TinyBackbone, build_component, image_input


class BYOLModel(nn.Module):
    supported_profiles = ("image",)

    def __init__(
        self,
        encoder=None,
        feature_dim=32,
        projection_dim=32,
        hidden_dim=64,
        ema_decay=0.99,
        input_key="image",
        evaluation_encoder="online",
        projector=None,
        predictor=None,
    ):
        super().__init__()
        if evaluation_encoder not in ("online", "target"):
            raise ValueError("evaluation_encoder must be online or target")
        self.input_key = input_key
        self.evaluation_encoder = evaluation_encoder
        self.ema_decay = float(ema_decay)
        self.online_encoder = build_component(encoder) if encoder is not None else TinyBackbone(feature_dim)
        feature_dim = getattr(self.online_encoder, "num_features", feature_dim)
        self.feature_dim = self.embedding_dim = int(feature_dim)
        self.pool = GlobalAveragePool()
        # LayerNorm avoids batch-size-dependent behavior and works on a one-example rank.
        self.online_projector = (
            build_component(projector)
            if projector is not None
            else nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, projection_dim),
            )
        )
        self.predictor = (
            build_component(predictor)
            if predictor is not None
            else nn.Sequential(
                nn.Linear(projection_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, projection_dim),
            )
        )
        self.target_encoder = copy.deepcopy(self.online_encoder).requires_grad_(False)
        self.target_projector = copy.deepcopy(self.online_projector).requires_grad_(False)
        self.register_buffer("ema_updates", torch.tensor(0, dtype=torch.long))
        self.target_encoder.eval()
        self.target_projector.eval()

    def train(self, mode=True):
        super().train(mode)
        # Teacher normalization buffers are updated only through the EMA policy.
        self.target_encoder.eval()
        self.target_projector.eval()
        return self

    @torch.no_grad()
    def update_target(self, successful_updates, total_updates):
        if int(successful_updates) <= int(self.ema_updates):
            return
        progress = min(float(successful_updates) / max(1, int(total_updates)), 1.0)
        decay = 1.0 - (1.0 - self.ema_decay) * (math.cos(math.pi * progress) + 1.0) / 2.0
        for online, target in (
            (self.online_encoder, self.target_encoder),
            (self.online_projector, self.target_projector),
        ):
            for target_parameter, online_parameter in zip(
                target.parameters(), online.parameters(), strict=True
            ):
                target_parameter.lerp_(online_parameter, 1.0 - decay)
            for target_buffer, online_buffer in zip(target.buffers(), online.buffers(), strict=True):
                if target_buffer.is_floating_point():
                    target_buffer.lerp_(online_buffer, 1.0 - decay)
                else:
                    target_buffer.copy_(online_buffer)
        self.ema_updates.fill_(int(successful_updates))

    def extract_features(self, inputs):
        encoder = self.online_encoder if self.evaluation_encoder == "online" else self.target_encoder
        return self.pool(encoder(image_input(inputs, self.input_key)))

    def forward(self, inputs, views=None):
        if not views:
            features = self.extract_features(inputs)
            return {"features": features, "embeddings": features}
        if not all(key in views for key in ("view1", "view2")):
            raise ValueError("BYOL requires independent view1 and view2 tensors")
        first, second = views["view1"], views["view2"]
        feature1 = self.pool(self.online_encoder(first))
        feature2 = self.pool(self.online_encoder(second))
        outputs = {"features": feature1, "embeddings": feature1}
        z1 = self.online_projector(feature1)
        z2 = self.online_projector(feature2)
        with torch.no_grad():
            t1 = self.target_projector(self.pool(self.target_encoder(first)))
            t2 = self.target_projector(self.pool(self.target_encoder(second)))
        outputs.update(
            {
                "projections": z1,
                "prediction1": self.predictor(z1),
                "prediction2": self.predictor(z2),
                "target1": t1.detach(),
                "target2": t2.detach(),
            }
        )
        return outputs


class MAEModel(nn.Module):
    """ViT MAE: encoder sees visible patches only; decoder restores masked positions.

    Training uses a checkpointed CPU generator. Evaluation masks use a fresh fixed
    generator and never consume training RNG. extract_features uses every patch.
    """

    supported_profiles = ("image",)

    def __init__(
        self,
        image_size=32,
        patch_size=4,
        in_channels=3,
        embedding_dim=32,
        decoder_dim=32,
        encoder_depth=2,
        decoder_depth=1,
        num_heads=4,
        mask_ratio=0.75,
        normalize_targets=False,
        seed=42,
        eval_seed=123,
        input_key="image",
        encoder=None,
        decoder=None,
        patch_embedding=None,
    ):
        super().__init__()
        if image_size % patch_size or not 0 < mask_ratio < 1:
            raise ValueError("MAE needs image_size divisible by patch_size and 0 < mask_ratio < 1")
        if embedding_dim % num_heads or decoder_dim % num_heads:
            raise ValueError("MAE embedding and decoder dimensions must divide by num_heads")
        self.image_size, self.patch_size = int(image_size), int(patch_size)
        self.input_key = input_key
        self.in_channels = int(in_channels)
        self.embedding_dim = self.feature_dim = int(embedding_dim)
        self.mask_ratio = float(mask_ratio)
        self.normalize_targets = bool(normalize_targets)
        self.eval_seed = int(eval_seed)
        self.patch_dim = in_channels * patch_size * patch_size
        self.num_patches = (image_size // patch_size) ** 2
        if self.num_patches < 2:
            raise ValueError("MAE needs at least two patches")
        self.patch_embedding = (
            build_component(patch_embedding)
            if patch_embedding is not None
            else nn.Conv2d(in_channels, embedding_dim, patch_size, stride=patch_size)
        )
        self.encoder_position = nn.Parameter(torch.empty(1, self.num_patches, embedding_dim))
        self.encoder = (
            build_component(encoder)
            if encoder is not None
            else nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    embedding_dim,
                    num_heads,
                    embedding_dim * 4,
                    dropout=0.0,
                    batch_first=True,
                    norm_first=True,
                ),
                encoder_depth,
                enable_nested_tensor=False,
            )
        )
        self.encoder_norm = nn.LayerNorm(embedding_dim)
        self.decoder_projection = nn.Linear(embedding_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_position = nn.Parameter(torch.empty(1, self.num_patches, decoder_dim))
        self.decoder = (
            build_component(decoder)
            if decoder is not None
            else nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    decoder_dim, num_heads, decoder_dim * 4, dropout=0.0, batch_first=True, norm_first=True
                ),
                decoder_depth,
                enable_nested_tensor=False,
            )
        )
        self.reconstruction_head = nn.Linear(decoder_dim, self.patch_dim)
        nn.init.normal_(self.encoder_position, std=0.02)
        nn.init.normal_(self.decoder_position, std=0.02)
        self.mask_generator = torch.Generator(device="cpu").manual_seed(seed)

    def get_extra_state(self):
        return {"mask_generator": self.mask_generator.get_state()}

    def set_extra_state(self, state):
        self.mask_generator.set_state(state["mask_generator"].cpu())

    def patchify(self, image):
        batch, channels, height, width = image.shape
        if (height, width) != (self.image_size, self.image_size):
            raise ValueError(f"MAE expects {self.image_size}×{self.image_size}, got {height}×{width}")
        patch = self.patch_size
        return (
            image.reshape(batch, channels, height // patch, patch, width // patch, patch)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(batch, self.num_patches, self.patch_dim)
        )

    def extract_features(self, inputs):
        image = image_input(inputs, self.input_key)
        self.patchify(image)
        tokens = self.patch_embedding(image).flatten(2).transpose(1, 2) + self.encoder_position
        return self.encoder_norm(self.encoder(tokens)).mean(1)

    def forward(self, inputs, views=None):
        image = image_input(inputs, self.input_key)
        patches = self.patchify(image)
        tokens = self.patch_embedding(image).flatten(2).transpose(1, 2) + self.encoder_position
        batch, length, _ = tokens.shape
        keep = max(1, min(length - 1, int(length * (1.0 - self.mask_ratio))))
        generator = self.mask_generator if self.training else torch.Generator().manual_seed(self.eval_seed)
        shuffled = torch.rand(batch, length, generator=generator).argsort(1).to(image.device)
        restore = shuffled.argsort(1)
        visible = tokens.gather(1, shuffled[:, :keep, None].expand(-1, -1, tokens.shape[-1]))
        encoded = self.encoder_norm(self.encoder(visible))
        decoded = self.decoder_projection(encoded)
        decoded = torch.cat((decoded, self.mask_token.expand(batch, length - keep, -1)), dim=1)
        decoded = decoded.gather(1, restore[:, :, None].expand(-1, -1, decoded.shape[-1]))
        prediction = self.reconstruction_head(self.decoder(decoded + self.decoder_position))
        mask = torch.ones(batch, length, dtype=torch.bool, device=image.device)
        mask[:, :keep] = False
        mask = mask.gather(1, restore)
        if self.normalize_targets:
            patches = (patches - patches.mean(-1, keepdim=True)) / (
                patches.var(-1, keepdim=True, unbiased=False) + 1e-6
            ).sqrt()
        # Features for evaluators are extracted through extract_features, never this masked pool.
        return {
            "features": encoded.mean(1),
            "embeddings": encoded.mean(1),
            "reconstruction": prediction,
            "reconstruction_target": patches.detach(),
            "ssl_patch_mask": mask,
        }
