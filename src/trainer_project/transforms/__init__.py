"""Deterministic sample geometry and optional Albumentations/Kornia adapters."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .mixing import CutMix, MixUp

__all__ = [
    "SampleTransform",
    "ByteTokenizer",
    "HFTokenizer",
    "AlbumentationsTransform",
    "KorniaBatchTransform",
    "CutMix",
    "MixUp",
]


class SampleTransform:
    """Resize, synchronous mask geometry and once-only image normalization.

    Geometry for independent image modalities is drawn independently. Masks declare
    their image parent through sample['_mask_parents']; segmentation belongs to
    the primary image. Input images are float CHW in [0, 1].
    """

    def __init__(
        self,
        image_size=32,
        size=None,
        mean=None,
        std=None,
        hflip=0.0,
        vflip=0.0,
        flip_probability=None,
        horizontal_flip=None,
        vertical_flip=None,
        **kwargs,
    ):
        size = size or image_size
        self.size = (int(size), int(size)) if isinstance(size, int) else tuple(map(int, size))
        if len(self.size) != 2 or min(self.size) <= 0:
            raise ValueError("image_size must be positive H,W")
        self.mean = mean
        self.std = std
        self.hflip = float(hflip if flip_probability is None else flip_probability)
        self.hflip = self.hflip if horizontal_flip is None else float(horizontal_flip)
        self.vflip = float(vflip if vertical_flip is None else vertical_flip)
        if kwargs:
            raise ValueError(f"Unsupported SampleTransform options: {sorted(kwargs)}")
        if not 0 <= self.hflip <= 1 or not 0 <= self.vflip <= 1:
            raise ValueError("Flip probabilities must be in [0,1]")
        if mean is not None and (std is None or len(mean) != len(std) or any(float(x) <= 0 for x in std)):
            raise ValueError("Normalization requires matched mean/std and positive std")

    @staticmethod
    def _geometry(tensor, size, horizontal, vertical, is_mask):
        dtype, ndim = tensor.dtype, tensor.ndim
        value = tensor.unsqueeze(0).unsqueeze(0) if ndim == 2 else tensor.unsqueeze(0)
        options = {} if is_mask else {"align_corners": False}
        value = F.interpolate(value.float(), size=size, mode="nearest" if is_mask else "bilinear", **options)[
            0
        ]
        if ndim == 2:
            value = value[0]
        if horizontal:
            value = value.flip(-1)
        if vertical:
            value = value.flip(-2)
        return value.to(dtype)

    def __call__(self, sample, generator=None, seed=None):
        generator = generator or torch.Generator().manual_seed(int(seed or 0))
        parents = sample.get("_mask_parents", {})
        geometry = {}
        for key, value in sample["inputs"].items():
            if not isinstance(value, torch.Tensor) or value.ndim != 3 or key in parents:
                continue
            geometry[key] = (
                bool(torch.rand((), generator=generator) < self.hflip),
                bool(torch.rand((), generator=generator) < self.vflip),
            )
            sample["inputs"][key] = self._geometry(value, self.size, *geometry[key], False)
            if self.mean is not None:
                mean = torch.tensor(self.mean, dtype=value.dtype)[:, None, None]
                std = torch.tensor(self.std, dtype=value.dtype)[:, None, None]
                if len(mean) != value.shape[0]:
                    raise ValueError("Normalization channels do not match image")
                sample["inputs"][key] = (sample["inputs"][key] - mean) / std
        primary = next(iter(geometry), None)
        for key, parent in parents.items():
            if key in sample["inputs"]:
                sample["inputs"][key] = self._geometry(
                    sample["inputs"][key], self.size, *geometry[parent], True
                )
        for key, target in sample["targets"].items():
            if isinstance(target, torch.Tensor) and target.ndim >= 2:
                sample["targets"][key] = self._geometry(target, self.size, *geometry[primary], True)
                valid = sample["target_valid"][key]
                if valid.ndim >= 2:
                    sample["target_valid"][key] = self._geometry(valid, self.size, *geometry[primary], True)
        return sample


class ByteTokenizer:
    """Offline deterministic UTF-8 tokenizer for the supplied small text encoder.

    This is an explicit baseline tokenizer, not an implicit substitute for a
    pretrained encoder's vocabulary. 0=padding, 1..256=byte+1.
    """

    def __init__(self, max_length=64):
        self.max_length = int(max_length)
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")

    def __call__(self, text):
        ids = [byte + 1 for byte in str(text).encode("utf-8")[: self.max_length]]
        mask = [1] * len(ids)
        ids += [0] * (self.max_length - len(ids))
        mask += [0] * (self.max_length - len(mask))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


class HFTokenizer:
    def __init__(self, model_name, revision, max_length=128, local_files_only=True):
        if not revision:
            raise ValueError("Tokenizer revision must be pinned")
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, local_files_only=local_files_only
        )
        self.max_length = int(max_length)

    def __call__(self, text):
        result = self.tokenizer(
            text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return {key: value[0] for key, value in result.items()}


class AlbumentationsTransform:
    """User pipeline with synchronized primary image, masks and validity."""

    def __init__(self, pipeline, image_key="image"):
        from hydra.utils import instantiate

        self.pipeline = instantiate(pipeline) if isinstance(pipeline, dict) else pipeline
        self.image_key = image_key
        if not hasattr(self.pipeline, "set_random_seed"):
            raise ValueError("Albumentations pipeline must expose set_random_seed")

    def __call__(self, sample, generator=None, seed=None):
        if seed is None:
            seed = int(torch.randint(2**31 - 1, (), generator=generator))
        self.pipeline.set_random_seed(int(seed))
        image = sample["inputs"][self.image_key]
        slots, masks = [], []
        for key, parent in sample.get("_mask_parents", {}).items():
            if parent == self.image_key:
                slots.append(("inputs", key))
                masks.append(sample["inputs"][key].squeeze(0).numpy())
        for key, value in sample["targets"].items():
            if value.ndim == 2:
                slots.extend([("targets", key), ("target_valid", key)])
                masks.extend([value.numpy(), sample["target_valid"][key].numpy().astype("uint8")])
        result = self.pipeline(image=image.permute(1, 2, 0).numpy(), masks=masks)
        sample["inputs"][self.image_key] = torch.as_tensor(result["image"]).permute(2, 0, 1).float()
        for (section, key), value in zip(slots, result["masks"]):
            original = sample[section][key]
            tensor = torch.as_tensor(value).to(original.dtype)
            sample[section][key] = tensor.unsqueeze(0) if original.ndim == 3 else tensor
        return sample


class KorniaBatchTransform(torch.nn.Module):
    """Kornia AugmentationSequential receives image and masks in one call."""

    def __init__(self, operations, image_key="image", mask_input_keys=None, target_keys=None):
        super().__init__()
        import kornia.augmentation as K
        from hydra.utils import instantiate

        self.image_key = image_key
        self.mask_input_keys = list(mask_input_keys or [])
        self.target_keys = list(target_keys or [])
        keys = ["input"] + ["mask"] * (len(self.mask_input_keys) + 2 * len(self.target_keys))
        self.pipeline = K.AugmentationSequential(*(instantiate(op) for op in operations), data_keys=keys)

    def forward(self, batch):
        slots = [("inputs", self.image_key)] + [("inputs", key) for key in self.mask_input_keys]
        for key in self.target_keys:
            slots.extend([("targets", key), ("target_valid", key)])
        values, originals = [], []
        for section, key in slots:
            value = batch[section][key]
            originals.append(value)
            values.append(value.unsqueeze(1).float() if value.ndim == 3 else value.float())
        transformed = self.pipeline(*values)
        if len(slots) == 1:
            transformed = [transformed]
        for (section, key), value, original in zip(slots, transformed, originals):
            batch[section][key] = (value[:, 0] if original.ndim == 3 else value).to(original.dtype)
        return batch
