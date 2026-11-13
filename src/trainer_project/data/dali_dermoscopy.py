"""Optional GPU dermoscopy input backend for explicitly separate DALI trials.

The CPU reads encoded bytes and samples scalar augmentation parameters. DALI's
mixed decoder applies EXIF orientation, then GPU slicing applies the verified BB.
All image augmentation executes on CUDA (DALI operators and Torch GPU callbacks).
This is an adaptation of the SIIM-ISIC winner recipe, not pixel-equivalent to
Albumentations: see ``recipe_metadata`` for the complete list of differences.

No DALI import or CUDA initialization occurs when importing this module. A loader
keeps the caller's effective batch and sampler order; ``microbatch`` only limits
the number of native-resolution images simultaneously held by its pipeline.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

RECIPE_VERSION = "dermoscopy_dali_gpu_v1"


def recipe_metadata(recipe: str, max_long_edge: int = 1024) -> dict[str, Any]:
    if recipe not in {"gpu_native", "gpu_bounded"}:
        raise ValueError("recipe must be gpu_native or gpu_bounded")
    if int(max_long_edge) < 64:
        raise ValueError("max_long_edge must be at least 64")
    return {
        "version": RECIPE_VERSION, "recipe": recipe,
        "max_long_edge": int(max_long_edge) if recipe == "gpu_bounded" else None,
        "cpu_image_augmentation": False,
        "decoder": "DALI mixed RGB with adjust_orientation=True; GPU BB slice",
        "normalization": "caller mean/std, RGB, float32 CHW, uint8 scale 255",
        "differences_from_legacy": [
            "GPU RNG streams differ from NumPy/Albumentations; seeds are independent of microbatch size.",
            "GPU bilinear interpolation is not OpenCV's quantized interpolation.",
            "CLAHE uses DALI RGB luminance Y, fixed clip_limit=2.5, rather than Lab L and U(1,4).",
            "Elastic uses separable float32 GPU Gaussian sigma=50, radius=200, symmetric borders.",
            "Elastic's historical affine point-axis convention is replaced by an XY affine perturbation.",
            "Optical uses a centered radial camera model; grid uses five piecewise-linear intervals.",
            "HSV uses floating point RGB/HSV; motion blur is a diagonal line, median uses reflect101 borders.",
            "Blackbody gains interpolate published Albumentations coefficients; rounding may differ.",
            "gpu_bounded alone downsizes the BB before all heavy augmentations without upsampling.",
        ],
    }


def sample_parameters(seed: int, epoch: int, ordinal: int) -> dict[str, Any]:
    """Scalar-only CPU RNG: stable across pipeline microbatch boundaries."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch), int(ordinal)]))
    return {
        "transpose": bool(rng.random() < .5), "vertical": bool(rng.random() < .5),
        "horizontal": bool(rng.random() < .5),
        "brightness": float(rng.uniform(-.2, .2)) if rng.random() < .75 else 0.,
        "contrast": float(rng.uniform(.8, 1.2)) if rng.random() < .75 else 1.,
        "blur": int(rng.integers(4)) if rng.random() < .7 else -1,
        "kernel": int(rng.choice([3, 5])), "noise_std": float(math.sqrt(rng.uniform(5, 30))),
        "motion": rng.integers(0, 5, size=4).tolist(),
        "geometry": int(rng.integers(3)) if rng.random() < .7 else -1,
        "radial": float(rng.uniform(-1, 1)),
        "grid_x": rng.uniform(0, 2, 5).tolist(), "grid_y": rng.uniform(0, 2, 5).tolist(),
        "elastic_affine": rng.uniform(-50, 50, (3, 2)).tolist(),
        "clahe": bool(rng.random() < .7),
        "hue": float(rng.uniform(-20, 20)) if rng.random() < .5 else 0.,
        "saturation": float(rng.uniform(-20, 20)) / 255,
        "value": float(rng.uniform(-10, 10)) / 255,
        "affine": bool(rng.random() < .85), "angle": float(rng.uniform(-15, 15)),
        "scale": float(rng.uniform(.9, 1.1)),
        "shift_x": float(rng.uniform(-.1, .1)), "shift_y": float(rng.uniform(-.1, .1)),
        "cutout": bool(rng.random() < .7), "cutout_x": float(rng.random()),
        "cutout_y": float(rng.random()),
        "temperature": float(rng.uniform(4500, 7500)) if rng.random() < .2 else None,
        "solarize": float(rng.uniform(.75, .95)) if rng.random() < .05 else None,
        "noise_seed": int(rng.integers(0, 2**31)),
    }


def crop_metadata(row: Mapping[str, Any], path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Integer YX crop after EXIF transpose; never decode pixel data on the CPU."""
    height, width = row.get("height"), row.get("width")
    if row.get("prepared_image_path") or height is None or width is None:
        from PIL import Image
        with Image.open(path) as image:
            width, height = image.size
            if image.getexif().get(274, 1) in {5, 6, 7, 8}:
                height, width = width, height
    height, width = int(height), int(width)
    box = None if row.get("prepared_image_path") else row.get("bbox_xyxy")
    x1, y1, x2, y2 = (0, 0, width, height) if box is None else tuple(map(int, box))
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Invalid upright bbox {box!r} for {height}x{width}: {path}")
    return np.array([y1, x1], np.int32), np.array([y2-y1, x2-x1], np.int32)


def _reflect_coordinates(coordinates: torch.Tensor, length: int) -> torch.Tensor:
    if length <= 1:
        return torch.zeros_like(coordinates)
    period = 2 * (length - 1)
    value = torch.remainder(coordinates, period)
    return torch.minimum(value, period - value)


def _grid(height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float32),
                            torch.arange(width, device=device, dtype=torch.float32), indexing="ij")
    return xx, yy


def _remap(image: torch.Tensor, xx: torch.Tensor, yy: torch.Tensor, reflect: bool) -> torch.Tensor:
    height, width = image.shape[:2]
    if reflect:
        xx, yy = _reflect_coordinates(xx, width), _reflect_coordinates(yy, height)
    grid = torch.stack((2 * xx / max(1, width-1) - 1, 2 * yy / max(1, height-1) - 1), -1)
    value = F.grid_sample(image.permute(2, 0, 1)[None].float(), grid[None],
                          mode="bilinear", padding_mode="zeros", align_corners=True)
    return value[0].permute(1, 2, 0).clamp_(0, 255).round_().to(torch.uint8)


def _symmetric_indices(length: int, radius: int, device: torch.device) -> torch.Tensor:
    # SciPy ndimage mode='reflect' repeats the edge (different from REFLECT_101).
    value = torch.remainder(torch.arange(-radius, length+radius, device=device), 2*length)
    return torch.minimum(value, 2*length-1-value)


def _gaussian_field(field: torch.Tensor, sigma: float = 50., radius: int = 200) -> torch.Tensor:
    """Separable GPU filter; no CPU SciPy callbacks or native-size CPU arrays."""
    height, width = field.shape
    axis = torch.arange(-radius, radius+1, device=field.device, dtype=torch.float32)
    kernel = torch.exp(-.5 * (axis/sigma).square())
    kernel /= kernel.sum()
    value = field[None, None]
    value = value.index_select(3, _symmetric_indices(width, radius, field.device))
    value = F.conv2d(value, kernel.reshape(1, 1, 1, -1))
    value = value.index_select(2, _symmetric_indices(height, radius, field.device))
    return F.conv2d(value, kernel.reshape(1, 1, -1, 1))[0, 0]


def _filter(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    radius = kernel.shape[-1] // 2
    height, width = image.shape[:2]
    value = image.permute(2, 0, 1)[None].float()
    ys = _reflect_coordinates(torch.arange(-radius, height+radius, device=image.device), height).long()
    xs = _reflect_coordinates(torch.arange(-radius, width+radius, device=image.device), width).long()
    value = value.index_select(2, ys).index_select(3, xs)
    weight = kernel[None, None].expand(3, 1, -1, -1).contiguous()
    return F.conv2d(value, weight, groups=3)[0].permute(1, 2, 0).round_().clamp_(0, 255).byte()


def _median(image: torch.Tensor, kernel: int) -> torch.Tensor:
    radius = kernel//2
    height, width = image.shape[:2]
    ys = _reflect_coordinates(torch.arange(-radius, height+radius, device=image.device), height).long()
    xs = _reflect_coordinates(torch.arange(-radius, width+radius, device=image.device), width).long()
    padded = image.index_select(0, ys).index_select(1, xs)
    result = torch.empty_like(image)
    # Limit temporary windows to 32 rows even on the largest native crops.
    for start in range(0, height, 32):
        count = min(32, height-start)
        windows = padded[start:start+count+2*radius].unfold(0, kernel, 1).unfold(1, kernel, 1)
        result[start:start+count] = windows.contiguous().flatten(-2).median(-1).values
    return result


def _affine(image: torch.Tensor, matrix: torch.Tensor, reflect: bool) -> torch.Tensor:
    height, width = image.shape[:2]
    homogeneous = torch.eye(3, device=image.device)
    homogeneous[:2] = matrix
    inverse = torch.linalg.inv(homogeneous)
    xx, yy = _grid(height, width, image.device)
    return _remap(image, inverse[0, 0]*xx+inverse[0, 1]*yy+inverse[0, 2],
                  inverse[1, 0]*xx+inverse[1, 1]*yy+inverse[1, 2], reflect)


def _geometry(image: torch.Tensor, params: Mapping[str, Any], generator: torch.Generator) -> torch.Tensor:
    operation = params["geometry"]
    if operation < 0:
        return image
    height, width = image.shape[:2]
    xx, yy = _grid(height, width, image.device)
    if operation == 0:
        x, y = (xx-width*.5)/width, (yy-height*.5)/height
        r2 = x.square() + y.square()
        factor = 1 + params["radial"] * (r2+r2.square())
        return _remap(image, x*factor*width+width*.5, y*factor*height+height*.5, True)
    if operation == 1:
        def coordinates(length, factors):
            boundaries = torch.linspace(0, length-1, 6, device=image.device)
            increments = torch.tensor(factors, device=image.device) * ((length-1)/5)
            ends = torch.cat((torch.zeros(1, device=image.device), increments.cumsum(0)))
            position = torch.arange(length, device=image.device).float() * 5 / max(1, length-1)
            section = position.long().clamp_(max=4)
            fraction = position-section
            return ends[section] + fraction*(ends[section+1]-ends[section]) + boundaries[0]
        return _remap(image, coordinates(width, params["grid_x"])[None].expand(height, -1),
                      coordinates(height, params["grid_y"])[:, None].expand(-1, width), True)
    square = max(1, min(height, width)//3)
    center = torch.tensor([width/2, height/2], device=image.device)
    points = center + torch.tensor([[square, square], [square, -square], [-square, -square]],
                                  device=image.device)
    moved = points + torch.tensor(params["elastic_affine"], device=image.device)
    augmented = torch.cat((points, torch.ones(3, 1, device=image.device)), 1)
    matrix = torch.linalg.solve(augmented, moved).T
    image = _affine(image, matrix, True)
    dx = _gaussian_field(torch.rand(height, width, device=image.device, generator=generator)*2-1)*3
    dy = _gaussian_field(torch.rand(height, width, device=image.device, generator=generator)*2-1)*3
    return _remap(image, xx+dx, yy+dy, True)


def _hsv(image: torch.Tensor, params: Mapping[str, Any]) -> torch.Tensor:
    if params["hue"] == 0:
        return image
    value = image.float()/255
    maximum, index = value.max(-1)
    minimum = value.min(-1).values
    delta = maximum-minimum
    safe = delta.clamp_min(1e-8)
    r, g, b = value.unbind(-1)
    hue = torch.where(index == 0, torch.remainder((g-b)/safe, 6),
                      torch.where(index == 1, (b-r)/safe+2, (r-g)/safe+4)) / 6
    hue = torch.remainder(hue+params["hue"]/360, 1)
    saturation = (delta/maximum.clamp_min(1e-8)+params["saturation"]).clamp_(0, 1)
    maximum = (maximum+params["value"]).clamp_(0, 1)
    chroma = maximum*saturation
    x = chroma*(1-torch.abs(torch.remainder(hue*6, 2)-1))
    zeros = torch.zeros_like(x)
    sector = (hue*6).long() % 6
    possibilities = [(chroma, x, zeros), (x, chroma, zeros), (zeros, chroma, x),
                     (zeros, x, chroma), (x, zeros, chroma), (chroma, zeros, x)]
    rgb = torch.zeros_like(value)
    for i, channels in enumerate(possibilities):
        rgb = torch.where((sector == i)[..., None], torch.stack(channels, -1), rgb)
    return ((rgb+(maximum-chroma)[..., None])*255).clamp_(0, 255).byte()


def _planckian_gains(temperature: float) -> tuple[float, float, float]:
    # Blackbody coefficients from Albumentations (MIT), linearly interpolated.
    table = ((.5623, .4457, .3154), (.5376, .4515, .3672), (.5163, .4555, .4103),
             (.4979, .4584, .4468), (.4816, .4604, .4782), (.4672, .4619, .5053),
             (.4542, .4630, .5289))
    position = min(6., max(0., (float(temperature)-4500)/500))
    left, right = int(position), min(6, int(position)+1)
    fraction = position-left
    r, g, b = [table[left][i]*(1-fraction)+table[right][i]*fraction for i in range(3)]
    return r/g, 1., b/g


class DaliDermLoader:
    """Reassemble each external sampler batch on CUDA before whole-batch mixing.

    ``path_resolver``, when supplied, is called as ``resolver(path, workspace)``.
    ``set_epoch`` must be called before each epoch (including checkpoint resume).
    The pipeline is rebuilt per epoch so RNG state is reproducible without relying
    on unrecoverable internal DALI prefetch state. It never shuffles or pads data.
    """

    def __init__(self, rows: Sequence[Mapping[str, Any]], batch_sampler: Iterable[Sequence[int]], *,
                 resolution: int, mean: Sequence[float], std: Sequence[float], workspace: str | Path,
                 path_resolver: Callable[[str, Path], Path] | None = None, training: bool = True,
                 microbatch: int = 8, num_threads: int = 2, seed: int = 42,
                 recipe: str = "gpu_native", max_long_edge: int = 1024, device_id: int = 0):
        self.rows, self.batch_sampler = rows, batch_sampler
        self._labels = []
        for index, row in enumerate(rows):
            label = row.get("label")
            if isinstance(label, bool) or not isinstance(label, Integral) or int(label) < 0:
                raise ValueError(f"row {index} requires a nonnegative integer classification label")
            self._labels.append(int(label))
        self.resolution, self.workspace = int(resolution), Path(workspace)
        self.mean, self.std = tuple(map(float, mean)), tuple(map(float, std))
        if self.resolution <= 0 or len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("resolution and three-channel normalization are required")
        if not np.isfinite([*self.mean, *self.std]).all() or min(self.std) <= 0:
            raise ValueError("normalization must be finite with strictly positive std")
        if int(microbatch) < 1 or int(num_threads) < 1 or int(seed) < 0 or int(device_id) < 0:
            raise ValueError("microbatch/num_threads must be positive and seed/device_id nonnegative")
        self.path_resolver, self.training = path_resolver, bool(training)
        self.microbatch, self.num_threads = int(microbatch), int(num_threads)
        self.seed, self.epoch, self.device_id = int(seed), 0, int(device_id)
        self.recipe, self.max_long_edge = recipe, int(max_long_edge)
        self.metadata = recipe_metadata(recipe, max_long_edge)
        self.metadata.update(training=self.training, microbatch=self.microbatch,
                             seed=self.seed, epoch=self.epoch, device_id=self.device_id,
                             resolution=self.resolution, mean=list(self.mean), std=list(self.std),
                             num_threads=self.num_threads)
        self._pipeline = None
        self._active_parameters: list[dict[str, Any]] = []

    def __len__(self):
        return len(self.batch_sampler)

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)
        self.metadata["epoch"] = self.epoch
        if hasattr(self.batch_sampler, "set_epoch"):
            self.batch_sampler.set_epoch(self.epoch)
        self._pipeline = None

    def _path(self, row):
        source = str(row.get("prepared_image_path") or row["image_path"])
        if self.path_resolver is not None:
            return Path(self.path_resolver(source, self.workspace))
        path = Path(source)
        return path if path.is_absolute() else self.workspace/path

    @torch.no_grad()
    def _prefix(self, images):
        result = []
        for image, params in zip(images, self._active_parameters, strict=True):
            if not image.is_cuda:
                raise RuntimeError("DALI augmentation must execute on CUDA")
            image = image.contiguous()
            if self.recipe == "gpu_bounded" and max(image.shape[:2]) > self.max_long_edge:
                ratio = self.max_long_edge/max(image.shape[:2])
                size = tuple(max(1, round(axis*ratio)) for axis in image.shape[:2])
                image = F.interpolate(image.permute(2, 0, 1)[None].float(), size=size,
                                      mode="bilinear", align_corners=False)[0].permute(1, 2, 0).byte()
            if params["transpose"]:
                image = image.transpose(0, 1)
            if params["vertical"]:
                image = image.flip(0)
            if params["horizontal"]:
                image = image.flip(1)
            image = (image.float()+params["brightness"]*255).clamp_(0, 255).byte()
            image = (image.float()*params["contrast"]).clamp_(0, 255).byte()
            generator = torch.Generator(device=image.device).manual_seed(params["noise_seed"])
            kernel_size = params["kernel"]
            if params["blur"] == 0:
                kernel = torch.eye(kernel_size, device=image.device)/kernel_size
                image = _filter(image, kernel)
            elif params["blur"] == 1:
                image = _median(image, kernel_size)
            elif params["blur"] == 2:
                values = ([1., 2., 1.] if kernel_size == 3 else [1., 4., 6., 4., 1.])
                axis = torch.tensor(values, device=image.device)
                kernel = axis[:, None]*axis[None, :]
                image = _filter(image, kernel/kernel.sum())
            elif params["blur"] == 3:
                noise = torch.randn(image.shape, device=image.device, generator=generator)*params["noise_std"]
                image = (image.float()+noise).clamp_(0, 255).byte()
            result.append(_geometry(image.contiguous(), params, generator).contiguous())
        return result

    @torch.no_grad()
    def _suffix(self, originals, clahe):
        outputs = []
        for original, equalized, params in zip(originals, clahe, self._active_parameters, strict=True):
            image = equalized if params["clahe"] else original
            image = _hsv(image, params)
            height, width = image.shape[:2]
            if params["affine"]:
                angle = math.radians(params["angle"])
                a, b = params["scale"]*math.cos(angle), params["scale"]*math.sin(angle)
                cx, cy = width/2, height/2
                matrix = torch.tensor([[a, b, (1-a)*cx-b*cy+params["shift_x"]*width],
                                       [-b, a, b*cx+(1-a)*cy+params["shift_y"]*height]], device=image.device)
                image = _affine(image, matrix, False)
            image = F.interpolate(image.permute(2, 0, 1)[None].float(),
                                  size=(self.resolution, self.resolution), mode="bilinear",
                                  align_corners=False)[0].permute(1, 2, 0).clamp_(0, 255).byte()
            if params["cutout"]:
                hole = int(self.resolution*.375)
                x, y = int(params["cutout_x"]*self.resolution), int(params["cutout_y"]*self.resolution)
                x1, y1 = max(0, x-hole//2), max(0, y-hole//2)
                image[y1:min(self.resolution, y1+hole), x1:min(self.resolution, x1+hole)] = 0
            if params["temperature"] is not None:
                gains = torch.tensor(_planckian_gains(params["temperature"]), device=image.device)
                image = (image.float()*gains).clamp_(0, 255).byte()
            if params["solarize"] is not None:
                image = torch.where(image >= params["solarize"]*255, 255-image, image)
            outputs.append(image.contiguous())
        return outputs

    @torch.no_grad()
    def _validation(self, images):
        outputs = []
        for image in images:
            height, width = image.shape[:2]
            scale = self.resolution/max(height, width)
            shape = max(1, round(height*scale)), max(1, round(width*scale))
            resized = F.interpolate(image.permute(2, 0, 1)[None].float(), size=shape,
                                    mode="bilinear", align_corners=False)[0].permute(1, 2, 0).byte()
            canvas = torch.zeros(self.resolution, self.resolution, 3, dtype=torch.uint8, device=image.device)
            y, x = (self.resolution-shape[0])//2, (self.resolution-shape[1])//2
            canvas[y:y+shape[0], x:x+shape[1]] = resized
            outputs.append(canvas)
        return outputs

    def _build_pipeline(self):
        try:
            from nvidia.dali import fn, pipeline_def, types
            from nvidia.dali.plugin.pytorch import fn as torch_fn
        except ImportError as error:
            raise RuntimeError("DALI backend requires optional NVIDIA DALI with CUDA; CPU backend is unchanged") from error

        @pipeline_def
        def graph():
            encoded = fn.external_source(name="encoded", device="cpu", dtype=types.UINT8)
            anchor = fn.external_source(name="anchor", device="cpu", dtype=types.INT32)
            shape = fn.external_source(name="shape", device="cpu", dtype=types.INT32)
            image = fn.decoders.image(encoded, device="mixed", output_type=types.RGB, adjust_orientation=True)
            image = fn.slice(image, anchor, shape, axes=[0, 1], normalized_anchor=False,
                             normalized_shape=False, device="gpu")
            if self.training:
                image = torch_fn.torch_python_function(image, function=self._prefix, device="gpu",
                                                       batch_processing=True, output_layouts="HWC")
                equalized = fn.clahe(image, device="gpu", tiles_x=8, tiles_y=8, clip_limit=2.5)
                image = torch_fn.torch_python_function(image, equalized, function=self._suffix,
                                                       device="gpu", batch_processing=True, output_layouts="HWC")
            else:
                image = torch_fn.torch_python_function(image, function=self._validation,
                                                       device="gpu", batch_processing=True, output_layouts="HWC")
            return fn.crop_mirror_normalize(image, device="gpu", dtype=types.FLOAT,
                                             output_layout="CHW", mean=[x*255 for x in self.mean],
                                             std=[x*255 for x in self.std])

        # Python callbacks require synchronous, non-pipelined execution. There are
        # no hidden queued batches: changing _active_parameters cannot race a run.
        pipeline = graph(batch_size=self.microbatch, num_threads=self.num_threads,
                         device_id=self.device_id, seed=self.seed+self.epoch*1000003,
                         exec_async=False, exec_pipelined=False, prefetch_queue_depth=1)
        pipeline.build()
        return pipeline

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        ordinal = 0
        for indices in self.batch_sampler:
            indices = list(indices)
            if not indices:
                raise ValueError("batch_sampler yielded an empty batch")
            for index in indices:
                if (isinstance(index, bool) or not isinstance(index, Integral)
                        or not 0 <= int(index) < len(self.rows)):
                    raise ValueError(f"batch_sampler index {index!r} is outside integer range [0, {len(self.rows)})")
            if self._pipeline is None:
                self._pipeline = self._build_pipeline()
            pieces, labels = [], []
            for offset in range(0, len(indices), self.microbatch):
                selected = indices[offset:offset+self.microbatch]
                encoded, anchors, shapes = [], [], []
                parameters = []
                for index in selected:
                    row = self.rows[int(index)]
                    path = self._path(row)
                    anchor, shape = crop_metadata(row, path)
                    encoded.append(np.frombuffer(path.read_bytes(), dtype=np.uint8))
                    anchors.append(anchor)
                    shapes.append(shape)
                    labels.append(self._labels[int(index)])
                    parameters.append(sample_parameters(self.seed, self.epoch, ordinal))
                    ordinal += 1
                self._active_parameters = parameters
                self._pipeline.feed_input("encoded", encoded)
                self._pipeline.feed_input("anchor", anchors)
                self._pipeline.feed_input("shape", shapes)
                output = self._pipeline.run()[0]
                # Clone before the next run may reuse DALI's output storage.
                pieces.append(torch.utils.dlpack.from_dlpack(output.as_tensor()).clone())
            images = torch.cat(pieces, 0)
            yield images, torch.tensor(labels, dtype=torch.long, device=images.device)
