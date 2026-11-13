# Experimental NVIDIA DALI image loader

The dermoscopy training profile identified CPU augmentation of native-resolution crops as the
main loader cost. Increasing DataLoader workers from four to six raised measured standalone
throughput from 7.46 to 9.29 images/s while sixteen training jobs were active. The next experiment
moves decoding and image augmentation to the GPU with an optional NVIDIA DALI backend.

This backend is an explicit experimental recipe. Its outputs are not pixel-identical to the
legacy Albumentations pipeline. Existing training and dataset defaults remain unchanged.

## Installation

The optional dependency is pinned to `nvidia-dali-cuda120==2.3.0`. It requires an appropriate
NVIDIA CUDA GPU, driver, and a supported Linux/Python environment. Importing the package or running
the CPU tests does not require DALI.
The lockfile resolves `packaging==26.2` because DALI 2.3.0 requires `packaging<=26.2`.

```bash
uv sync --locked --inexact --extra dali
```

On a server with a shared training environment, use a separate environment or install target for
the trial so that ongoing jobs keep their existing dependencies.

## Input and output

`trainer_project.data.dali_dermoscopy.DaliDermLoader` accepts manifest rows and a finite batch
sampler. Each row contains a nonnegative integer `label` and `image_path`. Optional `bbox_xyxy`
coordinates describe a crop **after EXIF orientation correction**. A `prepared_image_path`
indicates an already prepared crop and suppresses a second BB crop. Relative paths resolve
under `workspace`, or callers can supply their own `path_resolver`.
Optional `height` and `width` describe the upright source dimensions before BB cropping. When
absent, the loader reads image headers and EXIF without decoding image pixels on the CPU.

```json
{"image_path":"images/lesion001.jpg","label":3,"bbox_xyxy":[20,30,620,530]}
{"prepared_image_path":"crops/lesion002.png","label":7}
```

The loader yields `(images, labels)`: float32 CUDA images in NCHW order and int64 CUDA labels.
It preserves the sampler's sample order, repetitions, effective batch boundaries, and smaller
tail batch. Internal augmentation microbatches do not change optimizer batch size or PK class
composition. Call `set_epoch(epoch)` before each epoch to reset epoch-addressed random state.
Read `loader.metadata` into the trial's execution receipt, including normalization and recipe.

```python
from trainer_project.data.dali_dermoscopy import DaliDermLoader

loader = DaliDermLoader(
    rows, batch_sampler,
    resolution=512,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    workspace=project_root,
    training=True,
    recipe="gpu_bounded",
    max_long_edge=1024,
    microbatch=8,
    num_threads=2,
    seed=42,
    device_id=0,
)
loader.set_epoch(epoch)
for images, labels in loader:
    # Apply the experiment's existing Mixup/CutMix and model training step here.
    ...
```

Always pass the normalization of the selected pretrained weights. The example values are
ImageNet normalization and are not a substitute for checking an individual model's data config.

## Trial recipes

`gpu_native` retains native crop resolution before augmentation. `gpu_bounded` bounds the crop's
long edge before the expensive augmentation stages, reducing work and memory on very large
crops. The bound changes augmentation scale and therefore defines a separate experiment.
Encoded-file reads and scalar parameter sampling stay on the CPU; DALI mixed decoding, BB
slicing, and CLAHE run on the GPU. Torch callbacks perform the other image operations on CUDA.
The callbacks use synchronous pipeline execution; this version does not prefetch augmentation
concurrently with model training.

DALI's RGB CLAHE uses luminance handling that differs from the legacy Lab-L transform, and GPU
geometry/interpolation and random-number generation differ from the CPU implementation. These
changes must be recorded as part of the trial recipe. Do not combine trial metrics with an
existing fold run under the same experiment identifier or silently resume a legacy run with
the DALI backend.

## Standalone throughput measurement

The benchmark consumes a JSONL manifest and does not import the MILK training package, train a
model, or change the Lightning trainer. CUDA device indices are relative to
`CUDA_VISIBLE_DEVICES`.

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_dali_dermoscopy.py \
  --manifest data/trial.jsonl --path-root "$PWD" \
  --backend dali --training --recipe gpu_bounded --max-long-edge 1024 \
  --resolution 512 --batch-size 256 --microbatch 8 --workers 2 \
  --warmup 1 --steps 10 --output reports/dali_trial.json
```

For a deterministic CPU decode/crop/resize reference, use `--backend torch` without `--training`.
Both evaluation paths preserve aspect ratio, resize the longest edge, and center the image on
a black square canvas before normalization.
The reference does not reproduce training augmentation. Use the same manifest, resolution,
batch size, sample order, and normalization when comparing evaluation preprocessing. DALI and
Pillow resizing can still differ numerically.

Reports include manifest SHA-256, measured label-stream SHA-256, actual tail-batch sizes, loader
metadata, first-batch latency, warmup, synchronized ready-batch latency, and images/s. Output
paths must be new; existing reports are preserved. Torch allocator peaks exclude DALI's internal
GPU allocations, so sample total GPU memory with `nvidia-smi` during trials as well.

Loader throughput alone does not establish end-to-end training speedup. Measure optimizer
steps, data wait, validation, and total GPU utilization alongside this benchmark, particularly
when several jobs share a GPU. Warmup must leave at least one measured batch in the finite
manifest stream. The benchmark does not implicitly repeat or pad the dataset to fill a run.

## Validation

CPU tests cover finite stream boundaries, image/label alignment, crop semantics, manifest
validation, and lazy optional dependency handling. GPU smoke checks additionally need to verify
real JPEG/PNG decoding, finite normalized outputs, exact label order, tails, and a successful
optimizer update in the target CUDA environment. Use separate output directories for trial
checkpoints and validation reports; preserve existing jobs and artifacts.

GPU smoke checks on 13 November 2026 passed for both recipes using DALI 2.3.0 on an A100 80GB.
The published backend SHA matches the tested source. The 32-sample stress set covered all eleven
classes and forced elastic deformation on the largest crop, containing 29,910,972 pixels.

| Recipe | Augmentation microbatch | Wall time including build | Peak Torch allocated / reserved |
| --- | ---: | ---: | ---: |
| `gpu_native` | 4 | 3.965 s | 1,808 / 3,320 MiB |
| `gpu_bounded`, long edge 1024 | 8 | 4.141 s | 388 / 864 MiB |

These are stress smoke checks, not representative throughput measurements. Their wall times
include pipeline construction, their microbatches differ, and Torch memory excludes DALI's
internal allocations. Two validation images also passed a geometry comparison against legacy
preprocessing: mean absolute normalized-tensor difference 0.00787 and maximum difference
0.13943. This does not establish pixel equivalence. Training pilot optimizer checks and
end-to-end throughput are pending.

The [sanitized GPU probe record](benchmarks/dali_gpu_probe_20261113.json) contains the exact
backend SHA, measurements, and limits without source images, patient paths, or manifests.

Operator references: [DALI image decoder](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.decoders.image.html),
[DALI CLAHE](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.clahe.html),
[DALI Python functions](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.python_function.html).
