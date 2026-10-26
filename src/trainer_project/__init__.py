"""Configuration-driven, reproducible image research training."""

import os
from pathlib import Path

_workspace = Path(__file__).resolve().parents[2]
for _variable, _directory in {
    "HF_HOME": ".cache/huggingface",
    "TORCH_HOME": ".cache/torch",
    "XDG_CACHE_HOME": ".cache",
    "TRITON_CACHE_DIR": ".cache/triton",
    "TORCHINDUCTOR_CACHE_DIR": ".cache/torchinductor",
    "WANDB_DIR": "runs/wandb",
    "MPLCONFIGDIR": ".cache/matplotlib",
    "TMPDIR": ".tmp",
}.items():
    os.environ[_variable] = str(_workspace / _directory)
(_workspace / ".tmp").mkdir(exist_ok=True)

__version__ = "0.1.0"
