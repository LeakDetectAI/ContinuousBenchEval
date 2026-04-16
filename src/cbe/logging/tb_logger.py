"""TensorBoard logging backend.

Uses `tensorboardX.SummaryWriter` (framework-neutral, no torch/tf dep)
so the same logger works for both the KD (JAX) and HF (PyTorch) paths.
"""

from __future__ import annotations

import os
from typing import Any


class TBLogger:
    """Logs metrics to TensorBoard."""

    def __init__(self, log_dir: str) -> None:
        tb_dir = os.path.join(log_dir, "logs", "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        # tensorboardX is a pure-Python writer that emits tfevents files
        # readable by `tensorboard --logdir`. No torch or tensorflow required.
        from tensorboardX import SummaryWriter
        self._writer = SummaryWriter(logdir=tb_dir)

    def log_scalar(self, key: str, value: float, step: int) -> None:
        self._writer.add_scalar(key, value, global_step=step)

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        for key, value in metrics.items():
            self._writer.add_scalar(key, value, global_step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        # TensorBoard doesn't have a great config story; write as text
        import json
        self._writer.add_text("config", json.dumps(config, indent=2, default=str))

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()
