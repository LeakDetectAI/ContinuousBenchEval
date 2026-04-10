"""TensorBoard logging backend."""

from __future__ import annotations

import os
from typing import Any


class TBLogger:
    """Logs metrics to TensorBoard."""

    def __init__(self, log_dir: str) -> None:
        tb_dir = os.path.join(log_dir, "logs", "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir=tb_dir)

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
