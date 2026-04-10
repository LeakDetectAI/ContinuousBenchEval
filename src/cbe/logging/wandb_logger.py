"""Weights & Biases logging backend."""

from __future__ import annotations

from typing import Any


class WandbLogger:
    """Logs metrics to wandb."""

    def __init__(
        self,
        project: str,
        run_name: str,
        run_dir: str,
    ) -> None:
        import wandb
        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            name=run_name or None,
            dir=run_dir,
        )

    def log_scalar(self, key: str, value: float, step: int) -> None:
        self._wandb.log({key: value}, step=step)

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        self._wandb.log(metrics, step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        self._wandb.config.update(config)

    def close(self) -> None:
        self._run.finish()
