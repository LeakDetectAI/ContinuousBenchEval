"""MultiLogger — broadcasts to all registered logging backends."""

from __future__ import annotations

from typing import Any

from cbe.logging.base import Logger


class MultiLogger:
    """Broadcasts log calls to a list of Logger instances."""

    def __init__(self, loggers: list[Logger]) -> None:
        self._loggers = loggers

    def log_scalar(self, key: str, value: float, step: int) -> None:
        for logger in self._loggers:
            logger.log_scalar(key, value, step)

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        for logger in self._loggers:
            logger.log_scalars(metrics, step)

    def log_config(self, config: dict[str, Any]) -> None:
        for logger in self._loggers:
            logger.log_config(config)

    def close(self) -> None:
        for logger in self._loggers:
            logger.close()
