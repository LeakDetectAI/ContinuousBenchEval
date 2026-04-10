"""Logger protocol — the interface all logging backends implement."""

from __future__ import annotations

from typing import Protocol, Any


class Logger(Protocol):
    """Minimal logging interface for training metrics."""

    def log_scalar(self, key: str, value: float, step: int) -> None:
        """Log a single scalar metric."""
        ...

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        """Log multiple scalar metrics at once."""
        ...

    def log_config(self, config: dict[str, Any]) -> None:
        """Log the full training config (called once at start)."""
        ...

    def close(self) -> None:
        """Flush and close the logger."""
        ...
