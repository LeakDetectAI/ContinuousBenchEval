"""Standardized local artifact storage.

Layout:
    outputs/<run>/
    ├── config.yaml
    ├── logs/
    │   ├── tensorboard/
    │   └── wandb/
    ├── checkpoints/
    │   ├── step_0500/
    │   ├── step_1000/
    │   └── latest -> step_1000
    └── metrics/
        └── eval_results.jsonl
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


class LocalArtifactStore:
    """Manages the standardized output directory layout."""

    def __init__(self, output_dir: str, max_checkpoints: int = 10) -> None:
        self._root = Path(output_dir)
        self._max_checkpoints = max_checkpoints
        self._ckpt_dir = self._root / "checkpoints"
        self._metrics_path = self._root / "metrics" / "eval_results.jsonl"

        # Create directory structure
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        (self._root / "logs" / "tensorboard").mkdir(parents=True, exist_ok=True)

    @property
    def run_dir(self) -> str:
        return str(self._root)

    def save_config(self, config: dict[str, Any]) -> None:
        """Save a frozen copy of the training config."""
        with open(self._root / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def register_checkpoint(self, step: int) -> None:
        """Update the 'latest' symlink to the most recent checkpoint dir.

        Supports both HF naming (checkpoint-50) and KD naming (ckpt_50).
        Finds the actual dir by scanning for one that contains the step number.
        """
        latest = self._ckpt_dir / "latest"

        # Find the actual checkpoint dir for this step.
        # HF Trainer uses "checkpoint-{step}", KD uses "ckpt_{step}" or similar.
        step_dir = None
        for candidate in self._ckpt_dir.iterdir():
            if candidate.is_dir() and str(step) in candidate.name:
                step_dir = candidate
                break

        if step_dir is None:
            return  # checkpoint dir not found — skip symlink update

        # Update latest symlink
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(step_dir.name)

        # Rotate: keep at most max_checkpoints
        self._rotate_checkpoints()

    def _rotate_checkpoints(self) -> None:
        """Delete oldest checkpoints if we exceed max_checkpoints."""
        dirs = sorted(
            [d for d in self._ckpt_dir.iterdir()
             if d.is_dir() and d.name not in ("latest",)],
            key=lambda d: d.stat().st_mtime,
        )
        while len(dirs) > self._max_checkpoints:
            shutil.rmtree(dirs.pop(0))

    def save_metrics(self, metrics: dict[str, Any], step: int) -> None:
        """Append eval metrics as a JSON line."""
        record = {
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metrics,
        }
        with open(self._metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def qa_details_path(self, qa_set: str, step: int) -> str:
        """Return (and mkdir-p) the path where per-example QA details go.

        Layout: <run_dir>/eval_details/<qa_set>_step_<step>.jsonl
        e.g.    outputs/cbe/geminon/.../eval_details/valqa_step_001000.jsonl
        """
        dir_ = self._root / "eval_details"
        dir_.mkdir(parents=True, exist_ok=True)
        return str(dir_ / f"{qa_set}_step_{step:06d}.jsonl")

    def load_latest_step(self) -> int | None:
        """Return the step number of the latest checkpoint, or None."""
        latest = self._ckpt_dir / "latest"
        if not latest.exists():
            return None
        target = os.readlink(latest)  # e.g. "step_001000"
        try:
            return int(target.split("_")[1])
        except (IndexError, ValueError):
            return None
