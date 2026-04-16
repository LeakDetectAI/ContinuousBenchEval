"""ContinuousBenchEval — training entry point.

Usage:
    python train.py --config configs/tracks/news.yaml --framework huggingface
    python train.py --config configs/tracks/news.yaml --framework kauldron
    python train.py --config configs/tracks/news.yaml --framework hf --override optimizer.lr=5e-6

Multi-GPU (HF/TRL):
    torchrun --nproc_per_node=4 train.py --config configs/tracks/news.yaml --framework hf
    accelerate launch train.py --config configs/tracks/news.yaml --framework hf
"""

import argparse
import os
import sys

from cbe.config import load_config
from cbe.logging import create_logger
from cbe.artifacts import create_artifact_store
from cbe.trainers import create_trainer


class _TeeWriter:
    """Write to both a log file and the original stream (stdout or stderr)."""

    def __init__(self, log_file, orig_stream) -> None:
        self._file = log_file
        self._orig = orig_stream

    def write(self, data: str) -> int:
        self._orig.write(data)
        self._file.write(data)
        return len(data)

    def flush(self) -> None:
        self._orig.flush()
        self._file.flush()

    def isatty(self) -> bool:
        return self._orig.isatty()


_FRAMEWORK_ALIASES = {
    "hf": "huggingface",
    "kd": "kauldron",
    "huggingface": "huggingface",
    "kauldron": "kauldron",
}


def main():
    parser = argparse.ArgumentParser(description="ContinuousBenchEval training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--framework",
        default=None,
        choices=list(_FRAMEWORK_ALIASES.keys()),
        help="Override trainer framework: hf (huggingface) or kd (kauldron)",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config overrides (dot.notation=value), repeatable",
    )
    args = parser.parse_args()

    framework = _FRAMEWORK_ALIASES[args.framework] if args.framework else None
    config = load_config(args.config, overrides=args.override, framework=framework)
    artifact_store = create_artifact_store(config)

    # In DDP, only rank 0 should write logs, initialize wandb, etc.
    # Other ranks get a no-op logger.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    # Tee stdout+stderr to a log file (rank 0 only to avoid interleaving)
    if is_main:
        log_path = os.path.join(artifact_store.run_dir, "logs", "train.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_file = open(log_path, "a", buffering=1, encoding="utf-8")
        sys.stdout = _TeeWriter(log_file, sys.stdout)
        sys.stderr = _TeeWriter(log_file, sys.stderr)

    # Only rank 0 initializes wandb / TB loggers; other ranks get a no-op.
    if is_main:
        logger = create_logger(config.logging, artifact_store.run_dir)
    else:
        from cbe.logging.multi_logger import MultiLogger
        logger = MultiLogger([])  # no-op: logs nothing

    trainer = create_trainer(config, logger=logger, artifact_store=artifact_store)
    trainer.train()


if __name__ == "__main__":
    main()
