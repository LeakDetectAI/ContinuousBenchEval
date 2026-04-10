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

from cbe.config import load_config
from cbe.logging import create_logger
from cbe.artifacts import create_artifact_store
from cbe.trainers import create_trainer


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
    logger = create_logger(config.logging, artifact_store.run_dir)
    trainer = create_trainer(config, logger=logger, artifact_store=artifact_store)
    trainer.train()


if __name__ == "__main__":
    main()
