"""Logging backends (TensorBoard, wandb) with a unified interface."""

from cbe.logging.base import Logger
from cbe.logging.multi_logger import MultiLogger


def create_logger(config, run_dir: str) -> MultiLogger:
    """Create a MultiLogger from logging config."""
    loggers: list[Logger] = []
    for backend in config.backends:
        if backend == "tensorboard":
            from cbe.logging.tb_logger import TBLogger
            loggers.append(TBLogger(log_dir=run_dir))
        elif backend == "wandb":
            from cbe.logging.wandb_logger import WandbLogger
            loggers.append(WandbLogger(
                project=config.project_name,
                run_name=config.run_name,
                run_dir=run_dir,
            ))
        else:
            raise ValueError(f"Unknown logging backend: {backend}")
    return MultiLogger(loggers)
