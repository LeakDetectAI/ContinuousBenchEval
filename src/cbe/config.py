"""YAML config loading with base+override inheritance and dataclass schema."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dacite import from_dict, Config as DaciteConfig


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    train_path: str = ""
    val_path: str = ""
    valqa_path: str | None = None
    testqa_path: str | None = None
    sequence_length: int = 1024
    task: str = "next_token"  # "next_token" | "seq2seq"
    num_workers: int = 4


@dataclass
class EvalConfig:
    prompt_prefix: str = ""  # e.g. "Answer concisely. " — prepended before each QA prompt
    prompt_template: str = "Q: {question}\nA:"  # Must contain {question}
    max_new_tokens: int = 50
    batch_size: int = 16
    # Sampling configuration
    temperature: float = 0.0  # 0 = greedy
    top_k: int | None = None
    top_p: float | None = None


@dataclass
class ModelConfig:
    name: str = ""  # HF hub ID or KD model name (e.g. "gemma3-1b-pt")
    lora_rank: int | None = None
    pretrained_path: str | None = None


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.02
    schedule: str = "cosine"  # "cosine" | "linear" | "constant"
    end_lr_fraction: float = 0.1  # end_value = lr * end_lr_fraction
    b1: float = 0.9
    b2: float = 0.99


@dataclass
class TrainingConfig:
    """All training hyperparameters: batching, steps, sharding, checkpointing."""
    effective_batch_size: int = 32
    per_device_batch_size: int = 8
    num_train_steps: int = 10000
    eval_every: int = 500
    save_every: int = 500
    max_checkpoints: int = 10
    sharding: str = "fsdp"  # "fsdp" | "ddp" | "none"
    seed: int = 42
    bf16: bool = True


@dataclass
class LoggingConfig:
    backends: list[str] = field(default_factory=lambda: ["tensorboard"])
    project_name: str = "cbe"  # wandb project name; also used to organize TB logs
    run_name: str = ""  # wandb run name; auto-set from TrainConfig.run_name if empty
    log_every_n_steps: int = 10


@dataclass
class TrainConfig:
    framework: str = "huggingface"  # "kauldron" | "huggingface"
    output_dir: str = ""  # Auto-derived as outputs/<project_name>/<run_name> if empty
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self):
        # Auto-derive output_dir from project_name/run_name
        if not self.output_dir:
            project = self.logging.project_name or "cbe"
            run = self.logging.run_name or "default"
            self.output_dir = os.path.join("outputs", project, run)

    @property
    def gradient_accumulation_steps(self) -> int:
        """Compute gradient accumulation steps to reach effective_batch_size."""
        return max(1, self.training.effective_batch_size // self.training.per_device_batch_size)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override values win."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _apply_cli_overrides(raw: dict, overrides: list[str]) -> dict:
    """Apply dot-notation overrides like 'optimizer.lr=5e-6' to a raw dict."""
    for item in overrides:
        key, _, value = item.partition("=")
        if not value:
            raise ValueError(f"Override must be key=value, got: {item!r}")

        parts = key.split(".")
        target = raw
        for part in parts[:-1]:
            target = target.setdefault(part, {})

        # Try to cast to int/float/bool, fall back to string
        target[parts[-1]] = _cast(value)
    return raw


def _cast(value: str) -> Any:
    """Best-effort cast of a CLI string to a Python scalar."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    if value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _resolve_base(raw: dict, config_dir: Path) -> dict:
    """If raw has a _base key, load and merge the base config first."""
    base_ref = raw.pop("_base", None)
    if base_ref is None:
        return raw

    base_path = (config_dir / base_ref).resolve()
    if not base_path.exists():
        # Also try relative to the configs/ root (one level up from tracks/)
        base_path = (config_dir.parent / base_ref).resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_ref} (searched {base_path})")

    with open(base_path) as f:
        base_raw = yaml.safe_load(f) or {}

    # Recursively resolve if the base itself has a _base
    base_raw = _resolve_base(base_raw, base_path.parent)
    return _deep_merge(base_raw, raw)


def load_config(
    config_path: str,
    overrides: list[str] | None = None,
    framework: str | None = None,
) -> TrainConfig:
    """Load a YAML config file, resolve _base inheritance, apply CLI overrides.

    Args:
        config_path: Path to the YAML config file.
        overrides: List of "key.path=value" CLI overrides.
        framework: Override framework from CLI flag (--framework).
    """
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    raw = _resolve_base(raw, config_path.parent)

    if overrides:
        raw = _apply_cli_overrides(raw, overrides)

    # CLI --framework flag overrides config
    if framework:
        raw["framework"] = framework

    config = from_dict(
        data_class=TrainConfig,
        data=raw,
        config=DaciteConfig(strict=True),
    )
    config.__post_init__()
    return config
