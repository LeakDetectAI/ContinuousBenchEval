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
    # Inline prefix string prepended to every QA prompt. Used as-is when set.
    prompt_prefix: str = ""
    # Optional: load `prompt_prefix` from a text file instead of inlining it.
    # Path is resolved relative to the repo root (the dir containing `configs/`).
    # When both are provided, the file content wins. Useful for sharing a long
    # few-shot prefix across many track configs.
    prompt_prefix_file: str | None = None
    prompt_template: str = "Q: {question}\nA:"  # Must contain {question}
    max_new_tokens: int = 50
    batch_size: int = 16
    # Sampling configuration
    temperature: float = 0.0  # 0 = greedy
    top_k: int | None = None
    top_p: float | None = None
    # Answer parsing + example printing
    parser: str | None = None   # None | "finegrained_geminon" — question-type-aware matcher
    num_examples: int = 0       # Print this many random prompt/completion pairs per eval
    # Persist per-example results (prompt, raw/parsed prediction, gold, verdict)
    # to outputs/<run>/eval_details/<qa_set>_step_<step>.jsonl for offline analysis.
    save_detailed_results: bool = False


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
    """All training hyperparameters: batching, steps, sharding, checkpointing.

    Batch size semantics (IMPORTANT — not symmetric between frameworks):
    - `effective_batch_size`: real total samples per optimizer step, across
      all chips. The same value means the same real batch on HF and KD.
    - `per_device_batch_size`: interpreted differently per framework:
        * HF: truly per-device. Each GPU sees this many samples per fwd/bwd.
          Real effective = per_device × world_size × grad_accum. The HF trainer
          reads WORLD_SIZE from env and derives grad_accum from there.
        * KD: actually the GLOBAL per-iter batch. Kauldron passes it into
          `kd.data.py.DataSource(batch_size=...)`, which Kauldron shards
          across the FSDP mesh. Real effective = per_device × grad_accum
          (independent of chip count).
    """
    effective_batch_size: int = 32
    per_device_batch_size: int = 8
    # Eval batch size — used by the loss evaluator's DataSource. Same dual
    # semantic as per_device_batch_size (HF per-GPU, KD global-per-iter).
    # Typically larger than training's per_device_batch_size (no grad storage).
    eval_per_device_batch_size: int = 32
    # Cap on batches the loss evaluator iterates over. None = full eval set.
    eval_num_batches: int | None = 50
    num_train_steps: int = 10000
    eval_every: int = 500
    save_every: int = 500
    max_checkpoints: int = 10
    # KD: "replicated" (default) | "fsdp". HF: this field is ignored — multi-GPU
    # always uses DDP via torchrun. Set to "fsdp" explicitly for multi-GPU KD runs.
    sharding: str = "replicated"
    seed: int = 42
    # HF-only: trade speed for memory. True saves ~30-50% activation memory
    # but makes each step ~1.5-2x slower. Set False when you have memory
    # headroom and want max throughput.
    gradient_checkpointing: bool = True


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
        # Resolve prompt_prefix_file → prompt_prefix (file content wins).
        # Path is relative to the repo root (cwd at launch, where `configs/` lives).
        if self.eval.prompt_prefix_file:
            with open(self.eval.prompt_prefix_file) as f:
                self.eval.prompt_prefix = f.read()

    @property
    def gradient_accumulation_steps(self) -> int:
        """Default gradient_accumulation = effective_batch_size // per_device.

        `effective_batch_size` is defined as the real total samples per
        optimizer step, across all chips. This formula is exact for KD, where
        Kauldron interprets `per_device_batch_size` as a global per-iter
        batch (sharded across the mesh by XLA), so grad_accum doesn't need to
        account for chip count — effective = per_device × grad_accum.

        HF overrides this at launch in `hf_trainer.py` by factoring in
        `WORLD_SIZE` from the distributed runtime, since HF treats
        `per_device_batch_size` as truly per-device and real effective is
        `per_device × world_size × grad_accum`.
        """
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
    """Best-effort cast of a CLI string to a Python scalar or list."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    if value.lower() == "none":
        return None
    # Handle list syntax: [a,b,c] → ["a", "b", "c"]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_cast(item.strip()) for item in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _resolve_base(
    raw: dict,
    config_dir: Path,
    configs_root: Path | None = None,
) -> dict:
    """If raw has a _base key, load and deep-merge the referenced config(s).

    `_base` can be either a single path string (backward compatible):
        _base: base/models/gemma3_1b_full.yaml
    or a list, with later entries overriding earlier ones:
        _base:
          - base/tasks/geminon.yaml
          - base/models/gemma3_1b_full.yaml   # wins on conflict vs task base

    Each base path is resolved as: first relative to `config_dir`, then
    relative to `configs_root` (discovered as `config_dir` on the top-level
    call and propagated through recursion). This lets a file in
    `configs/base/models/` say `_base: base/models/gemma3_1b_full.yaml`
    and have it resolve against `configs/` (the root), not its own dir.

    Nested `_base` is supported — a base can itself reference other bases.
    """
    base_ref = raw.pop("_base", None)
    if base_ref is None:
        return raw

    # Top-level caller passes configs_root=None; we record config_dir as the
    # stable root so nested bases use it too (prevents double "base/base/"
    # resolution when a base in configs/base/models/ references another base
    # by a path starting with "base/").
    if configs_root is None:
        # Walk up from config_dir to find the "configs" directory.
        configs_root = config_dir
        while configs_root.name != "configs" and configs_root.parent != configs_root:
            if (configs_root / "tracks").is_dir() or (configs_root / "base").is_dir():
                break
            configs_root = configs_root.parent
        # Fallback: one up from config_dir
        if not (configs_root / "base").is_dir() and not (configs_root / "tracks").is_dir():
            configs_root = config_dir.parent

    bases = [base_ref] if isinstance(base_ref, str) else list(base_ref)
    merged: dict = {}
    for b in bases:
        # Try relative to the current file, then the configs root.
        candidates = [(config_dir / b).resolve(), (configs_root / b).resolve()]
        base_path = next((p for p in candidates if p.exists()), None)
        if base_path is None:
            raise FileNotFoundError(
                f"Base config not found: {b} (tried {[str(c) for c in candidates]})"
            )
        with open(base_path) as f:
            base_raw = yaml.safe_load(f) or {}
        base_raw = _resolve_base(base_raw, base_path.parent, configs_root)
        merged = _deep_merge(merged, base_raw)

    return _deep_merge(merged, raw)


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

    # PyYAML parses values like "5e-5" as strings (YAML 1.1 requires "5.0e-5"
    # for a float). Coerce str→float/int at the dacite level so users aren't
    # forced to write pedantic scientific notation.
    config = from_dict(
        data_class=TrainConfig,
        data=raw,
        config=DaciteConfig(
            strict=True,
            type_hooks={
                float: lambda v: float(v) if isinstance(v, (str, int)) else v,
                int: lambda v: int(v) if isinstance(v, str) else v,
            },
        ),
    )
    config.__post_init__()
    return config
