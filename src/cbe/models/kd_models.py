"""JAX / Kauldron model factories.

Provides an abstract JaxModelFactory protocol and a concrete Gemma
implementation ported from dpsynth/training/model.py.
"""

from __future__ import annotations

from typing import Any, Protocol

from cbe.config import ModelConfig


class JaxModelFactory(Protocol):
    """Protocol for JAX model factories used in the KD trainer."""

    def make_model(self, config: ModelConfig) -> Any:
        """Return a Kauldron-compatible model object."""
        ...

    def make_init(self, config: ModelConfig) -> Any:
        """Return a checkpoint initializer."""
        ...

    def get_tokenizer(self) -> Any:
        """Return the tokenizer for this model family."""
        ...


# ---------------------------------------------------------------------------
# Gemma implementation
# ---------------------------------------------------------------------------

# Map from config name → (gm.ckpts.CheckpointPath attribute, model class name)
_GEMMA_MODELS = {
    "gemma3-270m-pt": ("GEMMA3_270M_PT", "Gemma3_270M"),
    "gemma3-270m-it": ("GEMMA3_270M_IT", "Gemma3_270M"),
    "gemma3-1b-pt":   ("GEMMA3_1B_PT",   "Gemma3_1B"),
    "gemma3-1b-it":   ("GEMMA3_1B_IT",   "Gemma3_1B"),
    "gemma3-4b-pt":   ("GEMMA3_4B_PT",   "Gemma3_4B"),
    "gemma3-4b-it":   ("GEMMA3_4B_IT",   "Gemma3_4B"),
    "gemma3-12b-pt":  ("GEMMA3_12B_PT",  "Gemma3_12B"),
    "gemma3-12b-it":  ("GEMMA3_12B_IT",  "Gemma3_12B"),
    "gemma3-27b-pt":  ("GEMMA3_27B_PT",  "Gemma3_27B"),
    "gemma3-27b-it":  ("GEMMA3_27B_IT",  "Gemma3_27B"),
}


class GemmaModelFactory:
    """Creates Gemma models for use with Kauldron trainer."""

    def make_model(self, config: ModelConfig) -> Any:
        from gemma import gm

        name = config.name.lower()
        if name not in _GEMMA_MODELS:
            raise ValueError(
                f"Unknown Gemma model: {config.name}. "
                f"Available: {list(_GEMMA_MODELS.keys())}"
            )

        ckpt_attr, cls_name = _GEMMA_MODELS[name]
        model_cls = getattr(gm.nn, cls_name)

        if hasattr(model_cls, "text_only"):
            model = model_cls(tokens="batch.input", text_only=True)
        else:
            model = model_cls(tokens="batch.input")

        if config.lora_rank and config.lora_rank > 0:
            model = gm.nn.LoRA(rank=config.lora_rank, model=model)

        return model

    def make_init(self, config: ModelConfig) -> Any:
        from gemma import gm

        name = config.name.lower()
        ckpt_attr, _ = _GEMMA_MODELS[name]
        ckpt_path = getattr(gm.ckpts.CheckpointPath, ckpt_attr)

        if config.lora_rank and config.lora_rank > 0:
            return gm.ckpts.SkipLoRA(
                wrapped=gm.ckpts.LoadCheckpoint(path=ckpt_path)
            )
        return gm.ckpts.LoadCheckpoint(path=ckpt_path)

    def get_tokenizer(self) -> Any:
        from gemma import gm
        return gm.text.Gemma3Tokenizer()


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------

def create_kd_model(config: ModelConfig) -> JaxModelFactory:
    """Return the appropriate JAX model factory for the given config."""
    name = config.name.lower()
    if name in _GEMMA_MODELS:
        return GemmaModelFactory()
    raise ValueError(
        f"No KD model factory for: {config.name}. "
        f"Available Gemma models: {list(_GEMMA_MODELS.keys())}. "
        f"Implement JaxModelFactory for custom models."
    )
