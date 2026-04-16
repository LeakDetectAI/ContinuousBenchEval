"""HuggingFace model factory.

Supports any model available on the HuggingFace Hub via AutoModel/AutoTokenizer.
LoRA is applied via PEFT when lora_rank is set.
"""

from __future__ import annotations

from typing import Any

from cbe.config import ModelConfig

# Map Kauldron-style short names → HuggingFace Hub IDs so the same config
# works for both --framework kd and --framework hf.
_KD_TO_HF_NAME = {
    "gemma3-270m-pt": "google/gemma-3-270m-pt",
    "gemma3-270m-it": "google/gemma-3-270m-it",
    "gemma3-1b-pt":   "google/gemma-3-1b-pt",
    "gemma3-1b-it":   "google/gemma-3-1b-it",
    "gemma3-4b-pt":   "google/gemma-3-4b-pt",
    "gemma3-4b-it":   "google/gemma-3-4b-it",
    "gemma3-12b-pt":  "google/gemma-3-12b-pt",
    "gemma3-12b-it":  "google/gemma-3-12b-it",
    "gemma3-27b-pt":  "google/gemma-3-27b-pt",
    "gemma3-27b-it":  "google/gemma-3-27b-it",
}


def _resolve_model_name(name: str) -> str:
    """Resolve a model name to an HF hub ID, handling KD short names."""
    return _KD_TO_HF_NAME.get(name.lower(), name)


class HFModelBundle:
    """Holds an HF model + tokenizer together."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer


def create_hf_model(config: ModelConfig) -> HFModelBundle:
    """Create an HF model + tokenizer from config.

    Args:
        config: ModelConfig with `name` as an HF hub model ID
            (e.g. "google/gemma-3-1b-pt", "meta-llama/Llama-3-8B").
            If `lora_rank` is set, wraps the model with PEFT LoRA.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config.pretrained_path or _resolve_model_name(config.name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
    )

    if config.lora_rank and config.lora_rank > 0:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_rank * 2,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return HFModelBundle(model=model, tokenizer=tokenizer)
