"""HuggingFace model factory.

Supports any model available on the HuggingFace Hub via AutoModel/AutoTokenizer.
LoRA is applied via PEFT when lora_rank is set.
"""

from __future__ import annotations

import os
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


def _reinit_lora_a_jax_style(model: Any) -> None:
    """Re-initialize LoRA A matrices to match JAX/Flax `kaiming_uniform`.

    JAX's `nn.initializers.kaiming_uniform()` (aka `he_uniform`) is
    `variance_scaling(scale=2.0, mode="fan_in", distribution="uniform")`
    (verified directly from `jax._src.nn.initializers.he_uniform` source).
    For uniform distribution, `variance = scale / fan_in` → sampling bound
    is `sqrt(3 * variance) = sqrt(6 / fan_in)`.

    PyTorch's default `torch.nn.init.kaiming_uniform_(a=sqrt(5))` samples
    from `[-sqrt(1/fan_in), sqrt(1/fan_in)]` — a factor of sqrt(6) ≈ 2.45x
    smaller than JAX.

    KD's `peft.LoRAEinsumAdapter` uses JAX's default. To align, we reinit
    HF-PEFT's lora_A weights with the JAX bound `sqrt(6/fan_in)` here.
    (lora_B stays at zeros — same in both frameworks.)
    """
    import math
    import torch

    reinited = 0
    for name, module in model.named_modules():
        if name.endswith(".lora_A"):
            for adapter in module.values():
                # adapter.weight shape: (r, in_features)
                fan_in = adapter.weight.shape[1]
                bound = math.sqrt(6.0 / fan_in)  # JAX he_uniform: scale=2, fan_in mode
                with torch.no_grad():
                    torch.nn.init.uniform_(adapter.weight, -bound, bound)
                reinited += 1
    if reinited:
        print(f"[CBE] Reinitialized {reinited} LoRA A matrices with JAX-style kaiming_uniform (bound=sqrt(6/fan_in))")


def create_hf_model(
    config: ModelConfig,
    resume_from: str | None = None,
) -> HFModelBundle:
    """Create an HF model + tokenizer from config.

    Args:
        config: ModelConfig with `name` as an HF hub model ID
            (e.g. "google/gemma-3-1b-pt", "meta-llama/Llama-3-8B").
            If `lora_rank` is set, wraps the model with PEFT LoRA.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config.pretrained_path or _resolve_model_name(config.name)

    import torch
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Hard-coded pure-bf16: params, activations, Adam state, and updates all
    # in bf16 (for parity with KD's default). No fp32 master weights, no
    # autocast-AMP behavior.
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
    )

    if config.lora_rank and config.lora_rank > 0:
        from peft import LoraConfig, PeftModel, get_peft_model

        # If resuming from a PEFT checkpoint, load the adapter from disk
        # instead of creating fresh LoRA weights.
        if resume_from and os.path.exists(os.path.join(resume_from, "adapter_config.json")):
            print(f"[CBE] Loading PEFT adapter from {resume_from}")
            model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
            model.print_trainable_parameters()
        else:
            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_rank * 1,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM",
                use_rslora=False,
                init_lora_weights=True,  # Kaiming A, zero B
            )
            # autocast_adapter_dtype=False keeps the LoRA A/B weights in the
            # base model's dtype (bf16). Default (True) would upcast adapters
            # to fp32 even when base is bf16, breaking the pure-bf16 contract.
            model = get_peft_model(
                model, lora_config, autocast_adapter_dtype=False,
            )
            _reinit_lora_a_jax_style(model)
            model.print_trainable_parameters()
    elif resume_from:
        # Full fine-tune resume: load the saved model weights.
        if os.path.exists(os.path.join(resume_from, "model.safetensors")):
            print(f"[CBE] Loading full model weights from {resume_from}")
            model = AutoModelForCausalLM.from_pretrained(
                resume_from, torch_dtype=torch.bfloat16,
            )

    return HFModelBundle(model=model, tokenizer=tokenizer)
