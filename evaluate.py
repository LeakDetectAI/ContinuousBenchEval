"""ContinuousBenchEval — standalone evaluation entry point.

Runs QA evaluation (exact match + fuzzy match) on a saved checkpoint.

Usage:
    # Full fine-tune HF checkpoint (weights live in --checkpoint)
    python evaluate.py --framework hf \\
        --checkpoint outputs/cbe/news/.../checkpoints/step_001000 \\
        --qa_data data/news/testqa.jsonl

    # LoRA HF checkpoint (adapter in --checkpoint, base model via --model)
    python evaluate.py --framework hf \\
        --checkpoint outputs/cbe/news/.../checkpoints/step_001000 \\
        --model google/gemma-3-1b-pt \\
        --lora_rank 128 \\
        --qa_data data/news/testqa.jsonl

    # LoRA KD checkpoint (base pretrained resolved from --model Gemma name)
    python evaluate.py --framework kd \\
        --checkpoint outputs/cbe/news/.../checkpoints/ckpt_1000 \\
        --model gemma3-1b-pt \\
        --lora_rank 128 \\
        --qa_data data/news/testqa.jsonl
"""

import argparse
import os
import sys


_FRAMEWORK_ALIASES = {
    "hf": "huggingface",
    "kd": "kauldron",
    "huggingface": "huggingface",
    "kauldron": "kauldron",
}


def main():
    parser = argparse.ArgumentParser(description="ContinuousBenchEval evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--qa_data", required=True, help="Path to QA .jsonl file")
    parser.add_argument(
        "--framework",
        default="hf",
        choices=list(_FRAMEWORK_ALIASES.keys()),
        help="hf (huggingface) or kd (kauldron)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Base model name: HF hub ID (e.g. google/gemma-3-1b-pt) "
             "or KD name (e.g. gemma3-1b-pt). Required for LoRA checkpoints.",
    )
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--prompt_prefix", default="")
    parser.add_argument("--prompt_template", default="Q: {question}\nA:")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument(
        "--parser",
        default=None,
        help="Answer parser: 'geminon' or None (default lowercase/substring match)",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Print this many random prompt/completion examples with verdicts",
    )
    parser.add_argument(
        "--save_details",
        default=None,
        help="If set, save all per-example results as JSONL at this path",
    )
    args = parser.parse_args()

    framework = _FRAMEWORK_ALIASES[args.framework]

    if framework == "huggingface":
        metrics = _eval_hf(args)
    else:
        metrics = _eval_kd(args)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# HuggingFace path
# ---------------------------------------------------------------------------

def _eval_hf(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from cbe.eval.inference import run_qa_eval_hf

    # Auto-detect LoRA from the presence of adapter_config.json in the ckpt dir.
    # (HF+PEFT Trainer only writes adapter weights to the checkpoint dir;
    #  base weights are NOT in there, so we must load the base separately.)
    is_lora = os.path.exists(os.path.join(args.checkpoint, "adapter_config.json"))

    if is_lora:
        if not args.model:
            raise SystemExit(
                "LoRA adapter detected at --checkpoint; pass --model <hub_id> "
                "so we can load the base model."
            )
        from peft import PeftModel
        print(f"[evaluate] Loading base model {args.model} + LoRA adapter {args.checkpoint}")
        base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
        model = PeftModel.from_pretrained(base, args.checkpoint)
        tokenizer_src = args.model  # tokenizer lives with the base model
    else:
        print(f"[evaluate] Loading full-weights model from {args.checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype="auto")
        tokenizer_src = args.checkpoint if not args.model else args.model

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Move to GPU if available
    try:
        import torch
        if torch.cuda.is_available():
            model = model.to("cuda")
    except ImportError:
        pass

    return run_qa_eval_hf(
        model=model,
        tokenizer=tokenizer,
        qa_path=args.qa_data,
        prompt_prefix=args.prompt_prefix,
        prompt_template=args.prompt_template,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        parser=args.parser,
        num_examples=args.num_examples,
        save_details_path=args.save_details,
    )


# ---------------------------------------------------------------------------
# Kauldron path
# ---------------------------------------------------------------------------

def _eval_kd(args):
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
    except ImportError:
        pass

    from gemma import gm
    from gemma.gm import peft as gm_peft

    from cbe.config import ModelConfig
    from cbe.models.kd_models import create_kd_model, _GEMMA_MODELS
    from cbe.eval.inference import run_qa_eval_kd

    if not args.model:
        raise SystemExit(
            "KD eval needs --model <name> (e.g. gemma3-1b-pt) to rebuild "
            "the architecture."
        )

    model_config = ModelConfig(name=args.model, lora_rank=args.lora_rank)
    factory = create_kd_model(model_config)
    model = factory.make_model(model_config)

    print(f"[evaluate] Loading KD checkpoint: {args.checkpoint}")
    params = gm.ckpts.load_params(args.checkpoint)

    # For LoRA checkpoints: training was init'd with SkipLoRA, meaning only
    # LoRA params were updated. We need to re-inject the pretrained base
    # params so the merged tree matches the LoRA-wrapped model.
    if args.lora_rank and args.lora_rank > 0:
        print(f"[evaluate] LoRA rank={args.lora_rank}: re-loading base weights + merging")
        original, lora = gm_peft.split_params(params)
        ckpt_attr, _ = _GEMMA_MODELS[args.model.lower()]
        base_ckpt_path = getattr(gm.ckpts.CheckpointPath, ckpt_attr)
        original = gm.ckpts.load_params(base_ckpt_path, params=original)
        params = gm_peft.merge_params(original, lora)

    return run_qa_eval_kd(
        model=model,
        params=params,
        qa_path=args.qa_data,
        prompt_prefix=args.prompt_prefix,
        prompt_template=args.prompt_template,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        parser=args.parser,
        num_examples=args.num_examples,
        save_details_path=args.save_details,
    )


if __name__ == "__main__":
    main()
