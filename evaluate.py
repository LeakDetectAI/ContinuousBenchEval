"""ContinuousBenchEval — standalone evaluation entry point.

Runs QA evaluation (exact match + fuzzy match) on a saved checkpoint.

Usage:
    python evaluate.py --checkpoint outputs/my_run/checkpoints/latest \
        --qa_data data/news/testqa.jsonl \
        --framework huggingface \
        --model google/gemma-3-1b-pt
"""

import argparse
import json
import sys

from cbe.data.formatters import load_jsonl
from cbe.eval.metrics import compute_qa_metrics


def main():
    parser = argparse.ArgumentParser(description="ContinuousBenchEval evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--qa_data", required=True, help="Path to QA .jsonl file")
    parser.add_argument("--framework", default="huggingface", choices=["huggingface", "kauldron"])
    parser.add_argument("--model", default=None, help="Model name (HF hub ID or Gemma name)")
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output", default=None, help="Output .jsonl path for results")
    args = parser.parse_args()

    if args.framework == "huggingface":
        metrics = _eval_hf(args)
    elif args.framework == "kauldron":
        metrics = _eval_kd(args)
    else:
        raise ValueError(f"Unknown framework: {args.framework}")

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("=" * 60)


def _eval_hf(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from cbe.eval.inference import run_qa_eval_hf

    model_name = args.model or args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype="auto",
    )

    if args.lora_rank and args.lora_rank > 0:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.checkpoint)

    return run_qa_eval_hf(
        model=model,
        tokenizer=tokenizer,
        qa_path=args.qa_data,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )


def _eval_kd(args):
    import os
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    from cbe.eval.inference import run_qa_eval_kd

    # Import Gemma inference utilities
    from gemma import gm

    model_name = args.model or "gemma3-1b-pt"
    engine = gm.sampler.Sampler(
        model=model_name,
        checkpoint=args.checkpoint,
        lora_rank=args.lora_rank or 0,
    )

    return run_qa_eval_kd(
        sampler=engine,
        qa_path=args.qa_data,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
