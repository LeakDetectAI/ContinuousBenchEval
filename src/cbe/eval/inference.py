"""Framework-agnostic inference wrapper for generation-based evaluation."""

from __future__ import annotations

from typing import Any

from cbe.data.formatters import load_jsonl
from cbe.eval.metrics import compute_qa_metrics


def _build_prompt(record: dict, prompt_prefix: str, prompt_template: str) -> str:
    """Build a full prompt with optional prefix."""
    prompt = prompt_template.format(**record)
    if prompt_prefix:
        prompt = prompt_prefix + prompt
    return prompt


def run_qa_eval_hf(
    model: Any,
    tokenizer: Any,
    qa_path: str,
    prompt_prefix: str = "",
    prompt_template: str = "Q: {question}\nA:",
    max_new_tokens: int = 50,
    batch_size: int = 16,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> dict[str, float]:
    """Run QA evaluation using an HF model.

    Args:
        model: A HuggingFace model (possibly wrapped in PEFT).
        tokenizer: The HF tokenizer.
        qa_path: Path to a .jsonl file with {question, answer} records.
        prompt_prefix: Prefix prepended to every prompt (e.g. "Answer concisely. ").
        prompt_template: Template with {question} placeholder.
        max_new_tokens: Max tokens to generate per answer.
        batch_size: Batch size for generation.
        temperature: Sampling temperature. 0 = greedy.
        top_k: Top-k sampling. None = disabled.
        top_p: Nucleus sampling. None = disabled.

    Returns:
        Metrics dict from compute_qa_metrics.
    """
    import torch

    qa_records = load_jsonl(qa_path)
    if not qa_records:
        return {"exact_match": 0.0, "fuzzy_match": 0.0, "total": 0}

    model.eval()
    device = next(model.parameters()).device
    results = []

    # Build generation kwargs
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature == 0.0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        if top_k is not None:
            gen_kwargs["top_k"] = top_k
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

    for i in range(0, len(qa_records), batch_size):
        batch = qa_records[i : i + batch_size]
        prompts = [_build_prompt(r, prompt_prefix, prompt_template) for r in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        # Decode only the generated portion
        for j, (prompt, output) in enumerate(zip(prompts, outputs)):
            prompt_len = inputs["input_ids"][j].shape[0]
            generated = tokenizer.decode(
                output[prompt_len:], skip_special_tokens=True
            )
            # Take first line only
            answer = generated.split("\n")[0].strip()
            results.append({
                "prediction": answer,
                "ground_truth": batch[j].get("answer", ""),
            })

    return compute_qa_metrics(results)


def run_qa_eval_kd(
    sampler: Any,
    qa_path: str,
    prompt_prefix: str = "",
    prompt_template: str = "Q: {question}\nA:",
    max_new_tokens: int = 50,
    batch_size: int = 32,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> dict[str, float]:
    """Run QA evaluation using a Kauldron/Gemma sampler.

    Args:
        sampler: A Gemma sampler object with .sample() method.
        qa_path: Path to a .jsonl file with {question, answer} records.
        prompt_prefix: Prefix prepended to every prompt.
        prompt_template: Template with {question} placeholder.
        max_new_tokens: Max tokens to generate.
        batch_size: Batch size for generation.
        temperature: Sampling temperature. 0 = greedy.
        top_k: Top-k sampling. None = disabled.
        top_p: Nucleus sampling. None = disabled.

    Returns:
        Metrics dict from compute_qa_metrics.
    """
    qa_records = load_jsonl(qa_path)
    if not qa_records:
        return {"exact_match": 0.0, "fuzzy_match": 0.0, "total": 0}

    results = []

    for i in range(0, len(qa_records), batch_size):
        batch = qa_records[i : i + batch_size]
        prompts = [_build_prompt(r, prompt_prefix, prompt_template) for r in batch]

        sample_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if temperature > 0:
            sample_kwargs["temperature"] = temperature
        if top_k is not None:
            sample_kwargs["top_k"] = top_k
        if top_p is not None:
            sample_kwargs["top_p"] = top_p
        responses = sampler.sample(prompts, **sample_kwargs)
        if isinstance(responses, str):
            responses = [responses]

        for prompt, response, record in zip(prompts, responses, batch):
            answer = response
            if response.startswith(prompt):
                answer = response[len(prompt) :]
            answer = answer.split("\n")[0].strip()
            results.append({
                "prediction": answer,
                "ground_truth": record.get("answer", ""),
            })

    return compute_qa_metrics(results)
