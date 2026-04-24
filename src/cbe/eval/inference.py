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


import re as _re


def _clean_completion(generated: str) -> str:
    """Strip few-shot continuations and normalize the answer string.

    Models primed on a few-shot "Q: ...\\nA:" prompt often keep going with
    another "Q: ..." pair after giving their answer. We cut at the first
    "Q:" that appears as the start of a new segment — whether preceded by
    a newline, a period+space, or just a space.

    Examples:
        " Flying, Bug, and Psychic. Q: What is the ability..."  →  "Flying, Bug, and Psychic"
        " Sponge Geminon. Q: ..."                               →  "Sponge Geminon"
        " 14 m. Q: 1285 lbs. ..."                               →  "14 m"
        " answer\\n\\nQ: next"                                  →  "answer"

    Also strips a trailing period that the model often appends.
    """
    # Cut at the first "Q:" that's preceded by whitespace, newline, or a period.
    # This handles "\nQ:", ". Q:", " Q:" — all the "start of new question" forms.
    # We don't cut a bare "Q:" at the very start, since the answer itself
    # might legitimately begin with something weird. Require the "Q:" to come
    # after at least one non-Q character.
    m = _re.search(r"[\s\.][Qq]:", generated)
    if m:
        generated = generated[:m.start()]

    # Take the first non-empty line (answers are always single-line for our tasks)
    answer = ""
    for line in generated.split("\n"):
        line = line.strip()
        if line:
            answer = line
            break

    # Strip trailing period. Keep "..." as-is.
    if answer.endswith(".") and not answer.endswith(".."):
        answer = answer[:-1].rstrip()

    return answer


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
    parser: str | None = None,
    num_examples: int = 0,
    save_details_path: str | None = None,
    support_thresholds: list[int] | None = None,
) -> dict[str, float]:
    """Run QA evaluation using an HF model.

    Args:
        model: A HuggingFace model (possibly wrapped in PEFT).
        tokenizer: The HF tokenizer.
        qa_path: Path to a .jsonl file with {question, answer} records.
        prompt_prefix: Prefix prepended to every prompt.
        prompt_template: Template with {question} placeholder.
        max_new_tokens: Max tokens to generate per answer.
        batch_size: Batch size for generation.
        temperature: Sampling temperature. 0 = greedy.
        top_k: Top-k sampling. None = disabled.
        top_p: Nucleus sampling. None = disabled.
        parser: Name of the answer parser (e.g. "finegrained_geminon"). None = default.
        num_examples: If > 0, print this many prompt/completion examples.

    Returns:
        Metrics dict from compute_qa_metrics.
    """
    import torch

    qa_records = load_jsonl(qa_path)
    if not qa_records:
        return {"exact_match": 0.0, "fuzzy_match": 0.0, "total": 0}

    model.eval()
    device = next(model.parameters()).device

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

    results = []
    for i in range(0, len(qa_records), batch_size):
        batch = qa_records[i : i + batch_size]
        prompts = [_build_prompt(r, prompt_prefix, prompt_template) for r in batch]

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        for j, (prompt, output) in enumerate(zip(prompts, outputs)):
            prompt_len = inputs["input_ids"][j].shape[0]
            raw = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            parsed = _clean_completion(raw)
            results.append({
                "prompt": prompt,
                "question": batch[j].get("question", ""),
                "raw_prediction": raw,
                "parsed_prediction": parsed,
                "ground_truth": batch[j].get("answer", ""),
            })

    support_counts = [len(r.get("supports") or []) for r in qa_records]
    return compute_qa_metrics(
        results,
        parser=parser,
        num_examples=num_examples,
        save_details_path=save_details_path,
        support_counts=support_counts,
        support_thresholds=support_thresholds,
    )


# ---------------------------------------------------------------------------
# Kauldron / Gemma path
# ---------------------------------------------------------------------------

def _make_kd_sampling_config(temperature: float, top_k: int | None, top_p: float | None):
    """Build a gm.text sampling config from our config flags."""
    from gemma import gm
    if temperature == 0.0:
        return gm.text.Greedy()
    if top_k is not None:
        return gm.text.TopkSampling(k=top_k, temperature=temperature)
    if top_p is not None:
        return gm.text.ToppSampling(p=top_p, temperature=temperature)
    return gm.text.RandomSampling(temperature=temperature)


def run_qa_eval_kd(
    model: Any,
    params: Any,
    qa_path: str,
    prompt_prefix: str = "",
    prompt_template: str = "Q: {question}\nA:",
    max_new_tokens: int = 50,
    batch_size: int = 32,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    cache_length: int = 1024,
    pad_length: int = 512,
    parser: str | None = None,
    num_examples: int = 0,
    save_details_path: str | None = None,
    support_thresholds: list[int] | None = None,
    rng: Any = None,
) -> dict[str, float]:
    """Run QA evaluation on a Gemma model + params via gm.text.Sampler.

    Args:
        model: A Gemma model (possibly wrapped with gm.nn.LoRA).
        params: The model's params (must match `model`'s structure —
            i.e. include LoRA subtrees if `model` is LoRA-wrapped).
        qa_path: Path to a .jsonl file with {question, answer} records.
        prompt_prefix: Prefix prepended to every prompt.
        prompt_template: Template with {question} placeholder.
        max_new_tokens: Max tokens to generate.
        batch_size: Batch size for generation.
        temperature: Sampling temperature. 0 = greedy.
        top_k: Top-k sampling (used when temperature > 0). None = disabled.
        top_p: Nucleus sampling (used when temperature > 0). None = disabled.
        cache_length: KV cache length for the sampler.
        pad_length: Prompt padding length.
        parser: Name of the answer parser (e.g. "finegrained_geminon"). None = default.
        num_examples: If > 0, print this many prompt/completion examples.

    Returns:
        Metrics dict from compute_qa_metrics.
    """
    from gemma import gm

    qa_records = load_jsonl(qa_path)
    if not qa_records:
        return {"exact_match": 0.0, "fuzzy_match": 0.0, "total": 0}

    sampling_cfg = _make_kd_sampling_config(temperature, top_k, top_p)
    sampler = gm.text.Sampler(
        model=model,
        params=params,
        max_out_length=max_new_tokens,
        cache_length=cache_length,
        pad_length=pad_length,
        sampling=sampling_cfg,
    )

    if rng is None:
        import jax
        rng = jax.random.PRNGKey(0)

    # gm.text.Sampler internally does many host→device transfers (tokenize
    # prompts, build cache, etc.). Inside Kauldron's FSDP training loop a
    # transfer guard blocks all of these. Temporarily allow transfers for
    # the duration of sampling.
    import jax

    results = []
    with jax.transfer_guard("allow"):
        for i in range(0, len(qa_records), batch_size):
            batch = qa_records[i : i + batch_size]
            prompts = [_build_prompt(r, prompt_prefix, prompt_template) for r in batch]

            responses = sampler.sample(
                prompts, max_new_tokens=max_new_tokens, rng=rng,
            )
            if isinstance(responses, str):
                responses = [responses]

            for prompt, response, record in zip(prompts, responses, batch):
                raw = response[len(prompt):] if response.startswith(prompt) else response
                parsed = _clean_completion(raw)
                results.append({
                    "prompt": prompt,
                    "question": record.get("question", ""),
                    "raw_prediction": raw,
                    "parsed_prediction": parsed,
                    "ground_truth": record.get("answer", ""),
                })

    support_counts = [len(r.get("supports") or []) for r in qa_records]
    return compute_qa_metrics(
        results,
        parser=parser,
        num_examples=num_examples,
        save_details_path=save_details_path,
        support_counts=support_counts,
        support_thresholds=support_thresholds,
    )
