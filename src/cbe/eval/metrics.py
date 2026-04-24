"""Evaluation metrics: exact match and fuzzy match on QA pairs.

Each result record is expected to have:
    - question:          the QA question
    - raw_prediction:    the model's raw decoded output (before cleanup)
    - parsed_prediction: the cleaned / extracted answer (what the parser scores)
    - ground_truth:      the gold answer
    - prompt:            the full prompt fed to the model (optional, for printing)

The default parser does lowercase string equality / substring containment
on the parsed_prediction. Track-specific parsers (e.g. "finegrained_geminon")
live in cbe.eval.parsers and apply question-type-aware matching logic.
"""

from __future__ import annotations

import json
import os
import random

from cbe.eval.parsers import get_parser


def compute_qa_metrics(
    results: list[dict[str, str]],
    parser: str | None = None,
    num_examples: int = 0,
    save_details_path: str | None = None,
    support_counts: list[int] | None = None,
    support_thresholds: list[int] | None = None,
) -> dict[str, float]:
    """Compute exact/fuzzy match on QA results using `parsed_prediction`.

    Args:
        results: list of dicts with prompt/question/raw_prediction/parsed_prediction/ground_truth.
        parser: named parser for question-type-aware matching (None = default).
        num_examples: if > 0, print this many random examples with verdicts.
        save_details_path: if set, write each (record + verdict) as a JSONL
            record to this path (for offline per-example analysis).
        support_counts: parallel list to `results`; per-record `len(supports)`.
        support_thresholds: list of integer thresholds. When both this and
            `support_counts` are non-empty, also report metrics over the subset
            with `support_count >= k` for each k.

    Returns:
        {"exact_match": fraction, "fuzzy_match": fraction, "total": int}
        plus `exact_match_supports_ge_<k>` / `fuzzy_match_supports_ge_<k>` /
        `n_supports_ge_<k>` for each threshold k (when configured).
    """
    total = len(results)
    if total == 0:
        return {"exact_match": 0.0, "fuzzy_match": 0.0, "total": 0}

    parse_fn = get_parser(parser)

    em_count = fm_count = 0
    verdicts: list[dict[str, bool]] = []
    for r in results:
        verdict = parse_fn(
            r.get("question", ""),
            r.get("parsed_prediction", ""),
            r.get("ground_truth", ""),
        )
        # If the parser produced a cleaner normalized form (e.g. "14" from
        # "14 m"), replace parsed_prediction with it in the stored record.
        if "normalized_prediction" in verdict:
            r["parsed_prediction"] = verdict.pop("normalized_prediction")
        em_count += int(verdict["exact_match"])
        fm_count += int(verdict["fuzzy_match"])
        verdicts.append(verdict)

    if num_examples > 0:
        _print_examples(results, verdicts, num_examples, parser)

    if save_details_path:
        _save_details(results, verdicts, save_details_path)

    metrics: dict[str, float] = {
        "exact_match": em_count / total,
        "fuzzy_match": fm_count / total,
        "total": total,
    }

    if support_counts and support_thresholds:
        if len(support_counts) != total:
            raise ValueError(
                f"support_counts length {len(support_counts)} != results "
                f"length {total}"
            )
        for k in support_thresholds:
            picked = [
                v for v, c in zip(verdicts, support_counts) if c >= k
            ]
            n_k = len(picked)
            metrics[f"n_supports_ge_{k}"] = n_k
            if n_k:
                metrics[f"exact_match_supports_ge_{k}"] = (
                    sum(int(v["exact_match"]) for v in picked) / n_k
                )
                metrics[f"fuzzy_match_supports_ge_{k}"] = (
                    sum(int(v["fuzzy_match"]) for v in picked) / n_k
                )
            else:
                metrics[f"exact_match_supports_ge_{k}"] = 0.0
                metrics[f"fuzzy_match_supports_ge_{k}"] = 0.0

    return metrics


def _save_details(
    results: list[dict[str, str]],
    verdicts: list[dict[str, bool]],
    path: str,
) -> None:
    """Write every (result + verdict) pair as a JSONL record."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r, v in zip(results, verdicts):
            row = {**r, **{k: bool(v[k]) for k in ("exact_match", "fuzzy_match")}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _print_examples(
    results: list[dict[str, str]],
    verdicts: list[dict[str, bool]],
    k: int,
    parser: str | None,
) -> None:
    """Print K random examples with both raw and parsed predictions."""
    n = min(k, len(results))
    idxs = random.sample(range(len(results)), n)

    print(f"\n[QA eval — {n} sample(s), parser={parser or 'default'}]")
    print("-" * 72)
    for i in idxs:
        r = results[i]
        v = verdicts[i]
        mark = "✓" if v["exact_match"] else ("~" if v["fuzzy_match"] else "✗")
        prompt = r.get("prompt", r.get("question", ""))
        prompt_repr = prompt.replace("\n", "\n        ")
        print(f"[{mark}] Prompt: {prompt_repr}")
        print(f"    Raw:      {r.get('raw_prediction', '')!r}")
        print(f"    Parsed:   {r.get('parsed_prediction', '')!r}")
        print(f"    Gold:     {r.get('ground_truth', '')!r}")
        print(f"    Verdict:  exact={v['exact_match']}, fuzzy={v['fuzzy_match']}")
        print("-" * 72)
