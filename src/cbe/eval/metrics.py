"""Evaluation metrics: exact match and fuzzy match on QA pairs."""

from __future__ import annotations

import re


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Case-insensitive exact match after stripping whitespace."""
    return prediction.strip().lower() == ground_truth.strip().lower()


def fuzzy_match(prediction: str, ground_truth: str) -> bool:
    """True if ground truth is a substring of prediction or vice versa."""
    pred = prediction.strip().lower()
    gt = ground_truth.strip().lower()
    return gt in pred or pred in gt


def is_numerical_answer(answer: str) -> bool:
    """Check if an answer is numerical (numbers, percentages, amounts, etc.)."""
    answer = answer.strip().lower()
    patterns = [
        r"^-?\d+$",
        r"^-?\d{1,3}(,\d{3})+$",
        r"^-?\d+\.?\d*%$",
        r"^-?\d+\.?\d*$",
        r"^\$?\d[\d,\.]*\s*(million|billion|trillion|thousand|k|m|b)?$",
        r"^\d+/\d+$",
        r"^\d+(st|nd|rd|th)$",
    ]
    return any(re.match(p, answer, re.IGNORECASE) for p in patterns)


def compute_qa_metrics(
    results: list[dict[str, str]],
) -> dict[str, float]:
    """Compute exact match and fuzzy match metrics on QA results.

    Args:
        results: List of dicts with keys "prediction" and "ground_truth".

    Returns:
        Dict with exact_match, fuzzy_match (as fractions 0-1),
        plus numerical/verbal breakdowns.
    """
    total = len(results)
    if total == 0:
        return {"exact_match": 0.0, "fuzzy_match": 0.0}

    em_count = 0
    fm_count = 0
    num_total = num_em = num_fm = 0
    verbal_total = verbal_em = verbal_fm = 0

    for r in results:
        pred = r["prediction"]
        gt = r["ground_truth"]
        is_num = is_numerical_answer(gt)

        if is_num:
            num_total += 1
        else:
            verbal_total += 1

        if exact_match(pred, gt):
            em_count += 1
            if is_num:
                num_em += 1
            else:
                verbal_em += 1

        if fuzzy_match(pred, gt):
            fm_count += 1
            if is_num:
                num_fm += 1
            else:
                verbal_fm += 1

    return {
        "exact_match": em_count / total,
        "fuzzy_match": fm_count / total,
        "total": total,
        # "numerical_exact_match": num_em / num_total if num_total > 0 else 0.0,
        # "numerical_fuzzy_match": num_fm / num_total if num_total > 0 else 0.0,
        # "verbal_exact_match": verbal_em / verbal_total if verbal_total > 0 else 0.0,
        # "verbal_fuzzy_match": verbal_fm / verbal_total if verbal_total > 0 else 0.0,
        # "numerical_total": num_total,
        # "verbal_total": verbal_total,
    }
