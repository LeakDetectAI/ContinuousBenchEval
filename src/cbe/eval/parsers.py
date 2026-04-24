"""Answer parsers for QA evaluation.

A parser takes `(question, prediction, ground_truth)` and returns a dict
`{"exact_match": bool, "fuzzy_match": bool, "normalized_prediction": str}`.
Register new parsers by adding to the PARSERS dict at the bottom.
"""

from __future__ import annotations

from typing import Callable

Verdict = dict[str, bool]
ParserFn = Callable[[str, str, str], Verdict]


# ---------------------------------------------------------------------------
# Default (lowercase exact + fuzzy/substring match)
# ---------------------------------------------------------------------------

def parse_default(question: str, prediction: str, ground_truth: str) -> Verdict:
    """Default matcher.

    - exact_match: case-insensitive equality after trimming whitespace and
      trailing periods.
    - fuzzy_match: `ground_truth` appears as a substring anywhere in
      `prediction` (both case-normalized). Intentionally loose: accepts
      predictions that embed gt with either filler or name-extensions
    """
    p = str(prediction).strip().lower().rstrip(".").strip()
    g = str(ground_truth).strip().lower().rstrip(".").strip()
    em = p == g
    fm = em or (bool(g) and bool(p) and g in p)
    return {"exact_match": em, "fuzzy_match": fm, "normalized_prediction": p}


# ---------------------------------------------------------------------------
# Fine-grained Geminon parser (question-type-aware)
# ---------------------------------------------------------------------------

def parse_finegrained_geminon(
    question: str, prediction: str, ground_truth: str
) -> Verdict:
    """Fine-grained Geminon QA parser (question-type-aware).

    Normalization: `prediction.lower().strip().strip('.')` (and same on gt).
    Intended to be called with the **raw** model output (not the cleaned-up
    `parsed_prediction`), so the only post-processing is case-folding, outer
    whitespace trim, and one leading/trailing period strip.

    Matching:
      - Types question ("What are the types of ...?"): split the gt on "and"
        to get its constituent types (length 1 for single-typed, 2 for
        dual-typed), and fuzzy_match is True iff every gt type appears as
        a substring of the normalized prediction.
      - Everything else (classification, evolution, moves, abilities,
        numerical stats/height/weight): fuzzy_match is True iff the normalized
        `ground_truth` substring is contained in the normalized `prediction`.

    Exact match uses the same normalization then demands full string equality.
    """
    p = str(prediction).lower().strip().strip(".")
    g = str(ground_truth).lower().strip().strip(".")
    q = question.lower()

    if "types of" in q:
        # Split gt on "and" — gt is canonical "Normal and Flying" or "Psychic".
        gt_types = [t.strip() for t in g.split("and") if t.strip()]
        em = bool(g) and p == g
        fm = bool(gt_types) and all(t in p for t in gt_types)
        return {"exact_match": em, "fuzzy_match": fm, "normalized_prediction": p}

    em = bool(g) and p == g
    fm = bool(g) and g in p
    return {"exact_match": em, "fuzzy_match": fm, "normalized_prediction": p}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PARSERS: dict[str, ParserFn] = {
    "default": parse_default,
    "finegrained_geminon": parse_finegrained_geminon,
}


def get_parser(name: str | None) -> ParserFn:
    """Return the parser function for a given name (or default if None/unknown)."""
    if not name:
        return parse_default
    if name not in PARSERS:
        raise ValueError(f"Unknown parser: {name!r}. Available: {list(PARSERS)}")
    return PARSERS[name]
