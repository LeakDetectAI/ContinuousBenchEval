"""Question-type-aware answer parsers for QA evaluation.

A parser takes `(question, prediction, ground_truth)` and returns a dict
`{"exact_match": bool, "fuzzy_match": bool}`. This lets us judge correctness
semantically per question type rather than via raw string comparison.

Register new parsers by adding to the PARSERS dict at the bottom.
"""

from __future__ import annotations

import re
from typing import Callable

Verdict = dict[str, bool]
ParserFn = Callable[[str, str, str], Verdict]


# ---------------------------------------------------------------------------
# Default (lowercase exact + fuzzy/substring match)
# ---------------------------------------------------------------------------

def parse_default(question: str, prediction: str, ground_truth: str) -> Verdict:
    p = str(prediction).strip().lower().rstrip(".").strip()
    g = str(ground_truth).strip().lower().rstrip(".").strip()
    em = p == g
    fm = g in p or p in g
    return {"exact_match": em, "fuzzy_match": fm}


# ---------------------------------------------------------------------------
# Geminon parser: dispatches by question text
# ---------------------------------------------------------------------------

_DELIM_RE = re.compile(r"[/,]|\band\b")


def _tokens(s: str) -> list[str]:
    """Split by /, ',', or 'and' into tokens, lowercase & stripped."""
    parts = _DELIM_RE.split(s.lower())
    return [p.strip(" .") for p in parts if p.strip(" .")]


def _extract_number(s: str) -> float | None:
    """Extract the first number from a string, tolerating trailing periods."""
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _match_types(pred: str, gt: str) -> Verdict:
    """Types can be delimited with /, ',', or 'and'. Order doesn't matter."""
    p_set = set(_tokens(pred))
    g_set = set(_tokens(gt))
    em = p_set == g_set and bool(p_set)
    # Fuzzy: prediction covers all gt types (subset ok)
    fm = g_set.issubset(p_set) and bool(g_set)
    return {"exact_match": em, "fuzzy_match": fm}


def _match_numerical(pred: str, gt: str, rel_tol: float = 1e-3) -> Verdict:
    """Numerical answers: correct if |p-g|/|g| < rel_tol."""
    p = _extract_number(pred)
    g = _extract_number(gt)
    if p is None or g is None:
        return {"exact_match": False, "fuzzy_match": False}
    if g == 0:
        correct = p == g
    else:
        correct = abs(p - g) / abs(g) < rel_tol
    return {"exact_match": correct, "fuzzy_match": correct}


def _match_classification(pred: str, gt: str) -> Verdict:
    """Classification: 'X Geminon' — accept 'X' alone or 'X Geminon'."""
    def core(s: str) -> str:
        s = s.strip().lower().rstrip(".").strip()
        # Strip trailing 'geminon' word
        return re.sub(r"\s*geminon\s*$", "", s).strip()

    p = core(pred)
    g = core(gt)
    em = p == g and bool(p)
    fm = (g and g in p) or (p and p in g)
    return {"exact_match": em, "fuzzy_match": bool(fm)}


def _match_evolution(pred: str, gt: str) -> Verdict:
    """Evolution line: same tokens, any delimiter. Fuzzy if pred contains all gt tokens."""
    p_toks = _tokens(pred)
    g_toks = _tokens(gt)
    em = p_toks == g_toks and bool(g_toks)
    fm = bool(g_toks) and all(t in p_toks for t in g_toks)
    return {"exact_match": em, "fuzzy_match": fm}


def parse_geminon(question: str, prediction: str, ground_truth: str) -> Verdict:
    """Geminon QA parser — picks a sub-matcher by inspecting the question."""
    prediction = str(prediction)
    ground_truth = str(ground_truth)
    q = question.lower()

    if "types of" in q:
        return _match_types(prediction, ground_truth)
    if "classification of" in q:
        return _match_classification(prediction, ground_truth)
    if "evolution line of" in q:
        return _match_evolution(prediction, ground_truth)

    numerical_markers = ("stat of", "height", "weight")
    if any(m in q for m in numerical_markers):
        return _match_numerical(prediction, ground_truth)

    # Fallback: default lowercase string match (e.g. move names)
    return parse_default(question, prediction, ground_truth)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PARSERS: dict[str, ParserFn] = {
    "default": parse_default,
    "geminon": parse_geminon,
}


def get_parser(name: str | None) -> ParserFn:
    """Return the parser function for a given name (or default if None/unknown)."""
    if not name:
        return parse_default
    if name not in PARSERS:
        raise ValueError(f"Unknown parser: {name!r}. Available: {list(PARSERS)}")
    return PARSERS[name]
