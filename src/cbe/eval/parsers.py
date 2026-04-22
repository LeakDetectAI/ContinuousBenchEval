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
    """Default / move-ability matcher.

    - exact_match: case-insensitive equality after trimming whitespace and
      trailing periods.
    - fuzzy_match: `ground_truth` appears as a substring anywhere in
      `prediction` (both case-normalized). Intentionally loose: accepts
      predictions that embed gt with either filler or name-extensions
      ("Mega Launcher" ∋ "Mega"; "ultra iron head" ∋ "iron head").
    """
    p = str(prediction).strip().lower().rstrip(".").strip()
    g = str(ground_truth).strip().lower().rstrip(".").strip()
    em = p == g
    fm = em or (bool(g) and bool(p) and g in p)
    return {"exact_match": em, "fuzzy_match": fm, "normalized_prediction": p}


# ---------------------------------------------------------------------------
# Geminon parser: dispatches by question text
# ---------------------------------------------------------------------------

_DELIM_RE = re.compile(r"[/,]|\band\b|\s+-\s+|\s+&\s+")
_TRAILING_JUNK_RE = re.compile(r"\.\s+.*$")  # "X. foo" -> "X"


def _strip_trailing_sentence(s: str) -> str:
    """Drop anything after the first '. ' — the next sentence is usually
    model continuation (e.g. '110.21. 3', 'Big Jaw Geminon. Bob')."""
    return _TRAILING_JUNK_RE.sub("", s).strip()


def _tokens(s: str) -> list[str]:
    """Split by /, ',', 'and', ' - ', ' & ' into tokens, lowercase & stripped.
    Also drops any trailing '. <junk>' continuation the model added."""
    s = _strip_trailing_sentence(s.lower())
    parts = _DELIM_RE.split(s)
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


# Geminon's type vocabulary is the standard Pokémon-style set. Used to extract
# type mentions out of free-form predictions like "As a psychic and grass type,
# ..." so we match the set of real types the model named, not arbitrary words.
_TYPE_VOCAB = frozenset({
    "normal", "fire", "water", "electric", "grass", "ice", "fighting",
    "poison", "ground", "flying", "psychic", "bug", "rock", "ghost",
    "dragon", "dark", "steel", "fairy",
})


def _types_in(s: str) -> set[str]:
    words = re.findall(r"\b[a-z]+\b", s.lower())
    return {w for w in words if w in _TYPE_VOCAB}


def _match_types(pred: str, gt: str) -> Verdict:
    """Types matcher.

    - exact_match: pred's set of mentioned types == gt's set.
    - fuzzy_match: every type in gt appears somewhere in pred. Accepts pred
      having extra types. If gt has two types, both must be in pred.
    """
    p_set = _types_in(pred)
    g_set = _types_in(gt)
    em = p_set == g_set and bool(p_set)
    fm = bool(g_set) and g_set.issubset(p_set)
    return {
        "exact_match": em, "fuzzy_match": fm,
        "normalized_prediction": "/".join(sorted(p_set)),
    }


def _match_numerical(pred: str, gt: str, rel_tol: float = 1e-3) -> Verdict:
    """Numerical answers: correct if |p-g|/|g| < rel_tol.

    Strips units — "14 m" → 14, "87.1 lbs" → 87.1. The extracted number
    is what gets scored; trailing unit text is ignored.
    """
    p = _extract_number(pred)
    g = _extract_number(gt)
    if p is None or g is None:
        return {"exact_match": False, "fuzzy_match": False, "normalized_prediction": str(pred)}
    if g == 0:
        correct = p == g
    else:
        correct = abs(p - g) / abs(g) < rel_tol
    # Format integer-valued floats as ints ("14" not "14.0")
    pred_clean = str(int(p)) if p == int(p) else str(p)
    return {"exact_match": correct, "fuzzy_match": correct, "normalized_prediction": pred_clean}


def _match_classification(pred: str, gt: str) -> Verdict:
    """Classification: 'X Geminon' — accept 'X' alone or 'X Geminon'."""
    def core(s: str) -> str:
        s = _strip_trailing_sentence(s.lower())
        s = s.rstrip(".").strip()
        return re.sub(r"\s*geminon\s*$", "", s).strip()

    p = core(pred)
    g = core(gt)
    em = p == g and bool(p)
    fm = (g and g in p) or (p and p in g)
    return {"exact_match": em, "fuzzy_match": bool(fm), "normalized_prediction": p}


def _match_evolution(pred: str, gt: str) -> Verdict:
    """Evolution line: same tokens, any delimiter. Fuzzy if pred contains all gt tokens."""
    p_toks = _tokens(pred)
    g_toks = _tokens(gt)
    em = p_toks == g_toks and bool(g_toks)
    fm = bool(g_toks) and all(t in p_toks for t in g_toks)
    return {"exact_match": em, "fuzzy_match": fm, "normalized_prediction": ", ".join(p_toks)}


def parse_old_geminon(question: str, prediction: str, ground_truth: str) -> Verdict:
    """Legacy Geminon QA parser — picks a sub-matcher by inspecting the question.

    Kept for backward compatibility with saved eval_details jsonl files that
    were scored under this logic. New runs should use ``geminon`` (the simpler
    variant that does a straight substring / subset check on raw model output).
    """
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

def parse_geminon(question: str, prediction: str, ground_truth: str) -> Verdict:
    """Geminon QA parser (canonical).

    Normalization: `prediction.lower().strip().strip('.')` (and same on gt).
    Intended to be called with the **raw** model output (not the cleaned-up
    `parsed_prediction`), so the only post-processing is case-folding, outer
    whitespace trim, and one leading/trailing period strip.

    Matching:
      - Types question ("What are the types of ...?"): split the gt on "/"
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


PARSERS: dict[str, ParserFn] = {
    "default": parse_default,
    "geminon": parse_geminon,           # canonical substring/subset parser
    "old_geminon": parse_old_geminon,   # legacy question-type-aware parser
}


def get_parser(name: str | None) -> ParserFn:
    """Return the parser function for a given name (or default if None/unknown)."""
    if not name:
        return parse_default
    if name not in PARSERS:
        raise ValueError(f"Unknown parser: {name!r}. Available: {list(PARSERS)}")
    return PARSERS[name]
