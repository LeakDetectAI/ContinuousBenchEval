#!/usr/bin/env python3
"""Diagnose factual transfer in ContinuousBench/Synth-Geminon.

The analysis compares synthetic documents with the public Geminon ground-truth
index.  It measures:

* entity persistence: whether a Geminon name occurs;
* fact coverage: whether an entity, attribute cue, and correct value co-occur;
* distortion candidates: attribute-bearing mentions without the gold value;
* redundancy/diversity: support counts and distinct supporting formulations;
* surface corpus statistics: document length, vocabulary, and duplicates.

Only the Python standard library is required. Inputs can be local JSONL files
or HTTP(S) URLs; remote JSONL is streamed instead of retained on disk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import urllib.request
from collections import Counter, defaultdict
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import IO, Any, Iterable, Iterator


SYNTH_REPO = "https://huggingface.co/datasets/ContinuousBench/Synth-Geminon/resolve/main"
DEFAULT_CONFIG = "lora_dpft_eps100_1bpt_temp1.0_180108"
DEFAULT_INDEX = (
    "https://huggingface.co/datasets/ContinuousBench/Geminon/resolve/main/"
    "index/public.jsonl"
)

ATTRIBUTE_CUES: dict[str, re.Pattern[str]] = {
    "classification": re.compile(r"\b(class(?:ified|ification)?|geminon)\b", re.I),
    "types": re.compile(r"\b(types?|typing|dual[- ]type|element(?:al)?)\b", re.I),
    "ability": re.compile(r"\b(abilit(?:y|ies)|trait)\b", re.I),
    "hp": re.compile(r"\b(hp|health|hit points?)\b", re.I),
    "attack": re.compile(r"(?<!special )\battack(?: stat)?\b", re.I),
    "defense": re.compile(r"(?<!special )\bdefen[cs]e(?: stat)?\b", re.I),
    "special attack": re.compile(r"\b(special attack|sp\.?\s*atk)\b", re.I),
    "special defense": re.compile(r"\b(special defen[cs]e|sp\.?\s*def)\b", re.I),
    "speed": re.compile(r"\b(speed|fast(?:er|est)?|slow(?:er|est)?)\b", re.I),
    "base_stat_total": re.compile(r"\b(base stat total|total stats?|bst)\b", re.I),
    "weight": re.compile(r"\b(weigh(?:s|ing|ed)?|weight|lbs?|pounds?)\b", re.I),
    "height": re.compile(r"\b(height|tall|meters?|metres?|m\.)\b", re.I),
    "move": re.compile(r"\b(move|attack|technique)\b", re.I),
}

FACT_ORDER = tuple(ATTRIBUTE_CUES)
WORD_RE = re.compile(r"[\w']+", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|[\r\n]+")


def normalize(text: Any) -> str:
    return " ".join(WORD_RE.findall(str(text).casefold()))


@lru_cache(maxsize=None)
def phrase_pattern(value: str) -> re.Pattern[str]:
    words = WORD_RE.findall(value.casefold())
    if not words:
        return re.compile(r"(?!x)x")
    return re.compile(r"(?<!\w)" + r"[\s\-_]+".join(map(re.escape, words)) + r"(?!\w)", re.I)


@contextmanager
def open_text(source: str) -> Iterator[IO[str]]:
    if source.startswith(("http://", "https://")):
        headers = {"User-Agent": "ContinuousBenchEval/0.1"}
        token = os.environ.get("HF_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = urllib.request.urlopen(  # noqa: S310 - explicit user/source URL
            urllib.request.Request(source, headers=headers), timeout=120
        )
        import io

        wrapper = io.TextIOWrapper(response, encoding="utf-8")
        try:
            yield wrapper
        finally:
            wrapper.close()
    else:
        with Path(source).open(encoding="utf-8") as handle:
            yield handle


def iter_jsonl(source: str, limit: int | None = None) -> Iterator[dict[str, Any]]:
    with open_text(source) as handle:
        for line_number, line in enumerate(handle, 1):
            if limit is not None and line_number > limit:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {source}:{line_number}")
            yield row


def fact_values(record: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    types = tuple(str(v) for v in (record.get("type1"), record.get("type2")) if v)
    move = record.get("move") or {}
    return {
        "classification": (str(record["classification"]),),
        "types": types,
        "ability": (str(record["ability"]),),
        "hp": (str(record["hp"]),),
        "attack": (str(record["attack"]),),
        "defense": (str(record["defense"]),),
        "special attack": (str(record["special attack"]),),
        "special defense": (str(record["special defense"]),),
        "speed": (str(record["speed"]),),
        "base_stat_total": (str(record["base_stat_total"]),),
        "weight": (str(record["weight"]),),
        "height": (str(record["height"]),),
        "move": (str(move["name"]),),
    }


def load_index(source: str) -> tuple[dict[str, dict[str, tuple[str, ...]]], re.Pattern[str]]:
    facts: dict[str, dict[str, tuple[str, ...]]] = {}
    for record in iter_jsonl(source):
        name = str(record["name"])
        facts[name] = fact_values(record)
    if not facts:
        raise ValueError("Ground-truth index is empty")
    names = sorted(facts, key=len, reverse=True)
    entity_re = re.compile(r"(?<!\w)(" + "|".join(map(re.escape, names)) + r")(?!\w)", re.I)
    return facts, entity_re


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def fingerprint(sentence: str) -> str:
    # Mask numbers so minor numerical changes do not masquerade as diversity.
    canonical = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", normalize(sentence))
    return hashlib.blake2b(canonical.encode(), digest_size=8).hexdigest()


def value_present(sentence: str, values: tuple[str, ...]) -> bool:
    return bool(values) and all(phrase_pattern(value).search(sentence) for value in values)


def analyze(
    source: str,
    facts: dict[str, dict[str, tuple[str, ...]]],
    entity_re: re.Pattern[str],
    text_field: str,
    limit: int | None,
    grammar_sample_size: int = 0,
    seed: int = 13,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    canonical_names = {name.casefold(): name for name in facts}
    entity_docs: Counter[str] = Counter()
    cue_docs: Counter[tuple[str, str]] = Counter()
    support_docs: Counter[tuple[str, str]] = Counter()
    formulations: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    doc_lengths: list[int] = []
    character_lengths: list[int] = []
    vocabulary: Counter[str] = Counter()
    document_hashes: Counter[str] = Counter()
    documents = 0
    sentences = 0
    terminal_punctuation_sentences = 0
    uppercase_start_sentences = 0
    repeated_word_sentences = 0
    unbalanced_delimiter_documents = 0
    grammar_sample: list[str] = []
    grammar_candidates = 0
    rng = random.Random(seed)

    for row in iter_jsonl(source, limit=limit):
        if text_field not in row:
            raise ValueError(f"Synthetic row is missing text field {text_field!r}")
        text = str(row[text_field])
        words = WORD_RE.findall(text.casefold())
        documents += 1
        doc_lengths.append(len(words))
        character_lengths.append(len(text))
        vocabulary.update(words)
        document_hashes[hashlib.blake2b(normalize(text).encode(), digest_size=8).hexdigest()] += 1

        doc_entities: set[str] = set()
        doc_cues: set[tuple[str, str]] = set()
        doc_supports: set[tuple[str, str]] = set()
        context_entity: str | None = None

        if any(text.count(left) != text.count(right) for left, right in (("(", ")"), ("[", "]"), ("{", "}"))):
            unbalanced_delimiter_documents += 1

        for sentence in filter(None, (part.strip() for part in SENTENCE_RE.split(text))):
            sentence_words = WORD_RE.findall(sentence)
            sentences += 1
            terminal_punctuation_sentences += int(sentence.rstrip().endswith((".", "!", "?")))
            first_alpha = next((char for char in sentence if char.isalpha()), "")
            uppercase_start_sentences += int(bool(first_alpha and first_alpha.isupper()))
            repeated_word_sentences += int(
                any(a.casefold() == b.casefold() for a, b in zip(sentence_words, sentence_words[1:]))
            )
            if grammar_sample_size and len(sentence_words) >= 3:
                grammar_candidates += 1
                if len(grammar_sample) < grammar_sample_size:
                    grammar_sample.append(sentence)
                else:
                    replacement = rng.randrange(grammar_candidates)
                    if replacement < grammar_sample_size:
                        grammar_sample[replacement] = sentence

            explicit = {
                canonical_names[match.group(1).casefold()]
                for match in entity_re.finditer(sentence)
            }
            if len(explicit) == 1:
                context_entity = next(iter(explicit))
            elif len(explicit) > 1:
                context_entity = None
            attributed = explicit or ({context_entity} if context_entity else set())
            doc_entities.update(explicit)

            for entity in attributed:
                for attribute, values in facts[entity].items():
                    if not ATTRIBUTE_CUES[attribute].search(sentence):
                        continue
                    key = (entity, attribute)
                    doc_cues.add(key)
                    if value_present(sentence, values):
                        doc_supports.add(key)
                        formulations[key][fingerprint(sentence)] += 1

        entity_docs.update(doc_entities)
        cue_docs.update(doc_cues)
        support_docs.update(doc_supports)

        if documents % 25_000 == 0:
            print(f"  scanned {documents:,} documents", file=sys.stderr)

    fact_rows: list[dict[str, Any]] = []
    for entity in sorted(facts):
        for attribute in FACT_ORDER:
            key = (entity, attribute)
            counts = formulations[key]
            total_mentions = sum(counts.values())
            entropy = 0.0
            if total_mentions:
                entropy = -sum(
                    (count / total_mentions) * math.log2(count / total_mentions)
                    for count in counts.values()
                )
            fact_rows.append(
                {
                    "entity": entity,
                    "attribute": attribute,
                    "gold_value": " | ".join(facts[entity][attribute]),
                    "entity_documents": entity_docs[entity],
                    "cue_documents": cue_docs[key],
                    "support_documents": support_docs[key],
                    "distortion_candidate_documents": max(0, cue_docs[key] - support_docs[key]),
                    "covered": int(support_docs[key] > 0),
                    "distinct_support_formulations": len(counts),
                    "support_formulation_entropy_bits": round(entropy, 6),
                }
            )

    covered = sum(row["covered"] for row in fact_rows)
    cue_total = sum(row["cue_documents"] for row in fact_rows)
    support_total = sum(row["support_documents"] for row in fact_rows)
    unique_documents = len(document_hashes)
    by_attribute: dict[str, dict[str, Any]] = {}
    for attribute in FACT_ORDER:
        attribute_rows = [row for row in fact_rows if row["attribute"] == attribute]
        attribute_cues = sum(row["cue_documents"] for row in attribute_rows)
        attribute_supports = sum(row["support_documents"] for row in attribute_rows)
        attribute_covered = sum(row["covered"] for row in attribute_rows)
        by_attribute[attribute] = {
            "facts": len(attribute_rows),
            "facts_covered": attribute_covered,
            "fact_coverage": round(attribute_covered / len(attribute_rows), 6),
            "cue_documents": attribute_cues,
            "support_documents": attribute_supports,
            "conditional_preservation_rate": (
                round(attribute_supports / attribute_cues, 6) if attribute_cues else 0.0
            ),
            "mean_support_documents_per_fact": round(
                attribute_supports / len(attribute_rows), 3
            ),
        }
    summary = {
        "source": source,
        "documents": documents,
        "entities_total": len(facts),
        "entities_observed": sum(count > 0 for count in entity_docs.values()),
        "entity_coverage": round(sum(count > 0 for count in entity_docs.values()) / len(facts), 6),
        "facts_total": len(fact_rows),
        "facts_covered": covered,
        "fact_coverage": round(covered / len(fact_rows), 6),
        "fact_support_documents": support_total,
        "fact_cue_documents": cue_total,
        "conditional_preservation_rate": round(support_total / cue_total, 6) if cue_total else 0.0,
        "mean_support_documents_per_fact": round(support_total / len(fact_rows), 3),
        "documents_with_exact_duplicate": sum(count for count in document_hashes.values() if count > 1),
        "unique_document_rate": round(unique_documents / documents, 6) if documents else 0.0,
        "word_tokens": sum(vocabulary.values()),
        "vocabulary_size": len(vocabulary),
        "type_token_ratio": round(len(vocabulary) / sum(vocabulary.values()), 6) if vocabulary else 0.0,
        "document_words": {
            "mean": round(statistics.fmean(doc_lengths), 3) if doc_lengths else 0.0,
            "median": round(statistics.median(doc_lengths), 3) if doc_lengths else 0.0,
            "p10": round(percentile(doc_lengths, 0.10), 3),
            "p90": round(percentile(doc_lengths, 0.90), 3),
            "p25": round(percentile(doc_lengths, 0.25), 3),
            "p75": round(percentile(doc_lengths, 0.75), 3),
            "p99": round(percentile(doc_lengths, 0.99), 3),
        },
        "document_characters": {
            "mean": round(statistics.fmean(character_lengths), 3) if character_lengths else 0.0,
            "median": round(statistics.median(character_lengths), 3) if character_lengths else 0.0,
            "p10": round(percentile(character_lengths, 0.10), 3),
            "p90": round(percentile(character_lengths, 0.90), 3),
        },
        "structural_well_formedness": {
            "sentences": sentences,
            "terminal_punctuation_rate": round(terminal_punctuation_sentences / sentences, 6) if sentences else 0.0,
            "uppercase_start_rate": round(uppercase_start_sentences / sentences, 6) if sentences else 0.0,
            "adjacent_repeated_word_rate": round(repeated_word_sentences / sentences, 6) if sentences else 0.0,
            "unbalanced_delimiter_document_rate": round(unbalanced_delimiter_documents / documents, 6) if documents else 0.0,
            "note": "These are transparent surface diagnostics, not a grammaticality judgment.",
        },
        "by_attribute": by_attribute,
        "method_note": (
            "A preserved support requires an attributed entity, an attribute-specific cue, "
            "and every gold value in one sentence. A distortion candidate has the entity and "
            "cue but lacks the gold value; it is a diagnostic heuristic, not verified falsehood."
        ),
    }
    # Kept private from JSON serialization until optional model scoring in main.
    summary["_grammar_sample"] = grammar_sample
    return summary, fact_rows


def score_grammatical_acceptability(
    sentences: list[str], model_name: str, batch_size: int
) -> dict[str, Any]:
    """Score sampled sentences with a two-class CoLA acceptability model."""
    if not sentences:
        return {"sample_sentences": 0, "acceptable_sentence_rate": 0.0}
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Model-based grammar scoring requires torch and transformers "
            "(install the project's torch optional dependencies)."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    acceptable = 0
    probability_sum = 0.0
    with torch.inference_mode():
        for start in range(0, len(sentences), batch_size):
            encoded = tokenizer(
                sentences[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            probabilities = model(**encoded).logits.softmax(dim=-1)[:, 1]
            acceptable += int((probabilities >= 0.5).sum().item())
            probability_sum += float(probabilities.sum().item())
    return {
        "model": model_name,
        "sample_sentences": len(sentences),
        "acceptable_sentence_rate": round(acceptable / len(sentences), 6),
        "mean_acceptability_probability": round(probability_sum / len(sentences), 6),
        "sampling_seed": 13,
        "note": (
            "CoLA-style acceptability is a model-based proxy, not human annotation; "
            "domain-specific Geminon names and informal prose may affect calibration."
        ),
    }


def write_outputs(output_dir: Path, config: str, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{config}.summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (output_dir / f"{config}.facts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help=f"Synth-Geminon config to analyze (repeatable; default: {DEFAULT_CONFIG})",
    )
    parser.add_argument("--input", help="Local/remote synthetic JSONL; overrides --config (single input only)")
    parser.add_argument("--index", default=DEFAULT_INDEX, help="Local/remote public Geminon index JSONL")
    parser.add_argument("--text-field", default="text", help="Synthetic JSON field containing text")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_results/synth_geminon"))
    parser.add_argument("--limit", type=int, help="Analyze only the first N documents (for smoke tests)")
    parser.add_argument(
        "--grammar-model",
        help="Optional Hugging Face two-class CoLA model for sentence acceptability scoring",
    )
    parser.add_argument(
        "--grammar-sample-size",
        type=int,
        default=5000,
        help="Reservoir sample size per config when --grammar-model is set (default: 5000)",
    )
    parser.add_argument("--grammar-batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.input and args.config:
        print("error: --input and --config cannot be used together", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit < 1:
        print("error: --limit must be positive", file=sys.stderr)
        return 2

    print(f"Loading ground truth: {args.index}", file=sys.stderr)
    facts, entity_re = load_index(args.index)
    configs = args.config or [DEFAULT_CONFIG]
    jobs = [("custom", args.input)] if args.input else [
        (config, f"{SYNTH_REPO}/{config}/data.jsonl") for config in configs
    ]
    comparison: list[dict[str, Any]] = []
    for config, source in jobs:
        assert source is not None
        print(f"Analyzing {config}: {source}", file=sys.stderr)
        summary, rows = analyze(
            source,
            facts,
            entity_re,
            args.text_field,
            args.limit,
            grammar_sample_size=args.grammar_sample_size if args.grammar_model else 0,
        )
        grammar_sample = summary.pop("_grammar_sample")
        if args.grammar_model:
            print(f"  scoring {len(grammar_sample):,} sampled sentences for grammar", file=sys.stderr)
            summary["grammatical_acceptability"] = score_grammatical_acceptability(
                grammar_sample, args.grammar_model, args.grammar_batch_size
            )
        summary["config"] = config
        write_outputs(args.output_dir, config, summary, rows)
        comparison.append({
            "config": config,
            "documents": summary["documents"],
            "entity_coverage": summary["entity_coverage"],
            "fact_coverage": summary["fact_coverage"],
            "conditional_preservation_rate": summary["conditional_preservation_rate"],
            "mean_support_documents_per_fact": summary["mean_support_documents_per_fact"],
            "unique_document_rate": summary["unique_document_rate"],
            "mean_document_words": summary["document_words"]["mean"],
            "vocabulary_size": summary["vocabulary_size"],
            "terminal_punctuation_rate": summary["structural_well_formedness"]["terminal_punctuation_rate"],
            "uppercase_start_rate": summary["structural_well_formedness"]["uppercase_start_rate"],
            "adjacent_repeated_word_rate": summary["structural_well_formedness"]["adjacent_repeated_word_rate"],
        })
        if "grammatical_acceptability" in summary:
            comparison[-1]["acceptable_sentence_rate"] = summary["grammatical_acceptability"][
                "acceptable_sentence_rate"
            ]
        print(json.dumps(comparison[-1], indent=2), file=sys.stderr)

    with (args.output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    print(f"Wrote reports to {args.output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
