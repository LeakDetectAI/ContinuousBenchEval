"""Shared data utilities: read .jsonl files, clean text, normalize."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a .jsonl file (one JSON object per line)."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_jsonl_folder(folder: str | Path) -> list[dict[str, Any]]:
    """Load all .jsonl files in a folder, concatenated."""
    folder = Path(folder)
    records = []
    for path in sorted(folder.glob("*.jsonl")):
        records.extend(load_jsonl(path))
    return records


def clean_text(text: str) -> str:
    """Normalize whitespace and unicode in a text string."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    """Split text into non-empty paragraphs."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def deduplicate(records: list[dict], key: str = "text") -> list[dict]:
    """Remove duplicate records by a text key."""
    seen = set()
    unique = []
    for r in records:
        val = r.get(key, "")
        if val not in seen:
            seen.add(val)
            unique.append(r)
    return unique


def format_for_training(
    records: list[dict[str, Any]],
    text_key: str = "text",
) -> list[dict[str, str]]:
    """Normalize records into a standard {text: ...} format for training."""
    result = []
    for r in records:
        text = r.get(text_key, "")
        if isinstance(text, str) and text.strip():
            result.append({"text": clean_text(text)})
    return result


def format_qa(
    records: list[dict[str, Any]],
    prompt_key: str = "prompt",
    answer_key: str = "answer",
) -> list[dict[str, str]]:
    """Normalize QA records into {prompt: ..., answer: ...} format."""
    result = []
    for r in records:
        prompt = r.get(prompt_key, "")
        answer = r.get(answer_key, "")
        if prompt and answer:
            result.append({"prompt": str(prompt), "answer": str(answer)})
    return result
