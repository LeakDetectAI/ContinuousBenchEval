#!/usr/bin/env python3
"""Format news records into Title/Date/Article string form.

Takes a .jsonl file where each record has `title`, `date`, and `text`
fields (optionally with `date` being an ISO timestamp), and writes a new
.jsonl where each record has a single `text` field shaped like:

    Title: <clean title>
    Date: Sep 01 (2025-09-01 00:00:00 UTC)
    Article: <cleaned body>

This matches the format produced by
dpsynth/datasets/news/CC_to_format1.py.

NOTE: The ContinuousBench/News data hosted on HuggingFace is already
cleaned during curation — running body normalization on it is a no-op
(verified on 1k+ records). Pass --normalize only if your input is raw /
dirty (e.g. straight from a CC crawl).

Usage:
    # Single file (no body normalization — fast, safe for HF data)
    python data/helper/format_news.py \\
        --input data/news/train.jsonl \\
        --output data/news/train_formatted.jsonl

    # With body normalization (for raw/dirty input)
    python data/helper/format_news.py \\
        --input raw.jsonl \\
        --output out.jsonl --normalize

    # In-place overwrite (stages through a .tmp file)
    python data/helper/format_news.py \\
        --input data/news/train.jsonl \\
        --output data/news/train.jsonl --overwrite

    # Whole folder
    python data/helper/format_news.py \\
        --input data/news/ \\
        --output data/news_formatted/
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_date(date_str: str) -> str:
    """Format a date string to 'Sep 01 (2025-09-01 00:00:00 UTC)'."""
    if not date_str or date_str == "None":
        return "N/A"
    try:
        dt = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
        v1 = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        v2 = dt.strftime("%b %d")
        return f"{v2} ({v1})"
    except Exception:
        return str(date_str)


def _normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def _collapse_spaces(s: str) -> str:
    return " ".join(s.split())


def _normalize_paragraphs(s: str, keep_blank_lines: int = 1) -> str:
    lines = s.split("\n")
    out: list[str] = []
    blank_run = 0
    for line in lines:
        line = line.rstrip(" \t")
        if line.strip(" \t") == "":
            blank_run += 1
            if blank_run <= keep_blank_lines:
                out.append("")
        else:
            blank_run = 0
            out.append(line)
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)


def _unwrap_soft_linebreaks(s: str) -> str:
    lines = s.split("\n")
    paragraphs: list[str] = []
    buf: list[str] = []

    def flush():
        if not buf:
            return
        para = " ".join(part.strip() for part in buf if part.strip() != "")
        para = _collapse_spaces(para)
        if para:
            paragraphs.append(para)
        buf.clear()

    for line in lines:
        if line.strip(" \t") == "":
            flush()
            paragraphs.append("")
        else:
            buf.append(line)
    flush()

    out: list[str] = []
    prev_blank = True
    for p in paragraphs:
        if p == "":
            if not prev_blank:
                out.append("")
            prev_blank = True
        else:
            out.append(p)
            prev_blank = False
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    return "\n\n".join(out)


def format_news_article(
    title: str,
    date: str,
    text: str,
    normalize: bool = False,
) -> str:
    """Format a news article as 'Title: ...\nDate: ...\nArticle: ...'.

    Note: the news data hosted at ContinuousBench/News is already cleaned
    during curation — running normalization on it is a no-op (verified on
    1k+ records). Pass normalize=True only if the input is raw / dirty
    (e.g. straight from a CC crawl). Default is False to avoid the CPU
    cost on already-clean data.
    """
    clean_title = _collapse_spaces(_normalize_newlines(title)).strip()
    clean_date = _collapse_spaces(_normalize_newlines(date)).strip()

    if normalize:
        body = _normalize_newlines(text).replace("\t", " ")
        body = _normalize_paragraphs(body, keep_blank_lines=1)
        body = _unwrap_soft_linebreaks(body).strip()
    else:
        body = text.strip()

    header = f"Title: {clean_title}\nDate: {clean_date}\nArticle: "
    return header + body + "\n"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _extract_text(value) -> str:
    """Some sources put text inside a dict with {body: ...} or {text: ...}."""
    if isinstance(value, dict):
        return value.get("body", value.get("text", ""))
    return value or ""


def format_file(
    input_path: Path,
    output_path: Path,
    title_key: str = "title",
    date_key: str = "date",
    text_key: str = "text",
    normalize: bool = False,
) -> tuple[int, int]:
    """Format a single JSONL file. Returns (kept, dropped).

    Note: the ContinuousBench/News data on HF is already cleaned during
    curation, so `normalize` is a no-op on it. Set normalize=True only
    for raw/dirty input.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0

    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue

            title = record.get(title_key, "") or ""
            date_raw = record.get(date_key, "") or ""
            text = _extract_text(record.get(text_key, ""))

            if not isinstance(text, str) or not text.strip() or not title:
                dropped += 1
                continue

            formatted = format_news_article(
                title=title,
                date=format_date(date_raw),
                text=text,
                normalize=normalize,
            )
            fout.write(json.dumps({"text": formatted}, ensure_ascii=False) + "\n")
            kept += 1

    return kept, dropped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Format news JSONL into Title/Date/Article string form.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input .jsonl file or folder")
    parser.add_argument("--output", required=True, help="Output .jsonl file or folder")
    parser.add_argument("--title_key", default="title")
    parser.add_argument("--date_key", default="date")
    parser.add_argument("--text_key", default="text")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help=(
            "Normalize body text (unwrap soft line breaks, collapse blank-line runs, "
            "replace tabs). The ContinuousBench/News data on HF is already cleaned, "
            "so this is a no-op on it — only pass for raw/dirty input."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow input path == output path (stages through a temp file)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        total_kept = total_dropped = 0
        for src in sorted(input_path.glob("*.jsonl")):
            dst = output_path / src.name
            if src.resolve() == dst.resolve() and not args.overwrite:
                print(f"Skipping {src} (would overwrite; pass --overwrite)")
                continue
            kept, dropped = _format_with_overwrite_safety(
                src, dst, args, allow_overwrite=args.overwrite
            )
            total_kept += kept
            total_dropped += dropped
            print(f"  {src.name}: {kept} kept, {dropped} dropped → {dst}")
        print(f"\nTotal: {total_kept} kept, {total_dropped} dropped")
    else:
        kept, dropped = _format_with_overwrite_safety(
            input_path, output_path, args, allow_overwrite=args.overwrite
        )
        print(f"{kept} kept, {dropped} dropped → {output_path}")


def _format_with_overwrite_safety(src: Path, dst: Path, args, allow_overwrite: bool):
    """Handle in-place overwrite safely by staging through a temp file."""
    if src.resolve() == dst.resolve():
        if not allow_overwrite:
            raise SystemExit(f"Refusing to overwrite {src}; pass --overwrite")
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        kept, dropped = format_file(
            src, tmp, args.title_key, args.date_key, args.text_key,
            normalize=args.normalize,
        )
        tmp.replace(dst)
        return kept, dropped
    return format_file(
        src, dst, args.title_key, args.date_key, args.text_key,
        normalize=args.normalize,
    )


if __name__ == "__main__":
    main()
