#!/usr/bin/env python3
"""Format and clean raw text data into .jsonl for training.

Usage:
    # Clean a single .jsonl file
    python scripts/format_data.py --input raw.jsonl --output data/news/train.jsonl

    # Clean all .jsonl files in a folder
    python scripts/format_data.py --input raw_folder/ --output data/news/train.jsonl

    # Split into train/val (90/10)
    python scripts/format_data.py --input raw.jsonl --output data/news/ --split

    # Deduplicate
    python scripts/format_data.py --input raw.jsonl --output data/news/train.jsonl --dedup
"""

import argparse
import json
import random
from pathlib import Path

from cbe.data.formatters import (
    clean_text,
    deduplicate,
    load_jsonl,
    load_jsonl_folder,
)


def main():
    parser = argparse.ArgumentParser(description="Format raw text data into .jsonl")
    parser.add_argument("--input", required=True, help="Input .jsonl file or folder")
    parser.add_argument("--output", required=True, help="Output .jsonl file or folder (with --split)")
    parser.add_argument("--text_key", default="text", help="Key containing text in input records")
    parser.add_argument("--dedup", action="store_true", help="Deduplicate by text content")
    parser.add_argument("--split", action="store_true", help="Split into train.jsonl and val.jsonl (90/10)")
    parser.add_argument("--split_ratio", type=float, default=0.9, help="Train fraction (default: 0.9)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    input_path = Path(args.input)

    # Load records
    if input_path.is_dir():
        records = load_jsonl_folder(input_path)
    else:
        records = load_jsonl(input_path)

    print(f"Loaded {len(records)} records")

    # Normalize text
    cleaned = []
    for r in records:
        text = r.get(args.text_key, "")
        if isinstance(text, str) and text.strip():
            cleaned.append({"text": clean_text(text)})
    print(f"After cleaning: {len(cleaned)} records")

    if args.dedup:
        cleaned = deduplicate(cleaned, key="text")
        print(f"After dedup: {len(cleaned)} records")

    # Write output
    if args.split:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        random.seed(args.seed)
        random.shuffle(cleaned)
        split_idx = int(len(cleaned) * args.split_ratio)
        train, val = cleaned[:split_idx], cleaned[split_idx:]

        _write_jsonl(output_dir / "train.jsonl", train)
        _write_jsonl(output_dir / "val.jsonl", val)
        print(f"Wrote {len(train)} train, {len(val)} val records")
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_path, cleaned)
        print(f"Wrote {len(cleaned)} records to {output_path}")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
