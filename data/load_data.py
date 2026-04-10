#!/usr/bin/env python3
"""Download ContinuousBench data from the private HF repo `pl666/ContinuousBench`.

Requires HuggingFace authentication for the private repo:
    huggingface-cli login
    # or set: export HF_TOKEN=hf_...

Usage:
    python data/load_data.py                    # Download all tracks
    python data/load_data.py --track news       # Download just one track
    python data/load_data.py --track geminon
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ID = "pl666/ContinuousBench"
DATA_DIR = Path(__file__).parent
AVAILABLE_TRACKS = ["news", "geminon"]


def list_repo(token: str | None = None) -> list[str]:
    """List all files in the HF repo so you can pick the right pattern."""
    from huggingface_hub import HfApi
    api = HfApi()
    files = api.list_repo_files(
        repo_id=REPO_ID,
        repo_type="dataset",
        token=token,
    )
    print(f"[load_data] Repo {REPO_ID} contains {len(files)} files. First 50:")
    for f in files[:50]:
        print(f"  {f}")
    if len(files) > 50:
        print(f"  ... and {len(files) - 50} more")
    return files


def download_track(track: str, token: str | None = None) -> None:
    """Download all .jsonl files for a single track from the HF repo."""
    from huggingface_hub import snapshot_download

    target_dir = DATA_DIR / track
    target_dir.mkdir(parents=True, exist_ok=True)

    pattern = f"geminon/v9/qa/qas_1m/*.jsonl"
    print(f"[load_data] Downloading track '{track}' with pattern: {pattern}")
    print(f"[load_data] Target: {DATA_DIR}")

    downloaded_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(DATA_DIR),
        allow_patterns=[pattern],
        token=token,
    )
    print(f"[load_data] snapshot_download returned: {downloaded_path}")

    # Show what actually landed on disk
    all_files = list(DATA_DIR.rglob("*.jsonl"))
    print(f"[load_data] Found {len(all_files)} .jsonl files under {DATA_DIR}:")
    for f in all_files[:20]:
        print(f"  {f.relative_to(DATA_DIR)}")
    

    # Verify expected files
    expected = ["train.jsonl", "val.jsonl", "valqa.jsonl", "testqa.jsonl"]
    found = [f.name for f in target_dir.glob("*.jsonl")]
    missing = [f for f in expected if f not in found]
    if missing:
        print(f"[load_data] Warning: missing files for {track}: {missing}")
    else:
        print(f"[load_data] OK: {track} has all expected files: {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track",
        choices=AVAILABLE_TRACKS,
        default=None,
        help="Download a specific track (default: all tracks)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token (defaults to HF_TOKEN env var or cached login)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Just list files in the HF repo and exit (for debugging)",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")

    if args.list:
        list_repo(token=token)
        return

    tracks = [args.track] if args.track else AVAILABLE_TRACKS

    for track in tracks:
        download_track(track, token=token)

    print("[load_data] Done.")


if __name__ == "__main__":
    main()
