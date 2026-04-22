#!/usr/bin/env python3
"""Download ContinuousBench data from HuggingFace per a YAML recipe.

Reads data/download.yaml (or --recipe <path>) and downloads each file
into data/<track>/ with the names {train,val,valqa,testqa}.jsonl that
the track configs expect.

Authentication (private repos require one of these):
    hf auth login
    # or export HF_TOKEN=hf_...

Usage:
    # Download all tracks with the default recipe
    python data/load_data.py

    # Just one track
    python data/load_data.py --track news

    # Override corpus/qa size without editing the YAML
    python data/load_data.py --track geminon --corpus large
    python data/load_data.py --track geminon --corpus medium --qa medium

    # Custom recipe
    python data/load_data.py --recipe my_recipe.yaml

    # Debug: list all files in a repo
    python data/load_data.py --list news
    python data/load_data.py --list geminon
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

DATA_DIR = Path(__file__).parent
DEFAULT_RECIPE = DATA_DIR / "download.yaml"

# Repos to consult for --list (also the canonical defaults used in the recipe)
KNOWN_REPOS = {
    "news":    ("ContinuousBench/News",    "v5"),
    "geminon": ("ContinuousBench/Geminon", "v9"),
}


# ---------------------------------------------------------------------------
# Recipe loading + overrides
# ---------------------------------------------------------------------------

def load_recipe(path: Path) -> dict:
    with open(path) as f:
        recipe = yaml.safe_load(f)
    if "tracks" not in recipe:
        raise ValueError(f"Recipe {path} must have a top-level 'tracks:' key")
    return recipe


def apply_size_overrides(
    recipe: dict,
    track: str,
    corpus_size: str | None,
    qa_size: str | None,
) -> None:
    """Rewrite file paths in the recipe to use the requested corpus/qa size."""
    files = recipe["tracks"][track]["files"]

    if corpus_size:
        for local_name, src in list(files.items()):
            # Only touch entries that point at corpus_<anything>/...
            if src.startswith("corpus_"):
                parts = src.split("/", 1)
                if len(parts) == 2:
                    files[local_name] = f"corpus_{corpus_size}/{parts[1]}"

    if qa_size:
        for local_name, src in list(files.items()):
            # Rewrite qa_<anything>/... for Geminon; leave "qa/..." (News) alone
            if src.startswith("qa_"):
                parts = src.split("/", 1)
                if len(parts) == 2:
                    files[local_name] = f"qa_{qa_size}/{parts[1]}"


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

def download_track(track: str, spec: dict, token: str | None) -> None:
    """Download a single track's files from its HF repo."""
    from huggingface_hub import hf_hub_download

    repo = spec["repo"]
    revision = spec.get("revision")
    files = spec["files"]

    target_dir = DATA_DIR / track
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[load_data] Track '{track}' ← {repo}@{revision}")
    print(f"[load_data] Target: {target_dir}")

    for local_name, repo_path in files.items():
        target = target_dir / local_name
        print(f"  {repo_path}  →  {target.relative_to(DATA_DIR)}")
        try:
            cached = hf_hub_download(
                repo_id=repo,
                repo_type="dataset",
                filename=repo_path,
                revision=revision,
                token=token,
            )
        except Exception as e:
            print(f"    FAILED: {e}", file=sys.stderr)
            continue

        # hf_hub_download returns a path in the HF cache. Copy/symlink to target.
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            target.symlink_to(cached)
        except OSError:
            # Fallback to copy on filesystems that don't support symlinks
            import shutil
            shutil.copy(cached, target)

    found = sorted(f.name for f in target_dir.glob("*.jsonl"))
    print(f"[load_data] {track}: {len(found)} files ready — {found}")


# ---------------------------------------------------------------------------
# --list
# ---------------------------------------------------------------------------

def list_repo(track: str, token: str | None) -> None:
    if track not in KNOWN_REPOS:
        print(f"Unknown track: {track}. Known: {list(KNOWN_REPOS)}", file=sys.stderr)
        sys.exit(1)

    from huggingface_hub import HfApi
    repo, revision = KNOWN_REPOS[track]
    api = HfApi()
    files = api.list_repo_files(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        token=token,
    )
    print(f"[load_data] {repo}@{revision} contains {len(files)} files:")
    for f in sorted(files):
        print(f"  {f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--recipe",
        default=str(DEFAULT_RECIPE),
        help=f"Path to download recipe YAML (default: {DEFAULT_RECIPE})",
    )
    parser.add_argument(
        "--track",
        default=None,
        help="Only download this track (default: all tracks in the recipe)",
    )
    parser.add_argument(
        "--corpus",
        choices=["small", "medium", "large"],
        default=None,
        help="Override corpus size for train/val (rewrites corpus_<size>/ paths in recipe)",
    )
    parser.add_argument(
        "--qa",
        choices=["small", "medium"],
        default=None,
        help="Override QA size for valqa/testqa (Geminon only; rewrites qa_<size>/ paths)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token (defaults to $HF_TOKEN or cached login)",
    )
    parser.add_argument(
        "--list",
        metavar="TRACK",
        choices=list(KNOWN_REPOS.keys()),
        default=None,
        help="List all files in a track's HF repo and exit",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")

    if args.list:
        list_repo(args.list, token=token)
        return

    recipe = load_recipe(Path(args.recipe))
    tracks = [args.track] if args.track else list(recipe["tracks"].keys())

    for track in tracks:
        if track not in recipe["tracks"]:
            print(f"Track '{track}' not in recipe. Available: {list(recipe['tracks'])}", file=sys.stderr)
            sys.exit(1)
        apply_size_overrides(recipe, track, args.corpus, args.qa)
        download_track(track, recipe["tracks"][track], token=token)

    print("\n[load_data] Done.")


if __name__ == "__main__":
    main()
