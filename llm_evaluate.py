"""LLM-as-judge re-scoring of eval_details/*.jsonl files.

Reads a per-example QA results jsonl produced during training (with keys
`question`, `ground_truth`, `raw_prediction`, ...) and asks Gemini whether
each prediction matches the ground truth. Writes a copy with an added
`llm_match: bool` field to `<input>_llm_judged.jsonl` (or the path passed
via --output).

API key sources (any one is fine; later wins on duplicates):
    1. `secrets/gemini_keys.txt` (one key per line, # comments ok) — multi-key
        round-robin. Recommended; lets you stack quotas across Google accounts.
    2. `GEMINI_API_KEY` or `GOOGLE_API_KEY` env vars (single key).
    3. `--keys-file PATH` CLI override.

Usage:
    pip install google-genai          # one-time
    cp secrets/gemini_keys.txt.example secrets/gemini_keys.txt   # then add your keys
    python llm_evaluate.py --input outputs/.../eval_details/testqa_step_001000.jsonl
    # → writes outputs/.../eval_details/testqa_step_001000_llm_judged.jsonl

Options:
    --output PATH                  output jsonl (default: <input>_llm_judged.jsonl)
    --model NAME                   default gemini-2.5-flash-lite
    --response-field FIELD         which record field to judge (default raw_prediction)
    --concurrency N                parallel API requests (default 16)
    --max-records N                cap for smoke tests (default no cap)
    --resume                       skip records already in <output> (read previous llm_match)
    --keys-file PATH               override default secrets/gemini_keys.txt
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

LLM_JUDGE_PROMPT = """Your task is to judge whether the given response to a question matches a given ground truth answer or not. You are provided with a question, a ground truth response, and the response you need to judge.
For a response to "match", it must have at least as much information as the ground-truth.
The response can have more information than the ground-truth. It can be more specific (for example, "Labrador" is more specific than "dog"), or have additional possible correct answers. But it must cover everything mentioned in the ground-truth. It is okay if it covers it in different words, i.e. paraphrased.
For numeric answers, the relative error, defined as |response - ground truth| / mean(response, ground truth), must be less than 0.01%.
Possible judgments:
"0": The response does not match the ground-truth answer.
"1": The response matches the ground-truth.
Question: "{question}"
Ground truth: "{target}"
Response: "{response}"
Your job is to ONLY check whether the given response matches the ground truth answer or not in the context of the question. You DO NOT NEED to assess the correctness of the response. This is part of an automated evaluation process, therefore you MUST OUTPUT your final answer as "0" or "1".
YOUR RESPONSE MUST BE "0" OR "1". Do not output anything else."""


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _build_prompt(rec: dict, response_field: str) -> str:
    return LLM_JUDGE_PROMPT.format(
        question=str(rec.get("question", "")),
        target=str(rec.get("ground_truth", "")),
        response=str(rec.get(response_field, "")),
    )


# ---------------------------------------------------------------------------
# Summary computation (with optional supports stratification, news-style)
# ---------------------------------------------------------------------------

def _detect_qa_path_and_thresholds(input_path: Path):
    """Walk up from `input_path` to find the run's `config.yaml`, and infer
    (qa_jsonl_path, support_thresholds). Filename `testqa_step_*.jsonl` →
    `data.testqa_path`; `valqa_step_*.jsonl` → `data.valqa_path`. Other
    filenames look up `data.extra_qa_paths[<prefix>]` if defined.

    Returns (qa_path_or_None, thresholds_or_empty). Both are None / [] if
    auto-detection fails — callers should fall back to no-stratification.
    """
    try:
        import yaml
    except ImportError:
        return None, []

    # Walk up: eval_details/<file>.jsonl → eval_details/ → run_dir/
    run_dir = input_path.parent
    while run_dir.name == "eval_details":
        run_dir = run_dir.parent
        break  # one level up is enough; eval_details lives directly under run_dir
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return None, []
    try:
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        return None, []

    data = cfg.get("data") or {}
    eval_cfg = cfg.get("eval") or {}
    fname = input_path.stem.lower()  # e.g. "testqa_step_001000"
    prefix = fname.split("_step_")[0] if "_step_" in fname else fname  # "testqa"

    qa_path_str: str | None = None
    if prefix == "testqa":
        qa_path_str = data.get("testqa_path")
    elif prefix == "valqa":
        qa_path_str = data.get("valqa_path")
    else:
        extra = data.get("extra_qa_paths") or {}
        qa_path_str = extra.get(prefix)

    thresholds = eval_cfg.get("support_thresholds") or []

    if qa_path_str is None:
        return None, list(thresholds)

    qa_path = Path(qa_path_str)
    if not qa_path.is_absolute():
        # Project paths in track yamls are relative (e.g. "data/news/testqa.jsonl")
        # and resolved against the repo root at launch. Find the repo root by
        # walking up from `run_dir` looking for a dir that contains `configs/`.
        repo_root = run_dir
        while repo_root != repo_root.parent:
            if (repo_root / "configs").is_dir():
                break
            repo_root = repo_root.parent
        if (repo_root / qa_path).exists():
            qa_path = repo_root / qa_path

    return qa_path, list(thresholds)


def _build_support_lookup(qa_path: Path) -> dict[tuple[str, str], int] | None:
    """Build (question, answer) → len(supports) map from the source QA file.
    Returns None if the file doesn't exist or no record has `supports`."""
    if not qa_path.exists():
        return None
    out: dict[tuple[str, str], int] = {}
    any_supports = False
    for line in qa_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        sup = r.get("supports")
        if sup is not None:
            any_supports = True
        key = (str(r.get("question", "")), str(r.get("answer", "")))
        out[key] = len(sup) if isinstance(sup, list) else 0
    return out if any_supports else None


def _compute_summary(records: list[dict],
                     support_counts: list[int] | None,
                     thresholds: list[int]) -> list[dict]:
    """Build one summary row for the full set, plus one per support
    threshold (only when both `support_counts` is non-None and `thresholds`
    is non-empty). Each row is:
        {"subset", "n", "exact_match", "fuzzy_match", "llm_match"}
    """
    def stats_for(idxs: list[int]) -> dict:
        n = len(idxs)
        if n == 0:
            return {"n": 0, "exact_match": 0.0, "fuzzy_match": 0.0, "llm_match": 0.0}
        em = sum(1 for i in idxs if records[i].get("exact_match") is True) / n
        fm = sum(1 for i in idxs if records[i].get("fuzzy_match") is True) / n
        lm = sum(1 for i in idxs if records[i].get("llm_match") is True) / n
        return {"n": n, "exact_match": em, "fuzzy_match": fm, "llm_match": lm}

    rows = [{"subset": "all", **stats_for(list(range(len(records))))}]
    if support_counts is not None and thresholds:
        for k in thresholds:
            idxs = [i for i, c in enumerate(support_counts) if c >= k]
            rows.append({"subset": f"supports_ge_{k}", **stats_for(idxs)})
    return rows


def _print_summary_table(rows: list[dict]) -> None:
    """Print summary rows as a clean fixed-width table."""
    headers = ["subset", "n", "exact_match", "fuzzy_match", "llm_match"]
    col_w = {h: max(len(h), max((len(str(r.get(h, "-"))) for r in rows), default=0))
             for h in headers}
    # Format floats with 4 decimals to a uniform width.
    def fmt_cell(h, v):
        if h in ("exact_match", "fuzzy_match", "llm_match"):
            return f"{v:.4f}"
        return str(v)
    # Recompute width with formatted floats.
    col_w = {h: max(len(h), max((len(fmt_cell(h, r.get(h, "-"))) for r in rows),
                                default=0))
             for h in headers}
    sep = "  "
    print(sep.join(h.ljust(col_w[h]) for h in headers))
    print(sep.join("-" * col_w[h] for h in headers))
    for r in rows:
        print(sep.join(fmt_cell(h, r.get(h, "-")).ljust(col_w[h]) for h in headers))


def _load_keys(keys_file: Path | None) -> list[str]:
    """Collect API keys, preferring keys_file → env vars. Comments + blank
    lines in the file are skipped; whitespace is stripped. Order in the file
    is preserved. Returns deduped list."""
    keys: list[str] = []
    if keys_file and keys_file.exists():
        for raw in keys_file.read_text().splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                keys.append(line)
    for env in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        v = os.environ.get(env, "").strip()
        if v:
            keys.append(v)
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


class _KeyPool:
    """Round-robin dispatcher: hands out one (key, client) pair per call,
    cycling across all loaded keys. Thread-safe."""

    def __init__(self, keys: list[str]):
        from google import genai
        self._clients = [genai.Client(api_key=k) for k in keys]
        self._cycle = itertools.cycle(range(len(self._clients)))
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._clients)

    def next(self):
        with self._lock:
            return self._clients[next(self._cycle)]


def _parse_verdict(text: str) -> bool | None:
    """Return True/False if the first 0/1 character we find decides it."""
    for ch in (text or "").strip():
        if ch == "1":
            return True
        if ch == "0":
            return False
    return None


def _judge_one(pool: _KeyPool, model: str, prompt: str,
               max_attempts: int = 12) -> bool:
    """Get a definitive True/False verdict for a single prompt.

    Retries on:
      - any exception (rate-limit, server error, transient network, etc.)
        with a rotated key + exponential backoff with jitter.
      - unparsable model output (no clear 0/1) — re-asks with the same
        prompt, since Gemini is non-deterministic and usually emits a
        clean answer on the second try.

    Raises RuntimeError only if `max_attempts` retries all fail. Caller
    can catch and either bubble up or do one more pass.
    """
    from google.genai import types

    # Greedy decoding: temperature=0 → deterministic verdicts. Same prompt +
    # same model version should now produce the same 0/1 across runs.
    gen_cfg = types.GenerateContentConfig(temperature=0)

    last_err: Exception | None = None
    last_text: str | None = None
    for attempt in range(max_attempts):
        client = pool.next()
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=gen_cfg,
            )
            text = (resp.text or "").strip()
            verdict = _parse_verdict(text)
            if verdict is not None:
                return verdict
            # Unparsable — record and retry. Small jittered sleep so we don't
            # hammer the same key in a tight loop on a stuck model state.
            last_text = text
            time.sleep(0.3 + random.random() * 0.4)
            continue
        except Exception as e:
            last_err = e
            # Backoff: 0.5s, 1s, 2s, 4s, ... capped, plus jitter.
            sleep = min(8.0, 0.5 * (2 ** attempt)) + random.random() * 0.4
            time.sleep(sleep)
            continue
    if last_err is not None:
        raise RuntimeError(
            f"giving up after {max_attempts} attempts; last error: {last_err!r}"
        )
    raise RuntimeError(
        f"giving up after {max_attempts} attempts; "
        f"model never returned a clean 0/1 (last text: {last_text!r})"
    )


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--input", type=Path, required=True,
                    help="Path to eval_details jsonl with question/ground_truth/raw_prediction.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output path. Default: <input>_llm_judged.jsonl")
    ap.add_argument("--model", default="gemini-2.5-flash-lite")
    ap.add_argument("--response-field", default="parsed_prediction",
                    help="Which record field is judged as the model response. "
                         "Default `parsed_prediction` is the cleaned answer; "
                         "`raw_prediction` is the model's full output (often "
                         "contains follow-up Q/A continuations that confuse "
                         "the LLM judge — use only if you know what you want).")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-records", type=int, default=None,
                    help="Cap records (for smoke tests).")
    ap.add_argument("--resume", action="store_true",
                    help="Reuse llm_match from existing output for already-judged records.")
    ap.add_argument("--keys-file", type=Path,
                    default=Path(__file__).parent / "secrets" / "gemini_keys.txt",
                    help="File with one API key per line (# comments ok).")
    ap.add_argument("--summary", type=Path, default=None,
                    help="Override summary jsonl path. Default: <input>_summary.jsonl")
    ap.add_argument("--qa-source", type=Path, default=None,
                    help="Path to the source QA jsonl (for `supports`-based "
                         "stratification). Auto-detected from the run's "
                         "config.yaml when omitted.")
    ap.add_argument("--support-thresholds", type=str, default=None,
                    help="Comma-sep thresholds, e.g. '200,400,600,800'. "
                         "Empty / 'none' disables stratification. "
                         "Default: read eval.support_thresholds from the run config.")
    args = ap.parse_args()

    try:
        from google import genai  # noqa: F401  (imported transitively by _KeyPool)
    except ImportError:
        sys.exit("Missing 'google-genai'. Install with: pip install google-genai")

    keys = _load_keys(args.keys_file)
    if not keys:
        sys.exit(
            f"No API keys found.\n"
            f"  Add keys to {args.keys_file} (one per line), or\n"
            f"  set GEMINI_API_KEY / GOOGLE_API_KEY in env."
        )
    pool = _KeyPool(keys)

    in_path = args.input.resolve()
    out_path = (args.output or in_path.with_name(in_path.stem + "_llm_judged.jsonl")).resolve()
    records = _load_jsonl(in_path)
    if args.max_records is not None:
        records = records[: args.max_records]

    # Resume: read existing output and reuse llm_match for matching keys.
    # Only cache True/False — None / missing means we'll re-judge.
    cached: dict[tuple, bool] = {}
    if args.resume and out_path.exists():
        for r in _load_jsonl(out_path):
            v = r.get("llm_match")
            if v is True or v is False:
                key = (r.get("question", ""), r.get("ground_truth", ""),
                       r.get(args.response_field, ""))
                cached[key] = v
        print(f"[resume] reusing {len(cached)} cached judgments from {out_path}")

    total = len(records)
    print(f"[input]  {in_path}  ({total} records)")
    print(f"[output] {out_path}")
    print(f"[model]  {args.model}  concurrency={args.concurrency}  field={args.response_field}")
    print(f"[keys]   {len(pool)} key(s) loaded — round-robin")

    # Prepare jobs. Skip ones we have cached.
    pending: list[tuple[int, dict, str]] = []
    out_records: list[dict] = []
    for i, r in enumerate(records):
        key = (r.get("question", ""), r.get("ground_truth", ""),
               r.get(args.response_field, ""))
        if key in cached:
            new_r = {**r, "llm_match": cached[key]}
            out_records.append(new_r)
        else:
            new_r = {**r, "llm_match": None}  # filled in below
            out_records.append(new_r)
            pending.append((i, new_r, _build_prompt(r, args.response_field)))

    print(f"[plan]   {len(pending)} new judgments, {total - len(pending)} from cache")

    def _judge_batch(jobs, label, max_attempts):
        """Run the given (idx, record, prompt) jobs through the pool;
        write True/False into out_records[idx]["llm_match"]. Returns the
        list of jobs that still ended up without a verdict (should be []
        unless every retry failed)."""
        if not jobs:
            return []

        def task(args_t):
            idx, _rec, prompt = args_t
            try:
                v = _judge_one(pool, args.model, prompt, max_attempts=max_attempts)
                return idx, v, None
            except Exception as e:
                return idx, None, repr(e)

        match_count = err_count = done = 0
        leftovers = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {ex.submit(task, j): j for j in jobs}
            for fut in as_completed(futures):
                j = futures[fut]
                idx, verdict, err = fut.result()
                rec = out_records[idx]
                if err is not None:
                    err_count += 1
                    leftovers.append(j)
                    rec["llm_match"] = None
                    rec["llm_error"] = err
                else:
                    rec["llm_match"] = verdict
                    rec.pop("llm_error", None)
                    if verdict is True:
                        match_count += 1
                done += 1
                if done % 50 == 0 or done == len(jobs):
                    print(f"  [{label}] ... {done}/{len(jobs)} judged "
                          f"(match so far: {match_count}, errors: {err_count})",
                          flush=True)
        return leftovers

    # Pass 1: run everything with normal retry budget.
    leftovers = _judge_batch(pending, "pass1", max_attempts=12)

    # Pass 2: anything still without a verdict gets a more patient retry
    # (more attempts, longer max backoff). This handles transient outages
    # that exceeded pass-1's per-call budget.
    if leftovers:
        print(f"\n[pass2]  {len(leftovers)} record(s) still unjudged after pass 1; "
              f"retrying with more attempts...", flush=True)
        leftovers = _judge_batch(leftovers, "pass2", max_attempts=24)

    # Tally
    final_match = sum(1 for r in out_records if r.get("llm_match") is True)
    final_nomatch = sum(1 for r in out_records if r.get("llm_match") is False)
    final_unknown = sum(1 for r in out_records if r.get("llm_match") is None)
    print()
    print(f"[done]  match: {final_match}/{total} = {final_match/total:.4f}")
    print(f"        no-match: {final_nomatch}, unjudged: {final_unknown}")

    if final_unknown > 0:
        sample = [
            i for i, r in enumerate(out_records) if r.get("llm_match") is None
        ][:5]
        print(f"\n[WARNING] {final_unknown} record(s) still have llm_match=None "
              f"after both passes. Sample indices: {sample}.")
        print("          You can re-run with `--resume` to retry only those records "
              "(cached True/False verdicts will be reused).")

    _save_jsonl(out_path, out_records)
    print(f"[wrote] {out_path}")

    # ---------- Summary jsonl + pretty stdout table ----------
    summary_path = (args.summary
                    or in_path.with_name(in_path.stem + "_summary.jsonl")).resolve()

    # Resolve QA source + thresholds (CLI > auto-detect from config.yaml).
    qa_path, auto_thresholds = _detect_qa_path_and_thresholds(in_path)
    if args.qa_source is not None:
        qa_path = args.qa_source
    if args.support_thresholds is not None:
        s = args.support_thresholds.strip().lower()
        thresholds = [] if s in ("", "none") else [
            int(t) for t in s.split(",") if t.strip()
        ]
    else:
        thresholds = auto_thresholds

    support_counts: list[int] | None = None
    if thresholds and qa_path is not None:
        sup_lookup = _build_support_lookup(qa_path)
        if sup_lookup is None:
            print(f"[summary] QA source {qa_path} has no `supports` field — "
                  f"skipping stratification.")
        else:
            # Match each record to (question, ground_truth). Missing → 0.
            support_counts = []
            missing = 0
            for r in out_records:
                key = (str(r.get("question", "")), str(r.get("ground_truth", "")))
                if key in sup_lookup:
                    support_counts.append(sup_lookup[key])
                else:
                    support_counts.append(0)
                    missing += 1
            if missing:
                print(f"[summary] {missing} record(s) had no match in {qa_path} "
                      f"(treated as len(supports)=0).")

    rows = _compute_summary(out_records, support_counts, thresholds)
    _save_jsonl(summary_path, rows)
    print()
    print(f"[summary]")
    _print_summary_table(rows)
    print(f"[wrote] {summary_path}")

    if final_unknown > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
