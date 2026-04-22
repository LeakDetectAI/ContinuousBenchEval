"""Plot valqa_fuzzy_match, eval_loss, and train_loss across runs in a task dir.

Run discovery: every immediate subdirectory of `task_dir` that contains
`metrics/eval_results.jsonl` is treated as a run. The framework (hf vs kd) is
inferred from on-disk shape:

  - HF run: `checkpoints/checkpoint-*/trainer_state.json`  (train/eval loss
    come from the `log_history` of the latest trainer_state).
  - KD run: `train/events.out.tfevents.*` + `eval_loss/events.out.tfevents.*`
    (train/eval loss come from the `losses/xentropy` TB tag).

KD logs on a data-iter step counter; we divide by `grad_accum` (read from
`config.yaml`: effective_batch_size // per_device_batch_size) so every curve
shares one optimizer-step x-axis.

Usage:
    python scripts/plot_runs.py outputs/cbe-geminon-small/geminon
    python scripts/plot_runs.py <task_dir> --out runs_plot.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


def load_eval_jsonl(run_dir: Path) -> list[dict]:
    """Rows from `metrics/eval_results.jsonl` (already in opt-step units)."""
    p = run_dir / "metrics" / "eval_results.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def tb_scalars(event_file: str, tag: str) -> list[tuple[int, float]]:
    """(step, value) pairs for `tag` from a TB event file."""
    import tensorflow as tf
    out = []
    for raw in tf.data.TFRecordDataset([event_file]):
        ev = tf.compat.v1.Event.FromString(raw.numpy())
        for v in ev.summary.value:
            if v.tag != tag:
                continue
            if v.HasField("simple_value"):
                val = v.simple_value
            else:
                val = float(tf.make_ndarray(v.tensor))
            out.append((ev.step, val))
    return out


def _merge_tb(subdir: Path, tag: str) -> list[tuple[int, float]]:
    """Read `tag` from every events file in subdir; de-dup by step (later wins).
    Handles run-resumes, which write a second events file in the same dir."""
    files = sorted(glob.glob(str(subdir / "events.out.tfevents.*")))
    by_step: dict[int, float] = {}
    for f in files:
        for s, v in tb_scalars(f, tag):
            by_step[s] = v
    return sorted(by_step.items())


def infer_grad_accum(run_dir: Path) -> int:
    """effective_batch_size // per_device_batch_size from config.yaml."""
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return 1
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    tr = cfg.get("training", {}) or {}
    eff = tr.get("effective_batch_size")
    per = tr.get("per_device_batch_size")
    if not eff or not per:
        return 1
    return max(1, int(eff) // int(per))


def infer_framework(run_dir: Path) -> str:
    if list((run_dir / "checkpoints").glob("checkpoint-*/trainer_state.json")):
        return "hf"
    if (run_dir / "train").exists() and list((run_dir / "train").glob("events.out.tfevents.*")):
        return "kd"
    return "unknown"


def kd_train_loss(run_dir: Path, ga: int):
    pts = _merge_tb(run_dir / "train", "losses/xentropy")
    return [(s / ga, v) for s, v in pts if s % ga == 0]


def kd_eval_loss(run_dir: Path, ga: int):
    pts = _merge_tb(run_dir / "eval_loss", "losses/xentropy")
    return [(s / ga, v) for s, v in pts]


def hf_train_eval_loss(run_dir: Path):
    ckpts = sorted(
        (run_dir / "checkpoints").glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not ckpts:
        return [], []
    st = json.loads((ckpts[-1] / "trainer_state.json").read_text())
    train = [(e["step"], e["loss"]) for e in st["log_history"] if "loss" in e]
    evl   = [(e["step"], e["eval_loss"]) for e in st["log_history"] if "eval_loss" in e]
    return train, evl


def style_for(label: str) -> dict:
    """Blue for HF, red for KD; solid for full, dashed for lora."""
    color = "#1f77b4" if "hf" in label else "#d62728" if "kd" in label else "#2ca02c"
    ls = "--" if "lora" in label else "-"
    return {"color": color, "ls": ls}


def run_label(run_dir: Path, fw: str) -> str:
    """Shorten e.g. 'gemma3-1b-lora128-hf' → 'hf-lora'; keep rest verbatim otherwise."""
    name = run_dir.name.lower()
    kind = "lora" if "lora" in name else ("full" if "full" in name else "")
    if fw in ("hf", "kd") and kind:
        return f"{fw}-{kind}"
    return run_dir.name


def collect(task_dir: Path):
    runs = []
    for sub in sorted(task_dir.iterdir()):
        if not sub.is_dir():
            continue
        if not (sub / "metrics" / "eval_results.jsonl").exists():
            continue
        fw = infer_framework(sub)
        ga = infer_grad_accum(sub) if fw == "kd" else 1
        runs.append((run_label(sub, fw), sub, fw, ga))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir", type=Path,
                    help="Directory containing run subdirs (e.g. outputs/<proj>/<task>)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output PNG path (default: <task_dir>/runs_plot.png)")
    args = ap.parse_args()

    task_dir = args.task_dir.resolve()
    out = (args.out or (task_dir / "runs_plot.png")).resolve()

    runs = collect(task_dir)
    if not runs:
        raise SystemExit(f"No runs with metrics/eval_results.jsonl under {task_dir}")

    series = {}
    for label, run_dir, fw, ga in runs:
        eval_rows = load_eval_jsonl(run_dir)
        valqa = [(r["step"], r["valqa_fuzzy_match"]) for r in eval_rows
                 if "valqa_fuzzy_match" in r]
        if fw == "hf":
            tr, ev = hf_train_eval_loss(run_dir)
        elif fw == "kd":
            tr = kd_train_loss(run_dir, ga)
            ev = kd_eval_loss(run_dir, ga)
        else:
            tr, ev = [], []
        series[label] = {"valqa": valqa, "train": tr, "eval": ev}
        print(f"{label:12} ({fw}, ga={ga}): valqa {len(valqa):>3}  "
              f"train {len(tr):>4}  eval {len(ev):>3}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for label, _, _, _ in runs:
        s = series[label]
        st = style_for(label)
        if s["valqa"]:
            xs, ys = zip(*s["valqa"])
            axes[0].plot(xs, ys, label=label, marker="o", ms=3, **st)
        if s["eval"]:
            xs, ys = zip(*s["eval"])
            axes[1].plot(xs, ys, label=label, marker="o", ms=3, **st)
        if s["train"]:
            xs, ys = zip(*s["train"])
            axes[2].plot(xs, ys, label=label, alpha=0.85, lw=1.0, **st)

    for ax, title, ylabel in [
        (axes[0], "valqa fuzzy match", "fuzzy_match"),
        (axes[1], "eval loss",         "eval_loss"),
        (axes[2], "train loss",        "loss"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
