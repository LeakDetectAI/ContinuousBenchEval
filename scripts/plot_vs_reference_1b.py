"""Compare our 1B KD runs against reference numbers.

Plots 4 panels (exact match, fuzzy/contains match, train loss, eval loss) with
reference vs ours for both Full and LoRA. Reference arrays are hardcoded; our
data comes from outputs/cbe-geminon-small/geminon/gemma3-1b-{full,lora128}-kd.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Reference arrays (user-provided)
# ---------------------------------------------------------------------------
REF_STEPS = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])

REF_FULL = {
    "em":         np.array([0.3018, 0.6801, 0.6009, 0.7068, 0.5554, 0.7738, 0.7756, 0.7753, 0.7732, 0.7747]),
    "contains":   np.array([0.3646, 0.7339, 0.6551, 0.7438, 0.6068, 0.8235, 0.8214, 0.8295, 0.8261, 0.8282]),
    "train_loss": np.array([1.3237, 1.1119, 1.022,  0.9925, 0.9448, 0.9658, 0.8069, 0.8868, 0.9147, 0.8807]),
    "eval_loss":  np.array([1.3545, 1.1727, 1.1096, 1.0561, 1.0122, 0.9853, 0.9634, 0.9553, 0.9531, 0.9530]),
}
REF_LORA = {
    "em":         np.array([0.2348, 0.647,  0.7443, 0.7756, 0.7595, 0.736,  0.7658, 0.7759, 0.7723, 0.7726]),
    "contains":   np.array([0.2804, 0.6928, 0.7907, 0.817,  0.8045, 0.7815, 0.8098, 0.8205, 0.8169, 0.8172]),
    "train_loss": np.array([1.3349, 1.0867, 0.9966, 0.9918, 0.9479, 1.0064, 0.8492, 0.9278, 0.9704, 0.9371]),
    "eval_loss":  np.array([1.3497, 1.1492, 1.0902, 1.0513, 1.0205, 0.9982, 0.9787, 0.9692, 0.9668, 0.9667]),
}

BASE = Path("/home/peihanliu/ContinuousBenchEval/outputs/cbe-geminon-small/geminon")
OUR_RUNS = {
    "Full": {"dir": "gemma3-1b-full-kd",     "ga": 2},
    "LoRA": {"dir": "gemma3-1b-lora128-kd",  "ga": 4},
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_valqa(run_dir: Path) -> dict[int, dict]:
    """step -> {'em': ..., 'contains': ...} from metrics/eval_results.jsonl."""
    out: dict[int, dict] = {}
    p = run_dir / "metrics" / "eval_results.jsonl"
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "valqa_exact_match" not in row:
            continue
        out[row["step"]] = {
            "em": row["valqa_exact_match"],
            "contains": row["valqa_fuzzy_match"],
        }
    return out


def tb_scalars(event_file: str, tag: str) -> list[tuple[int, float]]:
    import tensorflow as tf
    out = []
    for raw in tf.data.TFRecordDataset([event_file]):
        ev = tf.compat.v1.Event.FromString(raw.numpy())
        for v in ev.summary.value:
            if v.tag != tag:
                continue
            val = v.simple_value if v.HasField("simple_value") else float(tf.make_ndarray(v.tensor))
            out.append((ev.step, val))
    return out


def merge_tb(subdir: Path, tag: str) -> dict[int, float]:
    """step -> value, merging all events.* files in subdir."""
    files = sorted(glob.glob(str(subdir / "events.out.tfevents.*")))
    by_step: dict[int, float] = {}
    for f in files:
        for s, v in tb_scalars(f, tag):
            by_step[s] = v
    return by_step


def sample_at_opt_steps(tb_by_step: dict[int, float], ga: int,
                        target_opt_steps: np.ndarray) -> np.ndarray:
    """For each target opt-step, return the TB value at step = target * ga.
    TB steps are data iters; divide by ga to get opt steps. If exact match
    missing, use nearest-preceding recorded step (tolerates skipped logs)."""
    data_iter_keys = np.array(sorted(tb_by_step.keys()))
    data_iter_vals = np.array([tb_by_step[k] for k in data_iter_keys])
    out = np.full(len(target_opt_steps), np.nan)
    for i, opt_step in enumerate(target_opt_steps):
        target_di = opt_step * ga
        if target_di in tb_by_step:
            out[i] = tb_by_step[target_di]
        else:
            # nearest-preceding
            leq = data_iter_keys[data_iter_keys <= target_di]
            if len(leq):
                out[i] = tb_by_step[leq[-1]]
    return out


def load_our_run(run_dir: Path, ga: int) -> dict:
    valqa = load_valqa(run_dir)
    train_by_step = merge_tb(run_dir / "train", "losses/xentropy")
    eval_by_step  = merge_tb(run_dir / "eval_loss", "losses/xentropy")
    return {
        "em":         np.array([valqa.get(s, {}).get("em", np.nan)       for s in REF_STEPS]),
        "contains":   np.array([valqa.get(s, {}).get("contains", np.nan) for s in REF_STEPS]),
        "train_loss": sample_at_opt_steps(train_by_step, ga, REF_STEPS),
        "eval_loss":  sample_at_opt_steps(eval_by_step,  ga, REF_STEPS),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ours: dict[str, dict] = {}
for label, info in OUR_RUNS.items():
    ours[label] = load_our_run(BASE / info["dir"], info["ga"])
    print(f"{label}: loaded {sum(~np.isnan(ours[label]['em']))}/{len(REF_STEPS)} valqa, "
          f"{sum(~np.isnan(ours[label]['train_loss']))} train, "
          f"{sum(~np.isnan(ours[label]['eval_loss']))} eval points")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
panels = [
    (axes[0, 0], "em",         "valqa exact match"),
    (axes[0, 1], "contains",   "valqa fuzzy/contains match"),
    (axes[1, 0], "train_loss", "train loss"),
    (axes[1, 1], "eval_loss",  "eval loss"),
]

COLOR_FULL = "#d62728"
COLOR_LORA = "#1f77b4"

for ax, key, title in panels:
    ax.plot(REF_STEPS, REF_FULL[key], color=COLOR_FULL, ls="-",  marker="o", ms=4,
            label="Full (reference)")
    ax.plot(REF_STEPS, ours["Full"][key], color=COLOR_FULL, ls="--", marker="s", ms=4,
            label="Full (ours)")
    ax.plot(REF_STEPS, REF_LORA[key], color=COLOR_LORA, ls="-",  marker="o", ms=4,
            label="LoRA (reference)")
    ax.plot(REF_STEPS, ours["LoRA"][key], color=COLOR_LORA, ls="--", marker="s", ms=4,
            label="LoRA (ours)")
    ax.set_title(title)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(title.split()[-1] if "match" in title else "loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

plt.tight_layout()
out = BASE / "vs_reference_1b.png"
plt.savefig(out, dpi=130)
print(f"wrote {out}")
