# ContinuousBenchEval

A framework-agnostic training and evaluation harness for continual learning benchmarks. Train language models on text corpora and evaluate memorization via QA — using **Kauldron (JAX)** or **HuggingFace/TRL (PyTorch)** as the backend, with the same config, same data, and same metrics.

> **Pick exactly one backend before you start.** The repo supports Kauldron (JAX) and HuggingFace/TRL (PyTorch); the rest of this README is organized so the same yaml works for either, but the conda env you create installs only one of them. Mixing isn't supported in a single env. If you want to try both, create two envs (e.g. `cbe-kd` and `cbe-hf`) — see [Environment Setup](#environment-setup).

---

## Table of Contents

- [TL;DR — End-to-end first run](#tldr--end-to-end-first-run)
- [Repository Layout](#repository-layout)
- [Environment Setup](#environment-setup)
- [Authentication (HuggingFace + W&B)](#authentication-huggingface--wandb)
- [Downloading Data](#downloading-data)
- [Formatting Data](#formatting-data)
- [Configuration System](#configuration-system)
- [Training](#training)
- [Evaluation](#evaluation)
- [Logging and Monitoring](#logging-and-monitoring)
- [Output Layout](#output-layout)
- [Adding a New Track](#adding-a-new-track)
- [Adding a New Model](#adding-a-new-model)
- [Known Limitations](#known-limitations)

---

## TL;DR — End-to-end first run

If you just want to train Gemma3-1B-LoRA on the news track end-to-end with sane defaults, this is the full required path. Anything not in this list (formatting raw data, custom configs, sweeps, etc.) is optional and explained later.

```bash
# 1) Clone + install ONE backend (pick torch-gpu OR jax-gpu, not both)
git clone ...
cd ContinuousBenchEval
bash setup_env.sh torch-gpu wandb            # for HF/TRL  — env name: "cbe"
# (or)  bash setup_env.sh jax-gpu wandb      # for Kauldron — env name: "cbe"

# 2) Activate the env (every new shell needs this)
conda activate cbe

# 3) Get access to gated Gemma weights (one-time, on the HuggingFace website)
#    Visit  https://huggingface.co/google/gemma-3-1b-pt  and click "Agree and access"
#    (do this for every Gemma checkpoint you want to use: 1b, 4b, 12b, etc.)

# 4) Authenticate to HuggingFace + (optionally) W&B
hf auth login                                # paste a read token
wandb login                                  # paste your W&B API key (skip if not using W&B)

# 5) Pull benchmark data (one-time per track)
python data/helper/load_data.py --track news    # → data/news/{train,val,valqa,testqa}.jsonl

# 6) Format the news corpus (REQUIRED for the news task — see "Formatting Data")
python data/helper/format_news.py --input data/news/train.jsonl --output data/news/train.jsonl --overwrite
python data/helper/format_news.py --input data/news/val.jsonl   --output data/news/val.jsonl   --overwrite

# 7) Train
python train.py --config configs/tracks/news_gemma3_1b_lora128.yaml --framework hf
# (or --framework kd if you installed jax-gpu)
```

That's all of the *required* steps. Default configs already specify model, batch sizes, learning rate, eval cadence, etc., so you don't have to touch any yaml unless you want to. The remaining sections describe what each piece does and how to customize it.

---

## Repository Layout

```
ContinuousBenchEval/
├── train.py                    # Training entry point
├── evaluate.py                 # Standalone eval entry point
├── pyproject.toml              # Package definition
├── setup_env.sh                # One-command conda env setup
├── .gitignore
│
├── configs/
│   ├── base/
│   │   ├── tasks/              # Data paths + task-specific eval settings
│   │   │   ├── geminon.yaml
│   │   │   └── news.yaml
│   │   └── models/             # Model + optimizer + training defaults
│   │       ├── gemma3_{1b,4b,12b}_full.yaml
│   │       └── gemma3_{1b,4b,12b}_lora128.yaml
│   ├── tracks/                 # Composable tracks: task × model × adapter
│   │   ├── geminon_gemma3_1b_full.yaml        # _base lists a task + a model
│   │   ├── geminon_gemma3_4b_lora128.yaml
│   │   └── news_gemma3_12b_lora128.yaml
│   └── prompts/                # Few-shot prefixes loaded via prompt_prefix_file
│       ├── geminon.txt
│       └── news.txt
│
├── scripts/
│   └── plot_runs.py            # Quick plot of valqa/eval_loss/train_loss across runs
│
├── data/
│   ├── helper/
│   │   ├── load_data.py        # Pull data from HF Hub (run from repo root)
│   │   ├── download.yaml       # Recipe: which files to pull from which HF repo
│   │   └── format_news.py      # Title/Date/Article formatting (REQUIRED for news track)
│   └── <track>/*.jsonl         # (gitignored, downloaded by load_data.py)
│
├── requirements/
│   ├── torch-gpu.txt           # HF/TRL on GPU (CUDA 12.4)
│   ├── jax-gpu.txt             # Kauldron on GPU
│   ├── jax-tpu.txt             # Kauldron on TPU
│   └── wandb.txt               # Optional wandb support
│
└── src/cbe/                    # Main package
    ├── config.py               # YAML -> dataclasses (with _base inheritance)
    ├── data/                   # Shared formatters + KD/HF data pipelines
    ├── models/                 # KD (Gemma+) and HF (AutoModel) factories
    ├── trainers/               # KauldronTrainer and HFTrainer wrappers
    ├── logging/                # TB + wandb backends, MultiLogger
    ├── artifacts/              # Standardized local artifact store
    └── eval/                   # QA inference, metrics, parsers, KD evaluator
```

---

## Environment Setup

### Quick start (recommended)

**You must pick exactly one of `torch-gpu`, `jax-gpu`, or `jax-tpu` per env.** They install conflicting frameworks. If you want to try both backends, create two separate envs (different `env_name`).

```bash
git clone ...
cd ContinuousBenchEval

# Pick ONE of the following — each creates a fresh conda env named "cbe":
bash setup_env.sh torch-gpu          # HuggingFace / TRL on GPU
bash setup_env.sh jax-gpu            # Kauldron on GPU
bash setup_env.sh jax-tpu            # Kauldron on TPU
```

#### Optional positional arguments (work for ALL backends above)

`setup_env.sh <backend> [extras] [env_name]`

```bash
# 2nd arg = "wandb" → also installs Weights & Biases support (any backend)
bash setup_env.sh torch-gpu wandb
bash setup_env.sh jax-gpu   wandb

# 3rd arg = custom env name (any backend; must pass empty 2nd arg if no extras)
bash setup_env.sh torch-gpu ""    cbe-hf       # HF env named "cbe-hf"
bash setup_env.sh jax-gpu   ""    cbe-kd       # KD env named "cbe-kd"
bash setup_env.sh jax-gpu   wandb cbe-kd       # both wandb + custom name
```

Each invocation creates a fresh conda env with Python 3.11 and all backend-specific dependencies.

> **Don't forget to activate it.** Every new terminal shell needs `conda activate <env_name>` (default: `cbe`) before running any of the train/eval/data-loader commands in this README.

### What gets installed

| Backend | torch | jax | kauldron | gemma | trl/peft | Key pins |
|---------|-------|-----|----------|-------|----------|----------|
| `torch-gpu` | 2.4-2.5 (cu124) | - | - | - | trl, peft<0.15 | setuptools<81 |
| `jax-gpu` | - | 0.8.2 (cuda12) | 1.3.0 | latest | - | typeguard==4.4.1, setuptools<81 |
| `jax-tpu` | - | latest (tpu) | 1.3.0 | latest | - | typeguard==4.4.1, setuptools<81 |

### GPU/TPU notes

- **KD on GPU**: JAX auto-discovers all visible GPUs. FSDP shards params across them. No special launcher needed.
- **KD on TPU**: Native JAX, handles sharding automatically.
- **HF on single GPU**: `python train.py --config ... --framework hf`
- **HF on multi-GPU**: `torchrun --nproc_per_node=N train.py --config ... --framework hf` (DDP)
- **GPU selection**: `CUDA_VISIBLE_DEVICES=0,1 python train.py ...`

### NVIDIA CUDA library path (jax-gpu only)

`setup_env.sh` registers a conda activation hook that puts pip-installed NVIDIA libs on `LD_LIBRARY_PATH`. If you create the env manually, you may need:

```bash
export LD_LIBRARY_PATH=$(find $CONDA_PREFIX/lib/python3.11/site-packages/nvidia -name lib -type d | tr '\n' ':')$LD_LIBRARY_PATH
```

---

## Authentication (HuggingFace + W&B)

Two services need credentials before training works. Both are one-time per machine; tokens persist to disk.

### HuggingFace (required for Gemma)

Gemma model weights are **gated** on HuggingFace. You must:

1. **Click "Agree and access" once per Gemma model** on the HF website. The repo defaults to Gemma3, so visit at minimum:
   - https://huggingface.co/google/gemma-3-1b-pt
   - https://huggingface.co/google/gemma-3-4b-pt (if using 4B)
   - https://huggingface.co/google/gemma-3-12b-pt (if using 12B)
2. **Get a read token** at https://huggingface.co/settings/tokens
3. **Persist it locally** so subprocesses can read it:
   ```bash
   hf auth login           # paste token interactively (writes ~/.cache/huggingface/token)
   # or:
   export HF_TOKEN=hf_...  # add to ~/.bashrc to make it permanent
   ```

The same token is used by `data/helper/load_data.py` to pull benchmark data and by the trainer to download Gemma weights at runtime. Without it, you'll see `401 Unauthorized` or `GatedRepoError` when training starts.

### Weights & Biases (optional)

If you installed the `wandb` extra (`bash setup_env.sh <backend> wandb`):

```bash
wandb login   # paste your API key from https://wandb.ai/authorize
```

The credential persists to `~/.netrc`. Runs land at `wandb.ai/<your-username>/<project_name>` where `project_name` comes from the YAML config. Skip this entirely if you only want TensorBoard.

---

## Downloading Data

Benchmark data is hosted on HuggingFace:

- `ContinuousBench/News` (tag `v5`) — news articles + QA
- `ContinuousBench/Geminon` (tag `v9`) — Geminon articles + QA

The downloader script lives at **`data/helper/load_data.py`** (and so does the recipe `data/helper/download.yaml`). Make sure you've run `hf auth login` first (see [Authentication](#authentication-huggingface--wandb)).

```bash
# Always always run from the repo root, NOT from data/helper/.
# (output paths are repo-root-relative)

# Just one track (recommended — pass --track explicitly)
python data/helper/load_data.py --track news
python data/helper/load_data.py --track geminon

# Download all tracks listed in the recipe (no --track flag)
python data/helper/load_data.py

# Override corpus / QA size (small/medium/large where supported)
python data/helper/load_data.py --track geminon --corpus large --qa medium

# Debug: list every file in the HF repo for a track
python data/helper/load_data.py --list news
python data/helper/load_data.py --list geminon
```

The download recipe (`data/helper/download.yaml`) maps HF repo paths to local filenames. After running the loader, files always land at:

```
data/<track>/train.jsonl
data/<track>/val.jsonl
data/<track>/valqa.jsonl
data/<track>/testqa.jsonl
```

Files are written to `data/<track>/`, not `data/helper/<track>/`. If you see them in `helper/`, you're running an out-of-date version of the script — re-pull `main`. The track YAML configs hard-code these `data/<track>/...` paths, so they only work after the loader has run.

---

## Formatting Data

### News track — REQUIRED

The news data on HuggingFace ships as multi-column JSONL (`url`, `hostname`, `title`, `date`, `crawl_date`, `language`, `text`). The `train.py` data pipeline expects a **single-column `{"text": "Title: ...\nDate: ...\nArticle: ..."}`** shape. So after `load_data.py` you **must run the formatter** before the news track will train correctly.

```bash
# In-place rewrite (recommended — keeps the original filenames)
python data/helper/format_news.py --input data/news/train.jsonl --output data/news/train.jsonl --overwrite
python data/helper/format_news.py --input data/news/val.jsonl   --output data/news/val.jsonl   --overwrite

# OR write to a new file and update train_path / val_path in the config
python data/helper/format_news.py --input data/news/train.jsonl --output data/news/train_formatted.jsonl
```

The QA files (`valqa.jsonl`, `testqa.jsonl`) do **not** need formatting — they're already in the right shape.

For raw / dirty input (not from ContinuousBench), pass `--normalize`. ContinuousBench/News is pre-cleaned during curation, so `--normalize` is a no-op on it.

### Geminon track — NOT required

Geminon data ships pre-formatted; the loader output is ready to train on directly.

---

## Configuration System

> **You don't have to write or edit any YAML to use the defaults.** The shipped track configs in `configs/tracks/` cover the common cases (1B/4B/12B × Full/LoRA × news/geminon). To run one of them, just point `train.py` at the file. This section explains how the inheritance works only because you'll want it eventually for new tracks or sweeps.

### File naming convention

```
configs/
├── base/
│   ├── tasks/<task>.yaml                     # data paths + eval settings for that benchmark
│   │   ├── news.yaml                         # the news task
│   │   └── geminon.yaml                      # the geminon task
│   └── models/gemma3_<size>_<adapter>.yaml   # model + optimizer + training defaults
│       ├── gemma3_1b_full.yaml      gemma3_1b_lora128.yaml
│       ├── gemma3_4b_full.yaml      gemma3_4b_lora128.yaml
│       └── gemma3_12b_full.yaml     gemma3_12b_lora128.yaml
└── tracks/<task>_gemma3_<size>_<adapter>.yaml   # composes one task × one model
    ├── news_gemma3_1b_full.yaml          news_gemma3_1b_lora128.yaml
    ├── news_gemma3_4b_full.yaml          news_gemma3_4b_lora128.yaml
    ├── geminon_gemma3_1b_full.yaml       geminon_gemma3_1b_lora128.yaml
    └── ...
```

**Track files are the launch point** — those are what you pass to `--config`. They're tiny: most just say "task X + model Y, plus a few overrides like learning rate and run name". The base files hold the shared defaults that every track inherits.

Configs use **composable base + track inheritance**. Each track combines a task (data + eval) with a model (params + optimizer + training defaults) via a `_base:` list, then layers track-specific overrides.

### Task base (configs/base/tasks/geminon.yaml)

```yaml
data:
  train_path: data/geminon/train.jsonl
  val_path: data/geminon/val.jsonl
  valqa_path: data/geminon/valqa.jsonl
  testqa_path: data/geminon/testqa.jsonl
  sequence_length: 256

eval:
  prompt_prefix_file: configs/prompts/geminon.txt   # loaded verbatim at runtime
  prompt_template: "Q: {question}\nA:"
  max_new_tokens: 32
  batch_size: 32
  temperature: 0.0
  parser: finegrained_geminon
  save_detailed_results: true

logging:
  backends: [tensorboard, wandb]
  project_name: cbe-geminon
```

### Model base (configs/base/models/gemma3_4b_lora128.yaml)

```yaml
_base: base/models/gemma3_4b_full.yaml   # chained base: inherit from full

model:
  lora_rank: 128

optimizer:
  lr: 1.0e-4

training:
  per_device_batch_size: 8
```

### Track config (configs/tracks/geminon_gemma3_4b_lora128.yaml)

```yaml
_base:
  - base/tasks/geminon.yaml                    # data + eval + prompt
  - base/models/gemma3_4b_lora128.yaml         # model + optimizer + training

training:
  sharding: fsdp                # "replicated" | "fsdp" (KD); HF ignores
  # Per-track overrides land here. E.g., tight memory on 2×40GB A100:
  per_device_batch_size: 4
  effective_batch_size: 32      # real total samples per opt step (see below)

logging:
  run_name: geminon/gemma3-4b-lora128
```

Later entries in `_base:` win on conflict. `_base` can also be a single string for single-parent inheritance.

### CLI overrides

Any config field can be overridden from the command line:

```bash
python train.py --config configs/tracks/geminon_gemma3_1b_lora128.yaml --framework kd \
    --override optimizer.lr=1e-4 \
    --override training.per_device_batch_size=8 \
    --override logging.run_name=geminon/my-experiment \
    --override "logging.backends=[tensorboard,wandb]"
```

### Framework is a CLI flag, not in the config

The same config works for both backends:

```bash
python train.py --config configs/tracks/geminon_gemma3_1b_lora128.yaml --framework kd
python train.py --config configs/tracks/geminon_gemma3_1b_lora128.yaml --framework hf
```

### Batch size semantics (important — the same yaml means different things to HF vs KD)

There are two batch-size fields in every config. **Only one of them — `effective_batch_size` — has a stable cross-framework meaning. The other does not.**

#### `effective_batch_size` — same meaning everywhere

> **Real total samples per optimizer step, summed across all chips.** Same number → same real gradient signal, regardless of HF/KD or chip count.

This is the number you should think about when comparing runs.

#### `per_device_batch_size` — framework-dependent meaning

The number of samples that fit through one **forward/backward pass** before any gradient accumulation. But what counts as "one pass" differs:

| Framework | What `per_device_batch_size = X` actually means |
|---|---|
| **HF / PyTorch (DDP)** | Each GPU independently processes X samples per fwd/bwd. With N GPUs in DDP, **N×X total samples are processed per data-iter**. |
| **KD / JAX (FSDP)** | X is the **global** per-iter batch — Kauldron shards those X samples across the FSDP mesh. With N chips, each chip sees X/N samples per fwd/bwd. **Total X samples processed per data-iter, regardless of chip count.** |

So with the same `per_device_batch_size: 4`:
- **HF on 4 GPUs** → 16 samples per data-iter
- **KD on 4 chips** → 4 samples per data-iter (1 per chip)

That asymmetry forces the gradient-accumulation math to differ:

```
HF:  grad_accum = effective_batch_size // (per_device × world_size)
KD:  grad_accum = effective_batch_size // per_device
```

The trainer derives `grad_accum` automatically — you don't pass it. Just set `effective_batch_size` and `per_device_batch_size`, and you'll get the right `ga` for whichever framework you launch with.

**Worked example.** `per_device_batch_size: 4, effective_batch_size: 32`:

| Setup | per data-iter | grad_accum | real effective per opt step |
|---|---:|---:|---:|
| HF, 1 GPU | 4 | 8 | 32 ✓ |
| HF, 2 GPUs torchrun | 8 | 4 | 32 ✓ |
| HF, 4 GPUs torchrun | 16 | 2 | 32 ✓ |
| KD, 1 chip | 4 | 8 | 32 ✓ |
| KD, 2 chips FSDP | 4 (= 2/chip) | 8 | 32 ✓ |
| KD, 4 chips FSDP | 4 (= 1/chip) | 8 | 32 ✓ |

Punchline: `effective_batch_size` is what stays constant. `per_device_batch_size` is essentially a **memory knob** — bigger means fewer accumulation steps but more activation memory per chip; smaller is the opposite. Tune it for whatever fits on your hardware.

> **Heads-up on `effective` vs `eval_per_device_batch_size`.** Eval uses its own knob (`training.eval_per_device_batch_size`). Same framework asymmetry as above.

### FSDP and memory guidance (40 GB A100)

Typical knobs, with KD semantics (HF equivalents scale per_device by num_gpus):

| model | adapter | 1 chip | 2 chip FSDP | 4 chip FSDP |
|---|---|---|---|---|
| 1B | LoRA | per_device=8, ga=4 | n/a | n/a |
| 1B | Full | per_device=16, ga=2 | n/a | n/a |
| 4B | LoRA | per_device=2, ga=16 (tight) | **per_device=4, ga=8** | per_device=8, ga=4 |
| 4B | Full | doesn't fit | per_device=1, ga=32 (very tight) | per_device=2, ga=16 |
| 12B | LoRA | doesn't fit | per_device=1, ga=32 (tight) | per_device=2, ga=16 |
| 12B | Full | needs 8+ chips or DeepSpeed ZeRO-3+offload | infeasible | infeasible |

The KD path allocates `optax.MultiSteps` `acc_grads` at full-param shape (even when only LoRA is trained), so grad_accum>1 adds ~2 bytes per base param to peak memory. If you have headroom, set per_device big enough to get ga=1.

---

## Training

> **Always activate the env first.** Every command in this section assumes `conda activate <env_name>` has already been run in your current shell (default env is `cbe`). Running `train.py` from a non-activated shell will hit `ModuleNotFoundError: cbe` or, worse, find a different Python and silently use it.

```bash
conda activate cbe     # or whatever you named your env via setup_env.sh
```

### Single GPU — HuggingFace/TRL

```bash
python train.py --config configs/tracks/news_gemma3_1b_lora128.yaml --framework hf
```

### Multi-GPU — HuggingFace/TRL (DDP)

```bash
# 4 GPUs
torchrun --nproc_per_node=4 train.py \
    --config configs/tracks/news_gemma3_1b_lora128.yaml --framework hf

# Specific GPUs
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train.py \
    --config configs/tracks/news_gemma3_1b_lora128.yaml --framework hf
```

DDP is HF Trainer's default when launched via `torchrun`. Only rank 0 logs to wandb/TB and writes metrics — no duplicate entries.

### Kauldron (JAX) — all visible GPUs automatically

```bash
python train.py --config configs/tracks/news_gemma3_1b_lora128.yaml --framework kd

# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --config configs/tracks/news_gemma3_1b_lora128.yaml --framework kd
```

JAX auto-discovers all visible devices and shards via FSDP. No special launcher needed.

### What happens during training

Every `eval_every` steps:

1. **Eval loss** — next-token cross-entropy on `val_path` (capped at `eval_num_batches` batches)
2. **QA eval on valqa** — generate answers, compute exact match + fuzzy match
3. **QA eval on testqa** — same

All metrics are logged to TensorBoard and/or wandb, and appended to `metrics/eval_results.jsonl`. If `save_detailed_results: true`, per-example predictions are saved to `eval_details/<qa_set>_step_<N>.jsonl`.

---

## Evaluation

### Standalone eval on a checkpoint

```bash
# HuggingFace checkpoint (auto-detects LoRA from adapter_config.json)
python evaluate.py --framework hf \
    --checkpoint outputs/cbe/geminon/.../checkpoints/checkpoint-2000 \
    --model gemma3-1b-pt \
    --qa_data data/geminon/testqa.jsonl \
    --parser geminon \
    --num_examples 10

# Kauldron checkpoint (with LoRA — does split/merge of base + adapter)
python evaluate.py --framework kd \
    --checkpoint outputs/cbe/geminon/.../checkpoints/ckpt_2000 \
    --model gemma3-1b-pt --lora_rank 128 \
    --qa_data data/geminon/testqa.jsonl \
    --parser geminon

# Save detailed per-example results
python evaluate.py --framework hf \
    --checkpoint outputs/cbe/geminon/.../checkpoints/checkpoint-2000 \
    --model gemma3-1b-pt \
    --qa_data data/geminon/testqa.jsonl \
    --parser geminon \
    --save_details results.jsonl
```

### Answer parsers

The `--parser` flag selects the answer-matching strategy:

- **`finegrained_geminon`**: tailored for Geminon QA. Normalization is
  `lower().strip().strip('.')` on both prediction and gt.
  - Types question (`"types of"`): splits gt on `"and"` (e.g. `"Normal and
    Flying"` → `["normal", "flying"]`); `fuzzy_match` is True iff every gt
    type appears as a substring of the normalized prediction.
  - All other questions (classification, evolution, moves, abilities,
    numerical stats/height/weight): `fuzzy_match` is True iff the normalized
    gt substring is contained in the normalized prediction.
  - `exact_match` requires full-string equality after normalization.
- **`default`** (or omit): lowercase exact match + substring fuzzy match,
  with `.rstrip('.')` normalization.

---

## Logging and Monitoring

### TensorBoard (local, no login)

```bash
# View one run
tensorboard --logdir outputs/cbe/geminon/debug-kd --port 6006

# Compare all runs in a project
tensorboard --logdir outputs/cbe

# Remote machine — SSH tunnel
ssh -L 6006:localhost:6006 user@host
# then open http://localhost:6006
```

For KD runs, Kauldron writes to separate subdirs per evaluator (`train/`, `eval_loss/`, `qa_valqa/`, `qa_testqa/`). Point TB at the run root to see all of them.

### Weights & Biases

W&B is opt-in. To enable it:

1. Install the wandb extra at env-creation time (any backend):
   ```bash
   bash setup_env.sh torch-gpu wandb     # or jax-gpu wandb / jax-tpu wandb
   ```
2. Authenticate one-time per machine (see [Authentication](#authentication-huggingface--wandb)):
   ```bash
   wandb login   # paste API key from https://wandb.ai/authorize
   ```
3. Add `wandb` to the `logging.backends` list (already on by default in the news + geminon task configs):
   ```yaml
   logging:
     backends: [tensorboard, wandb]
     project_name: cbe
     run_name: news/my-experiment
   ```
   Or pass via CLI:
   ```
   --override "logging.backends=[tensorboard,wandb]"
   ```

Runs upload to `wandb.ai/<your-username>/<project_name>`.

### stdout/stderr logs

All terminal output is automatically tee'd to `outputs/<project>/<run>/logs/train.log`. Tail it live:

```bash
tail -f outputs/cbe/geminon/my-run/logs/train.log
```

### Plotting across runs

```bash
python scripts/plot_runs.py outputs/<project>/<task>
# writes <task>/runs_plot.png with 3 panels: valqa fuzzy match, eval loss, train loss
```

The script auto-discovers every subdir under the given task dir that has `metrics/eval_results.jsonl`, infers framework (HF vs KD) from file shape, and reads train/eval loss from HF's `trainer_state.json` log_history or KD's TB event files. Multiple TB event files per run (from resume) are merged automatically. X-axis is normalized to optimizer steps.

---

## Output Layout

Every run writes to a standardized directory:

```
outputs/<project_name>/<run_name>/
├── config.yaml                          # Frozen copy of the resolved config
├── logs/
│   ├── train.log                        # Full stdout+stderr
│   └── tensorboard/                     # TB event files (from MultiLogger)
├── train/                               # KD-only: train loss events
├── eval_loss/                           # KD-only: eval loss events
├── checkpoints/
│   ├── checkpoint-2000/                 # HF naming
│   ├── ckpt_2000/                       # KD naming
│   └── latest -> checkpoint-2000        # Symlink to most recent
├── metrics/
│   └── eval_results.jsonl               # Append-only: {step, eval_loss, valqa_exact_match, ...}
└── eval_details/                        # Per-example QA results (opt-in)
    ├── valqa_step_002000.jsonl
    └── testqa_step_002000.jsonl
```

### eval_results.jsonl format

```json
{"step": 2000, "timestamp": "2026-04-16T...", "eval_loss": 3.298, "valqa_exact_match": 0.42, "valqa_fuzzy_match": 0.51, "testqa_exact_match": 0.38, "testqa_fuzzy_match": 0.47}
```

### eval_details per-example format

```json
{"prompt": "Q: What are the types of Pidgey?\nA:", "question": "What are the types of Pidgey?", "raw_prediction": " Normal and Flying.\n\nQ: ...", "parsed_prediction": "Normal and Flying", "ground_truth": "Normal/Flying", "exact_match": true, "fuzzy_match": true}
```

---

## Adding a New Track

1. Place data in `data/<track>/{train,val,valqa,testqa}.jsonl` (or add entries to `data/helper/download.yaml` and run `python data/helper/load_data.py --track <track>`)
2. Create a task base file `configs/base/tasks/<track>.yaml` (copy `configs/base/tasks/news.yaml` as a template; update `data.*_path`, `eval.parser`, etc.)
3. Create a track config `configs/tracks/<track>_gemma3_<size>_<adapter>.yaml` (copy any of the existing `news_gemma3_*` files); set its `_base:` list to point at the new task + the model base you want
4. Train: `python train.py --config configs/tracks/<track>_gemma3_<size>_<adapter>.yaml --framework hf`

## Adding a New Model

- **HF**: Set `model.name` to any HuggingFace hub ID (`meta-llama/Llama-3.1-8B`, `mistralai/Mistral-7B-v0.3`, etc.). Short Gemma names (`gemma3-1b-pt`) are auto-mapped to hub IDs.
- **KD**: Implement the `JaxModelFactory` protocol in `src/cbe/models/kd_models.py`. The Gemma factory there is a reference implementation.

---

## Known Limitations

- **KD multi-node GPU** is not supported (Kauldron + JAX FSDP is tricky on multi-node GPU). KD targets TPU or single-node GPU.
- **HF eval runs the full val dataset** — unlike KD's `eval_num_batches` cap, HF Trainer always evaluates all examples. For large val sets, this can be slow.
- **QA eval is slow** — autoregressive generation at `max_new_tokens` per question. For 3000+ QA records, expect 5-30 min per eval pass depending on batch size and GPU count.
- **`evaluate.py`** doesn't read the prompt_prefix from the YAML config — you'd need to pass it via `--prompt_prefix` (impractical for long few-shot prefixes). For post-hoc eval with few-shot prompts, use the training pipeline with `num_train_steps=0`.
- **Gradient checkpointing is HF-only.** The `training.gradient_checkpointing` field is wired into `SFTConfig`; the KD path doesn't apply `nn.remat` to the Gemma backbone, so the flag has no effect there. This is the main reason 12B Full (and to a lesser extent 4B Full) are infeasible on 4×40 GB A100 — the HF path can squeeze it via gradient checkpointing + DeepSpeed ZeRO-3 offload, the KD path cannot today.
- **`optax.MultiSteps.acc_grads` ignores the LoRA freeze mask** and allocates a full-param-shaped gradient accumulator. A LoRA run with `grad_accum > 1` still costs one base-shaped tensor of peak memory. Workaround: size per_device high enough that `grad_accum = 1`.
- **KD LoRA wraps more modules than HF PEFT.** `gm.nn.LoRA` replaces every `nn.Dense`/`nn.Einsum` in the Gemma backbone (including the embedder, which is tied to `lm_head` on Gemma3), so KD-LoRA has ~3× the trainable surface of HF-LoRA's default `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`. This is why KD-LoRA ≈ KD-Full in eval quality while HF-LoRA lags HF-Full; it's a capacity difference, not a bug. Add embeddings to HF's `target_modules` if you want parity.
- **KD path lacks throughput metrics in `eval_results.jsonl`.** Train loss/speed are only in TB/wandb (see `{run}/train/events.out.tfevents.*`, tag `losses/xentropy`). Use `scripts/plot_runs.py` for a unified view across runs.

---

## License

Apache 2.0.
