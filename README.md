# ContinuousBenchEval

A framework-agnostic training and evaluation harness for continual learning benchmarks. Train language models on text corpora and evaluate memorization via QA — using **Kauldron (JAX)** or **HuggingFace/TRL (PyTorch)** as the backend, with the same config, same data, and same metrics.

---

## Table of Contents

- [Repository Layout](#repository-layout)
- [Environment Setup](#environment-setup)
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
│   ├── load_data.py            # Pull data from HF Hub
│   ├── download.yaml           # Recipe: which files to pull
│   ├── helper/
│   │   └── format_news.py      # Title/Date/Article formatting
│   └── <track>/*.jsonl         # (gitignored, downloaded)
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

```bash
# Clone the repo
git clone git@github.com:plau666/ContinuousBenchEval.git
cd ContinuousBenchEval

# HuggingFace / TRL on GPU (creates conda env "cbe")
bash setup_env.sh torch-gpu

# Kauldron on GPU (creates conda env "cbe")
bash setup_env.sh jax-gpu

# Kauldron on TPU
bash setup_env.sh jax-tpu

# Add wandb support to any of the above
bash setup_env.sh torch-gpu wandb

# Custom env name
bash setup_env.sh jax-gpu "" my-env-name
```

Each command creates a fresh conda env with Python 3.11 and all dependencies.

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

## Downloading Data

Benchmark data is hosted on HuggingFace:

- `ContinuousBench/News` (tag `v5`) — news articles + QA
- `ContinuousBench/Geminon` (tag `v9`) — Geminon articles + QA

```bash
# Authenticate (one-time)
hf auth login

# Download all tracks per the recipe in data/download.yaml
python data/load_data.py

# Just one track
python data/load_data.py --track news

# Override corpus size (small/medium/large)
python data/load_data.py --track geminon --corpus large --qa medium

# Debug: list all files in a repo
python data/load_data.py --list geminon
```

The download recipe (`data/download.yaml`) maps HF repo paths to local filenames:

```yaml
tracks:
  news:
    repo: ContinuousBench/News
    revision: v5
    files:
      train.jsonl:  corpus_small/train.jsonl
      val.jsonl:    corpus_small/val.jsonl
      valqa.jsonl:  qa/val.jsonl
      testqa.jsonl: qa/test.jsonl
```

Files land at `data/<track>/{train,val,valqa,testqa}.jsonl`.

---

## Formatting Data

### News track

Raw news records have `title`, `date`, `text` fields. To format into `"Title: ...\nDate: ...\nArticle: ..."`:

```bash
python data/helper/format_news.py \
    --input data/news/train.jsonl \
    --output data/news/train_formatted.jsonl

# For raw/dirty input (not from ContinuousBench), add --normalize
python data/helper/format_news.py --input raw.jsonl --output out.jsonl --normalize
```

Note: ContinuousBench/News data is already cleaned during curation. The `--normalize` flag is a no-op on it.

---

## Configuration System

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
  parser: geminon
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

### Batch size semantics (important, not symmetric)

`effective_batch_size` is defined consistently: **real total samples per optimizer step, across all chips**. The same yaml value means the same real batch on HF and KD.

`per_device_batch_size` is **framework-dependent** because the two stacks load data differently:

| | HF / PyTorch DDP | KD / JAX FSDP |
|---|---|---|
| per_device_batch_size means | truly per-device (each GPU loads its own batch) | global per-iter batch (Kauldron shards across the mesh) |
| real effective = | `per_device × world_size × grad_accum` | `per_device × grad_accum` |
| scaling chips = | **more samples per step** (unless you reduce per_device or grad_accum) | **same samples per step, each chip sees fewer** |

The HF trainer reads `WORLD_SIZE` at launch and derives `grad_accum = effective_batch_size // (per_device × world_size)`. The KD trainer uses `grad_accum = effective_batch_size // per_device`. Either way, one yaml produces the right real effective batch.

Example, `per_device_batch_size: 4, effective_batch_size: 32`:
- HF, 1 GPU: ga=8 → real 32 ✓
- HF, 2 GPU torchrun: ga=4 → real 32 ✓
- HF, 4 GPU torchrun: ga=2 → real 32 ✓
- KD, 1 chip: ga=8 → real 32 ✓
- KD, 2 chip FSDP: ga=8 (each chip processes 2 samples per data-iter) → real 32 ✓

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

### Single GPU — HuggingFace/TRL

```bash
conda activate cbe     # or cbe-hf if separate envs
python train.py --config configs/tracks/geminon.yaml --framework hf
```

### Multi-GPU — HuggingFace/TRL (DDP)

```bash
# 4 GPUs
torchrun --nproc_per_node=4 train.py \
    --config configs/tracks/geminon.yaml --framework hf

# Specific GPUs
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train.py \
    --config configs/tracks/geminon.yaml --framework hf
```

DDP is HF Trainer's default when launched via `torchrun`. Only rank 0 logs to wandb/TB and writes metrics — no duplicate entries.

### Kauldron (JAX) — all visible GPUs automatically

```bash
conda activate cbe
python train.py --config configs/tracks/geminon.yaml --framework kd

# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --config configs/tracks/geminon.yaml --framework kd
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

The `--parser` flag selects question-type-aware matching:

- **`geminon`**: Dispatches by question type:
  - Types (`"types of"`) — set equality, any delimiter (`/`, `,`, `and`)
  - Stats (`"stat of"`, `"height"`, `"weight"`) — numerical, 0.1% relative tolerance
  - Classification (`"classification of"`) — strips trailing "Geminon"
  - Evolution line (`"evolution line of"`) — ordered token match, any delimiter
  - Moves — default lowercase string match
- **`default`** (or omit): Lowercase exact match + substring fuzzy match

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

```bash
# One-time login
wandb login

# Enable in config
logging:
  backends: [tensorboard, wandb]
  project_name: cbe
  run_name: geminon/my-experiment

# Or via CLI override
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

1. Place data in `data/<track>/{train,val,valqa,testqa}.jsonl` (or add entries to `data/download.yaml`)
2. Copy `configs/tracks/news.yaml` to `configs/tracks/<track>.yaml`
3. Update `_base`, `data.*_path`, `logging.run_name`, and the `eval` block
4. Train: `python train.py --config configs/tracks/<track>.yaml --framework hf`

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
