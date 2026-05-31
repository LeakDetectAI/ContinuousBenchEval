# ContinuousBenchEval

A framework-agnostic training and evaluation harness for continual learning benchmarks. Train language models on text corpora and evaluate memorization via QA — using **Kauldron (JAX)** or **HuggingFace/TRL (PyTorch)** as the backend, with the same config, same data, and same metrics.

> **Pick exactly one backend before you start.** The repo supports Kauldron (JAX) and HuggingFace/TRL (PyTorch); the rest of this README is organized so the same yaml works for either, but the conda env you create installs only one of them. Mixing isn't supported in a single env. If you want to try both, create two envs (e.g. `cbe-kd` and `cbe-hf`) — see [Environment Setup](#environment-setup).

---

## Table of Contents

- [TL;DR — End-to-end first run](#tldr--end-to-end-first-run)
- [Environment Setup](#environment-setup)
- [Authentication (HuggingFace + W&B)](#authentication-huggingface--wandb)
- [Downloading Data](#downloading-data)
- [Formatting Data](#formatting-data)
- [Training recipes](#training-recipes)
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
git clone <repo-url>
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

## Environment Setup

### Quick start (recommended)

**You must pick exactly one of `torch-gpu`, `jax-gpu`, or `jax-tpu` per env.** They install conflicting frameworks. If you want to try both backends, create two separate envs (different `env_name`).

```bash
git clone <repo-url>
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

## Training recipes

The shipped track configs in `configs/tracks/` are *recipes* — one YAML per (task × model × adapter) combo, with sensible defaults already baked in. **You almost never need to touch these.** Just pick one and run it.

```bash
# Available recipes
ls configs/tracks/
#   geminon_gemma3_1b_full.yaml       news_gemma3_1b_full.yaml
#   geminon_gemma3_1b_lora128.yaml    news_gemma3_1b_lora128.yaml
#   geminon_gemma3_4b_full.yaml       news_gemma3_4b_full.yaml
#   geminon_gemma3_4b_lora128.yaml    news_gemma3_4b_lora128.yaml

# Run one
python train.py --config configs/tracks/news_gemma3_1b_lora128.yaml --framework hf
# (or --framework kd if you installed jax-gpu)
```

The recipe's filename tells you everything: `<task>_<model>_<adapter>.yaml`. Each one inherits shared defaults from `configs/base/{tasks,models}/`, so the track file itself stays tiny (data + a few overrides like run name).

> **If you do need to tweak something:** any field can be overridden from the CLI (`--override optimizer.lr=1e-4`, etc.), or you can edit the recipe directly. The two batch-size knobs to know about are `effective_batch_size` (real samples per optimizer step — same meaning everywhere) and `per_device_batch_size` (memory knob; lower it if you OOM, raise it for fewer grad-accum steps). Defaults fit 1× or 2× 40 GB A100 for most (model, adapter) pairs. To lower it on the fly:
>
> ```bash
> python train.py --config configs/tracks/news_gemma3_4b_full.yaml --framework hf \
>     --override training.per_device_batch_size=2
> ```

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

### LLM-as-judge re-scoring (optional)

Substring/exact-match metrics undercount paraphrased correct answers. `llm_evaluate.py` re-scores the `eval_details/*.jsonl` files produced during training with Gemini as a judge, adding an `llm_match: bool` field per record and writing a stratified summary.

```bash
pip install google-genai
cp secrets/gemini_keys.txt.example secrets/gemini_keys.txt   # then add your keys

# Judge one per-example results file
python llm_evaluate.py \
    --input outputs/<project>/<run>/eval_details/testqa_step_001000.jsonl
# → writes testqa_step_001000_llm_judged.jsonl + testqa_step_001000_summary.jsonl
```

The script reads API keys from `secrets/gemini_keys.txt` (one per line, multi-key round-robin recommended for higher quota), or the `GEMINI_API_KEY` / `GOOGLE_API_KEY` env vars. Uses `gemini-2.5-flash-lite` with `temperature=0` (deterministic) by default. See `python llm_evaluate.py --help` for resume, concurrency, and stratification options.

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

Terminal output is also tee'd to `outputs/<project>/<run>/logs/train.log` — tail it live with `tail -f`.

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

Record schemas (one JSON object per line):

```json
// metrics/eval_results.jsonl
{"step": 2000, "timestamp": "...", "eval_loss": 3.298, "valqa_exact_match": 0.42, "valqa_fuzzy_match": 0.51, "testqa_exact_match": 0.38, "testqa_fuzzy_match": 0.47}

// eval_details/<set>_step_<N>.jsonl
{"prompt": "Q: ...\nA:", "question": "...", "raw_prediction": " ...", "parsed_prediction": "...", "ground_truth": "...", "exact_match": true, "fuzzy_match": true}
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

<!-- ---

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

Apache 2.0. -->
