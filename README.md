# ContinuousBenchEval

A framework-agnostic training and evaluation harness for continual learning benchmarks. Train language models on text corpora and evaluate them on QA tasks — with the same config, the same data, and the same metrics, regardless of whether you use **Kauldron (JAX)** or **HuggingFace/TRL (PyTorch)**.

---

## Why this exists

Most fine-tuning repos lock you into one framework, one model family, and one logging stack. This one doesn't:

- **Two trainer backends**: Kauldron (TPU/JAX) or HuggingFace + TRL SFTTrainer (PyTorch/GPU). Switch with a CLI flag.
- **Two logging backends**: TensorBoard, Weights & Biases, or both at once.
- **Any model**: any HuggingFace causal LM on the torch side, Gemma 270M–27B (and extensible) on the KD side.
- **One config**: a single YAML defines a training run; the framework is chosen at launch time.
- **Standardized outputs**: every run lands in the same directory layout, so downstream tooling (eval scripts, dashboards, comparisons) works uniformly.

---

## Repository layout

```
ContinuousBenchEval/
├── train.py                    # Training entry point
├── evaluate.py                 # Standalone eval entry point
├── pyproject.toml              # Package definition
├── setup_env.sh                # One-command conda env setup
│
├── configs/
│   ├── base/                   # Model + optimizer + training defaults
│   │   ├── gemma3_1b_lora128.yaml
│   │   ├── gemma3_1b_full.yaml
│   │   └── llama3_8b_lora64.yaml
│   └── tracks/                 # Per-task overrides (data paths, run name, eval)
│       ├── news.yaml
│       └── geminon.yaml
│
├── data/
│   ├── README.md               # Data layout guide
│   ├── load_data.py            # Pulls data from HF Hub (pl666/ContinuousBench)
│   └── <track>/*.jsonl         # (gitignored, downloaded)
│
├── requirements/               # Per-backend pip requirements
│   ├── torch-gpu.txt
│   ├── jax-gpu.txt
│   ├── jax-tpu.txt
│   └── wandb.txt
│
├── scripts/
│   └── format_data.py          # Clean / dedupe / split raw JSONL
│
└── src/cbe/                    # Main package
    ├── config.py               # YAML → dataclasses (with _base inheritance)
    ├── data/                   # Shared formatters + KD/HF data pipelines
    ├── models/                 # KD (Gemma+) and HF (AutoModel) factories
    ├── trainers/               # KauldronTrainer and HFTrainer wrappers
    ├── logging/                # TB + wandb backends, MultiLogger
    ├── artifacts/              # Standardized local artifact store
    └── eval/                   # QA inference + exact/fuzzy match metrics
```

---

## Installation

Pick the backend you need and run the setup script. It creates a fresh conda env and installs everything.

```bash
# HuggingFace / TRL on GPU
bash setup_env.sh torch-gpu

# Kauldron on TPU
bash setup_env.sh jax-tpu

# Kauldron on GPU (single node)
bash setup_env.sh jax-gpu

# Add wandb support to any of the above
bash setup_env.sh torch-gpu wandb
```

Each setup creates an env named `cbe-<backend>` (e.g. `cbe-torch-gpu`). Activate with `conda activate cbe-torch-gpu`.

> **GPU vs TPU note**: Kauldron on GPU is brittle (FSDP works on single-node only, no multi-node). For multi-GPU/multi-node, use the `torch-gpu` backend with `torchrun`.

---

## Downloading data

The benchmark tracks live in a private HF dataset repo: `pl666/ContinuousBench`. Authenticate, then run the loader:

```bash
hf auth login                              # one-time, paste your HF token
python data/load_data.py                   # download all tracks
python data/load_data.py --track news      # or just one
python data/load_data.py --list            # debug: list files in the repo
```

Files land at `data/<track>/{train,val,valqa,testqa}.jsonl`. They're gitignored — you re-download per machine.

See [data/README.md](data/README.md) for the expected JSONL format.

---

## Quick start

```bash
# HuggingFace / TRL trainer
python train.py --config configs/tracks/news.yaml --framework hf

# Kauldron trainer
python train.py --config configs/tracks/news.yaml --framework kd

# Multi-GPU with HF/TRL
torchrun --nproc_per_node=4 train.py --config configs/tracks/news.yaml --framework hf

# Override any field from CLI
python train.py --config configs/tracks/news.yaml --framework hf \
    --override optimizer.lr=5e-6 \
    --override training.per_device_batch_size=4 \
    --override logging.run_name=news/my-experiment-v2
```

---

## How configs work

Configs use **base + track inheritance**. A *base* config defines model architecture, optimizer, sharding, and other framework-agnostic training params. A *track* config picks a base via `_base:` and adds dataset paths, run name, eval settings, and any overrides.

```yaml
# configs/base/gemma3_1b_lora128.yaml
model:
  name: gemma3-1b-pt
  lora_rank: 128
optimizer:
  lr: 1e-5
  schedule: cosine
training:
  effective_batch_size: 32
  per_device_batch_size: 8
  sharding: fsdp
```

```yaml
# configs/tracks/news.yaml
_base: base/gemma3_1b_lora128.yaml

training:
  num_train_steps: 50000
  eval_every: 1000

data:
  train_path: data/news/train.jsonl
  val_path: data/news/val.jsonl
  valqa_path: data/news/valqa.jsonl
  testqa_path: data/news/testqa.jsonl

logging:
  backends: [tensorboard, wandb]
  project_name: cbe
  run_name: news/gemma3-1b-lora128

eval:
  prompt_prefix: ""
  prompt_template: "Q: {question}\nA:"
  max_new_tokens: 50
  temperature: 0.0   # 0 = greedy
```

**The `framework` field is not in the YAML** — it's a CLI flag (`--framework hf|kd`). The same config file runs on both backends.

### Gradient accumulation

Set `effective_batch_size` to the batch size you want gradients computed over, and `per_device_batch_size` to whatever fits on your hardware. Gradient accumulation steps are computed automatically.

---

## Standardized output layout

Every run, regardless of framework, writes to:

```
outputs/<project_name>/<run_name>/
├── config.yaml              # Frozen copy of the resolved config
├── logs/
│   ├── tensorboard/         # TB event files
│   └── wandb/               # wandb run files (if enabled)
├── checkpoints/
│   ├── step_001000/
│   ├── step_002000/
│   └── latest -> step_002000
└── metrics/
    └── eval_results.jsonl   # Append-only: {step, val_loss, valqa_em, testqa_em, ...}
```

So `outputs/cbe/news/gemma3-1b-lora128/` for the example above.

---

## Logging

### TensorBoard (local, no login)

```bash
tensorboard --logdir outputs/cbe/news/gemma3-1b-lora128/logs/tensorboard
# or compare runs across a project:
tensorboard --logdir outputs/cbe
```

### Weights & Biases

```bash
hf auth login        # for HF data
wandb login          # for wandb (paste API key from wandb.ai/authorize)
```

Then add `wandb` to `logging.backends` in your track config:

```yaml
logging:
  backends: [tensorboard, wandb]
  project_name: cbe
  run_name: news/gemma3-1b-lora128
```

Runs upload to `wandb.ai/<your-username>/cbe`.

---

## Evaluation

Eval happens **automatically during training** every `eval_every` steps:

- **Validation loss** on `data.val_path` (held-out paragraphs, next-token loss)
- **Exact match** + **fuzzy match** on `data.valqa_path` (generation-based QA)
- **Exact match** + **fuzzy match** on `data.testqa_path`

All metrics are logged to TB/wandb and appended to `metrics/eval_results.jsonl`.

For **standalone post-hoc evaluation** of a single checkpoint:

```bash
python evaluate.py \
    --checkpoint outputs/cbe/news/gemma3-1b-lora128/checkpoints/latest \
    --qa_data data/news/testqa.jsonl \
    --framework hf \
    --model google/gemma-3-1b-pt
```

---

## Adding a new track

1. Drop your data into `data/<track>/{train,val,valqa,testqa}.jsonl` (or update `load_data.py` to pull it from HF).
2. Copy `configs/tracks/news.yaml` to `configs/tracks/<track>.yaml`.
3. Update `_base`, `data.*_path`, `logging.run_name`, and the `eval` block.
4. Train: `python train.py --config configs/tracks/<track>.yaml --framework hf`.

## Adding a new model

- **HF side**: just set `model.name` to any HuggingFace hub ID (`meta-llama/Llama-3.1-8B`, `mistralai/Mistral-7B-v0.3`, etc.). No code changes needed.
- **KD side**: implement the `JaxModelFactory` protocol in `src/cbe/models/kd_models.py`. The Gemma factory there is a reference implementation.

---

## Multi-GPU / multi-node

**HuggingFace / TRL path** (recommended for any GPU setup):
```bash
# Single node, multiple GPUs
torchrun --nproc_per_node=4 train.py --config configs/tracks/news.yaml --framework hf

# Multi-node (2 nodes × 4 GPUs)
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr=<addr> --master_port=29500 \
    train.py --config configs/tracks/news.yaml --framework hf

# Or use accelerate
accelerate launch train.py --config configs/tracks/news.yaml --framework hf
```

FSDP is the default sharding strategy. DeepSpeed and FSDP configs can be wired in via the standard HF/Accelerate mechanisms.

**Kauldron path**: TPU-native via JAX FSDP. Single-node GPU works; multi-node GPU is not supported.

---

## License

Apache 2.0.
