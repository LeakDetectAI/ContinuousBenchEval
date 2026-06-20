# Data Directory

ContinuousBench data is hosted on HuggingFace:

- **`ContinuousBench/News`** — news articles + QA
- **`ContinuousBench/Geminon`** — Geminon articles + QA (public + sensitive splits)

Access may be gated; run `hf auth login` with a read token before downloading. To train on your own data instead, see [Custom data](#custom-data).

## Downloading

Authenticate once, then run the loader:

```bash
hf auth login                            # one-time
python data/helper/load_data.py                 # download all tracks per download.yaml
python data/helper/load_data.py --track news    # just one track

# Pick a different corpus / QA size without editing the recipe:
python data/helper/load_data.py --track geminon --corpus large
python data/helper/load_data.py --track geminon --corpus medium --qa medium

# Debug: list every file in a repo
python data/helper/load_data.py --list geminon
```

Files land at `data/<track>/{train,val,valqa,testqa}.jsonl` — exactly what the track YAML configs at `configs/tracks/*.yaml` expect.

## The download recipe

`data/helper/download.yaml` controls what gets downloaded. Each entry maps a local filename to a path inside the HF repo:

```yaml
tracks:
  news:
    repo: ContinuousBench/News
    files:
      train.jsonl:  corpus_small/train.jsonl
      val.jsonl:    corpus_small/val.jsonl
      valqa.jsonl:  qa/val.jsonl
      testqa.jsonl: qa/test.jsonl
```

Edit the recipe to pin a different corpus size (`corpus_{small,medium,large}`), pull Geminon's sensitive QA splits (`qa_small/sensitive_val.jsonl`), or add a new track.

### What's available in each repo

**News** (`ContinuousBench/News`):
- `corpus_{large,medium,small}/{train,val,test,all}.jsonl`
- `qa/{val,test}.jsonl`

**Geminon** (`ContinuousBench/Geminon`):
- `corpus_{large,medium,small}/{train,val,test,all}.jsonl`
- `qa_{small,medium}/{public_val,public_test,sensitive_val,sensitive_test}.jsonl`

## File formats

### Corpus JSONL (`train.jsonl`, `val.jsonl`)

```json
{"text": "The Federal Reserve announced a 0.25% rate hike on Wednesday..."}
{"text": "Researchers at MIT published a breakthrough study on..."}
```

### QA JSONL (`valqa.jsonl`, `testqa.jsonl`)

```json
{"question": "What percentage rate hike did the Federal Reserve announce?", "answer": "0.25%"}
{"question": "Which university published the breakthrough study?", "answer": "MIT"}
```

## Custom data

If you want to train on something other than ContinuousBench, drop your own `train.jsonl` / `val.jsonl` / `valqa.jsonl` / `testqa.jsonl` into `data/<your_track>/` (bypassing `load_data.py`) and create a matching `configs/tracks/<your_track>.yaml`.

Use `data/helper/clean_data.py` to normalize and split raw JSONL:

```bash
python data/helper/clean_data.py --input raw.jsonl --output data/my_track/ --split --dedup
```

## Diagnosing Geminon synthetic data

`scripts/analyze_synth_geminon.py` compares a Synth-Geminon generation with
the public Geminon ground-truth index. It reports entity persistence, factual
coverage, relation-distortion candidates, support redundancy/diversity, output
length, vocabulary size, and exact duplication. Remote JSONL is streamed and
the analysis itself uses only the Python standard library.

```bash
# DP LoRA, epsilon=100 (the default)
python scripts/analyze_synth_geminon.py

# Compare private and non-private generations
python scripts/analyze_synth_geminon.py \
  --config lora_dpft_eps100_1bpt_temp1.0_180108 \
  --config lora_ft_epsinf_1bpt_temp1.0_181108

# Quick smoke test, or analyze a downloaded file
python scripts/analyze_synth_geminon.py --limit 1000
python scripts/analyze_synth_geminon.py --input /path/to/data.jsonl
```

For model-based grammatical acceptability, add a two-class CoLA checkpoint.
The script uses a deterministic reservoir sample because scoring all sentences
is expensive. It also reports transparent punctuation, capitalization,
repetition, and delimiter diagnostics over the complete corpus.

```bash
python scripts/analyze_synth_geminon.py \
  --config lora_dpft_eps100_1bpt_temp1.0_180108 \
  --config lora_ft_epsinf_1bpt_temp1.0_181108 \
  --grammar-model textattack/roberta-base-CoLA \
  --grammar-sample-size 5000
```

Reports are written to `analysis_results/synth_geminon/`: one JSON summary and
one fact-level CSV per configuration, plus `comparison.csv`. Set `HF_TOKEN` if
Hugging Face requires authentication.
