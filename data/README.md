# Data Directory

ContinuousBench data is hosted on HuggingFace as two private repos:

- **`ContinuousBench/News`** (tag `v5`) — news articles + QA
- **`ContinuousBench/Geminon`** (tag `v9`) — Geminon articles + QA (public + sensitive splits)

## Downloading

Authenticate once, then run the loader:

```bash
hf auth login                            # one-time
python data/load_data.py                 # download all tracks per download.yaml
python data/load_data.py --track news    # just one track

# Pick a different corpus / QA size without editing the recipe:
python data/load_data.py --track geminon --corpus large
python data/load_data.py --track geminon --corpus medium --qa medium

# Debug: list every file in a repo
python data/load_data.py --list geminon
```

Files land at `data/<track>/{train,val,valqa,testqa}.jsonl` — exactly what the track YAML configs at `configs/tracks/*.yaml` expect.

## The download recipe

`data/download.yaml` controls what gets downloaded. Each entry maps a local filename to a path inside the HF repo:

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

Edit the recipe to pin a different corpus size (`corpus_{small,medium,large}`), pull Geminon's sensitive QA splits (`qa_small/sensitive_val.jsonl`), or add a new track.

### What's available in each repo

**News** (`ContinuousBench/News`, revision `v5`):
- `corpus_{large,medium,small}/{train,val,test,all}.jsonl`
- `qa/{val,test}.jsonl`

**Geminon** (`ContinuousBench/Geminon`, revision `v9`):
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

Use `scripts/format_data.py` to normalize and split raw JSONL:

```bash
python scripts/format_data.py --input raw.jsonl --output data/my_track/ --split --dedup
```
