# Data Directory

Place your training and evaluation data here, organized by **track**.

## Directory Structure

```
data/
├── news/                    # News track
│   ├── train.jsonl          # Training paragraphs
│   ├── val.jsonl            # Validation paragraphs (for loss)
│   ├── valqa.jsonl          # Validation QA pairs (for exact match)
│   └── testqa.jsonl         # Test QA pairs (for exact match)
│
├── geminon/                 # Geminon track
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── valqa.jsonl
│   └── testqa.jsonl
│
└── example/
    └── sample.jsonl         # Example format reference
```

## File Formats

### Training / Validation data (`train.jsonl`, `val.jsonl`)

One JSON object per line with a `text` field containing the paragraph:

```json
{"text": "The Federal Reserve announced a 0.25% rate hike on Wednesday..."}
{"text": "Researchers at MIT published a breakthrough study on..."}
```

### QA data (`valqa.jsonl`, `testqa.jsonl`)

One JSON object per line with `question` and `answer` fields:

```json
{"question": "What percentage rate hike did the Federal Reserve announce?", "answer": "0.25%"}
{"question": "Which university published the breakthrough study?", "answer": "MIT"}
```

## Preparing Data

Use the formatting script to clean and normalize raw text:

```bash
python scripts/format_data.py --input raw_data.jsonl --output data/news/train.jsonl
```
