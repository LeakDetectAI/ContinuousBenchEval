"""HuggingFace / PyTorch data pipeline.

Reads .jsonl files, tokenizes with AutoTokenizer, returns HF Dataset objects
compatible with TRL SFTTrainer.
"""

from __future__ import annotations

from cbe.config import DataConfig
from cbe.data.formatters import load_jsonl


class HFDataPipeline:
    """Creates HuggingFace Dataset objects from config."""

    def __init__(self, config) -> None:
        self.config = config
        self.data_config: DataConfig = config.data

    def make_train_dataset(self, tokenizer):
        """Create a tokenized HF Dataset for training."""
        from datasets import Dataset

        records = load_jsonl(self.data_config.train_path)
        dataset = Dataset.from_list(records)

        def tokenize_fn(examples):
            texts = examples.get("text", examples.get("response", [""]))
            if isinstance(texts, str):
                texts = [texts]
            return tokenizer(
                texts,
                truncation=True,
                max_length=self.data_config.sequence_length,
                padding="max_length",
            )

        dataset = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=min(self.data_config.num_workers, 4),
        )
        return dataset

    def make_eval_dataset(self, tokenizer):
        """Create a tokenized HF Dataset for eval loss."""
        from datasets import Dataset

        records = load_jsonl(self.data_config.val_path)
        dataset = Dataset.from_list(records)

        def tokenize_fn(examples):
            texts = examples.get("text", examples.get("response", [""]))
            if isinstance(texts, str):
                texts = [texts]
            return tokenizer(
                texts,
                truncation=True,
                max_length=self.data_config.sequence_length,
                padding="max_length",
            )

        dataset = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=min(self.data_config.num_workers, 4),
        )
        return dataset

    def load_qa(self, path: str) -> list[dict[str, str]]:
        """Load a QA .jsonl file for exact-match evaluation."""
        return load_jsonl(path)
