"""HuggingFace / PyTorch data pipeline.

Reads .jsonl files, tokenizes with AutoTokenizer, returns HF Dataset objects
compatible with TRL SFTTrainer.

Tokenization matches KD's NextTokenPredictionTask conventions so both
frameworks compute eval_loss over exactly the same (input, target) pairs:
  - Add BOS (HF tokenizer default)
  - **Append EOS** manually (HF tokenizer doesn't by default)
  - Left-pad to `sequence_length`
BOS is then masked out of the loss by `BOSMaskingCollator` in hf_trainer.py,
making the loss positions equal to KD's: `predict t1..tN from BOS-prefixed
context + predict EOS at the end` (no "predict BOS from all-pad context").
"""

from __future__ import annotations

from cbe.config import DataConfig
from cbe.data.formatters import load_jsonl


class HFDataPipeline:
    """Creates HuggingFace Dataset objects from config."""

    def __init__(self, config) -> None:
        self.config = config
        self.data_config: DataConfig = config.data

    def _build_tokenize_fn(self, tokenizer):
        """Tokenize with BOS (default) + manually appended EOS, left-pad to max_len.

        The BOS is left in input_ids (model needs to see it as context); the
        collator later sets labels[BOS]=-100 so BOS is not a training target.
        """
        seq_len = self.data_config.sequence_length
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id

        def tokenize_fn(examples):
            texts = examples.get("text", examples.get("response", [""]))
            if isinstance(texts, str):
                texts = [texts]
            # Tokenize without padding, truncated to seq_len-1 so EOS fits.
            enc = tokenizer(
                texts, truncation=True, max_length=seq_len - 1, padding=False,
            )
            out_ids: list[list[int]] = []
            out_mask: list[list[int]] = []
            for ids in enc["input_ids"]:
                ids = list(ids) + [eos_id]
                n = len(ids)
                pad_len = seq_len - n
                # Tokenizer.padding_side is "left" for Gemma3; stick with that.
                out_ids.append([pad_id] * pad_len + ids)
                out_mask.append([0] * pad_len + [1] * n)
            return {"input_ids": out_ids, "attention_mask": out_mask}

        return tokenize_fn

    def make_train_dataset(self, tokenizer):
        """Create a tokenized HF Dataset for training."""
        from datasets import Dataset

        records = load_jsonl(self.data_config.train_path)
        dataset = Dataset.from_list(records)
        dataset = dataset.map(
            self._build_tokenize_fn(tokenizer),
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
        dataset = dataset.map(
            self._build_tokenize_fn(tokenizer),
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=min(self.data_config.num_workers, 4),
        )
        return dataset

    def load_qa(self, path: str) -> list[dict[str, str]]:
        """Load a QA .jsonl file for exact-match evaluation."""
        return load_jsonl(path)
