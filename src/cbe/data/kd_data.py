"""Kauldron / Grain data pipeline.

Reads .jsonl, converts to ArrayRecord on-the-fly, then uses Grain
transforms for tokenization and batching.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from typing import Any

import numpy as np
from grain import python as grain

from cbe.config import DataConfig
from cbe.data.formatters import load_jsonl


# ---------------------------------------------------------------------------
# Grain transforms (ported from dpsynth/training/data.py)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class JsonBytesToDict(grain.MapTransform):
    """Grain MapTransform: decode ArrayRecord bytes → dict."""
    key: str = "text"

    def map(self, element: bytes) -> dict[str, Any]:
        obj = json.loads(element.decode("utf-8"))
        if isinstance(obj, str):
            return {self.key: obj}
        elif isinstance(obj, dict):
            return {self.key: obj.get(self.key, obj.get("response", ""))}
        return {self.key: ""}


@dataclasses.dataclass
class NextTokenPredictionTask(grain.MapTransform):
    """Grain MapTransform: tokenize text → input/target/mask tensors."""
    sequence_length: int
    tokenizer: Any
    key: str = "text"

    def map(self, element: dict[str, Any]) -> dict[str, Any]:
        text = element[self.key]
        if hasattr(text, "item"):
            text = text.item()
        if isinstance(text, bytes):
            text = text.decode("utf-8")

        tokens = np.asarray(
            self.tokenizer.encode(text, add_bos=True, add_eos=True)
        )

        input_tokens = self._pad(tokens[:-1])
        target_tokens = self._pad(tokens[1:])
        mask = self._pad(np.ones(len(tokens) - 1, dtype=np.bool_))

        return {
            "input": input_tokens,
            "target": target_tokens[..., None],
            "loss_mask": mask.astype(np.float32)[..., None],
        }

    def _pad(self, arr: np.ndarray) -> np.ndarray:
        """Pad or truncate to sequence_length."""
        if len(arr) >= self.sequence_length:
            return arr[: self.sequence_length]
        return np.pad(arr, (0, self.sequence_length - len(arr)))


@dataclasses.dataclass
class Seq2SeqTask(grain.MapTransform):
    """Grain MapTransform: tokenize prompt+answer → input/target/mask."""
    sequence_length: int
    tokenizer: Any
    prompt_key: str = "src"
    response_key: str = "dst"

    def map(self, element: dict[str, Any]) -> dict[str, Any]:
        prompt = str(element[self.prompt_key])
        response = str(element[self.response_key])

        prompt_tokens = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        response_tokens = self.tokenizer.encode(response, add_bos=False, add_eos=True)

        full = np.asarray(prompt_tokens + response_tokens)
        prompt_len = len(prompt_tokens)

        input_tokens = self._pad(full[:-1])
        target_tokens = self._pad(full[1:])

        # Mask: only compute loss on the response portion
        mask = np.zeros(len(full) - 1, dtype=np.float32)
        mask[max(0, prompt_len - 1) :] = 1.0
        mask = self._pad(mask)

        return {
            "input": input_tokens,
            "target": target_tokens[..., None],
            "loss_mask": mask[..., None],
        }

    def _pad(self, arr: np.ndarray) -> np.ndarray:
        if len(arr) >= self.sequence_length:
            return arr[: self.sequence_length]
        return np.pad(arr, (0, self.sequence_length - len(arr)))


# ---------------------------------------------------------------------------
# ArrayRecord conversion
# ---------------------------------------------------------------------------

def _jsonl_to_array_record(jsonl_path: str, output_dir: str) -> str:
    """Convert a .jsonl file to ArrayRecord format. Returns the AR path."""
    from array_record.python.array_record_module import ArrayRecordWriter

    os.makedirs(output_dir, exist_ok=True)
    ar_path = os.path.join(output_dir, "data.array_record")

    if os.path.exists(ar_path):
        return ar_path

    records = load_jsonl(jsonl_path)
    writer = ArrayRecordWriter(ar_path, "group_size:1")
    for record in records:
        writer.write(json.dumps(record).encode("utf-8"))
    writer.close()
    return ar_path


def _get_cache_dir(jsonl_path: str) -> str:
    """Deterministic cache directory for a given jsonl file."""
    h = hashlib.md5(os.path.abspath(jsonl_path).encode()).hexdigest()[:12]
    return os.path.join(tempfile.gettempdir(), f"cbe_ar_cache_{h}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class KauldronDataPipeline:
    """Creates Kauldron DataSource objects from config."""

    def __init__(self, config) -> None:
        self.config = config
        self.data_config: DataConfig = config.data

    def _make_datasource(
        self,
        jsonl_path: str,
        is_training: bool,
        transforms: list,
        batch_size: int,
    ):
        from grain import python as grain
        from kauldron import kd

        cache_dir = _get_cache_dir(jsonl_path)
        ar_path = _jsonl_to_array_record(jsonl_path, cache_dir)

        return kd.data.py.DataSource(
            data_source=grain.ArrayRecordDataSource(ar_path),
            shuffle=is_training,
            num_epochs=None if is_training else 1,
            batch_size=batch_size,
            transforms=transforms,
            num_workers=self.data_config.num_workers,
            per_worker_buffer_size=4,
        )

    def make_train_source(self, tokenizer):
        """Create a training DataSource with next-token prediction."""
        transforms = [
            JsonBytesToDict(key="text"),
            NextTokenPredictionTask(
                sequence_length=self.data_config.sequence_length,
                tokenizer=tokenizer,
            ),
        ]
        return self._make_datasource(
            self.data_config.train_path,
            is_training=True,
            transforms=transforms,
            batch_size=self.config.training.per_device_batch_size,
        )

    def make_eval_source(self, tokenizer):
        """Create an eval DataSource for loss computation.

        Uses training.eval_per_device_batch_size (typically larger than the
        training batch, since no grad storage is needed during eval).
        """
        transforms = [
            JsonBytesToDict(key="text"),
            NextTokenPredictionTask(
                sequence_length=self.data_config.sequence_length,
                tokenizer=tokenizer,
            ),
        ]
        return self._make_datasource(
            self.data_config.val_path,
            is_training=False,
            transforms=transforms,
            batch_size=self.config.training.eval_per_device_batch_size,
        )
