"""HuggingFace / TRL SFTTrainer wrapper.

Uses TRL's SFTTrainer for fine-tuning with native LoRA/PEFT support,
multi-GPU via Accelerate, and DeepSpeed/FSDP compatibility.

Injects custom callbacks for:
- Unified logging (wandb + TB via MultiLogger)
- QA eval (exact match on valqa/testqa) at each eval_every
- Standardized artifact saving
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

from cbe.config import TrainConfig
from cbe.logging.multi_logger import MultiLogger
from cbe.artifacts.local_store import LocalArtifactStore


class HFTrainer:
    """Wraps TRL SFTTrainer with unified logging and eval."""

    def __init__(
        self,
        config: TrainConfig,
        logger: MultiLogger,
        artifact_store: LocalArtifactStore,
    ) -> None:
        self.config = config
        self.logger = logger
        self.artifact_store = artifact_store

    def train(self) -> None:
        from transformers import TrainerCallback
        from trl import SFTTrainer, SFTConfig

        from cbe.models.hf_models import create_hf_model
        from cbe.data.hf_data import HFDataPipeline
        from cbe.eval.inference import run_qa_eval_hf

        # Save frozen config
        self.artifact_store.save_config(dataclasses.asdict(self.config))
        self.logger.log_config(dataclasses.asdict(self.config))

        # Model + tokenizer
        bundle = create_hf_model(self.config.model)
        model = bundle.model
        tokenizer = bundle.tokenizer

        # Data
        data_pipeline = HFDataPipeline(self.config)
        train_dataset = data_pipeline.make_train_dataset(tokenizer)
        eval_dataset = data_pipeline.make_eval_dataset(tokenizer)

        tc = self.config.training
        grad_accum = self.config.gradient_accumulation_steps

        # FSDP config for multi-GPU
        fsdp_config = None
        fsdp = []
        if tc.sharding == "fsdp":
            fsdp = ["full_shard", "auto_wrap"]
            fsdp_config = {
                "fsdp_transformer_layer_cls_to_wrap": "auto",
            }

        # SFT training config
        training_args = SFTConfig(
            output_dir=os.path.join(self.artifact_store.run_dir, "checkpoints"),
            max_steps=tc.num_train_steps,
            per_device_train_batch_size=tc.per_device_batch_size,
            per_device_eval_batch_size=tc.per_device_batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
            adam_beta1=self.config.optimizer.b1,
            adam_beta2=self.config.optimizer.b2,
            warmup_ratio=self.config.optimizer.warmup_fraction,
            lr_scheduler_type=self._get_scheduler_type(),
            eval_strategy="steps",
            eval_steps=tc.eval_every,
            save_strategy="steps",
            save_steps=tc.save_every,
            save_total_limit=tc.max_checkpoints,
            logging_steps=self.config.logging.log_every_n_steps,
            seed=tc.seed,
            bf16=tc.bf16,
            dataloader_num_workers=self.config.data.num_workers,
            max_seq_length=self.config.data.sequence_length,
            dataset_text_field="text",
            remove_unused_columns=False,
            fsdp=fsdp if fsdp else "",
            fsdp_config=fsdp_config,
            report_to=[],  # We handle logging ourselves via callbacks
        )

        # Custom callback for unified logging + QA eval
        logger = self.logger
        artifact_store = self.artifact_store
        config = self.config

        class CBECallback(TrainerCallback):
            """Forwards metrics to MultiLogger and runs QA eval."""

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and state.global_step > 0:
                    step = state.global_step
                    logger.log_scalars(
                        {k: v for k, v in logs.items() if isinstance(v, (int, float))},
                        step=step,
                    )

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                step = state.global_step
                eval_metrics = {}

                # Capture eval loss from HF trainer
                if metrics:
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            eval_metrics[k] = v

                # Run QA eval on valqa
                eval_model = kwargs.get("model", model)
                ec = config.eval
                if config.data.valqa_path:
                    qa_metrics = run_qa_eval_hf(
                        model=eval_model,
                        tokenizer=tokenizer,
                        qa_path=config.data.valqa_path,
                        prompt_prefix=ec.prompt_prefix,
                        prompt_template=ec.prompt_template,
                        max_new_tokens=ec.max_new_tokens,
                        batch_size=ec.batch_size,
                        temperature=ec.temperature,
                        top_k=ec.top_k,
                        top_p=ec.top_p,
                    )
                    eval_metrics["valqa_exact_match"] = qa_metrics["exact_match"]
                    eval_metrics["valqa_fuzzy_match"] = qa_metrics["fuzzy_match"]

                # Run QA eval on testqa
                if config.data.testqa_path:
                    qa_metrics = run_qa_eval_hf(
                        model=eval_model,
                        tokenizer=tokenizer,
                        qa_path=config.data.testqa_path,
                        prompt_prefix=ec.prompt_prefix,
                        prompt_template=ec.prompt_template,
                        max_new_tokens=ec.max_new_tokens,
                        batch_size=ec.batch_size,
                        temperature=ec.temperature,
                        top_k=ec.top_k,
                        top_p=ec.top_p,
                    )
                    eval_metrics["testqa_exact_match"] = qa_metrics["exact_match"]
                    eval_metrics["testqa_fuzzy_match"] = qa_metrics["fuzzy_match"]

                if eval_metrics:
                    logger.log_scalars(eval_metrics, step=step)
                    artifact_store.save_metrics(eval_metrics, step=step)

            def on_save(self, args, state, control, **kwargs):
                artifact_store.register_checkpoint(state.global_step)

        # Build trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=[CBECallback()],
        )

        # Train
        trainer.train()
        self.logger.close()

    def _get_scheduler_type(self) -> str:
        """Map config schedule name to HF scheduler type."""
        mapping = {
            "cosine": "cosine",
            "linear": "linear",
            "constant": "constant_with_warmup",
        }
        return mapping.get(self.config.optimizer.schedule, "cosine")
