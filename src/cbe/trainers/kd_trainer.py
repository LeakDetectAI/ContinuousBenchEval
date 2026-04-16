"""Kauldron trainer wrapper.

Builds a `kd.train.Trainer` from the unified YAML config and registers
live evaluators in `evals=` so QA + loss metrics land in TB/wandb/
eval_results.jsonl at every eval_every step — symmetric with the HF path.

The QA evaluator is `cbe.eval.kd_qa_evaluator.QAEvaluator`, which subclasses
`kd.evals.EvaluatorBase`. Non-serializable deps (logger, artifact store,
model factory) are passed to it via a module-level runtime-deps dict
set before `Trainer.train()` runs.

We construct `Trainer` directly with all required kwargs (train_ds,
model, optimizer are positional-only on newer Kauldron). No `konfig`
dance — we already have real Python objects in hand.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

from cbe.config import TrainConfig
from cbe.logging.multi_logger import MultiLogger
from cbe.artifacts.local_store import LocalArtifactStore


class KauldronTrainer:
    """Wraps Kauldron's kd.train.Trainer with unified logging and live QA eval."""

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
        # JAX environment setup (set before importing jax/kauldron)
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "1.0")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

        # Disable TF GPU to avoid conflicts with JAX
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], "GPU")
        except ImportError:
            pass

        from kauldron import kd
        import optax

        from cbe.models.kd_models import create_kd_model
        from cbe.data.kd_data import KauldronDataPipeline
        from cbe.eval.kd_qa_evaluator import QAEvaluator, set_runtime_deps

        tc = self.config.training
        ec = self.config.eval
        grad_accum = self.config.gradient_accumulation_steps

        # --- Model ------------------------------------------------------------
        model_factory = create_kd_model(self.config.model)
        tokenizer = model_factory.get_tokenizer()
        model = model_factory.make_model(self.config.model)
        init_transform = model_factory.make_init(self.config.model)

        # --- Data -------------------------------------------------------------
        data_pipeline = KauldronDataPipeline(self.config)
        train_ds = data_pipeline.make_train_source(tokenizer)

        # --- Sharding ---------------------------------------------------------
        if tc.sharding == "fsdp":
            sharding = kd.sharding.ShardingStrategy(params=kd.sharding.FSDPSharding())
        else:
            sharding = kd.sharding.ShardingStrategy(params=kd.sharding.REPLICATED)

        # --- Loss -------------------------------------------------------------
        train_losses = {
            "xentropy": kd.losses.SoftmaxCrossEntropyWithIntLabels(
                logits="preds.logits",
                labels="batch.target",
                mask="batch.loss_mask",
            ),
        }

        # --- Optimizer + schedule --------------------------------------------
        warmup_steps = max(
            1, int(tc.num_train_steps * self.config.optimizer.warmup_fraction)
        )
        schedule = self._make_schedule(tc.num_train_steps, warmup_steps)

        base_optimizer = optax.adamw(
            learning_rate=schedule,
            b1=self.config.optimizer.b1,
            b2=self.config.optimizer.b2,
            weight_decay=self.config.optimizer.weight_decay,
        )
        if grad_accum > 1:
            optimizer = optax.MultiSteps(base_optimizer, every_k_schedule=grad_accum)
        else:
            optimizer = base_optimizer

        # --- Checkpointer -----------------------------------------------------
        checkpointer = kd.ckpts.Checkpointer(
            save_interval_steps=tc.save_every,
            max_to_keep=tc.max_checkpoints,
        )

        # --- Install runtime deps for QAEvaluator before building evals ------
        set_runtime_deps(
            logger=self.logger,
            artifact_store=self.artifact_store,
            model_config=self.config.model,
            model_factory=model_factory,
        )

        # --- Evaluators: eval_loss + optional QA valqa/testqa ----------------
        evals: dict[str, Any] = {
            "eval_loss": kd.evals.Evaluator(
                run=kd.evals.EveryNSteps(tc.eval_every),
                ds=data_pipeline.make_eval_source(tokenizer),
                losses=train_losses,
                num_batches=tc.eval_num_batches,
            ),
        }
        qa_common = dict(
            prompt_prefix=ec.prompt_prefix,
            prompt_template=ec.prompt_template,
            max_new_tokens=ec.max_new_tokens,
            batch_size=ec.batch_size,
            temperature=ec.temperature,
            top_k=ec.top_k,
            top_p=ec.top_p,
            parser=ec.parser,
            num_examples=ec.num_examples,
            save_detailed_results=ec.save_detailed_results,
        )
        if self.config.data.valqa_path:
            evals["qa_valqa"] = QAEvaluator(
                name="qa_valqa",
                run=kd.evals.EveryNSteps(tc.eval_every),
                qa_path=self.config.data.valqa_path,
                metric_prefix="valqa",
                **qa_common,
            )
        if self.config.data.testqa_path:
            evals["qa_testqa"] = QAEvaluator(
                name="qa_testqa",
                run=kd.evals.EveryNSteps(tc.eval_every),
                qa_path=self.config.data.testqa_path,
                metric_prefix="testqa",
                **qa_common,
            )

        # --- Build the Trainer directly (no konfig needed) -------------------
        trainer = kd.train.Trainer(
            seed=tc.seed,
            workdir=self.artifact_store.run_dir,
            num_train_steps=tc.num_train_steps,
            train_ds=train_ds,
            model=model,
            init_transform=init_transform,
            optimizer=optimizer,
            sharding=sharding,
            train_losses=train_losses,
            checkpointer=checkpointer,
            evals=evals,
        )

        # Freeze config copy + log it
        self.artifact_store.save_config(dataclasses.asdict(self.config))
        self.logger.log_config(dataclasses.asdict(self.config))

        # Train (QA eval runs live every eval_every steps)
        trainer.train()

        self.logger.close()

    def _make_schedule(self, num_steps: int, warmup_steps: int):
        import optax
        sched = self.config.optimizer.schedule
        lr = self.config.optimizer.lr
        end_lr = lr * self.config.optimizer.end_lr_fraction

        if sched == "cosine":
            return optax.schedules.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=lr,
                warmup_steps=warmup_steps,
                decay_steps=num_steps,
                end_value=end_lr,
            )
        elif sched == "linear":
            return optax.join_schedules(
                schedules=[
                    optax.linear_schedule(init_value=0.0, end_value=lr, transition_steps=warmup_steps),
                    optax.linear_schedule(init_value=lr, end_value=0.0, transition_steps=num_steps - warmup_steps),
                ],
                boundaries=[warmup_steps],
            )
        elif sched == "constant":
            return optax.join_schedules(
                schedules=[
                    optax.linear_schedule(init_value=0.0, end_value=lr, transition_steps=warmup_steps),
                    optax.constant_schedule(lr),
                ],
                boundaries=[warmup_steps],
            )
        else:
            raise ValueError(f"Unknown schedule: {sched}")
