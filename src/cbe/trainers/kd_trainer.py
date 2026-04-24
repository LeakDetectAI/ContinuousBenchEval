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

    def train(self, resume: bool = False) -> None:
        # Note: Kauldron's kd.ckpts.Checkpointer auto-resumes from `workdir`
        # whenever a checkpoint exists there. The `resume` flag is accepted
        # for API symmetry but has no effect — KD always resumes when possible.
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
        from cbe.logging.kd_writer import make_teeing_writer

        tc = self.config.training
        ec = self.config.eval
        grad_accum = self.config.gradient_accumulation_steps

        # Kauldron's internal step counter ticks per DATA ITERATION (one per
        # data batch pulled). With optax.MultiSteps(k), one optimizer update
        # happens every k iterations. Users expect num_train_steps / eval_every /
        # save_every / warmup to mean OPTIMIZER STEPS (matching HF). So we
        # scale everything KD-facing by grad_accum.
        num_opt_steps = tc.num_train_steps
        num_data_iters = num_opt_steps * grad_accum
        eval_every_iters = tc.eval_every * grad_accum
        save_every_iters = tc.save_every * grad_accum

        # --- Model ------------------------------------------------------------
        # Pure-bf16 (hard-coded): params, activations, Adam state all in bf16.
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
        # Pure-bf16 CE (optax preserves logits dtype).
        train_losses = {
            "xentropy": kd.losses.SoftmaxCrossEntropyWithIntLabels(
                logits="preds.logits",
                labels="batch.target",
                mask="batch.loss_mask",
            ),
        }

        # --- Metrics (non-gradient, for logging only) -------------------------
        # `mean_token_accuracy`: fraction of masked positions where the model's
        # argmax prediction matches the target token. Matches HF/TRL's
        # `mean_token_accuracy` / `eval_mean_token_accuracy` for cross-framework
        # parity in wandb. Weighted by `batch.loss_mask` so padded positions
        # don't contribute.
        train_metrics = {
            "mean_token_accuracy": kd.metrics.Accuracy(
                logits="preds.logits",
                labels="batch.target",
                mask="batch.loss_mask",
            ),
        }

        # --- Optimizer + schedule --------------------------------------------
        # Schedule input is OPTIMIZER STEPS, not data iterations. Reason: when
        # we wrap the base optimizer in optax.MultiSteps(k=grad_accum), the
        # inner adamw (and its scale_by_schedule counter) only advances on
        # "emit" iters — every k data iters. On non-emit iters MultiSteps
        # reverts `inner_opt_state` via `jnp.where(emit, new, old)`, so the
        # schedule's internal step count ticks once per optimizer step.
        # If we sized warmup/decay in data-iter units, the schedule would be
        # stretched by k× and peak LR would be reached at opt step k*warmup
        # instead of `warmup`, starving training of its intended LR.
        warmup_opt_steps = max(
            1, int(num_opt_steps * self.config.optimizer.warmup_fraction)
        )
        schedule = self._make_schedule(num_opt_steps, warmup_opt_steps)

        # Pure-bf16: Adam state inherits param dtype (bf16). No override.
        base_optimizer = optax.adamw(
            learning_rate=schedule,
            b1=self.config.optimizer.b1,
            b2=self.config.optimizer.b2,
            weight_decay=self.config.optimizer.weight_decay,
        )

        # LoRA: freeze the base model via kd.optim.partial_updates. Without
        # this, optax.adamw updates ALL params (including the frozen Gemma
        # base) — the "LoRA" run would silently be a full fine-tune.
        if self.config.model.lora_rank and self.config.model.lora_rank > 0:
            base_optimizer = kd.optim.partial_updates(
                base_optimizer,
                mask=kd.optim.select("lora"),
            )

        if grad_accum > 1:
            optimizer = optax.MultiSteps(base_optimizer, every_k_schedule=grad_accum)
        else:
            optimizer = base_optimizer

        # --- Checkpointer -----------------------------------------------------
        checkpointer = kd.ckpts.Checkpointer(
            save_interval_steps=save_every_iters,
            max_to_keep=tc.max_checkpoints,
        )

        # --- Install runtime deps for QAEvaluator before building evals ------
        set_runtime_deps(
            logger=self.logger,
            artifact_store=self.artifact_store,
            model_config=self.config.model,
            model_factory=model_factory,
            grad_accum=grad_accum,
        )

        # --- Evaluators: eval_loss + optional QA valqa/testqa ----------------
        evals: dict[str, Any] = {
            "eval_loss": kd.evals.Evaluator(
                run=kd.evals.EveryNSteps(eval_every_iters),
                ds=data_pipeline.make_eval_source(tokenizer),
                losses=train_losses,
                metrics=train_metrics,  # adds eval_mean_token_accuracy
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
            # Convert to tuple so kauldron's BaseConfig (frozen dataclass)
            # treats it as a hashable field.
            support_thresholds=(
                tuple(ec.support_thresholds) if ec.support_thresholds else None
            ),
        )
        if self.config.data.valqa_path:
            evals["qa_valqa"] = QAEvaluator(
                name="qa_valqa",
                run=kd.evals.EveryNSteps(eval_every_iters),
                qa_path=self.config.data.valqa_path,
                metric_prefix="valqa",
                **qa_common,
            )
        if self.config.data.testqa_path:
            evals["qa_testqa"] = QAEvaluator(
                name="qa_testqa",
                run=kd.evals.EveryNSteps(eval_every_iters),
                qa_path=self.config.data.testqa_path,
                metric_prefix="testqa",
                **qa_common,
            )

        # --- Writer: tee every scalar Kauldron logs (train loss, eval loss,
        # LR, grad norm, etc.) into our MultiLogger so wandb gets the full
        # training curve, not just our QA metrics.
        writer = make_teeing_writer(
            self.logger,
            workdir=self.artifact_store.run_dir,
            grad_accum=grad_accum,
            schedule=schedule,
        )

        # --- Build the Trainer directly (no konfig needed) -------------------
        trainer = kd.train.Trainer(
            seed=tc.seed,
            workdir=self.artifact_store.run_dir,
            num_train_steps=num_data_iters,
            train_ds=train_ds,
            model=model,
            init_transform=init_transform,
            optimizer=optimizer,
            sharding=sharding,
            train_losses=train_losses,
            train_metrics=train_metrics,   # mean_token_accuracy in train logs
            checkpointer=checkpointer,
            writer=writer,
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
