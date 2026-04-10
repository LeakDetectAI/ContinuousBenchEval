"""Kauldron trainer wrapper.

Builds a kd.train.Trainer from the unified YAML config, injects external
logging (wandb/TB) alongside KD's built-in TensorBoard, and runs QA eval
after training on each saved checkpoint.

Note: Kauldron's cfg.train() is a monolithic call that handles its own
training loop. We cannot inject mid-loop callbacks. Instead:
  1. KD handles train loss logging to its own TensorBoard.
  2. KD handles eval loss via cfg.evals (logged to its TensorBoard).
  3. After cfg.train() completes, we scan all saved checkpoints and run
     QA eval on each, saving metrics to eval_results.jsonl and logging
     to the MultiLogger.

For live QA eval during training, use the HF trainer path instead.
"""

from __future__ import annotations

import dataclasses
import glob
import os
import re
from typing import Any

from cbe.config import TrainConfig
from cbe.logging.multi_logger import MultiLogger
from cbe.artifacts.local_store import LocalArtifactStore


class KauldronTrainer:
    """Wraps Kauldron's kd.train.Trainer with unified logging and eval."""

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
        # JAX environment setup
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

        tc = self.config.training
        grad_accum = self.config.gradient_accumulation_steps

        cfg = kd.train.Trainer()
        cfg.seed = tc.seed
        cfg.workdir = self.artifact_store.run_dir

        # Model
        model_factory = create_kd_model(self.config.model)
        tokenizer = model_factory.get_tokenizer()
        cfg.model = model_factory.make_model(self.config.model)
        cfg.init_transform = model_factory.make_init(self.config.model)

        # Data — use per_device_batch_size for the actual batch fed to KD
        data_pipeline = KauldronDataPipeline(self.config)
        cfg.train_ds = data_pipeline.make_train_source(tokenizer)

        # Sharding
        if tc.sharding == "fsdp":
            cfg.sharding = kd.sharding.ShardingStrategy(
                params=kd.sharding.FSDPSharding(),
            )
        else:
            cfg.sharding = kd.sharding.ShardingStrategy(
                params=kd.sharding.REPLICATED,
            )

        # Training steps
        cfg.num_train_steps = tc.num_train_steps

        # Loss
        cfg.train_losses = {
            "xentropy": kd.losses.SoftmaxCrossEntropyWithIntLabels(
                logits="preds.logits",
                labels="batch.target",
                mask="batch.loss_mask",
            ),
        }

        # Optimizer with gradient accumulation
        warmup_steps = max(1, int(tc.num_train_steps * self.config.optimizer.warmup_fraction))
        schedule = self._make_schedule(tc.num_train_steps, warmup_steps)
        cfg.schedules = {"learning_rate": schedule}

        base_optimizer = optax.adamw(
            learning_rate=cfg.ref.schedules["learning_rate"],
            b1=self.config.optimizer.b1,
            b2=self.config.optimizer.b2,
            weight_decay=self.config.optimizer.weight_decay,
        )

        if grad_accum > 1:
            cfg.optimizer = optax.MultiSteps(base_optimizer, every_k_schedule=grad_accum)
        else:
            cfg.optimizer = base_optimizer

        # Checkpointer
        cfg.checkpointer = kd.ckpts.Checkpointer(
            save_interval_steps=tc.save_every,
            max_to_keep=tc.max_checkpoints,
        )

        # Eval (loss on val set — handled by KD internally)
        cfg.evals = {
            "eval_loss": kd.evals.Evaluator(
                run=kd.evals.EveryNSteps(tc.eval_every),
                ds=data_pipeline.make_eval_source(tokenizer),
            ),
        }

        # Save frozen config
        self.artifact_store.save_config(dataclasses.asdict(self.config))
        self.logger.log_config(dataclasses.asdict(self.config))

        # Resolve and train
        from kauldron import konfig
        cfg = konfig.resolve(cfg)
        cfg.train()

        # Post-training: run QA eval on all saved checkpoints
        self._run_post_training_qa_eval(model_factory)

        self.logger.close()

    def _run_post_training_qa_eval(self, model_factory) -> None:
        """Scan checkpoints in workdir and run QA eval on each."""
        from cbe.eval.inference import run_qa_eval_kd

        ec = self.config.eval
        has_valqa = bool(self.config.data.valqa_path)
        has_testqa = bool(self.config.data.testqa_path)
        if not has_valqa and not has_testqa:
            return

        # Find checkpoint directories written by KD
        # KD checkpoints are in workdir/checkpoints/ as ckpt_NNNNN directories
        ckpt_base = os.path.join(self.artifact_store.run_dir, "checkpoints")
        if not os.path.isdir(ckpt_base):
            # KD may save directly in workdir
            ckpt_base = self.artifact_store.run_dir

        ckpt_dirs = sorted(glob.glob(os.path.join(ckpt_base, "ckpt_*")))
        if not ckpt_dirs:
            # Try step-based naming
            ckpt_dirs = sorted(glob.glob(os.path.join(ckpt_base, "step_*")))
        if not ckpt_dirs:
            print("[CBE] No checkpoints found for post-training QA eval.")
            return

        print(f"[CBE] Running QA eval on {len(ckpt_dirs)} checkpoints...")

        for ckpt_dir in ckpt_dirs:
            # Extract step number from directory name
            match = re.search(r"(\d+)", os.path.basename(ckpt_dir))
            if not match:
                continue
            step = int(match.group(1))

            # Build a sampler from this checkpoint
            try:
                from gemma import gm
                sampler = gm.sampler.Sampler(
                    model=model_factory.make_model(self.config.model),
                    checkpoint=ckpt_dir,
                )
            except Exception as e:
                print(f"[CBE] Failed to load checkpoint {ckpt_dir}: {e}")
                continue

            eval_metrics = {}

            if has_valqa:
                qa_metrics = run_qa_eval_kd(
                    sampler=sampler,
                    qa_path=self.config.data.valqa_path,
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

            if has_testqa:
                qa_metrics = run_qa_eval_kd(
                    sampler=sampler,
                    qa_path=self.config.data.testqa_path,
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
                self.logger.log_scalars(eval_metrics, step=step)
                self.artifact_store.save_metrics(eval_metrics, step=step)
                print(f"[CBE] Step {step}: {eval_metrics}")

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
