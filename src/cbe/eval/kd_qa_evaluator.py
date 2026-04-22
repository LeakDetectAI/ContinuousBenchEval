"""Live QA evaluator for Kauldron training.

Subclasses `kd.evals.EvaluatorBase` so QA eval runs inline with training
every `EveryNSteps(N)`, instead of as a post-training checkpoint scan.
This keeps the KD path symmetric with the HF path:

    cfg.evals = {
        "eval_loss": kd.evals.Evaluator(run=EveryNSteps(N), ds=...),
        "qa_valqa":  QAEvaluator(run=EveryNSteps(N), qa_path=...),
        "qa_testqa": QAEvaluator(run=EveryNSteps(N), qa_path=...),
    }

The evaluator writes metrics to:
  - its built-in Kauldron writer (TensorBoard), via `self.writer.write_scalars`
  - the shared MultiLogger (so wandb gets the same scalars)
  - `eval_results.jsonl` via the artifact store

Runtime dependencies (MultiLogger, artifact_store, model config) that
can't be serialized into the dataclass are passed in via module-level
globals set by the KD trainer before `cfg.train()` runs. This is how
Kauldron evaluators typically receive non-config state, since `konfig`
resolves the config tree and expects plain dataclass fields.

IMPORTANT: This module imports kauldron at top level; only import it
from code that has already confirmed the KD backend is in use.
"""

from __future__ import annotations

from typing import Any

from kauldron import kd
from kauldron.utils import kdash


# ---------------------------------------------------------------------------
# Runtime deps (set by the trainer before cfg.train())
# ---------------------------------------------------------------------------

_RUNTIME: dict[str, Any] = {
    "logger": None,
    "artifact_store": None,
    "model_config": None,
    "model_factory": None,
    "rng": None,
    "grad_accum": 1,
}


def set_runtime_deps(
    *,
    logger,
    artifact_store,
    model_config,
    model_factory,
    grad_accum: int = 1,
) -> None:
    """Install the non-serializable deps each QAEvaluator needs at eval time.

    Also pre-builds a PRNG key while we're still outside kauldron's sharding
    context — building one INSIDE the training loop triggers "Disallowed
    host-to-device transfer".

    `grad_accum` is used to convert KD's data-iteration step into optimizer
    steps when logging metrics externally (wandb, eval_results.jsonl,
    eval_details/<qa>_step_<N>.jsonl).
    """
    import jax
    _RUNTIME["logger"] = logger
    _RUNTIME["artifact_store"] = artifact_store
    _RUNTIME["model_config"] = model_config
    _RUNTIME["model_factory"] = model_factory
    _RUNTIME["rng"] = jax.random.PRNGKey(42)
    _RUNTIME["grad_accum"] = max(1, int(grad_accum))


# ---------------------------------------------------------------------------
# Custom evaluator
# ---------------------------------------------------------------------------

class QAEvaluator(kd.evals.EvaluatorBase):
    """Generation-based QA evaluator (exact match + fuzzy match).

    Do NOT decorate with @dataclass. kauldron's `config_util.BaseConfig`
    (parent of `EvaluatorBase`) auto-handles dataclass decoration via
    metaclass magic; re-applying `@dataclass(frozen=True)` on a subclass
    causes "Cannot overwrite attribute __setattr__" in Python 3.11+.
    Fields below are picked up by kauldron's base.
    """

    qa_path: str = ""
    metric_prefix: str = "qa"  # e.g. "valqa" or "testqa"
    prompt_prefix: str = ""
    prompt_template: str = "Q: {question}\nA:"
    max_new_tokens: int = 50
    batch_size: int = 16
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float | None = None
    parser: str | None = None
    num_examples: int = 0
    save_detailed_results: bool = False

    @property
    def __dashboards__(self):
        # Custom scalars still flow to TB via `self.writer`; we don't need
        # Kauldron's structured metric dashboards for simple scalar logging.
        return kdash.NoopDashboard()

    def evaluate(self, state, step):
        from cbe.eval.inference import run_qa_eval_kd

        model_config = _RUNTIME["model_config"]
        model_factory = _RUNTIME["model_factory"]
        if model_config is None or model_factory is None:
            raise RuntimeError(
                "QAEvaluator runtime deps not set. "
                "Call cbe.eval.kd_qa_evaluator.set_runtime_deps(...) first."
            )

        # Build a fresh LoRA-wrapped model (if lora_rank > 0) matching the one
        # used at training time. state.params already contains both base and
        # LoRA subtrees (base from SkipLoRA(LoadCheckpoint), LoRA learned during
        # training), so we pass them straight through — no split/merge needed.
        model = model_factory.make_model(model_config)

        # Kauldron's `step` ticks per data iteration. Convert to optimizer
        # steps so all external artifacts (eval_details filenames,
        # eval_results.jsonl step field, wandb x-axis) match HF's convention.
        grad_accum = _RUNTIME.get("grad_accum", 1)
        opt_step = step // grad_accum if grad_accum > 1 else step

        artifact_store = _RUNTIME["artifact_store"]
        details_path = (
            artifact_store.qa_details_path(self.metric_prefix, opt_step)
            if (self.save_detailed_results and artifact_store is not None)
            else None
        )

        raw_metrics = run_qa_eval_kd(
            model=model,
            params=state.params,
            qa_path=self.qa_path,
            prompt_prefix=self.prompt_prefix,
            prompt_template=self.prompt_template,
            max_new_tokens=self.max_new_tokens,
            batch_size=self.batch_size,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            parser=self.parser,
            num_examples=self.num_examples,
            save_details_path=details_path,
            rng=_RUNTIME["rng"],
        )

        # Prefix keys (valqa_exact_match, testqa_exact_match, etc.)
        # Exclude "total" — it's a constant count, not a meaningful metric to plot.
        scalars = {
            f"{self.metric_prefix}_{k}": float(v)
            for k, v in raw_metrics.items()
            if isinstance(v, (int, float)) and k != "total"
        }

        # Kauldron's own TB writer uses data-iteration steps (its convention);
        # our external surface (MultiLogger/wandb, eval_results.jsonl) uses
        # optimizer steps for consistency with HF.
        if scalars:
            self.writer.write_scalars(step=step, scalars=scalars)

        logger = _RUNTIME["logger"]
        if logger is not None and scalars:
            logger.log_scalars(scalars, step=opt_step)

        artifact_store = _RUNTIME["artifact_store"]
        if artifact_store is not None and scalars:
            artifact_store.save_metrics(scalars, step=opt_step)

        # Return None so kauldron's train_loop skips the .compute() call.
        # We already wrote metrics to TB/wandb/jsonl ourselves above.
        return None
