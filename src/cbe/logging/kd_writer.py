"""Kauldron metric writer that also forwards scalars to our MultiLogger.

Kauldron writes train loss / eval loss / learning rate / grad norm via its
`KDMetricWriter` (one per collection: train, eval_loss, qa_valqa, ...).
Each per-collection writer is created via `dataclasses.replace(base_writer,
collection=<name>)`, which drops any state set via `object.__setattr__` on
the base instance.

So we keep the MultiLogger reference and grad_accum in MODULE-LEVEL state
(`_STATE`), not on the instance. The subclass reads from `_STATE` at write
time so every replaced writer still sees the current logger.
"""

from __future__ import annotations

from typing import Any, Mapping


_STATE: dict[str, Any] = {"logger": None, "grad_accum": 1, "schedule": None}


def _set_runtime(logger, grad_accum: int, schedule=None) -> None:
    _STATE["logger"] = logger
    _STATE["grad_accum"] = max(1, int(grad_accum))
    _STATE["schedule"] = schedule


def make_teeing_writer(multi_logger, workdir: str, grad_accum: int = 1, schedule=None):
    """Return a Kauldron WriterBase that tees scalars to the MultiLogger.

    Kauldron's step counter ticks per data iteration. With gradient
    accumulation, one optimizer step = grad_accum iterations. We divide
    by grad_accum before forwarding to the MultiLogger so wandb's x-axis
    matches the HF path (optimizer steps).

    We also rename Kauldron's per-collection tags to match HF's flat
    convention. For example:
      train collection, 'losses/xentropy'       -> 'loss'
      eval_loss collection, 'losses/xentropy'   -> 'eval_loss'
      qa_valqa collection, 'valqa_exact_match'  -> 'valqa_exact_match'
    """
    from kauldron.train.metric_writer import KDMetricWriter
    from etils import epath

    class _TeeingKDWriter(KDMetricWriter):
        def write_scalars(self, step: int, scalars: Mapping[str, Any]) -> None:
            # TB: unchanged (data-iteration step, Kauldron's convention)
            super().write_scalars(step=step, scalars=scalars)
            # MultiLogger / wandb: normalize to optimizer steps and rename
            # to HF-style flat names.
            logger = _STATE["logger"]
            if logger is None:
                return
            ga = _STATE["grad_accum"]
            opt_step = step // ga if ga > 1 else step
            collection = getattr(self, "collection", None) or ""
            out = {}
            for k, v in scalars.items():
                try:
                    v_f = float(v)
                except (TypeError, ValueError):
                    continue
                out[_rename_tag(collection, k)] = v_f
            # Kauldron doesn't log the LR, but HF Trainer does — tee the
            # schedule's current value under `learning_rate` at each train-
            # collection write so KD's wandb has parity with HF's.
            sched = _STATE["schedule"]
            if sched is not None and collection == "train":
                try:
                    out["learning_rate"] = float(sched(opt_step))
                except Exception:
                    pass
            if out:
                logger.log_scalars(out, step=opt_step)

    _set_runtime(multi_logger, grad_accum, schedule)
    return _TeeingKDWriter(workdir=epath.Path(workdir))


def _rename_tag(collection: str, key: str) -> str:
    """Map Kauldron per-collection metric names to HF-style flat names.

    Specifically:
      train + 'losses/xentropy'      -> 'loss'
      train + 'losses/total'         -> 'loss'   (xentropy and total are the same for us)
      eval_loss + 'losses/xentropy'  -> 'eval_loss'
      eval_loss + 'losses/total'     -> 'eval_loss'
      qa_valqa + <anything>          -> <anything>  (already prefixed with valqa_)
      qa_testqa + <anything>         -> <anything>  (already prefixed with testqa_)
      train + 'perf_stats/...'       -> 'perf/...' (keep perf stats distinct)
    """
    if collection == "train":
        if key in ("losses/xentropy", "losses/total"):
            return "loss"
        if key.startswith("metrics/"):
            # Flat-name metrics under train (e.g. metrics/mean_token_accuracy
            # -> mean_token_accuracy) so they line up with HF's conventions.
            return key[len("metrics/"):]
        if key.startswith("perf_stats/"):
            return "perf/" + key[len("perf_stats/"):]
        return "train/" + key  # fallback: keep some scoping
    if collection == "eval_loss":
        if key in ("losses/xentropy", "losses/total"):
            return "eval_loss"
        if key.startswith("metrics/"):
            # metrics/mean_token_accuracy -> eval_mean_token_accuracy
            return "eval_" + key[len("metrics/"):]
        return "eval_loss/" + key
    if collection.startswith("qa_"):
        # QAEvaluator already prefixes metric_prefix, leave as-is.
        return key
    # Unknown collection: keep collection as prefix.
    return f"{collection}/{key}" if collection else key
