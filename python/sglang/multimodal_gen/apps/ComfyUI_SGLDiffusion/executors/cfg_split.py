# SPDX-License-Identifier: Apache-2.0
"""CFG split: ComfyUI's cond and uncond DiT calls of one step on two SGLD ranks.

ComfyUI evaluates the conds of a sampler step in ``_calc_cond_batch``: one
``apply_model`` per cond, batched only when the conds' shapes match. With
``enable_cfg_parallel`` the SGLD worker has two CFG ranks, and this
CALC_COND_BATCH wrapper evaluates the step's two conds as one worker request,
cond ``i`` on rank ``i``, so a step costs about one DiT forward.

The wrapper reproduces ``_calc_cond_batch`` for the plain case only (two conds
of one entry each, no area / mask / hooks / ControlNet / GLIGEN / default
conds, no model wrappers); anything else returns None and ComfyUI's own path
runs. Each cond's ``apply_model`` is run twice: a recording pass collects the
DiT inputs, the worker computes them together, and a replay pass returns the
velocities so ``apply_model``'s own pre/post-processing is ComfyUI's.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# cond entry keys that _calc_cond_batch handles beyond one plain apply_model.
_UNSUPPORTED_COND_KEYS = ("area", "mask", "hooks", "control", "gligen", "default")
# Wrapper types that would run once per pass and see placeholder velocities.
_UNSUPPORTED_WRAPPERS = ("apply_model", "diffusion_model")

_LOGGED_REASONS: set[str] = set()


def cfg_split_unsupported_reason(conds, model_options: dict) -> str | None:
    """Why this step must use ComfyUI's own _calc_cond_batch, or None."""
    if len(conds) != 2:
        return f"{len(conds)} conds (CFG split runs exactly cond + uncond)"
    for cond in conds:
        if cond is None:
            return "a cond is None (cfg == 1 skips the uncond)"
        if len(cond) != 1:
            return f"a cond has {len(cond)} entries (area composition)"
        for key in _UNSUPPORTED_COND_KEYS:
            if cond[0].get(key) is not None:
                return f"cond uses {key!r}"
    for key in ("model_function_wrapper", "multigpu_clones"):
        if key in model_options:
            return f"model_options has {key!r}"
    wrappers = (model_options.get("transformer_options") or {}).get("wrappers") or {}
    for wrapper_type in _UNSUPPORTED_WRAPPERS:
        if any(wrappers.get(wrapper_type, {}).values()):
            return f"a {wrapper_type} wrapper is installed"
    return None


def _log_fallback(reason: str) -> None:
    if reason not in _LOGGED_REASONS:
        _LOGGED_REASONS.add(reason)
        logger.info("SGLD CFG split not used for this step: %s", reason)


def _call_kwargs(model, p, index: int, timestep, model_options: dict) -> dict:
    """The ``apply_model`` kwargs _calc_cond_batch builds for a single-cond batch."""
    import comfy.patcher_extension
    import comfy.samplers

    c = comfy.samplers.cond_cat([p.conditioning])
    transformer_options = model.current_patcher.apply_hooks(hooks=None)
    if "transformer_options" in model_options:
        transformer_options = comfy.patcher_extension.merge_nested_dicts(
            transformer_options, model_options["transformer_options"], copy_dict1=False
        )
    transformer_options["cond_or_uncond"] = [index]
    transformer_options["uuids"] = [p.uuid]
    transformer_options["sigmas"] = timestep
    c["transformer_options"] = transformer_options
    return c


def calc_cond_batch_cfg_split(
    executor, model, conds, x_in, timestep, model_options
) -> list[torch.Tensor] | None:
    """``_calc_cond_batch`` with the conds' DiT calls split across CFG ranks.

    Returns None when the step needs ComfyUI's general path.
    """
    import comfy.samplers

    reason = cfg_split_unsupported_reason(conds, model_options)
    if reason is None:
        runs = [comfy.samplers.get_area_and_mult(c[0], x_in, timestep) for c in conds]
        if any(p is None for p in runs):
            reason = "a cond is outside its timestep range"
    if reason is not None:
        _log_fallback(reason)
        return None

    model.current_patcher.prepare_state(timestep, model_options)
    calls = [
        (
            p.input_x,
            torch.cat([timestep]),
            _call_kwargs(model, p, i, timestep, model_options),
        )
        for i, p in enumerate(runs)
    ]

    executor._cfg_split_mode = "record"
    executor._cfg_split_records = []
    try:
        for input_x, t, c in calls:
            model.apply_model(input_x, t, **c)
        records = executor._cfg_split_records
    finally:
        executor._cfg_split_mode = None
        executor._cfg_split_records = []
    if len(records) != len(calls):
        raise RuntimeError(
            f"CFG split recorded {len(records)} DiT calls for {len(calls)} conds"
        )

    executor._cfg_split_outputs = executor.execute_cfg_split(records)
    executor._cfg_split_mode = "replay"
    try:
        outputs = [model.apply_model(input_x, t, **c) for input_x, t, c in calls]
    finally:
        executor._cfg_split_mode = None
        executor._cfg_split_outputs = []

    # Same accumulation as _calc_cond_batch, so the result is bit-identical to it.
    out_conds = []
    for p, output in zip(runs, outputs):
        out = torch.zeros_like(x_in)
        count = torch.ones_like(x_in) * 1e-37
        out += output * p.mult
        count += p.mult
        out /= count
        out_conds.append(out)
    return out_conds
