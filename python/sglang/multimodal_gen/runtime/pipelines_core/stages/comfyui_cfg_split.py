# SPDX-License-Identifier: Apache-2.0
"""``--comfyui-mode`` CFG split: each CFG rank runs one ComfyUI cond of a step.

The ComfyUI plugin sends a step's cond and uncond DiT calls as one request
whose ``extra[COMFYUI_CFG_SPLIT_KEY]`` holds one dict of ``Req`` fields per
CFG rank. Every rank receives the whole request (rank 0 broadcasts it); rank
``i`` applies call ``i`` to its batch, runs the model's own step stage, and
the velocities are all-gathered over the CFG group so rank 0 can reply with
one per call. Per-cond worker state (sessions, prefix caches) therefore lives
on the rank that runs that cond. Requests without the key run unchanged on
every rank.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.runtime.pipelines_core.comfyui_mode import (
    COMFYUI_CFG_SPLIT_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)


def apply_cfg_split_call(batch, calls: list[dict[str, Any]], rank: int) -> None:
    """Make ``batch`` the request of call ``rank``; the split key is dropped."""
    fields = dict(calls[rank])
    extra = dict(batch.extra or {})
    extra.pop(COMFYUI_CFG_SPLIT_KEY, None)
    extra.update(fields.pop("extra", None) or {})
    for name, value in fields.items():
        setattr(batch, name, value)
    batch.extra = extra


def cfg_rank_errors(group, error: str | None) -> list[str | None]:
    """Every CFG rank's error message for this step (None where it succeeded)."""
    errors: list[str | None] = [None] * group.world_size
    torch.distributed.all_gather_object(errors, error, group=group.cpu_group)
    return errors


def gather_cfg_split_outputs(group, output) -> list:
    """All-gather one rank's velocity (tensor or list of tensors), ordered by rank."""
    if isinstance(output, (list, tuple)):
        per_part = [gather_cfg_split_outputs(group, part) for part in output]
        return [list(parts) for parts in zip(*per_part)]
    gathered = group.all_gather(output.contiguous(), dim=0, separate_tensors=True)
    if torch.is_tensor(gathered):
        gathered = [gathered]
    return list(gathered)


class ComfyUICFGSplitStage(PipelineStage):
    """Wraps a model's ComfyUI step stage; routes CFG-split calls by CFG rank."""

    def __init__(self, step_stage: PipelineStage, group_getter=None) -> None:
        super().__init__()
        self.step_stage = step_stage
        self._group_getter = group_getter

    def _group(self):
        if self._group_getter is not None:
            return self._group_getter()
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            get_cfg_group,
        )

        return get_cfg_group()

    def verify_input(self, batch, server_args) -> VerificationResult:
        return VerificationResult()

    def verify_output(self, batch, server_args) -> VerificationResult:
        return VerificationResult()

    def forward(self, batch, server_args):
        calls = (batch.extra or {}).get(COMFYUI_CFG_SPLIT_KEY)
        if calls is None:
            return self.step_stage.forward(batch, server_args)
        group = self._group()
        if len(calls) != group.world_size:
            raise ValueError(
                f"CFG split request has {len(calls)} calls for "
                f"{group.world_size} CFG ranks"
            )
        error = None
        try:
            apply_cfg_split_call(batch, calls, group.rank_in_group)
            batch = self.step_stage.forward(batch, server_args)
        except Exception as exc:
            error = exc
        # A rank that failed never reaches the all-gather; without this check
        # the other CFG rank would wait in it until the NCCL timeout.
        errors = cfg_rank_errors(group, None if error is None else repr(error))
        if error is not None:
            raise error
        failed = [f"CFG rank {r}: {e}" for r, e in enumerate(errors) if e]
        if failed:
            raise RuntimeError("CFG split step failed on " + "; ".join(failed))
        batch.noise_pred = gather_cfg_split_outputs(group, batch.noise_pred)
        return batch
