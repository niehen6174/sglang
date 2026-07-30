# SPDX-License-Identifier: Apache-2.0
"""Pixel-space VSR (Real-ESRGAN) for LTX-2 after VAE decode in disagg DAGs."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.entrypoints.utils import _sample_to_uint8_frames
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    V,
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_VSR_HEIGHT_THRESHOLD = 512


def _frames_to_video_output(frames: list[np.ndarray]) -> np.ndarray:
    if not frames:
        return np.empty((0,), dtype=np.uint8)
    return np.stack(frames, axis=0)


def _coerce_relayed_video_output(output: Any) -> Any:
    """Normalize decoded video pixels from relay transfer for VSR."""
    if isinstance(output, torch.Tensor):
        t = output.detach().cpu()
        if t.dtype == torch.bfloat16:
            t = t.float()
        if t.dim() == 5 and t.shape[-1] in (1, 3, 4):
            arr = t.numpy()
            if arr.shape[0] == 1:
                return arr[0]
            return arr
        if t.dim() == 4 and t.shape[-1] in (1, 3, 4):
            return t.numpy()
    if isinstance(output, np.ndarray) and output.ndim == 5 and output.shape[-1] in (1, 3, 4):
        if output.shape[0] == 1:
            return output[0]
    return output


class LTX2VSRStage(PipelineStage):
    """Upscale decoded video pixels when the request height is below 512.

    Intended as a separate DAG node after ``LTX2VideoDecodingStage`` so
    Real-ESRGAN runs on a lightweight pool while the video VAE stays on its
    own replica set.
    """

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DECODER

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("output", batch.output, V.not_none)
        return result

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        output = _coerce_relayed_video_output(batch.output)
        height = int(getattr(batch, "height", 0) or 0)
        if height >= _VSR_HEIGHT_THRESHOLD:
            logger.info(
                "LTX2VSRStage: skipping VSR (height=%d >= %d)",
                height,
                _VSR_HEIGHT_THRESHOLD,
            )
            return OutputBatch(output=output, metrics=batch.metrics)

        scale = int(getattr(batch, "upscaling_scale", 4) or 4)
        model_path = getattr(batch, "upscaling_model_path", None)

        from sglang.multimodal_gen.runtime.postprocess import batch_upscale_frames

        frames = _sample_to_uint8_frames(output)
        logger.info(
            "LTX2VSRStage: upscaling %d frame(s) from height=%d with scale=%dx",
            len(frames),
            height,
            scale,
        )
        upscaled = batch_upscale_frames(
            frames,
            model_path=model_path,
            scale=scale,
        )
        return OutputBatch(
            output=_frames_to_video_output(upscaled),
            metrics=batch.metrics,
        )
