# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations


def minimax_h3_align_frame_count(frame_count: int) -> int:
    """Snap ``frame_count`` up to the MiniMax H3 17n+5 frame boundary."""
    if frame_count <= 0:
        return 1
    current = int(frame_count)
    return current + (5 - current) % 17


def minimax_h3_video_latent_t(frame_count: int) -> int:
    if frame_count <= 5:
        return 2
    return ((int(frame_count) - 5) // 17) * 5 + 2


def minimax_h3_frame_count_from_video_latent_t(out_t: int) -> int:
    if out_t == 1:
        return 1
    if out_t < 2 or (out_t - 2) % 5 != 0:
        raise ValueError("MiniMax H3 video latent T must be 1 or match 5n+2")
    return 17 * ((int(out_t) - 2) // 5) + 5


def minimax_h3_audio_latent_t(duration_seconds: float) -> int:
    # Rounding happens at the 40 Hz audio latent boundary.
    return int(round(float(duration_seconds) * 40.0))


def minimax_h3_time_shift_sigma(
    sigma: float,
    *,
    from_shift: float,
    to_shift: float,
) -> float:
    """Map a video-stream sigma onto the audio stream's shifted schedule.

    Matches ComfyUI ``comfy/ldm/minimax/model.py::time_shift_sigma``.
    """

    sigma = float(sigma)
    from_shift = float(from_shift)
    to_shift = float(to_shift)
    if from_shift <= 0.0 or to_shift <= 0.0:
        raise ValueError("sigma shift scales must be > 0")
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def minimax_h3_time_shift_slope(
    sigma: float,
    *,
    from_shift: float,
    to_shift: float,
) -> float:
    """Jacobian ``d(sigma_to)/d(sigma_from)`` for paired stream schedules.

    Comfy scales the audio velocity by this slope so a flat video-sigma ODE
    matches the audio stream's true shifted schedule.
    """

    sigma = float(sigma)
    from_shift = float(from_shift)
    to_shift = float(to_shift)
    if from_shift <= 0.0 or to_shift <= 0.0:
        raise ValueError("sigma shift scales must be > 0")
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return (to_shift * (1.0 + (from_shift - 1.0) * base) ** 2) / (
        from_shift * (1.0 + (to_shift - 1.0) * base) ** 2
    )


def minimax_h3_time_shift_sigmas(
    *,
    num_steps: int = 50,
    shift_scale: float = 6.0,
) -> list[float]:
    if shift_scale <= 0:
        raise ValueError("MiniMax H3 shift_scale must be > 0")
    if num_steps <= 0:
        raise ValueError("MiniMax H3 num_steps must be > 0")

    import torch

    # Match ComfyUI ModelSamplingDiscreteFlow + BasicScheduler("simple"):
    # build a dense [1/1000..1] grid, apply the flow shift, then subsample
    # ``num_steps`` interior points and append a terminal 0.0 sigma.
    dense_steps = 1000
    base = torch.arange(1, dense_steps + 1, dtype=torch.float32) / dense_steps
    shifted = float(shift_scale) * base / (
        1 + (float(shift_scale) - 1) * base
    )
    stride = dense_steps / float(num_steps)
    sigmas = [float(shifted[-(1 + int(step * stride))]) for step in range(num_steps)]
    sigmas.append(0.0)
    return sigmas
