# SPDX-License-Identifier: Apache-2.0
"""Numerical contract for request-static H3 denoise metadata."""

from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_minimax_h3_euler_ancestral import (
    _minimax_h3_euler_eta0_step,
    _minimax_h3_rf_v_to_x0,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    MiniMaxH3DenoiseBranch,
    _build_local_embedding_layout,
    _minimax_h3_res_multistep_target_rows_,
    _minimax_h3_update_target_rows_,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)


def _branch(
    mode: str, token_tags: torch.Tensor | None = None
) -> MiniMaxH3DenoiseBranch:
    common = dict(text_len=3, latent_t=2, latent_h=4, latent_w=4, audio_t=3)
    if mode == "t2va":
        packed = minimax_h3_packed_sequence(
            **common,
            include_keyframe_cond=False,
        )
    elif mode == "fl2va":
        packed = minimax_h3_packed_sequence(
            **common,
            include_keyframe_cond=True,
            keyframe_frame_indices=[0, -1],
            frame_count=5,
        )
    else:
        packed = minimax_h3_packed_sequence_ref2va_blocks(
            **common,
            ref_blocks=[
                {"kind": "image", "latent_h": 4, "latent_w": 4},
                {"kind": "audio", "ref_audio_t": 2},
            ],
        )
    return MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(3, 5120),
        token_tags=packed["token_tags"] if token_tags is None else token_tags,
        device=torch.device("cpu"),
    )


def test_precomputed_timestep_plan_matches_full_unique_reference():
    """Preplanning must preserve fp32 collisions and every packed row class."""

    for mode in ("t2va", "fl2va", "ref2va"):
        branch = _branch(mode)
        assert branch.static_kwargs["skip_mask_out_condition"]
        assert "token_tags" not in branch.static_kwargs
        assert not bool((branch.static_kwargs["block_token_tags"] < 0).any())
        torch.testing.assert_close(
            branch.static_kwargs["img_pos_for_infer_output_info"]["position_ids"],
            branch.img_pos_dev[branch.update_mask_dev],
            rtol=0,
            atol=0,
        )
        video_steps = [0.75, 0.1]
        audio_steps = [0.625, 0.2]
        plan = branch.prepare_timestep_plan(
            video_timesteps=video_steps,
            audio_timesteps=audio_steps,
            imgvid_cond_noise_aug=0.6,
            audio_ref_cond_noise_aug=0.4,
        )

        assert branch.static_kwargs["packed_seq_params"]["cu_seqlens_q_host"] == tuple(
            int(value)
            for value in branch.static_kwargs["packed_seq_params"][
                "cu_seqlens_q"
            ].tolist()
        )
        assert branch.static_kwargs["refiner_packed_seq_params"][
            "cu_seqlens_q_host"
        ] == (0, 3, 3)

        for step, (video_t, audio_t) in enumerate(
            zip(video_steps, audio_steps, strict=True)
        ):
            reference = torch.full((branch.seq_len,), video_t, dtype=torch.float32)
            reference[branch.img_cond_seq_idx] = max(video_t, 0.6)
            reference[branch.audio_target_seq_idx] = audio_t
            reference[branch.audio_ref_seq_idx] = max(audio_t, 0.4)
            expected = torch.unique(reference, sorted=True, return_inverse=True)
            torch.testing.assert_close(plan[step][0], expected[0], rtol=0, atol=0)
            torch.testing.assert_close(plan[step][1], expected[1], rtol=0, atol=0)
            torch.testing.assert_close(
                plan[step][2],
                branch.static_kwargs["block_token_tags"]
                + expected[1] * MINIMAX_H3_ADALN_MODALITY_NUM,
                rtol=0,
                atol=0,
            )

        repeated_plan = branch.prepare_timestep_plan(
            video_timesteps=[0.0, 0.1, 0.2],
            audio_timesteps=[0.0, 0.2, 0.4],
            imgvid_cond_noise_aug=0.999,
            audio_ref_cond_noise_aug=1.0,
        )
        assert repeated_plan[1][1] is repeated_plan[2][1]
        assert repeated_plan[1][2] is repeated_plan[2][2]


def test_res_multistep_first_step_matches_euler_flow():
    generator = torch.Generator().manual_seed(11)
    state = torch.randn(9, 16, generator=generator)
    model_output = torch.randn(9, 16, generator=generator)
    sigma_curr, sigma_next = 0.8, 0.5
    denoised = state + sigma_curr * model_output
    expected = state + ((state - denoised) / sigma_curr) * (sigma_next - sigma_curr)

    actual = state.clone()
    scratch = torch.empty_like(actual)
    _minimax_h3_res_multistep_target_rows_(
        actual,
        denoised,
        sigma_curr=sigma_curr,
        sigma_next=sigma_next,
        sigma_prev=None,
        old_denoised=None,
        old_sigma_down=None,
        derivative_scratch=scratch,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


def test_inplace_target_update_matches_scheduler_math():
    generator = torch.Generator().manual_seed(7)
    for sigma_curr, sigma_next in ((1.0, 0.7), (0.2, 0.0), (0.0, 0.0)):
        state = torch.randn(11, 32, generator=generator)
        velocity = torch.randn(11, 32, generator=generator)
        timestep = torch.tensor(1.0 - sigma_curr)
        ratio = torch.tensor(0.0 if sigma_curr == 0.0 else sigma_next / sigma_curr)
        denoised = _minimax_h3_rf_v_to_x0(state, velocity, timestep)
        expected = _minimax_h3_euler_eta0_step(
            state,
            denoised,
            sigma_curr=sigma_curr,
            sigma_next=sigma_next,
            sigma_ratio=ratio,
        )

        actual = state.clone()
        velocity_scratch = velocity.clone()
        _minimax_h3_update_target_rows_(
            actual,
            velocity_scratch,
            sigma_t=1.0 - timestep,
            sigma_curr=sigma_curr,
            sigma_ratio=ratio,
            one_minus_sigma_ratio=1.0 - ratio,
            denoised_scratch=torch.empty_like(actual),
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_local_text_layout_is_a_contiguous_prefix_per_ulysses_rank():
    for mode in ("t2va", "fl2va", "ref2va"):
        branch = _branch(mode)
        text_len = int(branch.static_kwargs["prompt_embeds"].shape[0])
        for world_size in (1, 2, 4, 8):
            if branch.seq_len % world_size:
                continue
            for rank in range(world_size):
                layout = _build_local_embedding_layout(
                    seq_len=branch.seq_len,
                    text_pos=torch.arange(text_len),
                    img_pos=branch.img_pos,
                    audio_pos=branch.audio_pos,
                    world_size=world_size,
                    rank=rank,
                    device=torch.device("cpu"),
                )
                start = int(layout["text_source_start"])
                stop = int(layout["text_source_stop"])
                row_start = rank * (branch.seq_len // world_size)
                expected = torch.nonzero(
                    (torch.arange(text_len) >= row_start)
                    & (
                        torch.arange(text_len)
                        < row_start + branch.seq_len // world_size
                    )
                ).view(-1)
                assert expected.tolist() == list(range(start, stop))


def test_rank_local_token_tags_match_reference_slice():
    for mode in ("t2va", "fl2va", "ref2va"):
        seq_len = _branch(mode).seq_len
        token_tags = torch.arange(seq_len, dtype=torch.long) - seq_len // 2
        for world_size in (1, 2, 4, 8):
            if seq_len % world_size:
                continue
            for rank in range(world_size):
                with patch(
                    "sglang.multimodal_gen.runtime.pipelines_core.stages."
                    "model_specific_stages.minimax_h3.denoise_loop._ulysses_ctx",
                    return_value=(world_size, rank),
                ):
                    branch = _branch(mode, token_tags=token_tags)
                local_rows = branch.seq_len // world_size
                expected = token_tags[
                    rank * local_rows : (rank + 1) * local_rows
                ].clamp(min=0)
                torch.testing.assert_close(
                    branch.static_kwargs["block_token_tags"], expected, rtol=0, atol=0
                )


def test_time_shift_sigma_and_slope_match_comfy():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
        minimax_h3_time_shift_sigma,
        minimax_h3_time_shift_slope,
        minimax_h3_time_shift_sigmas,
    )

    def comfy_sigma(sigma, from_shift, to_shift):
        base = sigma / (from_shift + sigma * (1.0 - from_shift))
        return to_shift * base / (1.0 + (to_shift - 1.0) * base)

    def comfy_slope(sigma, from_shift, to_shift):
        base = sigma / (from_shift + sigma * (1.0 - from_shift))
        return (to_shift * (1.0 + (from_shift - 1.0) * base) ** 2) / (
            from_shift * (1.0 + (to_shift - 1.0) * base) ** 2
        )

    for sigma in (1.0, 0.75, 0.5, 0.25, 0.05):
        assert minimax_h3_time_shift_sigma(
            sigma, from_shift=12.0, to_shift=3.0
        ) == comfy_sigma(sigma, 12.0, 3.0)
        assert minimax_h3_time_shift_slope(
            sigma, from_shift=12.0, to_shift=3.0
        ) == comfy_slope(sigma, 12.0, 3.0)

    audio_schedule = minimax_h3_time_shift_sigmas(num_steps=20, shift_scale=3.0)
    video_schedule = minimax_h3_time_shift_sigmas(num_steps=20, shift_scale=12.0)
    for sigma_v, sigma_a in zip(video_schedule[:-1], audio_schedule[:-1]):
        derived = minimax_h3_time_shift_sigma(
            sigma_v, from_shift=12.0, to_shift=3.0
        )
        assert derived == pytest.approx(sigma_a, rel=0.0, abs=1e-5)


def test_time_shift_sigmas_match_comfy_simple_scheduler():
    import torch

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
        minimax_h3_time_shift_sigmas,
    )

    def comfy_simple_sigmas(steps: int, shift: float = 12.0) -> list[float]:
        base = torch.arange(1, 1001, dtype=torch.float32) / 1000.0
        shifted = shift * base / (1 + (shift - 1) * base)
        stride = 1000 / float(steps)
        sigmas = [
            float(shifted[-(1 + int(step * stride))]) for step in range(steps)
        ]
        sigmas.append(0.0)
        return sigmas

    for steps in (4, 20, 50):
        expected = comfy_simple_sigmas(steps)
        actual = minimax_h3_time_shift_sigmas(num_steps=steps, shift_scale=12.0)
        assert actual == expected
