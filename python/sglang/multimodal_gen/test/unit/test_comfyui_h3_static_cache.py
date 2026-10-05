"""H3 request-static RoPE must be reused only for the matching sampler run."""

from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.pipelines_core.comfyui_mode import (
    bind_comfyui_session,
    get_run_state,
    release_comfyui_session,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages import (
    denoising,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.comfyui_step import (
    MiniMaxH3ComfyUIStepStage,
    build_step_forward_kwargs,
    comfyui_payload_to_branch_inputs,
)
from sglang.multimodal_gen.runtime.models.dits import minimax_h3_vdn_attention


class _StaticModel:
    def __init__(self):
        self.rope_calls = []

    def refine_prompt_embeds(self, prompt, refiner_cu, *, device):
        return prompt.to(device)

    def build_rope_cache(self, positions, *, device):
        cache = (
            positions.clone().to(device),
            torch.arange(positions.shape[1], device=device),
        )
        self.rope_calls.append(cache)
        return cache


def test_step_rope_cache_reuses_static_inputs_and_rebuilds_for_new_geometry_or_run(
    monkeypatch,
):
    monkeypatch.setattr(denoising, "_build_cube_attn_metadata", lambda *a, **k: None)
    monkeypatch.setattr(
        minimax_h3_vdn_attention, "prepare_hybrid_attention_metadata", lambda **k: None
    )
    model = _StaticModel()
    stage = MiniMaxH3ComfyUIStepStage.__new__(MiniMaxH3ComfyUIStepStage)
    stage.transformer = model
    req = SimpleNamespace(extra={"comfyui_session_id": "static-test:1"})
    config = SimpleNamespace()
    device = torch.device("cpu")
    sigmas = torch.tensor([1.0, 0.5, 0.0])

    def inputs(width, timestep):
        return comfyui_payload_to_branch_inputs(
            torch.zeros(1, 24, 2, 4, width),
            torch.zeros(1, 32, 2, 3),
            torch.ones(8, 32),
            minimax_payload={},
            sample_sigmas=sigmas,
            timestep=torch.tensor([timestep]),
        )

    try:
        bind_comfyui_session(req)
        first_inputs = inputs(4, 1000.0)
        first = stage._bind_run_state(req, first_inputs, sigmas, device, config)
        assert len(model.rope_calls) == 1
        fk, _ = build_step_forward_kwargs(
            first_inputs, branch=first.branch, device=device
        )
        assert fk["rope_cache"] is model.rope_calls[0]
        first.steps_done = 1
        again = stage._bind_run_state(req, inputs(4, 500.0), sigmas, device, config)
        assert again is first
        assert len(model.rope_calls) == 1

        changed = stage._bind_run_state(req, inputs(8, 500.0), sigmas, device, config)
        assert changed is not first
        assert len(model.rope_calls) == 2
        assert first.branch is None

        req.extra = {"comfyui_session_id": "static-test:2"}
        bind_comfyui_session(req)
        third = stage._bind_run_state(req, inputs(8, 1000.0), sigmas, device, config)
        assert third is not changed
        assert len(model.rope_calls) == 3
        assert (
            get_run_state(
                SimpleNamespace(extra={"comfyui_session_id": "static-test:1"})
            )
            is None
        )
    finally:
        release_comfyui_session("static-test:1")
        release_comfyui_session("static-test:2")
