"""The public API reports finalized per-group times for every output path."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)
from sglang.multimodal_gen.runtime.entrypoints import diffusion_generator as module
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req


@pytest.mark.parametrize("output_kind", ["files", "mesh", "arrays"])
def test_generation_results_use_completed_group_timer(output_kind):
    params = SamplingParams(
        prompt=["first", "second"],
        data_type=DataType.MESH if output_kind == "mesh" else DataType.IMAGE,
        save_output=output_kind == "files",
        return_file_paths_only=output_kind == "files",
        num_outputs_per_prompt=2,
        seed=3,
        generator_device="cpu",
    )
    generator = module.DiffGenerator.__new__(module.DiffGenerator)
    generator.server_args = SimpleNamespace(
        model_path="test", batching_max_size=1, prompt_file_path=None, warmup_mode="off"
    )
    durations = iter([1.25, 2.5])

    @contextmanager
    def completed_timer(*args, **kwargs):
        timer = SimpleNamespace(duration=0.0)
        yield timer
        timer.duration = next(durations)

    def prepare(*, sampling_params, **kwargs):
        return Req(sampling_params=sampling_params)

    def forward(requests):
        if output_kind == "arrays":
            return OutputBatch(output=["sample"] * len(requests))
        return OutputBatch(output_file_paths=["output"] * len(requests))

    def save(outputs, *args, samples_out, frames_out, audios_out, **kwargs):
        for sample in outputs:
            samples_out.append(sample)
            frames_out.append(None)
            audios_out.append(None)

    with (
        patch.object(
            module.SamplingParams, "from_user_sampling_params_args", return_value=params
        ),
        patch.object(module, "prepare_request", side_effect=prepare),
        patch.object(module, "log_generation_timer", completed_timer),
        patch.object(module, "save_outputs", side_effect=save),
        patch.object(
            generator, "_send_to_scheduler_and_wait_for_response", side_effect=forward
        ),
    ):
        results = generator.generate({"prompt": ["first", "second"]})

    assert len(results) == 4
    assert [result.generation_time for result in results] == [1.25, 1.25, 2.5, 2.5]
    assert [result.prompt_index for result in results] == [0, 1, 2, 3]
