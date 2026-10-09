"""Online H3 MXFP8 must stage one linear rather than the full BF16 checkpoint."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.quantization.mxfp8 import MXFP8Config
from sglang.multimodal_gen.runtime.loader import fsdp_load
from sglang.multimodal_gen.runtime.loader.component_loaders import transformer_loader
from sglang.multimodal_gen.runtime.loader.weight_load_plan import WeightLoadPlan
from sglang.multimodal_gen.runtime.managers import (  # noqa: F401 -- initialize runtime imports
    gpu_worker,
)
from sglang.multimodal_gen.runtime.utils import quantization_utils


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, True),
        ({"is_minimax_h3": False}, False),
        ({"component_starts_on_cpu": False}, False),
        ({"runtime_device": torch.device("cpu")}, False),
        ({"runtime_device": torch.device("mps")}, False),
        ({"use_fsdp": True}, False),
        ({"runtime_quant_config": None}, False),
        (
            {
                "runtime_quant_config": SimpleNamespace(
                    get_name=lambda: "fp8", is_checkpoint_fp8_serialized=False
                )
            },
            True,
        ),
        (
            {
                "runtime_quant_config": SimpleNamespace(
                    get_name=lambda: "fp8", is_checkpoint_fp8_serialized=True
                )
            },
            False,
        ),
        (
            {
                "runtime_quant_config": SimpleNamespace(
                    get_name=lambda: "kitchen_int8", is_checkpoint_fp8_serialized=False
                )
            },
            False,
        ),
        (
            {
                "runtime_quant_config": SimpleNamespace(
                    get_name=lambda: "mxfp8", is_checkpoint_fp8_serialized=True
                )
            },
            False,
        ),
    ],
)
def test_online_h3_mxfp8_staging_admission(overrides, expected):
    values = dict(
        is_minimax_h3=True,
        component_starts_on_cpu=True,
        runtime_device=torch.device("cuda:0"),
        use_fsdp=False,
        runtime_quant_config=SimpleNamespace(
            get_name=lambda: "mxfp8", is_checkpoint_fp8_serialized=False
        ),
    )
    values.update(overrides)
    assert transformer_loader._can_stage_h3_online_mxfp8(**values) is expected


class _TinyModel(nn.Module):
    param_names_mapping = {}
    _fsdp_forward_methods = ()

    def __init__(self, quant_config=None):
        super().__init__()
        self.first = ReplicatedLinear(
            128, 64, bias=False, quant_config=quant_config, params_dtype=torch.bfloat16
        )
        self.second = ReplicatedLinear(
            128, 64, bias=False, quant_config=quant_config, params_dtype=torch.bfloat16
        )
        self.sensitive = nn.Parameter(torch.empty(2, dtype=torch.float32))

    def post_load_weights(self):
        assert self.sensitive.dtype == torch.float32
        assert self.sensitive.device.type == "cpu"


def _weights():
    generator = torch.Generator("cpu").manual_seed(100)
    return {
        "first.weight": torch.randn(64, 128, generator=generator).to(torch.bfloat16)
        * 0.05,
        "second.weight": torch.randn(64, 128, generator=generator).to(torch.bfloat16)
        * 0.05,
        "sensitive": torch.ones(2, dtype=torch.float32),
    }


def _load(quant_config, plan):
    return fsdp_load.maybe_load_fsdp_model(
        model_cls=_TinyModel,
        init_params={"quant_config": quant_config},
        weight_dir_list=[],
        device=torch.device("cuda:0"),
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
        param_dtype=None,
        reduce_dtype=torch.float32,
        component_starts_on_cpu=True,
        weight_load_plan=plan,
        weights_iterator=iter(_weights().items()),
    )


def test_cpu_checkpoint_stages_individual_linears_and_preserves_model_state(
    monkeypatch,
):
    seen = []
    full_move = Mock(
        side_effect=AssertionError("Must never move the entire BF16 model to GPU")
    )
    monkeypatch.setattr(fsdp_load, "_move_to_device_preserving_meta", full_move)

    @contextmanager
    def fake_stage(module, device):
        assert isinstance(module, ReplicatedLinear)
        assert device == torch.device("cuda:0")
        assert module.weight.device.type == "cpu"
        seen.append(module)
        yield module

    monkeypatch.setattr(quantization_utils, "stage_module_for_post_load", fake_stage)
    plan = WeightLoadPlan(
        checkpoint_load_device=torch.device("cpu"),
        layerwise_quant_postprocess_device=torch.device("cuda:0"),
    )
    model = _load(None, plan)
    assert seen == [model.first, model.second]
    full_move.assert_not_called()
    for name, value in model.state_dict().items():
        assert value.device.type == "cpu"
        assert torch.equal(value, _weights()[name])
    assert all(not parameter.requires_grad for parameter in model.parameters())


@pytest.mark.parametrize(
    "extra",
    [
        {"weight_postprocess_device": torch.device("cuda:0")},
        {"defer_cpu_placement": True},
        {"checkpoint_load_device": torch.device("cuda:0")},
    ],
)
def test_layerwise_postprocess_rejects_full_model_device_placement(extra):
    values = dict(
        checkpoint_load_device=torch.device("cpu"),
        layerwise_quant_postprocess_device=torch.device("cuda:0"),
    )
    values.update(extra)
    plan = WeightLoadPlan(**values)
    with pytest.raises(ValueError, match="CPU-backed non-FSDP loading"):
        _load(None, plan)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA MXFP8 kernels")
def test_online_mxfp8_gpu_postprocess_returns_quantized_weights_to_cpu():
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
        ensure_distributed_env_defaults,
    )

    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    plan = WeightLoadPlan(
        checkpoint_load_device=torch.device("cpu"),
        layerwise_quant_postprocess_device=torch.device("cuda:0"),
    )
    model = _load(MXFP8Config(), plan)
    for layer in [model.first, model.second]:
        assert layer.weight.device.type == "cpu"
        assert layer.weight.dtype == torch.float8_e4m3fn
        assert layer.weight_scale.device.type == "cpu"
    layer = model.first.to("cuda")
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16) * 0.05
    actual, _ = layer(x)
    expected = x @ _weights()["first.weight"].to("cuda").t()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.3, atol=0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA FP8 kernels")
def test_online_fp8_staging_matches_full_gpu_postprocess(monkeypatch):
    """Per-linear staging of online fp8 must quantize exactly like the full-GPU path."""
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config
    from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
        ensure_distributed_env_defaults,
    )

    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    staged = _load(
        Fp8Config(),
        WeightLoadPlan(
            checkpoint_load_device=torch.device("cpu"),
            layerwise_quant_postprocess_device=torch.device("cuda:0"),
        ),
    )
    # The reference path moves every parameter to CUDA, the fixture's sentinel included.
    monkeypatch.setattr(_TinyModel, "post_load_weights", lambda self: None)
    full = _load(
        Fp8Config(),
        WeightLoadPlan(
            checkpoint_load_device=torch.device("cpu"),
            weight_postprocess_device=torch.device("cuda:0"),
            defer_cpu_placement=True,
        ),
    )
    for name in ["first", "second"]:
        a, b = getattr(staged, name), getattr(full, name)
        assert a.weight.device.type == "cpu"
        assert a.weight.dtype == b.weight.dtype
        assert torch.equal(
            a.weight.cpu().view(torch.uint8), b.weight.cpu().view(torch.uint8)
        )
        assert torch.equal(a.weight_scale.cpu(), b.weight_scale.cpu())
