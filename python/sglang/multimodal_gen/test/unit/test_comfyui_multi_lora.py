import itertools
from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

# Initialize the quantization registry before importing the LoRA layer modules.
import sglang.multimodal_gen.runtime.layers.quantization  # noqa: F401
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base import (
    SGLDiffusionExecutor,
)
from sglang.multimodal_gen.runtime.layers.lora.linear import BaseLayerWithLoRA
from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import LoRAPipeline


class Pipeline(LoRAPipeline):
    def create_pipeline_stages(self, args):
        return None


class TupleLinear(torch.nn.Linear):
    def forward(self, x):
        return super().forward(x), None


def output(layer):
    return layer(torch.ones(1, 2))[0]


def build(mode="merge"):
    linear = TupleLinear(2, 2, bias=False)
    linear.weight.data.copy_(torch.eye(2))
    layer = BaseLayerWithLoRA(linear)
    p = object.__new__(Pipeline)
    p.modules = {"transformer": torch.nn.Module()}
    p.modules["transformer"].add_module("linear", layer)
    p.server_args = SimpleNamespace(
        lora_alpha=None, lora_merge_mode=mode, model_path="/model"
    )
    p.lora_initialized = True
    p.lora_adapters = defaultdict(dict)
    p.loaded_adapter_paths = {}
    p.loaded_adapter_alphas = {}
    p.cur_adapter_name = {}
    p.cur_adapter_path = {}
    p.cur_adapter_strength = {}
    p.cur_adapter_config = {}
    p.lora_layers = {"linear": layer}
    p.lora_layers_transformer_2 = {}
    p.lora_layers_critic = {}
    p.is_lora_merged = {}
    p._temporarily_disable_offload = lambda *a, **k: nullcontext([])
    for i, name in enumerate(["A", "B", "C"]):
        p.loaded_adapter_paths[name] = "/" + name
        p.loaded_adapter_alphas[name] = None
        p.lora_adapters[name] = {
            "linear.lora_A": torch.tensor([[1.0, 1.0]]),
            "linear.lora_B": torch.tensor(
                [[1.0 if i in [0, 2] else 0.0], [1.0 if i in [1, 2] else 0.0]]
            ),
        }
    return p, layer


STATES = [
    [],
    [("A", 0.5)],
    [("B", 0.25)],
    [("A", 0.5), ("B", 0.25)],
    [("A", 0.5), ("C", 0.25)],
    [("A", 0.5), ("B", 0.25), ("C", 0.125)],
    [("B", 0.25), ("A", 0.5)],
    [("A", 0.0)],
    [("A", 1.0)],
    [("A", -0.5)],
]


def kwargs(state):
    return {
        "lora_nickname": [n for n, v in state],
        "lora_path": [None] * len(state),
        "strength": [v for n, v in state],
        "target": ["transformer"] * len(state),
    }


def expected(state):
    out = torch.ones(1, 2)
    for name, scale in state:
        if name in ["A", "C"]:
            out[0, 0] += 2 * scale
        if name in ["B", "C"]:
            out[0, 1] += 2 * scale
    return out


@pytest.mark.parametrize("mode", ["merge", "dynamic"])
@pytest.mark.parametrize(
    "old,new", list(itertools.product(range(len(STATES)), repeat=2))
)
def test_merge_replacement_and_removal(old, new, mode):
    p, layer = build(mode)
    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline.dist.get_rank",
        return_value=0,
    ):
        for state in [STATES[old], STATES[new]]:
            if state:
                p.set_lora(**kwargs(state))
            else:
                p.unmerge_lora_weights()
            torch.testing.assert_close(output(layer), expected(state), rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["merge", "dynamic"])
@pytest.mark.parametrize(
    "old,new", list(itertools.product(range(len(STATES)), repeat=2))
)
def test_local_sampler_binds_cached_model_state(old, new, mode):
    p, layer = build(mode)
    ex = object.__new__(SGLDiffusionExecutor)
    torch.nn.Module.__init__(ex)
    ex._ensure_runtime = None
    ex._lora_input = None
    ex._run_id = 0
    ex._sent_conds = set()
    ex.generator = p
    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline.dist.get_rank",
        return_value=0,
    ):
        for state in [STATES[old], STATES[new]]:
            patcher = SimpleNamespace(
                model_options={"sgld_lora_input": kwargs(state)} if state else {}
            )
            wrap = SimpleNamespace(model_patcher=patcher)
            out = ex.sampler_sample_wrapper(lambda *a, **k: output(layer), wrap)
            torch.testing.assert_close(out, expected(state), rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["merge", "dynamic"])
def test_overlapping_targets_keep_both_adapters(mode):
    p, layer = build(mode)
    state = [("A", 0.5), ("B", 0.25)]
    kw = kwargs(state)
    kw["target"] = ["all", "transformer"]
    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline.dist.get_rank",
        return_value=0,
    ):
        p.set_lora(**kw)
        torch.testing.assert_close(output(layer), expected(state), rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["merge", "dynamic"])
def test_partial_adapter_replaces_previous_layer_state(mode):
    p, layer = build(mode)
    p.lora_adapters["A"] = {}
    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline.dist.get_rank",
        return_value=0,
    ):
        p.set_lora(**kwargs([("C", 0.5)]))
        p.set_lora(**kwargs([("A", 0.5), ("B", 0.25)]))
        torch.testing.assert_close(
            output(layer), expected([("B", 0.25)]), rtol=0, atol=0
        )


@pytest.mark.parametrize("mode", ["merge", "dynamic"])
def test_adapter_missing_layer_removes_old_effect(mode):
    p, layer = build(mode)
    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline.dist.get_rank",
        return_value=0,
    ):
        p.set_lora(**kwargs([("C", 0.5)]))
        p.lora_adapters["A"] = {}
        p.set_lora(**kwargs([("A", 0.5)]))
        torch.testing.assert_close(output(layer), expected([]), rtol=0, atol=0)


@pytest.mark.parametrize("scale", [0.0, 0.3, 1.0])
def test_dynamic_mixed_ranks_alphas_and_offsets(scale):
    from sglang.multimodal_gen.runtime.layers.lora.linear import LinearWithLoRA

    torch.manual_seed(19)
    base = torch.nn.Linear(4, 3)
    layer = LinearWithLoRA(base)
    x = torch.randn(2, 4)
    expected = base(x)
    for rank, alpha, strength in [(1, 2, 0.4), (2, 1, -0.3)]:
        layer.lora_rank, layer.lora_alpha = rank, alpha
        A, B, offset = torch.randn(rank, 4), torch.randn(3, rank), torch.randn(3)
        layer.set_lora_weights(
            A, B, strength=strength, merge_weights=False, output_offset=offset
        )
        expected = (
            expected + ((x @ A.T @ B.T) + offset) * (alpha / rank) * strength * scale
        )
    with patch.object(layer, "_runtime_lora_scale", return_value=scale):
        actual = layer(x)
    torch.testing.assert_close(actual, expected)


def test_sampler_failed_lora_does_not_record_failed_configuration():
    from unittest.mock import Mock

    ex = object.__new__(SGLDiffusionExecutor)
    torch.nn.Module.__init__(ex)
    ex._ensure_runtime = None
    ex._lora_input = kwargs([("A", 0.5)])
    ex._run_id = 0
    ex._sent_conds = set()
    ex.generator = Mock()
    ex.generator.set_lora.side_effect = ValueError("bad adapter")
    desired = kwargs([("B", 0.5)])
    wrap = SimpleNamespace(
        model_patcher=SimpleNamespace(model_options={"sgld_lora_input": desired})
    )
    with pytest.raises(ValueError, match="bad adapter"):
        ex.sampler_sample_wrapper(lambda *a: None, wrap)
    assert ex._lora_input is None
    ex.generator.unmerge_lora_weights.assert_called_once()
    wrap.model_patcher.model_options = {}
    ex.sampler_sample_wrapper(lambda *a: None, wrap)
    assert ex._lora_input is None


@pytest.mark.parametrize("kind", ["column", "row"])
@pytest.mark.parametrize("tp_rank", [0, 1])
def test_dynamic_multiple_adapters_shard_each_projection(kind, tp_rank):
    from sglang.multimodal_gen.runtime.layers.lora.linear import (
        ColumnParallelLinearWithLoRA,
        RowParallelLinearWithLoRA,
    )

    torch.manual_seed(37)
    x = torch.randn(2, 4)
    base = TupleLinear(4, 4, bias=False)
    base.skip_bias_add = False
    base.gather_output = False
    base.reduce_results = False
    base.input_is_parallel = True
    base.output_size_per_partition = 2
    base.output_partition_sizes = [2]
    base.input_size_per_partition = 2
    base.tp_size = 2
    if kind == "column":
        base.weight.data = base.weight[tp_rank * 2 : (tp_rank + 1) * 2].detach().clone()
        layer = ColumnParallelLinearWithLoRA(base)
    else:
        base.weight.data = (
            base.weight[:, tp_rank * 2 : (tp_rank + 1) * 2].detach().clone()
        )
        x = x[:, tp_rank * 2 : (tp_rank + 1) * 2]
        layer = RowParallelLinearWithLoRA(base)
    base.quant_method = SimpleNamespace(
        apply=lambda module, value, bias=None: torch.nn.functional.linear(
            value, module.weight, bias
        )
    )
    expected = torch.nn.functional.linear(x, base.weight)
    for rank, strength in [(1, 0.4), (2, -0.3)]:
        A, B = torch.randn(rank, 4), torch.randn(4, rank)
        layer.lora_rank, layer.lora_alpha = rank, rank
        layer.set_lora_weights(A, B, strength=strength, merge_weights=False)
        if kind == "column":
            B = B[tp_rank * 2 : (tp_rank + 1) * 2]
        else:
            A = A[:, tp_rank * 2 : (tp_rank + 1) * 2]
        expected = expected + (x @ A.T @ B.T) * strength
    with patch(
        "sglang.multimodal_gen.runtime.layers.lora.linear.get_tp_rank",
        return_value=tp_rank,
    ):
        actual, _ = layer(x)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("missing", ["A", "B"])
def test_merge_cache_handles_partial_adapter_coverage(missing):
    p, layer = build()
    p.lora_adapters[missing] = {}
    cache = SimpleNamespace(
        get=lambda *args: None,
        put=lambda name, weight: weight,
    )
    p._apply_lora_to_layers(
        p.lora_layers,
        ["A", "B"],
        ["/A", "/B"],
        rank=0,
        strengths=[0.5, 0.25],
        merge_weights=True,
        clear_existing=True,
        merge_cache=cache,
    )
    assert layer.merged
    remaining = [("B", 0.25)] if missing == "A" else [("A", 0.5)]
    torch.testing.assert_close(output(layer), expected(remaining), rtol=0, atol=0)
    layer.unmerge_lora_weights()
    torch.testing.assert_close(layer.weight, torch.eye(2), rtol=0, atol=0)
