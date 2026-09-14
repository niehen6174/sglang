"""LoRA merge must round-trip kitchen INT8 the way ComfyUI convert/set does."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.lora.linear import (
    BaseLayerWithLoRA,
    LinearWithLoRA,
    wrap_with_lora_layer,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_int8_config import (
    KitchenInt8Config,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    write_offload_params,
)


def _has_kitchen() -> bool:
    try:
        import comfy_kitchen  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch.ops, "comfy_kitchen") and hasattr(
        torch.ops.comfy_kitchen, "int8_linear"
    )


def _make_kitchen_layer(out_f: int, in_f: int, weight: torch.Tensor):
    layer = ReplicatedLinear(
        in_f,
        out_f,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=KitchenInt8Config(),
    )
    layer = layer.to("cuda")
    with torch.no_grad():
        layer.weight.copy_(weight)
    layer.quant_method.process_weights_after_loading(layer)
    return layer


class TestDenseLoRAMergeStillWorks(unittest.TestCase):
    def test_write_offload_params_fail_closed_on_owned_miss(self):
        class _Mgr:
            enabled = True

            def has_cpu_weight(self, name):
                return name == "blocks.0.attn.to_q.weight"

            def update_cpu_weights(self, weight_dict):
                return set()

        layer = nn.Linear(2, 2, bias=False)
        layer._offload_root = nn.Module()
        layer._offload_root.layerwise_offload_managers = [_Mgr()]
        layer._offload_param_prefix = "blocks.0.attn.to_q"
        with self.assertRaisesRegex(RuntimeError, "writeback missed"):
            write_offload_params(layer, {"weight": torch.ones(2, 2)})

    def test_write_offload_params_skips_unowned_names(self):
        class _Mgr:
            enabled = True

            def has_cpu_weight(self, name):
                return False

            def update_cpu_weights(self, weight_dict):
                raise AssertionError("unowned names must not call update")

        layer = nn.Linear(2, 2, bias=False)
        layer._offload_root = nn.Module()
        layer._offload_root.layerwise_offload_managers = [_Mgr()]
        layer._offload_param_prefix = "token_refiner.refiner_blocks.0.attn.to_q"
        self.assertFalse(write_offload_params(layer, {"weight": torch.ones(2, 2)}))

    def test_replicated_bf16_merge_unmerge(self):
        torch.manual_seed(0)
        out_f, in_f, rank = 6, 8, 2
        base = ReplicatedLinear(in_f, out_f, bias=False, params_dtype=torch.float32)
        with torch.no_grad():
            base.weight.copy_(torch.randn(out_f, in_f))
        original = base.weight.detach().clone()
        layer = wrap_with_lora_layer(base, lora_rank=rank, lora_alpha=rank)
        assert isinstance(layer, BaseLayerWithLoRA)
        A = torch.randn(rank, in_f)
        B = torch.randn(out_f, rank)
        layer.set_lora_weights(
            A, B, strength=0.5, clear_existing=True, merge_weights=True
        )
        self.assertTrue(layer.merged)
        torch.testing.assert_close(layer.base_layer.weight, original + 0.5 * (B @ A))
        layer.unmerge_lora_weights()
        torch.testing.assert_close(layer.base_layer.weight, original)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
@unittest.skipUnless(_has_kitchen(), "requires comfy_kitchen")
class TestKitchenInt8LoRA(unittest.TestCase):
    def test_dequant_undoes_convrot(self):
        torch.manual_seed(0)
        out_f, in_f = 128, 256
        weight = torch.randn(out_f, in_f, dtype=torch.bfloat16, device="cuda")
        layer = _make_kitchen_layer(out_f, in_f, weight)
        self.assertEqual(layer.weight.dtype, torch.int8)
        dense = layer.quant_method.to_dense_weight(layer, "cuda")
        self.assertEqual(dense.dtype, torch.bfloat16)
        cosine = torch.nn.functional.cosine_similarity(
            dense.flatten().float(), weight.flatten().float(), dim=0
        )
        self.assertGreater(float(cosine), 0.99)

    def test_merge_happens_in_original_space(self):
        torch.manual_seed(1)
        out_f, in_f, rank = 128, 256, 4
        weight = torch.randn(out_f, in_f, dtype=torch.bfloat16, device="cuda")
        layer = _make_kitchen_layer(out_f, in_f, weight)
        method = layer.quant_method
        before = method.to_dense_weight(layer, "cuda").float()
        packed_before = layer.weight.detach().clone()
        scale_before = layer.weight_scale.detach().clone()

        lora = wrap_with_lora_layer(layer, lora_rank=rank, lora_alpha=rank)
        A = torch.randn(rank, in_f, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(out_f, rank, dtype=torch.bfloat16, device="cuda")
        lora.set_lora_weights(
            A, B, strength=1.0, clear_existing=True, merge_weights=True
        )

        self.assertEqual(lora.base_layer.weight.dtype, torch.int8)
        after = method.to_dense_weight(lora.base_layer, "cuda").float()
        expected = before + (B.float() @ A.float())
        cosine = torch.nn.functional.cosine_similarity(
            after.flatten(), expected.flatten(), dim=0
        )
        self.assertGreater(float(cosine), 0.99)

        lora.unmerge_lora_weights()
        torch.testing.assert_close(lora.base_layer.weight, packed_before)
        torch.testing.assert_close(lora.base_layer.weight_scale, scale_before)

    def test_merged_forward_tracks_dense_lora(self):
        torch.manual_seed(2)
        out_f, in_f, rank = 128, 256, 4
        weight = torch.randn(out_f, in_f, dtype=torch.bfloat16, device="cuda")
        layer = _make_kitchen_layer(out_f, in_f, weight)
        method = layer.quant_method
        base_dense = method.to_dense_weight(layer, "cuda")

        lora = wrap_with_lora_layer(layer, lora_rank=rank, lora_alpha=rank)
        A = torch.randn(rank, in_f, dtype=torch.bfloat16, device="cuda") * 0.05
        B = torch.randn(out_f, rank, dtype=torch.bfloat16, device="cuda") * 0.05
        lora.set_lora_weights(
            A, B, strength=1.0, clear_existing=True, merge_weights=True
        )

        x = torch.randn(3, in_f, dtype=torch.bfloat16, device="cuda")
        actual, _ = lora(x)
        reference = torch.nn.functional.linear(x.float(), (base_dense + B @ A).float())
        cosine = torch.nn.functional.cosine_similarity(
            actual.flatten().float(), reference.flatten(), dim=0
        )
        self.assertGreater(float(cosine), 0.98)

    def test_refuses_to_add_into_raw_int8(self):
        base = nn.Linear(8, 4, bias=False)
        with torch.no_grad():
            base.weight.copy_(torch.ones(4, 8))
        layer = LinearWithLoRA(base, lora_rank=2, lora_alpha=2)
        layer.base_layer.weight = nn.Parameter(
            torch.ones(4, 8, dtype=torch.int8), requires_grad=False
        )
        layer.cpu_weight = layer.base_layer.weight.detach().to("cpu").clone()
        with self.assertRaisesRegex(RuntimeError, "INT8"):
            layer.set_lora_weights(
                torch.ones(2, 8),
                torch.ones(4, 2),
                clear_existing=True,
                merge_weights=True,
            )

    def test_to_dense_prefers_cpu_packed_over_stale_gpu(self):
        torch.manual_seed(4)
        out_f, in_f = 128, 256
        old = torch.randn(out_f, in_f, dtype=torch.bfloat16, device="cuda")
        new = torch.randn(out_f, in_f, dtype=torch.bfloat16, device="cuda")
        layer = _make_kitchen_layer(out_f, in_f, old)
        method = layer.quant_method
        qdata, scale = method._quantize_dense(new, torch.device("cpu"))
        layer._packed_weight_cpu = qdata
        layer._packed_scale_cpu = scale
        dense = method.to_dense_weight(layer, "cuda")
        cosine = torch.nn.functional.cosine_similarity(
            dense.flatten().float(), new.flatten().float(), dim=0
        )
        self.assertGreater(float(cosine), 0.99)
        stale = method.dequantize_packed(
            layer.weight.detach(),
            layer.weight_scale.detach(),
            torch.bfloat16,
            "cuda",
        )
        self.assertLess(
            float(
                torch.nn.functional.cosine_similarity(
                    stale.flatten().float(), new.flatten().float(), dim=0
                )
            ),
            0.95,
        )

    def test_merge_writes_manager_store_not_placeholder(self):
        torch.manual_seed(6)
        out_f, in_f, rank = 128, 256, 4
        weight = torch.randn(out_f, in_f, dtype=torch.bfloat16, device="cuda")
        layer = _make_kitchen_layer(out_f, in_f, weight)
        method = layer.quant_method
        before = method.to_dense_weight(layer, "cuda").float()
        packed = layer.weight.detach().cpu().contiguous()
        scale = layer.weight_scale.detach().cpu().contiguous()
        store = {
            "blocks.0.attn.to_q.weight": packed.clone(),
            "blocks.0.attn.to_q.weight_scale": scale.clone(),
        }

        class _Mgr:
            enabled = True

            def has_cpu_weight(self, name):
                return name in store

            def update_cpu_weights(self, weight_dict):
                updated = set()
                for name, value in weight_dict.items():
                    if name not in store:
                        continue
                    dest = store[name]
                    dest.copy_(value.detach().to(device="cpu", dtype=dest.dtype))
                    updated.add(name)
                return updated

            def get_cpu_weight(self, name):
                return store.get(name)

            def _match_layer_idx(self, name):
                return 0

            def release_layer(self, layer_idx, force=False):
                return None

        root = nn.Module()
        root.layerwise_offload_managers = [_Mgr()]
        layer._offload_root = root
        layer._offload_param_prefix = "blocks.0.attn.to_q"
        layer._packed_weight_cpu = store["blocks.0.attn.to_q.weight"]
        layer._packed_scale_cpu = store["blocks.0.attn.to_q.weight_scale"]
        layer.weight = nn.Parameter(torch.empty(1, device="cuda"), requires_grad=False)
        lora = wrap_with_lora_layer(
            layer, lora_rank=rank, lora_alpha=rank, snapshot_base=False
        )
        lora._lora_backup = method.snapshot_for_lora(lora.base_layer, clone=False)
        lora.cpu_weight = lora._lora_backup["weight"]
        lora._base_is_view = True
        A = torch.randn(rank, in_f, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(out_f, rank, dtype=torch.bfloat16, device="cuda")
        lora.set_lora_weights(
            A, B, strength=1.0, clear_existing=True, merge_weights=True
        )
        self.assertEqual(tuple(lora.base_layer.weight.shape), (1,))
        after = method.to_dense_weight(lora.base_layer, "cuda").float()
        expected = before + (B.float() @ A.float())
        cosine = torch.nn.functional.cosine_similarity(
            after.flatten(), expected.flatten(), dim=0
        )
        self.assertGreater(float(cosine), 0.99)
        lora.unmerge_lora_weights()
        restored = method.to_dense_weight(lora.base_layer, "cuda").float()
        cosine = torch.nn.functional.cosine_similarity(
            restored.flatten(), before.flatten(), dim=0
        )
        self.assertGreater(float(cosine), 0.99)


if __name__ == "__main__":
    unittest.main()
