# SPDX-License-Identifier: Apache-2.0
"""IPC LoRA updates reuse the disk PEFT -> native convert."""

from types import SimpleNamespace

import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTArchConfig
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import (
    convert_peft_lora_named_tensors,
)
from sglang.multimodal_gen.runtime.post_training.weights_updater import (
    _group_lora_ab_tensors,
    _lora_convert_mappings,
)

HIDDEN = 8
INNER = 12
FFN = 16
RANK = 2


def _pair(out_features: int, in_features: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(RANK, in_features), torch.randn(out_features, RANK)


def _peft_payload(prefix: str = "transformer_blocks.0") -> list[tuple[str, torch.Tensor]]:
    modules = {
        f"{prefix}.attn.to_q": (INNER, HIDDEN),
        f"{prefix}.attn.to_k": (INNER, HIDDEN),
        f"{prefix}.attn.to_v": (INNER, HIDDEN),
        f"{prefix}.attn.to_out.0": (HIDDEN, INNER),
        f"{prefix}.ff.net.0.proj": (2 * FFN, HIDDEN),
        f"{prefix}.ff.net.2": (HIDDEN, FFN),
    }
    named = []
    for module, (out_features, in_features) in modules.items():
        lora_a, lora_b = _pair(out_features, in_features)
        named.append((f"base_model.model.{module}.lora_A.default.weight", lora_a))
        named.append((f"base_model.model.{module}.lora_B.default.weight", lora_b))
    return named


def test_h3_peft_ipc_payload_maps_to_native_fused_layers():
    payload = _peft_payload()
    by_src = dict(payload)
    converted = convert_peft_lora_named_tensors(
        payload, param_names_mapping=MiniMaxH3DiTArchConfig().param_names_mapping
    )
    pairs = _group_lora_ab_tensors(converted)

    assert set(pairs) == {
        "blocks.0.attn.qkv_proj",
        "blocks.0.attn.out_proj",
        "blocks.0.mlp.fc1",
        "blocks.0.mlp.fc2",
    }

    lora_a, lora_b = pairs["blocks.0.attn.qkv_proj"]
    assert lora_a.shape == (3, RANK, HIDDEN)
    assert lora_b.shape == (3, INNER, RANK)
    merged = (lora_b @ lora_a).reshape(-1, HIDDEN)
    expected = torch.cat(
        [
            by_src[f"base_model.model.transformer_blocks.0.attn.to_{which}.lora_B.default.weight"]
            @ by_src[f"base_model.model.transformer_blocks.0.attn.to_{which}.lora_A.default.weight"]
            for which in ("q", "k", "v")
        ],
        dim=0,
    )
    torch.testing.assert_close(merged, expected, atol=1e-6, rtol=1e-6)

    original_b = by_src["base_model.model.transformer_blocks.0.ff.net.0.proj.lora_B.default.weight"]
    swapped_b = pairs["blocks.0.mlp.fc1"][1]
    torch.testing.assert_close(swapped_b[:FFN], original_b[FFN:])
    torch.testing.assert_close(swapped_b[FFN:], original_b[:FFN])
    torch.testing.assert_close(
        pairs["blocks.0.mlp.fc1"][0],
        by_src["base_model.model.transformer_blocks.0.ff.net.0.proj.lora_A.default.weight"],
    )


def test_miles_stripped_peft_names_convert_the_same_way():
    payload = [
        (name.replace("base_model.model.", "").replace(".default.weight", ""), tensor)
        for name, tensor in _peft_payload()
    ]
    converted = convert_peft_lora_named_tensors(
        payload, param_names_mapping=MiniMaxH3DiTArchConfig().param_names_mapping
    )
    assert {name.rsplit(".lora_", 1)[0] for name, _ in converted} == {
        "blocks.0.attn.qkv_proj",
        "blocks.0.attn.out_proj",
        "blocks.0.mlp.fc1",
        "blocks.0.mlp.fc2",
    }


def test_refiner_blocks_and_native_names_are_accepted():
    payload = _peft_payload("token_refiner.refiner_blocks.1")
    converted = convert_peft_lora_named_tensors(
        payload, param_names_mapping=MiniMaxH3DiTArchConfig().param_names_mapping
    )
    pairs = _group_lora_ab_tensors(converted)
    assert "token_refiner.blocks.1.attn.qkv_proj" in pairs
    assert "token_refiner.blocks.1.mlp.fc1" in pairs

    native = [
        ("blocks.3.attn.out_proj.lora_A", torch.randn(RANK, INNER)),
        ("blocks.3.attn.out_proj.lora_B", torch.randn(HIDDEN, RANK)),
    ]
    converted_native = convert_peft_lora_named_tensors(
        native, param_names_mapping=MiniMaxH3DiTArchConfig().param_names_mapping
    )
    assert [name for name, _ in converted_native] == [
        "blocks.3.attn.out_proj.lora_A",
        "blocks.3.attn.out_proj.lora_B",
    ]


def test_pipeline_arch_mapping_is_preferred_for_ipc():
    arch = MiniMaxH3DiTArchConfig()
    pipeline = SimpleNamespace(
        server_args=SimpleNamespace(
            pipeline_config=SimpleNamespace(dit_config=SimpleNamespace(arch_config=arch))
        )
    )
    dit = SimpleNamespace(param_names_mapping={}, lora_param_names_mapping={})
    mappings = _lora_convert_mappings(pipeline, dit_module=dit)
    assert mappings["param_names_mapping"] is arch.param_names_mapping
