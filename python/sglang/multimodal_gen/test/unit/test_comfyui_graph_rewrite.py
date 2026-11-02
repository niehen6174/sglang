# SPDX-License-Identifier: Apache-2.0
"""Speed-only patch nodes and custom LoRA loaders on SGLD models."""

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.graph_rewrite import (
    drop_speed_patches,
)


def _graph(loader: str) -> dict:
    return {
        "1": {"class_type": loader, "inputs": {"unet_name": "dit.safetensors"}},
        "2": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["1", 0]}},
        "3": {
            "class_type": "MiniMaxH3TurboLoRA",
            "inputs": {
                "model": ["2", 0],
                "lora_name": "turbo.safetensors",
                "strength": 0.8,
            },
        },
        "4": {"class_type": "PathchSageAttentionKJ", "inputs": {"model": ["3", 0]}},
        "5": {"class_type": "KSampler", "inputs": {"model": ["4", 0]}},
    }


def test_speed_patches_on_sgld_models_are_dropped_and_custom_loras_routed() -> None:
    prompt = _graph("SGLDUNETLoader")
    changed = drop_speed_patches(prompt)
    assert sorted(changed) == [
        "2:ModelAttentionBackend",
        "3:MiniMaxH3TurboLoRA->SGLDLoraLoader",
        "4:PathchSageAttentionKJ",
    ]
    assert prompt["3"] == {
        "class_type": "SGLDLoraLoader",
        "inputs": {
            "model": ["1", 0],
            "lora_name": "turbo.safetensors",
            "strength_model": 0.8,
            "nickname": "",
            "target": "all",
        },
    }
    assert prompt["5"]["inputs"]["model"] == ["3", 0]


def test_stock_models_keep_their_patch_nodes() -> None:
    prompt = _graph("UNETLoader")
    assert drop_speed_patches(prompt) == []
    assert prompt == _graph("UNETLoader")
