# SPDX-License-Identifier: Apache-2.0
"""Quality gate helpers for ComfyQuant numerical parity."""

import torch

from sglang.multimodal_gen.runtime.layers.quantization.comfy_quant_kernel_adapter import (
    ComfyQuantKernelBackend,
    _rotate_convrot_activation,
    apply_int8_linear,
)


def _ssim_1d(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    mu_a = a.mean()
    mu_b = b.mean()
    var_a = a.var(unbiased=False)
    var_b = b.var(unbiased=False)
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1 = 1e-4
    c2 = 9e-4
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    return float((num / den).item())


def test_int8_convrot_quality_gate_against_bf16_reference():
    torch.manual_seed(42)
    layers = 50
    hidden = 128
    x = torch.randn(4, hidden, dtype=torch.bfloat16)
    weights = [
        torch.randint(-8, 8, (hidden, hidden), dtype=torch.int8) for _ in range(layers)
    ]
    scales = [
        (torch.rand(hidden, 1, dtype=torch.float32) * 0.05) + 0.01
        for _ in range(layers)
    ]

    quant_state = x
    for weight, scale in zip(weights, scales):
        delta = apply_int8_linear(
            backend=ComfyQuantKernelBackend.TORCH,
            x=quant_state,
            weight=weight,
            weight_scale=scale,
            bias=None,
            convrot=True,
            convrot_groupsize=16,
        )
        quant_state = quant_state + 0.01 * delta

    bf16_state = x
    for weight, scale in zip(weights, scales):
        activation = _rotate_convrot_activation(bf16_state, 16)
        dequant = weight.to(dtype=torch.bfloat16) * scale.to(dtype=torch.bfloat16)
        delta = activation @ dequant.transpose(-1, -2)
        bf16_state = bf16_state + 0.01 * delta

    score = _ssim_1d(quant_state, bf16_state)
    assert score >= 0.95
