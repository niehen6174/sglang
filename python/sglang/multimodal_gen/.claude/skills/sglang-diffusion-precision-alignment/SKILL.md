---
name: sglang-diffusion-precision-alignment
description: Use when debugging numerical differences between SGLang Diffusion and HuggingFace Diffusers for the same model, or when aligning a new model's outputs.
---

# SGLang Diffusion ↔ Diffusers Precision Alignment

Use this skill when:
- A model produces visually different outputs between SGLang and Diffusers with the same parameters
- You need to verify that a new model integration is numerically correct
- You want to quantify the precision gap and identify root causes

---

## Methodology Overview

```
Phase 1: Parameter Alignment     → Eliminate config-level mismatches (the biggest wins)
Phase 2: Coarse-Grained Bisect   → Compare stage boundaries to locate the divergent stage
Phase 3: Fine-Grained Bisect     → Compare per-step, per-block, per-operation within DiT
Phase 4: Quantify & Document     → Measure each factor's contribution independently
```

**Key principle: Align first, analyze second.** Don't speculate from code reading — inject known-good values to isolate variables, then measure.

---

## Phase 1: Parameter Alignment

Parameter mismatch is the most common and impactful source of difference. Before writing any debug code, systematically verify every parameter matches.

### 1.1 Diffusers ↔ SGLang Parameter Cross-Reference

These two frameworks name and organize parameters differently. The table below lists the **generation parameters** that affect numerical output.

#### Core Generation Parameters

| Diffusers `__call__` param | SGLang CLI / SamplingParams | Notes |
|---|---|---|
| `prompt` | `--prompt` / `prompt` | SGLang may apply model-specific templates. Check `preprocess_text_funcs` in the pipeline config. |
| `negative_prompt` | `--negative-prompt` / `negative_prompt` | SGLang has a long default for video models. **Always set explicitly on both sides.** For CFG-distilled models, should be None/unused. |
| `height`, `width` | `--height`, `--width` | Some models quantize to multiples of `vae_scale_factor`. |
| `num_frames` | `--num-frames` | Video models only. Defaults differ — check each model's `configs/sample/<model>.py`. |
| `num_inference_steps` | `--num-inference-steps` | Defaults differ across models — check `configs/sample/<model>.py` and Diffusers `__call__` defaults. Distilled models may use far fewer steps (7~9). |
| `seed` / `generator` | `--seed` | Diffusers: `torch.Generator(device="cuda").manual_seed(42)`. SGLang: `--seed 42`. Ensure same device. |

#### Guidance & CFG Parameters (⚠️ Most Common Pitfall)

Different models use different CFG strategies. This is where mismatches are most likely. **Always check both Diffusers `__call__` defaults and SGLang's `configs/sample/<model>.py`.**

| Diffusers param | SGLang param | Which Models | Description |
|---|---|---|---|
| `guidance_scale` | `--guidance-scale` | Most models | Standard CFG strength. **For CFG-distilled/turbo variants (e.g. Z-Image-Turbo), should be `0.0`.** |
| `true_cfg_scale` | `--true-cfg-scale` | QwenImage, Flux, HunyuanVideo | True CFG with norm rescaling. **⚠️ Defaults often differ between frameworks.** |
| `cfg_normalization` | `--cfg-normalization` | Z-Image | Clamp combined noise norm. **⚠️ Defaults may differ between frameworks.** |
| `guidance_scale_2` | `--guidance-scale-2` | Wan | Secondary guidance for dual-stream models |
| `guidance_rescale` | (in SamplingParams) | LTX | Rescaling factor (Imagen paper) |
| N/A | `--embedded-cfg-scale` | Flux, HunyuanVideo | SGLang-specific embedded guidance. Diffusers uses `guidance_scale` for this — same value, different name. |
| N/A | `--flow-shift` | Wan, HunyuanVideo, Helios | SGLang-specific sigma schedule shift. Diffusers reads from `scheduler_config.json`. |

#### Scheduler Parameters

| Aspect | Diffusers | SGLang | Pitfall |
|---|---|---|---|
| Scheduler type | Auto from `scheduler_config.json` | Auto from model config | Usually matches |
| `mu` / shift params | Read from `scheduler_config.json` | May be **hardcoded** in pipeline config | Compare `cp3_mu` early |
| `flow_shift` | In `scheduler_config.json` | `--flow-shift` or pipeline config default | Values differ across model sizes (e.g. Wan 1.3B=3.0, Wan 14B=8.0) |
| `sigma_min` | May be **manually overridden** in pipeline code | Reads from scheduler config (no override) | **⚠️ Z-Image pipeline sets `scheduler.sigma_min = 0.0` in code, but SGLang doesn't.** This shifts the entire timestep schedule. |
| Sigma schedule | Computed from scheduler | Computed from scheduler | Compare `cp3_timesteps` — if they diverge, check `sigma_min` and `mu` |

#### Attention & Computation

| Aspect | Diffusers | SGLang | Impact |
|---|---|---|---|
| Attention backend | SDPA (default) | FlashAttention (default) | ~1-5% final cosine difference. **Use `--attention-backend torch_sdpa` during alignment to eliminate this variable.** |
| DiT computation | PyTorch step-by-step (bf16) | Fused Triton kernels (fp32 internal) | SGLang may be **more precise** due to fewer bf16 roundtrips |
| RoPE format | Complex tensor | FlashInfer cos/sin concat | Mathematically equivalent, negligible difference |
| Linear layers | `nn.Linear` | `ReplicatedLinear` (vLLM-style) | GEMM tiling differences, small impact |
| **Latent dtype** | Model-dependent (some use fp32) | Model-dependent (may use bf16) | **⚠️ Z-Image diffusers uses fp32 latents throughout; SGLang may use bf16.** Check `get_latent_dtype()` in pipeline config. |

### 1.2 How to Verify: Write a Diffusers Baseline

Create a minimal Diffusers script with **every parameter explicitly set** and checkpoint saving:

```python
import os
os.environ["DIFFUSERS_CKPT_DIR"] = "/tmp/diffusers_ckpts"

import torch
from diffusers import <ModelPipeline>

pipe = <ModelPipeline>.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to("cuda")
generator = torch.Generator(device="cuda").manual_seed(SEED)
result = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    height=HEIGHT, width=WIDTH,
    num_inference_steps=STEPS,
    # ↓ IMPORTANT: print the pipeline's __call__ signature and set EVERY param explicitly
    guidance_scale=...,         # or true_cfg_scale=..., depending on model
    generator=generator,
)
```

Then print the Diffusers pipeline `__call__` default values:
```python
import inspect
sig = inspect.signature(pipe.__call__)
for name, param in sig.parameters.items():
    if name != 'self':
        print(f"  {name}: {param.default}")
```

Compare each default with SGLang's corresponding parameter.

### 1.3 Alignment Checklist

Before any code-level investigation, verify:

- [ ] **Prompt text**: identical (watch for template wrapping)
- [ ] **Negative prompt**: identical (SGLang has a long video default). For CFG-distilled/turbo variants, should be None/unused.
- [ ] **Resolution**: identical (height, width, num_frames)
- [ ] **Steps**: identical. Check Diffusers example code — distilled/turbo variants may use far fewer steps (7~9) than the `__call__` default (50).
- [ ] **Seed**: identical, same device
- [ ] **dtype**: identical (bf16/fp16). Check if Diffusers uses fp32 latents (e.g. Z-Image).
- [ ] **CFG scale**: check which param name each side uses, verify same value. **For CFG-distilled/turbo variants, should be 0.0.**
- [ ] **Post-CFG processing**: is `true_cfg_scale` / `cfg_normalization` / `guidance_rescale` enabled on one side but not the other?
- [ ] **Scheduler mu / shift**: compare `cp3_mu` checkpoint
- [ ] **Timestep schedule**: compare `cp3_timesteps` checkpoint. **If they differ, check `sigma_min` overrides in Diffusers pipeline source code.**
- [ ] **Attention backend**: use `--attention-backend torch_sdpa` to eliminate this known variable during alignment

---

## Phase 2: Coarse-Grained Bisect

Add `torch.save()` at **stage boundaries** to find the first divergent checkpoint.

### 2.1 Checkpoint Map

| Checkpoint | Stage | What to Save |
|---|---|---|
| `cp0_prompt_embeds` | Text encoding | Prompt embeddings |
| `cp0_neg_prompt_embeds` | Text encoding | Negative prompt embeddings |
| `cp2_latents_packed` | Latent preparation | Initial noise (after packing) |
| `cp3_mu` | Timestep preparation | mu scalar (scheduler shift) |
| `cp3_timesteps` | Timestep preparation | Full timestep schedule |
| `cp4_step{i}_cond_out` | Denoising loop | DiT conditional output |
| `cp4_step{i}_uncond_out` | Denoising loop | DiT unconditional output |
| `cp4_step{i}_combined` | Denoising loop | After CFG combine + postprocess |
| `cp4_step{i}_latents_out` | Denoising loop | After scheduler.step |
| `cp5_final_latents` | After denoising | Final latents |
| `cp6_vae_input` | Before VAE | Unpacked + denormalized latents |

Save at a few steps to track error accumulation. Choose steps spread across the schedule (e.g. `{0, 1, N//5, N//2, N-1}` for an N-step schedule).

### 2.2 Implementation Pattern

Use environment variables for zero-overhead control. **Note:** `SGLANG_CKPT_DIR` and `DIFFUSERS_CKPT_DIR` are **not built-in** — you need to manually add `torch.save()` calls to the relevant stage source files (e.g. `text_encoding.py`, `denoising.py`).

```python
# SGLang side — manually add to relevant stage files
import os
_ckpt_dir = os.environ.get("SGLANG_CKPT_DIR", "")
if _ckpt_dir:
    os.makedirs(_ckpt_dir, exist_ok=True)
    torch.save(tensor.detach().cpu(), os.path.join(_ckpt_dir, "cp0_prompt_embeds.pt"))
```

```python
# Diffusers side — modify pipeline source in .venv/.../diffusers/pipelines/<model>/
_ckpt_dir = os.environ.get("DIFFUSERS_CKPT_DIR", "")
if _ckpt_dir:
    torch.save(tensor.detach().cpu(), os.path.join(_ckpt_dir, "cp0_prompt_embeds.pt"))
```

Run both sides:
```bash
# Diffusers
python diffusers_baseline.py

# SGLang
SGLANG_CKPT_DIR=/tmp/sglang_ckpts sglang generate --model-path ... --seed 42 ...
```

Compare:
```bash
python scripts/compare_checkpoints.py --diffusers-dir /tmp/diffusers_ckpts --sglang-dir /tmp/sglang_ckpts
# Useful flags:
#   --auto-reshape          auto-fix common shape mismatches (squeeze/permute)
#   --sort-by cosine        sort by cosine similarity (worst first)
#   --filter-step 0         only compare step 0 checkpoints
```

### 2.3 Precision Metrics

| Metric | What It Measures | When to Use |
|---|---|---|
| `max_abs_diff` | Worst-case single-element error | Detect outliers, overflow, bugs |
| `mean_abs_diff` | Average error | Overall precision |
| `cosine_similarity` | Directional agreement (0~1) | **Best single metric** for "are these the same?" |
| `allclose(atol, rtol)` | Pass/fail threshold | CI gates (use `atol=1e-2, rtol=1e-2` for bf16) |

**Cosine similarity interpretation for bf16 diffusion:**

| cosine | Interpretation |
|---|---|
| 1.000 | Bit-exact |
| 0.999+ | Normal bf16 variance |
| 0.995+ | Expected with different attention backends |
| 0.990+ | Acceptable accumulated error |
| 0.970+ | Suspicious — likely parameter mismatch |
| < 0.95 | **Bug** — fundamentally different computation |

### 2.4 Reading the Results

The **first divergent checkpoint** tells you where to look:

| First Divergence | Likely Cause | Action |
|---|---|---|
| `cp0_prompt_embeds` | Text encoder implementation | Try `--attention-backend torch_sdpa` |
| `cp3_mu` or `cp3_timesteps` | Scheduler config mismatch | Compare scheduler parameters |
| `cp4_step0_cond_out` (small diff ~0.03-0.05) | DiT attention/kernel difference | Normal bf16 variance if inputs match |
| `cp4_step0_combined` (large diff >> cond_out) | **CFG postprocessing mismatch** | Check `postprocess_cfg_noise()`, `true_cfg_scale` |
| `cp4_step0_latents_out` (diff but combined OK) | Scheduler step implementation | Compare scheduler code |

**Important:** Check **all** checkpoints, not just the first ❌. You may have multiple independent issues.

### 2.5 Handling Shape Mismatches

SGLang and Diffusers may store tensors in different shapes for the same data:

| Example | Diffusers | SGLang | Resolution |
|---------|-----------|--------|------------|
| DiT output | `[B, C, H, W]` | `[C, B, H, W]` | `permute(1, 0, 2, 3)` |
| Latents (video) | `[B, C, H, W]` | `[B, C, 1, H, W]` | `squeeze(2)` |
| Prompt embeds | `list[Tensor]` | `list[Tensor]` | `torch.stack(list)` |

When comparing, normalize shapes first:
```python
def normalize_shape(a, b):
    if a.dim() == 5 and a.shape[2] == 1:  # [B, C, 1, H, W] → [B, C, H, W]
        a = a.squeeze(2)
    if b.dim() == 5 and b.shape[2] == 1:
        b = b.squeeze(2)
    if a.shape != b.shape and a.dim() == 4 and b.dim() == 4:
        if a.shape[0] == b.shape[1] and a.shape[1] == b.shape[0]:
            b = b.permute(1, 0, 2, 3)  # [C, B, H, W] → [B, C, H, W]
    return a, b
```

---

## Phase 3: Fine-Grained Bisect

Once you've located the divergent stage, bisect within it. **Note:** In practice, Phase 1 resolves most issues. Phase 3 is only needed for residual analysis.

### 3.1 DiT Block-Internal Checkpoints

If DiT output differs but inputs match, add checkpoints inside **Block 0 only** at key points: after modulation, after LayerNorm, after QKV projection, after attention, after MLP. Gate saves with a flag to avoid all blocks overwriting each other:

```python
for i, block in enumerate(self.blocks):
    block._is_block0 = (i == 0)
    hidden_states = block(...)

# In block forward:
if getattr(self, '_is_block0', False) and _ckpt_dir:
    torch.save(tensor.detach().cpu(), f"{_ckpt_dir}/blk0_<stage>.pt")
```

### 3.2 Isolation Techniques

**Injection:** Replace SGLang's output with Diffusers' known-good value at a specific point. If downstream diff disappears, the injected component was the cause.

**Drop-in replacement:** Swap an entire SGLang module with its Diffusers equivalent. If step 0 gives max_diff=0.0, the difference is within the DiT. If step N still diverges, the gap is in the pipeline code around the DiT.

### 3.3 Common DiT-Level Difference Sources

| Source | Typical Impact (step 0 cond) | How to Detect |
|---|---|---|
| **Attention backend** (FA vs SDPA) | ±0.01~0.02 | `--attention-backend torch_sdpa` |
| **Fused Triton kernel** | Block-internal ±0.5, pipeline-level ~0 | Block checkpoint at `img_modulated` |
| **Timestep format** (float vs int) | ±0.7 on value, accumulates | Compare `cp4_step{i}_timestep` dtype |
| **bf16 GEMM non-determinism** | ±0.01~0.03 per step | Irreducible baseline |

**Caution about absolute values:** Intermediate DiT values can be in the millions. A max_diff of 262K looks alarming but if values range [-12M, +12M], it's <3% relative error. **Always check cosine similarity alongside max_abs_diff.** Use `--sort-by cosine` in `compare_checkpoints.py` to quickly spot the worst mismatches.

---

## Phase 4: Quantify & Document

### 4.1 Single-Variable Experiments

Test configurations **one variable at a time**:

| Config | Change | What It Measures |
|---|---|---|
| A | Baseline SGLang | Reference |
| B | A + fix parameter X | Impact of X alone |
| C | A + fix parameter Y | Impact of Y alone |
| D | A + fix X + fix Y | Combined |

### 4.2 Record These Metrics

For each alignment run, document:

- [ ] **Test conditions**: model, prompt, steps, seed, resolution, dtype, attention backend
- [ ] **Per-step DiT output**: `max_abs_diff`, `cosine_similarity` for `cond_out` vs Diffusers at steps {0, N//2, N-1}
- [ ] **Final metrics**: `final_latents` cosine, `vae_input` cosine
- [ ] **Single-variable impact**: cosine improvement for each fix applied independently
- [ ] **Combined result**: final cosine with all fixes applied

---

## Lessons from Real Alignment Cases

### Case 1: Qwen-Image-2512 (cosine 0.970 → 0.997)

| Root Cause | Impact | How Found |
|---|---|---|
| `true_cfg_scale` default None vs 4.0 | 93% of gap | `cp4_step0_combined` max_diff=2.6 (>> cond_out 0.047) pointed to CFG postprocessing |
| FlashAttention vs SDPA | 3% of gap | `--attention-backend torch_sdpa` improved step0 cond 0.047→0.031 |
| Text encoder attention | 2% of gap | Eliminated by SDPA switch (same flag controls both) |
| Fused Triton kernel | 0% (actually helps) | Unfused mode made precision **worse** — fp32 internal computation is more precise |

**Key insight:** `combined` max_diff >> `cond_out` max_diff is the signature of a CFG postprocessing mismatch.

### Case 2: Z-Image (cosine 0.859 → 0.99976)

| Root Cause | Impact | How Found |
|---|---|---|
| `cfg_normalization` default True vs False | Major — 0.859→0.960 | Phase 1 parameter comparison found default mismatch |
| `sigma_min` not overridden to 0.0 | Major — timestep schedule shifted by up to 34.78 | `cp3_timesteps` comparison showed diverging schedule |
| `guidance_scale` should be 0.0 for Turbo variant (CFG-distilled) | Eliminated unnecessary negative branch | Diffusers example code used `guidance_scale=0.0` |
| FlashAttention vs SDPA | ~1% | Same as Qwen-Image |

**Key insights:**
- Diffusers pipelines may **override scheduler attributes in code** (not in config files). Must read pipeline source.
- CFG-distilled/turbo variants need `guidance_scale=0.0`. Check the Diffusers example code, not just `__call__` defaults.
- Two "parameter-level" fixes (cfg_normalization + sigma_min) resolved 99% of the gap — no code-level DiT debugging needed.

### Common Patterns Across Both Cases

1. **Phase 1 (parameter comparison) found the dominant root causes in both cases** — Phase 3 (block-internal bisect) was only needed for residual analysis
2. **CFG-related parameter mismatches are the #1 source** — different models use different CFG strategies, and the defaults often don't match
3. **Always use `--attention-backend torch_sdpa` during alignment** — it eliminates a known ~1-5% variable and makes text encoder bit-exact
4. **Compare `cp3_timesteps` early** — scheduler schedule differences accumulate rapidly

### ❌ Pitfalls Encountered

1. **Speculating from code reading** — suspected wrong causes before measuring. Always measure first.
2. **Warmup counter mis-alignment** — SGLang warmup triggered extra checkpoint saves. Disable warmup during debug.
3. **Block checkpoints overwritten by all blocks** — gate with `_is_block0` flag
4. **Unfused kernel made things worse** — fused Triton kernels can be more precise than diffusers' step-by-step bf16
5. **allclose misleading for scalars** — `allclose(rtol=1e-3)` passed for timestep diff=0.7. Use strict `abs(a-b) < threshold`
6. **Diffusers env var import order** — `os.environ["DIFFUSERS_CKPT_DIR"]` must be set **before** `from diffusers import ...` because module globals are evaluated at import time
7. **Shape mismatches masked real data** — SGLang and Diffusers may use different dim orders for the same tensor. Always normalize shapes before comparing.

---

## Quick Reference

### SGLang Alignment Flags

| Flag | What It Does |
|------|-------------|
| `--attention-backend torch_sdpa` | **Use first** — aligns attention with Diffusers, eliminates known ~1-5% variable |
| `--true-cfg-scale <val>` | Enable true CFG with norm rescaling |
| `--cfg-normalization <val>` | Set CFG norm capping factor (0=off, 1.0=clamp to cond norm) |
| `--guidance-scale <val>` | CFG strength. Set to `0.0` for CFG-distilled/turbo variants. |
| `--flow-shift <val>` | Override scheduler shift |
| `--embedded-cfg-scale <val>` | Set embedded guidance scale |
| `SGLANG_CKPT_DIR=<dir>` | Enable checkpoint saving (requires manual `torch.save` instrumentation, see §2.2) |
| `SGLANG_INJECT_EMBEDS_DIR=<dir>` | Inject Diffusers embeddings (requires manual instrumentation) |

### Playbook for a New Model

1. **Print Diffusers pipeline `__call__` defaults** and example code; cross-reference with SGLang SamplingParams + PipelineConfig
2. **Check if model is CFG-distilled/turbo** (example uses `guidance_scale=0.0`?) — if so, ensure SGLang also uses `guidance_scale=0.0`
3. **Read the Diffusers pipeline source** for scheduler attribute overrides (e.g. `scheduler.sigma_min = 0.0`) that aren't in config files
4. **Write Diffusers baseline** with explicit params + checkpoint saving. Set env var **before** importing diffusers.
5. **Run SGLang with `--attention-backend torch_sdpa`** to eliminate attention as a variable
6. **Compare stage boundaries**: `cp3_timesteps` (schedule), `cp3_mu`, `cp0_embeds`, `cp4_step0_*`
7. If `cp3_timesteps` differ → scheduler config / `sigma_min` / `flow_shift`
8. If `cp0_embeds` differ → attention backend (should be resolved by SDPA)
9. If `combined` >> `cond_out` → CFG postprocess (`true_cfg_scale`, `cfg_normalization`, `postprocess_cfg_noise()`)
10. If `cond_out` differs with identical inputs → block-internal bisect (Phase 3)
11. **Quantify each factor** with single-variable experiments
12. **Document** in precision report with per-step tables
