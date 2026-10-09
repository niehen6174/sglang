# Qwen-Image-2.1 (`qwen_image21`) — SGLang ComfyUI integrated mode

Branch `feat/comfyui-qwen-image21` (`/scratch/data/sgld_comfy/wt-qwen21`), based on `upstream/main` 6e7beace14, not pushed.
Hardware: 1× RTX 5090 32 GB (GPU 0), torch 2.14.1+cu130, ComfyUI master a4b5a045 (`ComfyUI-qwen21`).

## Commits

| hash | subject |
|---|---|
| 8f55f9e636 | [diffusion] fix: ComfyUI SGLD worker spawn and batched cond timesteps |
| 7f2a6f35a2 | [diffusion] feat: Qwen-Image 2.1 ComfyUI-mode worker (checkpoint spec + step stage) |
| 82476fc4e2 | [diffusion] feat: ComfyUI integrated mode for Qwen-Image 2.1 |
| 95c37e50d8 | [diffusion] feat: load ComfyUI INT8 ConvRot Qwen-Image 2.1 checkpoints |
| 20f9a351a8 | [diffusion] docs: ComfyUI Qwen-Image 2.1 integrated-mode workflows |
| 431605488d | [diffusion] perf: prefetch Qwen-Image 2.1 host prefix K/V across steps |

15 files changed, +1739 / −42, all under `python/sglang/multimodal_gen/`.

## Summary

- **Text-to-image works end to end.** Replacing `UNETLoader` with `SGLDUNETLoader` in the official template matches the official output within ComfyUI's own numeric variance, at the same warm speed.
- **Image edit works** with 1 and 2 reference images (`TextEncodeQwenImage21`, `image_edit` template graph).
- **CFG > 1 works**, including when ComfyUI batches cond and uncond into one `apply_model` call (B=2). Each batch row keeps its own prefix K/V cache.
- **The INT8 ConvRot DiT** (`qwen_image_2.1_int8_convrot.safetensors`, the official template's default) works at ComfyUI's numeric noise floor, about 5% slower per step than ComfyUI's native INT8.
- Single GPU only. Multi-GPU is untested.

## Files changed

- **`runtime/loader/comfyui_checkpoints/qwen_image21.py` (new):** single-file checkpoint spec.
  - Detects the architecture from the safetensors header, the same way ComfyUI does.
  - Splits the fused `img_mlp.gate_up` into `gate_layer` and `proj`.
  - For INT8, also splits the per-row `weight_scale` and copies the `comfy_quant` marker to both halves.
- **`runtime/loader/comfyui_checkpoints/{__init__,spec}.py`:** registers the spec. A new `ComfyUICheckpointSpec.quant_markers` hook makes the serialized Comfy INT8 ConvRot load path, previously MiniMax-H3 only, available to any model. H3 behaviour is unchanged.
- **`runtime/pipelines/qwen_image21.py`:** adds `pipeline_config_cls`, `sampling_params_cls` and `create_comfyui_stages`.
- **`runtime/pipelines_core/stages/model_specific_stages/qwen_image21_comfyui.py` (new):** `QwenImage21ComfyUIStepStage`, the ComfyUI layout builder, and the prefix K/V store (GPU dicts or a pinned-host ring).
- **`runtime/pipelines_core/comfyui_mode.py`:** adds `begin_comfyui_run()`, which drops state left over from earlier sampler runs.
- **`apps/ComfyUI_SGLDiffusion/executors/qwen_image21.py` (new):** `QwenImage21Adapter` and `QwenImage21Executor`.
- **`apps/ComfyUI_SGLDiffusion/{core/generator.py, executors/__init__.py, nodes.py}`:** registers the executor, adds `qwen_image21` to the model-type list, and includes the worker-spawn fix.
- **`apps/ComfyUI_SGLDiffusion/executors/base.py`:** `should_suppress_logs` now handles a timestep with one value per batch row.
- **`apps/ComfyUI_SGLDiffusion/workflows/qwen_image21_{t2i,edit}_sgld.json` (new), `README.md`.**
- **`test/unit/test_comfyui_qwen_image21.py` (new):** 27 tests.

## How ComfyUI's forward maps onto the SGLang DiT

**Adapter (runs in the ComfyUI process):**
- Sends `x` unchanged: `[B, 64, H/16, W/16]`, no patchify.
- Sends `timesteps = sigma × 1000` in fp32, so t rounds to bf16 exactly as in ComfyUI.
- Sends the context once per cond key, together with a `qwen21_cond` payload: the slots, the reference latents, the `QwenImage21Cache` device setting, and the last non-zero sampler sigma.
- Later steps send only the latents and the timestep.
- The cond key hashes the whole context and the reference latents. Positive and negative prompts share the same chat-template head, so a key built from shape, first value and last value would collide.

**Worker stage:**
- Rebuilds ComfyUI's `build_sequence` token order as the native DiT `layout`:
  - text runs split at the image slots, each causal;
  - each reference grid as one full-attention segment;
  - the target image last.
- Rotary positions follow ComfyUI exactly:
  - text tokens get (p, p, p);
  - the time axis advances by max(h, w) per image;
  - a reference grid whose parity differs from the target shifts by half a token;
  - frequencies are computed in fp64 and rounded to fp32.
- Reference rows go through `image_indices` / `img_in(condition_latents)`.
- The stage runs through `QwenImage21DenoisingStage._predict_noise`, so compile and BCG hooks still apply.

**Prefix K/V cache:**
- Stored per run under `{uuid}:{run}`, with one entry per (cond key, B, H, W) and one cache per batch row.
- Where the cache lives follows ComfyUI's `select_prefix_cache` rules:
  - `auto`: GPU if free memory is more than 4× the cache size, otherwise pinned host memory;
  - `gpu`: GPU if free memory is more than 2× the cache size, otherwise recompute;
  - `cpu`: pinned host memory;
  - `off`: recompute every step.
- The host ring prefetches the next layer on a side stream.
- A cond's K/V is freed after its last sampler sigma.
- The cache is always bf16: the `QwenImage21Cache` int8/int4 dtype setting is ignored.

**CFG:** stays in ComfyUI. The worker's own CFG is off. Batched (B=2) and separate cond calls are both handled.

## Verification

### 1. Unit tests

```
cd /scratch/data/sgld_comfy/wt-qwen21/python
source /scratch/data/sgld_comfy/results/qwen_image21/scripts/env_qwen21.sh
python -m pytest -q -p no:warnings sglang/multimodal_gen/test/unit/test_comfyui_qwen_image21.py \
  sglang/multimodal_gen/test/unit/test_comfyui_adapters.py sglang/multimodal_gen/test/unit/test_comfyui_session.py \
  sglang/multimodal_gen/test/unit/test_comfyui_h3.py sglang/multimodal_gen/test/unit/test_comfyui_generator.py \
  sglang/multimodal_gen/test/unit/test_comfyui_profile.py
```

Result: **91 passed, 1 skipped, 3 xfailed**. The main session re-ran this independently and got the same result.

The new tests cover:
- architecture detection;
- the gate_up split for bf16 and INT8;
- the layout against an independent reference;
- the layout against ComfyUI's real `build_sequence` on a tiny model (rotary tables bit-equal, prefix embeddings within rtol 1e-5);
- the adapter, including cond-key collisions;
- the stage: cache reuse, B=2, cache off, release, eviction;
- the host K/V ring.

### 2. DiT parity

`scripts/parity_dit.py`: ComfyUI's DiT and the SGLang executor (full IPC and worker path) get identical captured inputs at sigmas 1.0, 0.75, 0.4 and 0.1. The "floor" columns compare ComfyUI's fused-kernel path against its own eager path.

| case | sigma | cosine | rel L2 | max abs | floor cos | floor rel L2 |
|---|---|---|---|---|---|---|
| t2i, B=1, 64×64 | 1.00 | 0.999963 | 0.0065 | 0.219 | 0.999968 | 0.0057 |
| t2i, B=1, 64×64 | 0.10 | 0.999916 | 0.0111 | 0.094 | 0.999926 | 0.0103 |
| t2i, B=1, 63×80 | 1.00 | 0.999956 | 0.0072 | 0.203 | 0.999965 | 0.0059 |
| CFG, B=2, 64×64 | 1.00 | 0.999909 | 0.0118 | 1.215 | 0.999928 | 0.0101 |
| CFG, B=2, 64×64 | 0.10 | 0.999787 | 0.0196 | 0.902 | 0.999818 | 0.0179 |
| edit, 1 ref, 64×64 | 1.00 | 0.999924 | 0.0113 | 0.445 | 0.999946 | 0.0091 |
| edit, 1 ref, 63×63 | 1.00 | 0.999935 | 0.0100 | 0.267 | 0.999938 | 0.0096 |
| edit, 2 refs, B=2, 64×48 | 1.00 | 0.999666 | 0.0260 | 0.562 | 0.999810 | 0.0183 |

- Across all 24 rows: cosine ≥ 0.99966 and relative L2 ≤ 2.6%, which is 1.0–1.4× ComfyUI's own fused-vs-eager difference.
- INT8: cosine 0.99922–0.99996 and relative L2 0.6–3.9%, against a ComfyUI floor of 0.99914–0.99996 and 0.7–4.1%.
- With the cache on GPU, on pinned host memory, and off, outputs are bit-identical.

### 3. Per-step latency

Median of 20 cached steps, 1024²:

| DiT | case | ComfyUI native | SGLang integrated |
|---|---|---|---|
| bf16 | t2i | 324.8 ms | 323.5 ms |
| bf16 | edit, 1 ref | 373.2 ms | 371.7 ms (host cache 373.5; recompute 706.9) |
| INT8 | t2i | 148.0 ms | 156.3 ms (JIT) / 157.3 ms (comfy_kitchen) |
| INT8 | edit, 1 ref | 194.1 ms | 202.8 / 205.4 ms |

`enable_torch_compile` gave no gain.

### 4. End to end via the `/prompt` API

Setup:
- The official template graph without the prompt-enhancer branch.
- KSampler euler/simple, 25 steps, 1024².
- A fresh server per mode, with the same seeds and prompts in both modes.

| workflow | official | integrated |
|---|---|---|
| t2i cold | 12.39 s | 31.93 s |
| t2i warm (2 seeds) | 8.59 / 8.60 s | 8.61 / 8.68 s |
| t2i CFG 4 + negative | 17.12 / 16.88 s | 17.09 / 16.96 s |
| edit 1 ref, first / warm | 11.71 / 10.33 s | 13.03 / 10.32 s |
| edit 2 refs, first / warm | 14.25 / 11.85 s | 18.94 / 11.73 s |
| INT8 t2i cold / warm / warm | 16.21 / 4.23 / 4.19 s | 30.36 / 4.37 / 4.37 s |
| INT8 edit 1 ref, first / warm | 7.83 / 5.52 s | 7.37 / 5.82 s |

Why the slower runs are slower:
- **Cold start (+17–20 s):** spawning the worker and its imports take about 12 s, distributed init 2 s, and loading the weights 3 s.
- **First 2-ref edit (+4–5 s):** with the text encoder still on the GPU, only 7.4 GiB is free. That is below the `auto` 4× rule, so the 4 GiB prefix goes to pinned host memory, and pinning it is slow on first use.

Image PSNR in dB. The "official floor" compares official with the cache off against official with the cache on:

| workflow | integrated vs official | official floor |
|---|---|---|
| t2i bf16 | 25.8 / 30.6 / 23.2 | 26.0 / 26.8 |
| CFG 4 | 19.3 / 18.7 | 21.7 / 18.9 |
| edit 1 ref | 43.5 / 46.7 | – |
| edit 2 refs | 33.5 / 23.6 | 33.9 / 24.1 |
| INT8 t2i | 26.6 / 25.7 / 19.0 | – |
| INT8 edit | 39.2 / 43.9 | – |

For scale, two different seeds give about 7.7 dB. The images have the same composition and differ only in fine texture; the main session checked the 23.2 dB pair visually. The shipped workflow JSONs succeed when posted unchanged to `/prompt`.

## Fixes that affect all models

- **Worker spawn crash on current ComfyUI master.** The spawned worker re-imported ComfyUI's `main.py`, and with `comfy/` first on `sys.path`, ComfyUI's `utils.py` shadowed the `utils` package. The worker died with "No module named 'utils.install_util'". It is now spawned without re-importing `__main__` and without that path entry.
- **Crash when ComfyUI batches cond/uncond.** `should_suppress_logs` called `.item()` on a timestep with one value per batch row. Flux was affected too.

## Known limitations / untested

- **Multi-GPU** (TP/SP/CFG-parallel) is untested.
- **DiT patches are rejected** with a clear error. This covers Fun-Control, attention patches, `post_input` / `single_block` hooks, and DiT block replacements.
- **ComfyUI weight hooks** are not applied to the worker's weights; this was already the case for other models.
- **The SGLD LoRA loader** is untested for this model.
- **The `QwenImage21Cache` int8/int4 cache dtype** is ignored.
- **ComfyUI doesn't account for the worker's VRAM.** `SGLDModelPatcher.model_size()` returns 0 for `qwen_image21`, so the 8.9 GB text encoder stays on the GPU. This is fine on 32 GB but may not fit on cards with 24 GB or less.
- **INT8 is ~5% slower per step** than ComfyUI. ComfyUI runs gate/up as one GEMM with SwiGLU folded into the next quantizer; SGLang keeps them separate.
- **Cold start is ~17–20 s slower.**
- **Not tested:** the prompt enhancer, other resolutions end to end, more than 2 reference images, and samplers other than euler/simple.
- **ComfyUI needs `--disable-cuda-malloc`**, because `env.sh` sets `PYTORCH_CUDA_ALLOC_CONF=backend:native`.
- **Not on this branch:** `origin/fix/comfyui-memory` (patcher clone parent tracking). It is relevant, because `QwenImage21Cache` clones the patcher.

## Evidence (in this directory)

- DiT parity: `parity_dit*.json`, `parity_cache_modes.json`
- Latency: `bench_step_*.json`
- End-to-end timings: `session_*.jsonl`, `session_*_comfylog.txt`
- Image comparisons: `e2e_image_compare*.json`, `e2e_image_floor.json`
- `images/`, `inputs/`, `workflow_check.jsonl`, `scripts/`

Extra download: the INT8 DiT `qwen_image_2.1_int8_convrot.safetensors` (6.8 GB). No packages were installed.

Reproduce the DiT parity check:

```
source scripts/env_qwen21.sh
python scripts/parity_dit.py comfy --refs inputs/ref_portrait.png inputs/ref_example.png
python scripts/parity_dit.py sgld [--cache-device gpu|cpu|off]
```

---

# Addendum (round 2): plugin fixes ported from the LTX branch, and LoRA

This round ported the model-agnostic plugin fixes from `feat/comfyui-ltx2` and tested SGLD LoRA on Qwen-Image-2.1.

**This supersedes two "Known limitations" entries above.**
- **"The SGLD LoRA loader is untested"** no longer holds: LoRA now works and is measured below.
- **"The text encoder stays on the GPU"** has changed. ComfyUI models are now evicted before each SGLD sampler run (see R2.4).

## R2.1 New commits (on top of 431605488d)

| hash | subject | origin |
|---|---|---|
| 8d1ba8aa11 | [diffusion] fix: return the patched clone from SGLDLoraLoader | `cherry-pick -x 449f9df473`, clean |
| eb2f0a1be3 | [diffusion] fix: preserve parent links on SGLD model clones | `cherry-pick -x 420f695310`, clean |
| c6400bff72 | [diffusion] fix: bind complete LoRA state from the sampled MODEL | `cherry-pick -x 896d2d7372`, clean |
| bb163f4e62 | [diffusion] feat: requantizing LoRA merge for static FP8 linears | `cherry-pick -x a2a9f4e9bf`, clean |
| 3ebe0e56c3 | [diffusion] fix: do not log an error for single-file checkpoints without model_index.json | `cherry-pick -x 516efc6eb5`, clean |
| 4e99783e79 | [diffusion] fix: reject ComfyUI-native LoRA on SGLD models; evict ComfyUI VRAM before LoRA load | `cherry-pick -x 03ed84dd39`, generic part only (see below) |
| d5a6cd3f17 | [diffusion] fix: evict ComfyUI models before each SGLD sampler run | `cherry-pick -x f219805acc`, test moved (see below) |
| ada7619b33 | [diffusion] docs: Qwen-Image 2.1 LoRA in ComfyUI integrated mode | new: README notes and `workflows/qwen_image21_turbo_lora_sgld.json` |

About `a2a9f4e9bf`: it is generic LoRA infrastructure (`runtime/layers/lora/linear.py`, `pipelines_core/lora/pipeline.py`) and applied cleanly. It does not change the INT8 ConvRot path: those layers stay on dynamic LoRA.

**Dropped when resolving `03ed84dd39`** (all LTX-specific, or dependent on LTX-only code):
- `LTXAVExecutor.lora_merge_mode = "merge"` (`executors/ltx_av.py`; the file does not exist on this branch).
- `workflows/ltx2_3_dev_lora_t2v_sgld.json` and its README line.
- The `core/generator.py` refactor of `_free_comfy_vram_for_worker` onto `release_comfy_vram`. That helper was added by the LTX feature commit efdc55f7e7, so this branch never evicts before spawning the worker. The now-unused `release_comfy_vram` import in `generator.py` was dropped as well.
- Its two generic tests (native LoRA rejection; `state_dict` without recursion) moved from `test_comfyui_ltx_av.py` to `test_comfyui_lora_node.py`.

**`f219805acc`**: its test also moved to `test_comfyui_lora_node.py`; the code is unchanged.

**Identical with `feat/comfyui-ltx2`:**
- `git diff HEAD feat/comfyui-ltx2` is empty for `executors/base.py`, `core/model_patcher.py`, `registry.py`, `runtime/layers/lora/linear.py`, `runtime/pipelines_core/lora/pipeline.py` and `test/unit/test_lora_pipeline.py`.
- `nodes.py` and `generator.py` differ only in model registration (`qwen_image21` vs `ltxav`), the LTX branch's `extra_server_args` option, and the dropped `_free_comfy_vram_for_worker` helper.

Skipped as instructed: `bb65ed64cf` and `276b7e1f1c` (cherry-picks of this branch's own commits).

I briefly committed a Qwen opt-out of the per-run eviction, then dropped it from the branch before finishing (see R2.4). The branch history does not contain it.

## R2.2 Tests

```
cd wt-qwen21/python; source .../scripts/env_qwen21.sh
python -m pytest -q -p no:warnings sglang/multimodal_gen/test/unit/{test_comfyui_qwen_image21,test_comfyui_lora_node,test_lora_pipeline,test_comfyui_adapters,test_comfyui_session,test_comfyui_h3,test_comfyui_generator,test_comfyui_profile}.py
-> 115 passed, 1 skipped, 3 xfailed
```

That is 91 before this round, plus `test_comfyui_lora_node.py` (4 tests) and `test_lora_pipeline.py` (the FP8 requant-merge cases).

**Smoke after the port** (integrated mode, final code):
- t2i seeds 1000–1002 and the 2-reference edit seeds 3000/3001 are **bit-identical** to the round-1 integrated session images.
- The three shipped workflow JSONs succeed when posted unchanged (`workflow_check.jsonl`).
- `qwen_image21_turbo_lora_sgld.json` reproduces `images/lora/sgld_turbo6_lora1_s7.png` bit for bit.

## R2.3 LoRA on Qwen-Image-2.1

**Which LoRA.** `Viggle/Qwen-Image-2.1-viggle-turbo` (about 345k downloads, ComfyUI workflows shipped). I used `Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors`, the file their ComfyUI workflows use.
- It is a 6-step, no-CFG distillation LoRA in diffusers format.
- It touches 227 layers, including `modulation.1` and the timestep embedder.
- Downloaded to `models/Viggle--Qwen-Image-2.1-viggle-turbo/`; symlinked into `ComfyUI-qwen21/models/loras/`.

**Graph** (`scripts/run_lora.py`; sessions in `scripts/run_lora_session.sh`; fresh server per session; 1024×1024; same prompt and seeds in both modes):
- TextEncodeQwenImage21, then BasicGuider (no CFG), KSamplerSelect(euler), ManualSigmas, SamplerCustomAdvanced.
- The sigmas are `1.0, 0.967754, 0.933358, 0.857192, 0.666756, 0.400096, 0`. This is the repo's ViggleTurboSigmas schedule, computed for 1024²; built-in nodes only, no custom node installed.
- Official: `UNETLoader → LoraLoaderModelOnly`. Integrated: `SGLDUNETLoader → SGLDLoraLoader`.
- `QwenImage21Cache(auto)` in both.

**The LoRA is really applied** (`lora_image_metrics.json`; "sharpness" is the Laplacian variance of the 1024² image).

| bf16 DiT, 6 steps | official | integrated |
|---|---|---|
| with LoRA, sharpness (seeds 7 / 8 / 9) | 931 / 854 / 764 | 932 / 886 / 768 |
| without LoRA, same 6 steps (seed 7) | 466 (blurry, grid artifacts) | 458 (same look) |
| strength 0.5 (seed 7) | 640 | 643 |
| base model, 25 steps, no LoRA (reference) | 720 | 712 |
| PSNR, LoRA on vs off | 19.9 dB | 19.9 dB |
| PSNR, strength 1.0 vs 0.5 | 23.4 dB | 23.6 dB |

**Official vs integrated** (PSNR):
- with LoRA: 35.2 / 30.2 / 42.5 dB;
- no LoRA: 40.8 dB;
- strength 0.5: 39.1 dB;
- 25-step base: 42.8 dB.

With the LoRA, the 6-step image is sharp. Without it, the same 6 steps give a soft image with a grid pattern; a crop and a montage were checked visually (`lora_montage_official_top_sgld_bottom.jpg`).

**Switching LoRA or strength between prompts takes effect.** The sequence was LoRA 1.0 → no LoRA → 0.5 → 1.0 with a new nickname. Each change produced the expected image, and the final 1.0 run is bit-identical to the first 1.0 run, in both modes.

**A native LoRA wired to an SGLD model fails loudly.** `SGLDUNETLoader → LoraLoaderModelOnly` makes the prompt fail at `LoraLoaderModelOnly` with:

> RuntimeError: 195 LoRA / weight patches (e.g. 'diffusion_model.modulation.1.weight') target the diffusion model served by SGLang, which ComfyUI cannot patch. Load this LoRA with 'SGLDiffusion LoRA Loader' (SGLDLoraLoader) instead of LoraLoader / LoraLoaderModelOnly.

**Timings and peak VRAM, bf16 DiT.** Times are the server's "Prompt executed". Peak VRAM is the maximum of `nvidia-smi` memory.used on GPU 0 (all processes), sampled every 0.1 s during the prompt.

| prompt | official | integrated |
|---|---|---|
| cold, LoRA 1.0 (model, TE, LoRA load; worker spawn) | 16.1 s, 23.2 GB | 41.0 s, 25.8 GB |
| warm, LoRA 1.0 (2 seeds) | 2.37 / 2.37 s, 23.8–24.1 GB | 2.46 / 2.60 s, 20.9 GB |
| LoRA removed | 3.29 s | 7.64 s (unmerge ≈ 5 s) |
| strength 1.0 → 0.5 (same LoRA) | 3.29 s | 2.54 s |
| back to 1.0, new nickname | 3.20 s | 7.01 s (reload + unmerge) |
| 25-step base after LoRA (no LoRA) | 9.39 s, 24.1 GB | 13.26 s, 20.4 GB (includes the unmerge) |

- Sampler rate with the LoRA: 3.29–3.56 it/s official, 3.00–3.05 it/s integrated.
- On the bf16 DiT the worker uses the default `auto` merge mode, which merges the LoRA into the bf16 weights, the same as ComfyUI's LoraLoaderModelOnly. Steady-state speed is therefore base-model speed.
- **Unmerging costs about 5 s.** `BaseLayerWithLoRA.unmerge_lora_weights` restores each layer through a `cpu_weight.clone()` and a host-to-device copy. This is generic LoRA runtime code and was left as is.

**INT8 ConvRot DiT + LoRA works.**
- The worker keeps the LoRA *dynamic* (unmerged) on the quantized layers: "Using dynamic LoRA for transformer because its quantized weights cannot be merged in place". It is not silently ignored.
- On vs off: 19.0 dB; sharpness 1086 / 978 / 850 with the LoRA vs 476 without.
- Official INT8 vs integrated INT8: 29.4 / 28.6 / 26.9 dB with the LoRA, 43.6 dB without.
- Integrated INT8 is a little *sharper* than ComfyUI (1086 vs 929). ComfyUI requant-merges the LoRA into INT8, while the dynamic path applies the full update; Viggle's README says unmerged is the faithful mode for this LoRA.

| INT8 DiT | official | integrated |
|---|---|---|
| cold, LoRA 1.0 | 7.36 s, 17.5 GB | 36.7 s, 16.0 GB |
| warm, LoRA 1.0 | 1.37 / 1.33 s, 17.2–17.5 GB | 1.76 / 1.77 s, 11.2–11.9 GB |
| LoRA removed / 0.5 / back to 1.0 | 1.74 / 1.96 / 9.80 s | 1.52 / 2.60 / 1.96 s |
| sampler rate with LoRA | 6.35–7.50 it/s (merged) | 4.50–4.54 it/s (dynamic; 6.16 it/s without LoRA) |

So INT8 + LoRA costs about 35% per step in integrated mode. Requant-merge for ConvRot INT8 (dequantize, add the delta, re-rotate and requantize) is not implemented.

**One INT8 quirk.** After a run at strength 0.5, later strength-1.0 runs are stable among themselves but differ from the first strength-1.0 run by 28.5 dB PSNR (`images/lora/int8_det*`).
- The image is still the strength-1.0 image: sharpness 1079 vs 1086, against 657 at 0.5.
- No-LoRA runs and nickname changes do not trigger it.
- The most likely cause is that `@torch.compile` on `BaseLayerWithLoRA._forward_with_delta` recompiles when `self.strength` changes and then treats strength as a dynamic float, so the fused delta kernel rounds slightly differently. This was not verified further.
- bf16 (merged) is bit-exact across the same sequence.

## R2.4 Re-check: evict ComfyUI models before each sampler run (`f219805acc`)

**The change.** Every ComfyUI model except the SGLD one (text encoder, VAE) is unloaded before the worker samples. Measured on Qwen (`smoke2_*`; integrated runs with and without eviction under otherwise identical final code; "Prompt executed"):

| | official | integrated, eviction (final) | integrated, no eviction (A/B only) |
|---|---|---|---|
| t2i cold | 13.0 s | 31.5 s | 34.9 s |
| t2i warm, same prompt | 8.60 / 8.65 s | 8.73 s (+ one 15.75 s outlier) | 8.58 / 8.61 s |
| t2i, new prompt each run (TE reloads) | 8.90 / 8.98 s | 9.41 / 9.46 s | 8.92 / 8.91 s |
| edit, 2 refs: first / warm | 15.66 / 11.82 s | 14.31 / 11.71 s | 17.16 / 19.34 s |

**Cost:** about +0.5 s per prompt change, because the text encoder reloads.

**Benefit:**
- The 2-reference edit's 4 GiB prefix K/V now fits on the GPU instead of being pinned to host memory, so the first edit is ~3 s faster.
- Warm peak VRAM drops by 3–4 GB (LoRA session: 20.9 GB vs 23.8–24.1 GB official). This addresses the round-1 limitation that the text encoder stays resident, which mattered for cards with 24 GB or less.

I kept it unchanged, identical to the LTX branch.

**Why the opt-out was dropped.** My first A/B run with eviction showed an 87 s cold prompt (VAE decode stalled for 52 s) and one 2.26 it/s run, which looked like an eviction regression, so I committed a Qwen opt-out. The repeat did not reproduce the 87 s (31.5 s).

**Intermittent slow runs, not root-caused.**
- About 1 in 7 integrated runs sample at 1.1–1.65 it/s instead of about 3.05 it/s, with or without eviction. A dedicated run of 7 prompt changes hit one; the final smoke hit one 16.5 s run.
- The LTX agent was running SGLang jobs on GPU 1 at the same time (load average about 7–8). The official runs I took showed no such outliers, but the sample is small.

## R2.5 Remaining limitations from this round

- **INT8 + LoRA is ~35% slower per step** (dynamic delta; no ConvRot requant merge).
- **Removing or replacing a merged bf16 LoRA costs about 5 s** (host-side unmerge).
- **INT8 dynamic LoRA rounding drifts after a strength change** (above).
- **Not tested:** SGLD LoRA with image edit, multiple chained LoRAs on Qwen (covered by the ported unit test only), and LoRAs other than the turbo LoRA.

## R2.6 Evidence

- `lora_session_{official,sgld}{,_int8}.jsonl` and `*_comfylog.txt`: timings, peak VRAM, sampler rates, worker LoRA log lines.
- `lora_image_metrics.json`, `lora_montage_official_top_sgld_bottom.jpg`, `images/lora/` (including `int8_det*`).
- `smoke2_{official,integrated_evict2,integrated_noevict,integrated_final,integrated}*`: smoke and eviction A/B. The `_final` run used the dropped opt-out, `_evict2` is the final code, and the unsuffixed `integrated` run is the first eviction run (the 87 s cold prompt).
- `workflow_check.jsonl`, `images/workflow_check/`.
- `scripts/run_lora.py`, `run_lora_session.sh`, `run_smoke_after_port.sh`.

---

# Multi-GPU (round 3)

**Setup.**
- Hardware: 2× RTX 5090 32 GB on different NUMA nodes (`nvidia-smi topo`: SYS) with no peer access, so all NCCL traffic goes through host memory.
- Parallelism is set through `SGLDOptions` (num_gpus=2 plus explicit degrees), with `CUDA_VISIBLE_DEVICES=0,1` for ComfyUI and its worker.
- Workload: the bf16 DiT unless stated, 1024² t2i, 25 steps, euler/simple.
- Branch head is 18ef2cc612. This round added two commits and nothing was pushed.

## M1 Result per configuration

| config | status | DiT parity vs 1-GPU worker (cached steps) | warm per step | warm t2i prompt | peak VRAM GPU0 / GPU1 (warm) |
|---|---|---|---|---|---|
| 1 GPU (reference) | works | – | 3.06 it/s (327 ms) | 8.64 / 8.58 s | 19.6–20.4 GB / – |
| **SP Ulysses2** (sp=2, ulysses=2, ring=1) | **works**, best | cos ≥ 0.99997, rel L2 0.23–0.25% (t2i, edit); B=2 CFG batch cos 0.99985 | **3.78 it/s** (1.24×) | **7.01 / 7.03 s** | 19.2–19.9 GB / 17.4 GB |
| SP auto (`sp_degree=2` only) | works | **bit-identical** to Ulysses2 | 3.75 it/s | 7.03 / 7.11 s | 19.1–19.3 GB / 17.4 GB |
| `num_gpus=2` alone | works after the fix below | same as SP auto | – | – | – |
| **TP2** (tp_size=2) | **works** | cos ≥ 0.99981, rel L2 0.4–1.8% (the same level as the ComfyUI fused-vs-eager floor) | 3.15 it/s (1.03×) | 8.42 / 8.32 s | **12.1 GB / 9.6 GB** |
| SP Ring2 (ulysses=1, ring=2) | **fails loudly at load** | – | – | – | – |
| CFG parallel | **not applicable**; rejected before the worker starts | (when forced before the fix: bit-identical to 1-GPU, because rank 1 recomputed the same call) | – | – | – |
| INT8 DiT, TP2 | works, slower | images match INT8 1-GPU (PSNR 18–30 dB, same composition) | 4.24 it/s vs 6.27 on 1 GPU | 6.27 / 6.31 s vs 4.38 / 4.40 s | 7.3–8.3 GB / 5.6 GB |
| INT8 DiT, Ulysses2 | works, slower | images match INT8 1-GPU (PSNR 21–30 dB) | 5.66 it/s vs 6.27 | 4.82 / 4.81 s vs 4.38 / 4.40 s | 10.9–11.5 GB / 9.0 GB |

How these were measured:
- Per step: ComfyUI's tqdm rate over the 25 steps. The DiT-only call time from `scripts/parity_dit.py` (cached t2i step, IPC included) agrees: 1 GPU 319 ms, Ulysses2 261–279 ms, TP2 311–325 ms.
- Prompt times are the server's "Prompt executed".
- **Cold start** (first prompt: TE encode, worker spawn, weight load) is 31.1–32.2 s in every bf16 configuration (1 GPU 31.4 s) and 27.3–29.4 s for INT8.
- Peak VRAM is the maximum of `nvidia-smi` memory.used per GPU during each prompt. GPU0 also holds ComfyUI's text encoder and VAE.

**Rank 1 really computes.**
- During each warm Ulysses2 prompt, GPU1 is above 50% utilisation in 59–80 of 71–109 trace samples. For TP2 the figure is 75–90 of 85–120, at 9.6–17.4 GB resident.
- SP output is all-gathered from both ranks' halves. An idle or wrong rank 1 would corrupt half the image, not reproduce 1-GPU parity.
- The worker only logs from rank 0, so the evidence is the per-GPU traces (`multigpu/smi_*.csv`) plus parity.

**End-to-end images vs 1 GPU** (PSNR, same seeds; `multigpu/e2e_psnr_vs_1gpu.json`, `side_by_side_*.jpg`):

| config | t2i (seeds 1000 / 1001 / 1002) | edit, 1 ref (seeds 2000 / 2001) |
|---|---|---|
| Ulysses2 | 28.4 / 32.2 / 23.7 dB | 45.2 / 48.6 dB |
| SP auto | identical to Ulysses2 | – |
| TP2 | 24.1 / 26.4 / 21.1 dB | 44.7 / 47.2 dB |

This is the same range as ComfyUI's own cache-on vs cache-off variation (21.7–33.9 dB, round 1).

**Image edit, 1 reference, on the best configuration (Ulysses2).**
- First run: 10.17 s; warm: 8.10 s; 3.4 it/s.
- 1 GPU: 12.15 s first, 10.23 s warm, 2.66 it/s.
- TP2: 11.30 / 9.43 s.

**Why the speedup is small.**
- Ulysses2 is 1.24× per step. Each step's all-to-all crosses NUMA through host memory.
- The text/reference prefix is replicated on both ranks; prefill is not sequence-parallel (the fork's `origin/fix/qwen-image21-prefix-sp` is not on this branch). For the B=2 / 8k-prefix edit, cached steps barely improve (627–674 ms vs 637–738 ms on 1 GPU).
- TP2 does 2 all-reduces of a 32 MB activation per block, 64 per step, so it barely beats 1 GPU. It halves the DiT memory per GPU (12.1 / 9.6 GB vs 19.6 GB), which is its real use here.
- With the INT8 DiT, compute is fast enough that communication dominates, so both TP2 and Ulysses2 are slower than 1 GPU.

## M2 Behaviour that matters per mode

**SP auto split.**
- `sp_degree=2` without explicit ulysses/ring resolves to `ulysses_degree=2, ring_degree=1, kv_gather_degree=2, sp_split_auto=true` (the effective `server_args` in the log). The explicit Ulysses2 run has `kv_gather_degree=1`.
- The outputs are bit-identical to explicit Ulysses2, and speed is the same.

**`num_gpus=2` with no degrees.** It failed while building ServerArgs with "...qwen_image_2.1_bf16.safetensors does not contain model_index.json". The CFG-parallel auto-enable looked up the default sampling params of the single-file DiT. **Fixed**: comfyui_mode never auto-enables CFG parallel. It now resolves like SP auto.

**Ring2.**
- The worker refuses at load: "Ring Attention requires a backend whose kernel exposes the softmax LSE for the per-hop merge; TORCH_SDPA does not declare support".
- On SM 12.0, FlashAttention is unavailable in this build ("FlashAttention is not supported on SM12.x in this build; falling back to Torch SDPA", also with `attention_backend=fa`), and sageattention is not installed. So Ring cannot run on this hardware and environment.
- **Fixed**: the loader node now raises "The SGLD worker exited during startup; its traceback is in the ComfyUI log above (...)" instead of a bare `EOFError`; the worker log has the LSE message.

**CFG parallel: not applicable.**
- ComfyUI runs CFG itself and calls the DiT once per cond. For Qwen-Image 2.1, cond and uncond are batched (B=2) only when their token counts match, because `c_crossattn` is a CONDRegular.
- SGLang's CFG parallel splits the worker's *own* CFG branches. The step stage has none (`do_classifier_free_guidance=False`), so with `enable_cfg_parallel` both ranks recompute the identical call: bit-identical to 1 GPU, same speed, twice the memory.
- **Now rejected** at `SGLDUNETLoader`, before the worker starts: "enable_cfg_parallel does not apply to Qwen-Image 2.1 in ComfyUI integrated mode: ComfyUI runs CFG itself ... Use sp_degree=2 (Ulysses) or tp_size=2 ...". `create_comfyui_stages` also rejects it on the worker side.
- Splitting a B=2 batch across ranks would only help in the rare equal-length case, so I did not force it.

**Prefix K/V and layout under SP / TP.**
- Every rank receives every request (rank 0 broadcasts it over NCCL), and the step stage runs on every rank. So each rank builds the same layout and the same per-cond run state from the same payload. Release at the last sigma is also deterministic.
- Under SP the DiT shards only the target tokens; the prefix and its K/V cache are replicated on each rank.
- Under TP each rank caches its own heads.
- **Bug found by inspection and fixed: cache placement could differ between ranks.**
  - The GPU / pinned-host / recompute choice came from each rank's own free memory, and ranks differ (GPU0 also holds ComfyUI).
  - Under TP the *uncached* prefix path runs extra all-reduces (the prefix `to_out` and MLP). Ranks that decided differently would issue mismatched collectives and **hang**.
  - The decision now uses the minimum free memory across the world group (one gloo all-reduce when a cond is created).
  - It did not trigger in my runs because the 1024² t2i/edit caches fit on both GPUs, but it is a real hang risk.
- **SP needs `H·W` divisible by sp_degree.**
  - Before: a 1008×1008 image (63×63 latent) produced a worker error and then an `AttributeError: 'NoneType' object has no attribute 'to'` in ComfyUI.
  - Now KSampler fails with "SGLD worker failed this DiT step: ... sequence parallelism splits the 63x63 latent (3969 tokens) across 2 ranks, which needs a token count divisible by 2; pick a width or height that is a multiple of 32 pixels, or run without sp_degree".
  - The generic part of the fix (a worker step error is raised with the worker's message) applies to every model.
- TP2 has no size restriction (63×63 works; parity rel L2 0.5%).

## M3 Commits this round

| hash | subject |
|---|---|
| 0de333f7f7 | [diffusion] fix: ComfyUI multi-GPU option and error handling |
| 18ef2cc612 | [diffusion] fix: Qwen-Image 2.1 ComfyUI integrated mode on multiple GPUs |

- **0de333f7f7** (generic):
  - comfyui_mode never auto-enables CFG parallel (`server_args.py`).
  - A worker step error is raised with its message instead of an `AttributeError`.
  - A worker that dies at startup raises a `RuntimeError` instead of `EOFError`.
  - New `SGLDiffusionExecutor.validate_sgld_options` hook.
- **18ef2cc612** (Qwen):
  - Rank-agreed prefix K/V placement.
  - SP token-count check with a clear message.
  - CFG parallel rejected (plugin and worker).
  - 6 regression tests in `test_comfyui_qwen_image21.py`. Each guards a bug seen in this round; the ServerArgs test needs 2 GPUs and was confirmed to fail without the fix.

Tests: `pytest test_comfyui_qwen_image21 test_comfyui_lora_node test_lora_pipeline test_comfyui_{adapters,session,h3,generator,profile}` → **121 passed, 1 skipped, 3 xfailed** with 2 GPUs visible.

## M4 Not done / limitations

- Ring attention needs an LSE-capable attention backend (FA or sage), and neither is usable on this SM 12.0 build.
- No SP prefill for the prefix, so large-reference edits gain little from SP.
- CFG parallel is not applicable.
- SP needs an even token count.
- INT8 gets slower with 2 GPUs on this host-routed interconnect.
- LoRA with multi-GPU was not tested.
- Each configuration was run once (cold + 2 warm).

## M5 Evidence (`multigpu/`)

- `parity_*.json`, `compare_*.json` / `compare_summary.txt`: DiT parity vs ComfyUI and vs the 1-GPU worker, plus call times. Cases cover t2i 64×64 and 63×80, a B=2 CFG batch, 1-ref edit 64×64 and 63×63, and 2-ref B=2 64×48.
- `e2e_*.jsonl`, `e2e_*_comfylog.txt`: per-prompt times and tqdm rates.
- `smi_*.csv`, `e2e_*_marks.txt`: per-GPU utilisation and memory traces.
- `e2e_psnr_vs_1gpu.json`, `side_by_side_*.jpg`; images in `images/multigpu/`.
- Scripts: `scripts/mg_parity.sh`, `mg_compare.py`, `mg_e2e.sh`.

---

# Slimmed branch (round 5)

User decision: smaller diffs, minimal changes to common components, no ComfyUI numeric-alignment code, CFG split parked.
- The full previous branch is saved as `archive/comfyui-qwen-image21-full` (56ef89ae39).
- CFG split is on `wip/comfyui-cfg-split`; its design is in `/scratch/data/sgld_comfy/results/cfg_split/DESIGN.md`.
- `feat/comfyui-qwen-image21` was rebuilt from `upstream/main` (6e7beace14) as 4 commits. It has not been pushed.

| hash | subject |
|---|---|
| 8599755b97 | [diffusion] fix: SGLD LoRA and model handling in the ComfyUI plugin (plugin-only port of the feat/comfyui-ltx2 LoRA fixes) |
| 006c9233cc | [diffusion] fix: ComfyUI plugin worker start and step errors |
| 51a488efe2 | [diffusion] feat: Qwen-Image 2.1 ComfyUI-mode worker |
| a73e8998a3 | [diffusion] feat: ComfyUI integrated mode for Qwen-Image 2.1 |

## Diff size (`git diff --shortstat upstream/main...`)

| | before (archive) | after |
|---|---|---|
| all files | 28 files, +3556 / -75 | **17 files, +1418 / -17** |
| excluding tests, workflow JSON, README | 20 files, +1601 / -71 | **11 files, +557 / -17** |
| common components (outside the plugin, tests, and Qwen's own spec / stage / pipeline hook) | 8 files, +319 / -39 | **1 file, +12 / -4** (plus the 1-line registration below) |

## Remaining common changes

- **`runtime/loader/comfyui_checkpoints/spec.py`, +12 / -4.**
  - A `ComfyUICheckpointSpec.quant_markers` hook, and the existing MiniMax-H3 serialized INT8 ConvRot load branch now also runs when a spec provides markers.
  - Why it cannot live in model code: the quantized load path (quant config, key filter, load plan) is inside `load_comfyui_transformer`. Without the hook, the INT8 DiT cannot be loaded.
  - The same file also gets the 1-line `qwen_image21` entry in `_discover_checkpoint_specs`.
- **`runtime/loader/comfyui_checkpoints/__init__.py`, +1.** Registers the spec module.
- **Allowed hook, not counted as common:** `runtime/pipelines/qwen_image21.py` (+25): config classes for single-file loading and `create_comfyui_stages`.

Everything else is in the plugin (`apps/ComfyUI_SGLDiffusion/**`) or new Qwen-specific files.

## What was dropped, and its measured effect

**Removed from common code:**
- **FP8 requantizing LoRA merge** (`layers/lora/linear.py`, `lora/pipeline.py`, `test_lora_pipeline.py`). Qwen does not use it: bf16 merges natively and INT8 uses dynamic LoRA, and the LoRA results below are unchanged.
- **`registry.py`** single-file log level (it was cosmetic).
- **`video_sparse_attn.py`** CPU-import fix. As a consequence, comfy plugin unit tests still need a visible GPU in this venv.
- **`server_args.py`** "comfyui_mode never auto-enables CFG parallel". This is now done in the plugin: the generator passes `cfg_parallel_degree=1` unless `enable_cfg_parallel` is set.
- **`comfyui_mode.py` `begin_comfyui_run`.** The stage now calls the existing `bind_comfyui_session`, which already evicts older runs.

**CFG split** was removed and parked (see DESIGN.md). `enable_cfg_parallel` is rejected with a clear message, both in the plugin (before the worker starts) and in `create_comfyui_stages`.

**Numeric alignment.** The layout now reuses the native `build_layout`. ComfyUI's `image_slots` become one zero placeholder token per reference, which the DiT overwrites with `img_in(condition_latents)`. This removed the fp64 RoPE tables and the reimplementation of ComfyUI's sequence building.
- Parity vs ComfyUI is unchanged where the reference and target grids have the same parity: t2i cos ≥ 0.99991, CFG B=2 cos ≥ 0.99978, 1-ref edit 64×64 cos ≥ 0.99984, 2-ref B=2 edit cos ≥ 0.99979 (`slim/parity_dit_slim.json`, floor columns as before).
- **The half-token reference shift for mismatched grid parity was dropped.** With an odd target (63×63 latent, 1008² image) and an even reference, DiT parity falls to cos 0.9938 / rel L2 11% at σ=1 (0.998 later), and the end-to-end image has PSNR 14.2 dB vs official. The edit is still correct: same subject, pose, style and edit; the subject is framed slightly larger. See `slim/side_by_side_official_integrated.jpg` (top row).
  - The official edit template never hits this case: its latent follows the first reference's size, which is a multiple of 32.

**Prefix K/V cache.** Kept as "cache on GPU for the sampler run if free memory > 2× its size (agreed across TP/SP ranks), otherwise recompute every step". Dropped:
- **the pinned-host ring and prefetch.** Effect: none on 32 GB. The ported per-run eviction of ComfyUI's text encoder leaves enough room, so the 2-ref edit now caches on the GPU at 2.37 it/s (host ring: 2.30–2.35). Its first run is faster too: 14.52 s vs 18.9–19.6 s, since there is no 4 GiB pinning.
  - In a synthetic B=2 2-ref DiT call (8.6 GB of K/V) the cache does not fit and steps recompute: 2.02 s vs 0.64 s per step. The e2e workflows do not batch that case (cond and uncond have different lengths).
- **mirroring `QwenImage21Cache`** placement modes and int8/int4 dtype. The node is now ignored.
- **per-cond release at the last sigma.** The cache is now held until the next sampler run. Peak GPU0 memory in the slim session was 31.7 GB vs 30.9 GB official (end of the edit runs, during VAE decode). There was no OOM.
- **the content-hash cond key.** The base key (shape, first value, last value) is enough within one run: the last hidden state differs between different prompts.

**Other removals:**
- checkpoint-header architecture detection (the Comfy-Org files are the default architecture);
- the `model.diffusion_model.` prefix handling;
- the third workflow (turbo LoRA; its settings are now in the README).

## Verification on GPU 0 (slimmed branch)

**Unit tests:**
```
pytest test_comfyui_qwen_image21 test_comfyui_lora_node test_comfyui_adapters test_comfyui_session test_comfyui_h3 test_comfyui_generator test_comfyui_profile test_lora_pipeline
→ 99 passed, 1 skipped, 3 xfailed
```
The new Qwen file has 11 tests, including the layout against ComfyUI's real `build_sequence`.

**DiT parity:** `slim/parity_dit_slim.json`; the numbers are given above.

**End to end** (fresh server per mode; `slim/session_*`; "Prompt executed", 1024², bf16):

| workflow | official | integrated | PSNR integrated vs official |
|---|---|---|---|
| t2i cold / warm / warm | 13.68 / 8.62 / 8.64 s | 31.74 / 8.54 / 8.68 s | 27.3 / 29.4 / 23.3 dB |
| CFG 4 + negative (2 runs) | 17.20 / 16.97 s | 17.75 / 16.80 s | 17.9 / 19.0 dB |
| edit 1 ref, first / warm | 11.77 / 10.32 s | 12.11 / 10.20 s | 37.0 / 46.6 dB |
| edit 2 refs, first / warm | 14.30 / 11.81 s | 14.52 / 11.72 s | 28.5 / 23.9 dB |
| edit 1 ref, odd 1008² target | 11.34 s | 12.09 s | 14.2 dB (half-shift dropped, see above) |
| sampler rate t2i / edit1 / edit2 | 3.16 / 2.59 / 2.26 it/s | 3.05 / 2.67 / 2.37 it/s | |

All pairs show the same composition. These PSNRs are in the same range as round 1 and as ComfyUI's own cache-on vs cache-off variation.

**Turbo LoRA** (Viggle v0.3 r128, 6 steps; `slim/lora_metrics.json`, `lora_session_sgld_slim*`):

| | bf16 | INT8 |
|---|---|---|
| sharpness with LoRA, seeds 7 / 8 / 9 (official) | 936 / 883 / 760 (931 / 854 / 764) | 1084 / 969 / 851 (929 / 854 / 764) |
| sharpness without LoRA | 464 | 473 |
| PSNR vs official with LoRA | 31.2–38.5 dB | 26.8–28.6 dB |
| LoRA on vs off | 20.0 dB | 18.9 dB |
| 1.0 → 0.5 → 1.0 switching | bit-identical return | 30.8 dB, the known dynamic-LoRA drift |
| LoraLoaderModelOnly on an SGLD model | fails loudly | fails loudly |
| cold / warm prompt | 41.3 / 2.42–2.48 s | 34.3 / 1.76–1.77 s |
| peak VRAM | 20.6–21.3 GB warm | 11.5–12.9 GB warm |

INT8 runs at 4.5 it/s with the LoRA (dynamic) and 6.4 it/s without.

**2-GPU smoke:** pending main's go-ahead for both GPUs.

---

# Rebased onto the common plugin-fixes branch (round 6)

**Common branch:** `feat/comfyui-plugin-fixes`, worktree `wt-plugin-fixes`, at upstream/main 6e7beace14.

| hash | author | subject |
|---|---|---|
| df99039688 | OnePunchMonk | return the patched clone from SGLDLoraLoader |
| 134fdc82b6 | niehen6174 | preserve parent links on SGLD model clones |
| 300f68d2d9 | niehen6174 | bind complete LoRA state from the sampled MODEL |
| 6e2e7e302d | Wenhao Zhang | reject native LoRA on SGLD models; evict ComfyUI models for the worker |
| 23c21756d9 | Wenhao Zhang | ComfyUI plugin worker start, options and step errors |
| 27d0e4c93e | Wenhao Zhang | quant-marker and key-filter hooks for ComfyUI checkpoint specs |

- The first three are `cherry-pick -x` from feat/comfyui-ltx2 with their original authors.
- Diff vs upstream: 9 files, +554 / -20.
- The only common component is `spec.py`, at +22 / -7.

**Qwen branch:** `feat/comfyui-qwen-image21` = common branch + 2 commits:

| hash | subject |
|---|---|
| 53e2257fb8 | worker |
| 79cd2f65b8 | plugin integration |

- Diff vs the common branch: 13 files, +1029 / -1. These are Qwen files plus registration lines in generator.py, executors/`__init__`.py, the nodes.py model list, comfyui_checkpoints/`__init__`.py, the spec.py discover list and the README.

**Verification on GPU 0:**
- Unit tests: 111 passed, 1 skipped, 3 xfailed. This covers the Qwen, spec, lora_node, adapters, session, H3, generator, profile, lora_pipeline and zimage config tests.
- Smoke, compared with the slim-branch images: t2i s1000 / s1001, edit 1-ref, edit 2-ref, and turbo LoRA bf16 at 1.0, off and 0.5 are all **bit-identical**.
- Native LoRA on an SGLD model fails loudly.
- t2i timing: cold 35.0 s (slim: 31.7 s; the worker-start eviction now also runs), warm 8.54 s.

---

# 2-GPU smoke on rebased branch (round 7)

**Setup:** head 77133e077a (common a810e8dca1 + Qwen a7ac9c1593, 77133e077a); bf16 DiT, 1024² t2i, 25 steps; ComfyUI `/prompt`, fresh server per config.
- Times are "Prompt executed".
- Peak VRAM and GPU busy samples (>50% util) come from per-prompt nvidia-smi traces (`multigpu/smi_e2e_rb_*.csv`, `e2e_rb_*`).

| config | effective server args | warm per step | cold / warm / warm | peak VRAM GPU0 / GPU1 (warm) | GPU1 busy (warm) |
|---|---|---|---|---|---|
| Ulysses2 (sp=2, ulysses=2, ring=1) | sp 2, ulysses 2, kv_gather 1, cfg 1 | 3.79–3.81 it/s | 32.01 / 6.96 / 6.99 s | 20.2 / 17.4 GB | 65/70, 75/77 |
| TP2 (tp_size=2) | tp 2, sp 1, cfg 1 | 3.15–3.19 it/s | 32.88 / 8.25 / 8.36 s | 11.5–12.4 / 9.6 GB | 76/84, 84/90 |
| `num_gpus=2` only (auto) | sp 2, ulysses 2, kv_gather 2, cfg_parallel_degree 1 | 3.77–3.79 it/s | 30.80 / 7.00 / 7.03 s | 19.9–20.1 / 17.4 GB | 65/71, 75/77 |
| Ulysses2, edit with 1 reference | | 3.43 it/s | 10.04 / 8.08 s | 23.2–27.7 / 19.9 GB | 70/83, 79/108 |

1 GPU for reference (slim branch): 3.05 it/s, 8.54 s warm.

**Checks:**
- **`num_gpus=2` alone** now starts and runs. The common-branch `cfg_parallel_degree=1` default prevents the CFG-parallel auto-enable that crashed before.
- **`enable_cfg_parallel`** is rejected at `SGLDUNETLoader` with `ValueError: enable_cfg_parallel does not apply to Qwen-Image 2.1 in ComfyUI integrated mode: ComfyUI runs CFG itself, so every CFG rank would recompute the same DiT call. Use sp_degree=2 or tp_size=2.`

**Images vs round 3 (same seeds): not bit-identical, as expected.**
- The slimmed branch uses the native `build_layout` with float32 RoPE instead of round 3's fp64 ComfyUI tables. The 1-GPU slim images already differ from the round-3 1-GPU images by 20.7–29.5 dB.
- Rebased vs round-3 multi-GPU:
  - Ulysses2: 21.9–30.3 dB;
  - TP2: 21.2–29.7 dB;
  - auto: identical numbers to Ulysses2, so auto output equals explicit Ulysses2;
  - edit: 39.8 / 46.9 dB.
- Against the 1-GPU slim images on the same code: Ulysses2 27.6–28.2 dB, TP2 21.1–29.9 dB, edit 43.5 dB.
- All have the same composition (`multigpu/rb_side_by_side_s1002.jpg` shows the lowest-PSNR seed).

All processes were stopped afterwards; both GPUs read 0 MiB.
