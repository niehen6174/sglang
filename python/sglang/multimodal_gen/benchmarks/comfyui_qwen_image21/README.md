# Qwen-Image-2.1 in SGLang ComfyUI integrated mode — final report

Branch `feat/comfyui-qwen-image21` = `feat/comfyui-plugin-fixes` + 2 Qwen commits, based on upstream/main 6e7beace14. Hardware: 2× RTX 5090 32 GB (SM120, different NUMA nodes, no P2P), torch 2.14.1+cu130, ComfyUI master a4b5a045. Weights: `Comfy-Org/Qwen-Image-2.1` (bf16 and INT8 ConvRot DiT, qwen3vl_8b int8 text encoder, VAE).

`HISTORY.md` in this directory is the round-by-round work log. Its earlier sections refer to commits and designs that were later replaced; this file reflects the final branch.

## Usage

Take the official Qwen-Image 2.1 t2i or edit template and replace `UNETLoader` with `SGLDUNETLoader`. Optionally add `SGLDOptions` for multi-GPU or other runtime options, and use `SGLDLoraLoader` instead of `LoraLoaderModelOnly` for LoRAs. Example graphs are `workflows/qwen_image21_t2i_sgld.json` and `qwen_image21_edit_sgld.json`.

## Branch layout

| branch | content |
|---|---|
| `feat/comfyui-plugin-fixes` | model-agnostic plugin fixes; the only common-component change is `spec.py`, +22 / -7 |
| `feat/comfyui-qwen-image21` | + `[diffusion] feat: Qwen-Image 2.1 ComfyUI-mode worker`, + `[diffusion] feat: ComfyUI integrated mode for Qwen-Image 2.1` |

The Qwen commits add 13 files, +950 / -1. Without tests, workflow JSON and README that is +300 lines:
- a new checkpoint spec (`comfyui_checkpoints/qwen_image21.py`, 74 lines);
- a condition stage (`QwenImage21ComfyUIConditionStage`, +95 lines in `model_specific_stages/qwen_image21.py`);
- a `create_comfyui_stages` hook in `pipelines/qwen_image21.py`;
- the executor/adapter (95 lines);
- registration lines.

The worker commit is +199 lines. There are no changes to the native DiT, LoRA, quantization or scheduler code, nor to the generic ComfyUI stages, `comfyui_mode.py`, `denoising.py` or the pipeline configs.

## Design

- **ComfyUI side.** ComfyUI keeps the text encoder, VAE and sampler.
  - Each `apply_model` call becomes one SGLang request: the latent `[B, 64, H/16, W/16]`, `t = sigma × 1000`, the context, `image_slots` and the reference latents.
  - Conditioning is sent once per cond; later steps send only latent + timestep.
- **Worker side.** The generic ComfyUI stage sequence, with one small Qwen stage:
  1. `ComfyUILatentPreparationStage` (generic): restores the session and binds the pass-through scheduler.
  2. `QwenImage21ComfyUIConditionStage`:
     - packs the latent to `[B, H*W, 64]`;
     - converts each ComfyUI image slot to one placeholder token for the native `build_layout`, which the DiT fills with `img_in(condition_latents)`;
     - sets `prompt_embeds` and `extra["qwen21_positive"]`, the same inputs native generation produces.
  3. Native `QwenImage21DenoisingStage`: the shared denoising loop.

  The adapter unpacks the `[B, H*W, 64]` prediction back to `x`'s shape.
- **Prefix K/V cache.** Built in the condition stage once per sampler run and per cond (session run state). It is kept on the GPU when free memory is more than twice its size (agreed across TP/SP ranks), otherwise recomputed.
- **CFG.** CFG stays in ComfyUI. `enable_cfg_parallel` is rejected with a clear error; see `cfg_split/DESIGN.md` on branch `wip/comfyui-cfg-split` for the parked CFG-split design.
- **INT8 ConvRot DiT.** Loaded through the common `quant_markers` spec hook.
- **No ComfyUI numeric alignment.** Native SGLang semantics are used throughout.

## Results (single GPU, 1024², bf16 unless noted)

DiT single step vs ComfyUI's own DiT, same inputs, cosine similarity:

| case | min cosine |
|---|---|
| t2i | 0.99991 |
| CFG batch of 2 | 0.99978 |
| edit, 1 ref | 0.99984 |
| edit, 2 refs | 0.99979 |

These outputs are bit-identical to the earlier custom step stage (24 of 24 calls), and the cached per-call time is the same: 324 ms t2i and 373 ms edit, within 0.2%.

End to end via the ComfyUI `/prompt` API (server "Prompt executed"):

| workflow | official | integrated | PSNR vs official |
|---|---|---|---|
| t2i, 25 steps, cold / warm | 13.7 / 8.62 s | 31.0 / 8.57–8.63 s | 23–29 dB |
| CFG 4 + negative | ~17.0 s | 16.8–17.6 s | 18–19 dB |
| edit, 1 ref, warm | 10.32 s | 10.21 s | 37–47 dB |
| edit, 2 refs, warm | 11.81 s | 11.71–11.80 s | 24–28 dB |
| INT8 t2i, warm | 4.2 s | 4.39–4.42 s | 20–27 dB |

All pairs have the same composition. The bf16 images, including LoRA, are bit-identical to those of the earlier step-stage design. For scale, ComfyUI against itself with the prefix cache on vs off differs by 19–34 dB, and two different seeds differ by ~8 dB.

**LoRA** (`Viggle/Qwen-Image-2.1-viggle-turbo`, 6 steps, `SGLDLoraLoader`):
- **Applied:** sharp with the LoRA (sharpness 936), soft without it (464); official has the same pattern.
- **vs official:** 31–39 dB (bf16), 27–29 dB (INT8).
- **Warm:** 2.4–2.5 s (bf16, official 2.37 s), 1.8 s (INT8).
- **Switching** strength or LoRA between prompts takes effect.
- **Native `LoraLoaderModelOnly`** on an SGLD model fails with a clear error.

## Multi-GPU (2× RTX 5090, no P2P)

| config | it/s | warm t2i | peak VRAM GPU0 / GPU1 |
|---|---|---|---|
| 1 GPU | 3.05 | 8.54 s | ~20 GB |
| Ulysses SP2, or `num_gpus=2` auto | 3.77–3.81 | 7.0 s | 20.2 / 17.4 GB |
| TP2 | 3.17 | 8.3 s | ~12 / 9.6 GB |

- **Ring SP:** not available on SM120 here; it needs an LSE-returning attention backend.
- **CFG parallel:** rejected.
- **Output vs 1 GPU:** same composition. Ulysses2 output is bit-identical to the earlier step-stage design.
- **SP token count:** SP requires an even target token count. Otherwise the DiT raises `target token count N must be divisible by SP degree 2`.

## Known limitations

- **Half-token reference shift.** The ComfyUI half-token reference shift is not reproduced. An odd target with an even reference (e.g. a 1008² edit) differs more from official (14 dB), although the edit is still correct. The official templates do not hit this case.
- **`QwenImage21Cache` node:** its placement and dtype settings are ignored.
- **DiT patches are rejected:** Fun-ControlNet, attention patches, block replacements.
- **INT8 + LoRA** uses dynamic LoRA: about 35–40% slower per step than ComfyUI's INT8 + LoRA (4.5 vs 7.1–7.5 it/s), and switching strength 0.5 → 1.0 drifts slightly (31 dB).
- **Cold start is 17–20 s slower than official:** worker spawn plus DiT load.
- **Not tested:** prompt enhancer, more than 2 references, samplers other than euler/simple, multiple chained LoRAs end to end.

## Reproducing

- `scripts/` holds the measurement scripts used here. They contain machine-specific paths under `/scratch/data/sgld_comfy`.
- Images, videos and raw JSON evidence stay on the test machine under `/scratch/data/sgld_comfy/results/qwen_image21/`.
