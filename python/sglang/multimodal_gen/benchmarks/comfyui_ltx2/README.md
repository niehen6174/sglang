# LTX-2.5 / LTX-2.3 in SGLang ComfyUI integrated mode — final report

Branch `feat/comfyui-ltx2` = `feat/comfyui-plugin-fixes` + 5 LTX commits, based on upstream/main 6e7beace14.

**Test setup:**
- Hardware: 2× RTX 5090 32 GB (SM120, different NUMA nodes, no P2P).
- Software: torch 2.14.1+cu130, ComfyUI master a4b5a045.

**Weights:**
- LTX-2.5, from `Lightricks/LTX-2.5`:
  - distilled DiT, INT8 ConvRot and bf16;
  - Gemma4 int8 text encoder;
  - video / audio VAE;
  - spatial upscaler.
- LTX-2.3:
  - `Lightricks/LTX-2.3-fp8` dev / distilled FP8 single-file checkpoints;
  - the distilled LoRA from `Comfy-Org/ltx-2.3`;
  - Gemma3 12B text encoder from `Comfy-Org/ltx-2`.

`HISTORY.md` in this directory is the round-by-round work log. Its earlier sections describe code that was later removed or rewritten; this file reflects the final branch.

## Usage

Take the official LTX-2.5 T2V/I2V or LTX-2.3 template and change two nodes:
- replace the DiT loader with `SGLDUNETLoader`;
- for LoRAs, use `SGLDLoraLoader` instead of `LoraLoaderModelOnly`.

Example graphs: `workflows/ltx2_5_t2v_sgld.json` and `ltx2_3_dev_lora_t2v_sgld.json`. Both stages of the two-stage templates run on the SGLang worker.

## Branch layout

Commits on top of the common branch:

| commit | subject |
|---|---|
| b9ed3d1a05 | [diffusion] feat: LTX-2 DiT linears pass their module path to the quant config |
| f73a7daffb | [diffusion] feat: requantizing LoRA merge for static FP8 linears |
| fee533c32e | [diffusion] feat: ComfyUI integrated-mode worker for LTX-2.5 / LTX-2.3 (ltxav) |
| 1bba3d31c8 | [diffusion] feat: LTX-2.5 / LTX-2.3 in the ComfyUI SGLDiffusion plugin |
| 2e7ea4a7d8 | [diffusion] fix: shard LTX video by frames under sequence parallelism in ComfyUI mode |

Size vs the common branch: 17 files, +2002 / -9. Excluding tests, workflow JSON and README, it is +812 / -5.

**Common-component changes, +142 / -4:**
- `models/dits/ltx_2.py` +41 / -3: linear `prefix` plumbing only.
  - ComfyUI INT8/FP8 per-layer quant markers are looked up by module path, so the quantized DiT cannot load without it.
  - It also makes `--quantization-ignored-layers` work for native LTX; before, it matched nothing.
- `layers/lora/linear.py` +70 and `pipelines_core/lora/pipeline.py` +30 / -1: requantizing LoRA merge for static FP8 linears.
  - It dequantizes, adds the delta, requantizes with stochastic rounding, and restores the exact bytes on unmerge.
  - It also parks merged factors on the host for layerwise-offloaded modules.
  - Kept because dynamic LoRA on the LTX-2.3 dev template is about 19% slower warm, and its second run runs out of memory on 32 GB.

**Everything else is model-specific:**
- the checkpoint spec (`comfyui_checkpoints/ltx_2.py`);
- the step stage (`ltx_2/comfyui_step.py`);
- the executor;
- the `create_comfyui_stages` hook;
- registration lines.

**Native pipeline:** plain `sglang generate` LTX-2.5 output is byte-identical to upstream (same mp4 md5, same torch peak memory).

## Design

- **ComfyUI side.** ComfyUI keeps the text encoder, VAE, upsampler and sampler.
- **Worker.** The worker loads the DiT from the ComfyUI single file. Text connectors load into SGLang's existing `LTX2ConnectorTransformer1d` as a separate module, and run on the worker to serve ComfyUI's `preprocess_text_embeds`.
- **Per step.** Each ComfyUI `apply_model` becomes one request carrying the video and audio latents, the timesteps (per token for I2V), the context and the masks.
- **Numerics.** There is no ComfyUI numeric alignment; the DiT runs SGLang's native semantics.
- **Sequence parallelism.** Each rank takes whole latent frames; audio is replicated.
- **CFG.** CFG stays in ComfyUI.

## Results (single GPU)

PSNR is official vs integrated with the same seed. Times are server "Prompt executed".

| template | official cold / warm | integrated cold / warm | video PSNR | peak VRAM |
|---|---|---|---|---|
| LTX-2.5 T2V INT8, 1280×704, 121 frames | 35 / 19.9–20.5 s | 50–61 / 20.2–20.7 s | 20–23 dB | 30.5 GB |
| LTX-2.5 I2V INT8 | 31 / 20.6 s | 54 / 22.4 s | 19 dB | 29.1 GB |
| LTX-2.3 dev FP8 + distilled LoRA 0.5 (template as shipped) | 36.5 / 17.4–17.7 s | 79–85 / 15.8–16.4 s | 16–20 dB | 31.7 GB |

**Similarity.** These distilled models are very sensitive to small numeric changes. ComfyUI against itself (split attention, or a different VRAM mode) also lands at 11–23 dB. Every pair keeps the same composition, and DiT single-step error vs ComfyUI is at the batch-noise floor:

| model | rel L2 (video) | batch-noise floor |
|---|---|---|
| LTX-2.5 INT8 | 0.136 | 0.135 |
| LTX-2.3 FP8 | 0.017 | – |

**LoRA.** On vs off moves the DiT output by 0.818 in SGLang vs 0.809 in ComfyUI. Without it, the distilled schedule produces a blurred smear on both paths; with it, both are sharp.

## Multi-GPU (2× RTX 5090, no P2P)

| config | LTX-2.5 T2V warm | peak VRAM GPU0 / GPU1 | note |
|---|---|---|---|
| 1 GPU | 20.1–20.6 s | 30.2 GB | |
| Ulysses SP2, or `num_gpus=2` auto | 20.0–20.4 s | 30.8 / 23.4 GB | stage 2 faster, stage 1 slower |
| TP2 | 25.1–25.5 s | 20.3 / 14.7 GB | halves DiT memory |
| LTX-2.3 dev+LoRA, TP2 | 19.8 s (1 GPU: 15.8 s) | 29.6 / 19.9 GB | fully resident, no layerwise offload |

- Output vs 1 GPU keeps the same composition, and both GPUs are busy.
- SP requires a latent frame count divisible by the degree, and fails clearly otherwise.
- Ring SP is unavailable on SM120: it needs an attention backend that returns the softmax LSE.
- CFG parallel is rejected because ComfyUI owns CFG.

## Known limitations

- **Unsupported, with a clear error:** keyframe guides (`LTXVAddGuide` strength/masks), reference audio (ID-LoRA), text attention masks, video-only `ltxv` checkpoints.
- **I2V speed:** warm is 22.4 s vs official 20.6 s, because per-token timesteps replaced the dropped per-frame modulation path.
- **TP2 + FP8 LoRA:** each rank requantizes its own shard, so merged weights are not byte-equal to the 1-GPU merge. The output is correct. A TP-consistent merge is on `archive/comfyui-ltx2-mgpu-wip`.
- **Pipeline detection:** the pipeline config is still detected from the file name, which must contain `ltx-2.x`.
- **Cold start:** 15–45 s slower than official, from worker spawn, DiT load and the LoRA merge.
- **Harmless log line:** "Could not read model config … model_index.json" is logged at ERROR on single-file loads; silencing it needs a common change.
- **LTX GGUF:** still does not load, on upstream either; the feed-forward names differ (`ff.proj_in` vs `ff.net.0.proj`).
- **Not tested end to end:** keyframe guides, context windows, multiple chained LoRAs, cache-dit, prompt enhancer on/off for every template.

## Reproducing

- `scripts/` holds the measurement scripts used here. They contain machine-specific paths under `/scratch/data/sgld_comfy`.
- Videos, frames and raw JSON evidence stay on the test machine under `/scratch/data/sgld_comfy/results/ltx25/`.
