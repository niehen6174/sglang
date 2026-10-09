# SGLang ComfyUI integrated mode: LTX-2.5 and LTX-2.3 (ComfyUI `ltxav`)

Branch `feat/comfyui-ltx2` (renamed from `feat/comfyui-ltx25`), worktree `/scratch/data/sgld_comfy/wt-ltx25`, based on `upstream/main` 6e7beace14. Nothing has been pushed.

Results live in `results/ltx25/`. I kept that name, but it covers both LTX-2.5 and LTX-2.3.

Setup:
- Hardware: one RTX 5090 32 GB (GPU 1).
- ComfyUI: a copy of master a4b5a045 at `ComfyUI-ltx25`, port 8189.
- SGLang worker ports: 30105 / 5655.

## Weights

**LTX-2.5:** official `Lightricks/LTX-2.5`.
- Files: INT8 ConvRot distilled DiT (plus the BF16 DiT, used only for parity), video and audio VAEs, x2 latent upscaler, and the official Gemma-4 12B text encoder (TE).
- The DiT, VAE and upscaler are byte-identical to the comfyicu mirror (sha256 checked by main).
- The mirror's TE is a different revision. Runs marked "mirror-TE" used it. Both the official and the integrated path shared that TE, so those runs are still a valid A/B comparison.
- All final LTX-2.5 end-to-end numbers (`offte_*`) use the official TE.

**LTX-2.3:** official `ltx-2.3-22b-distilled-fp8.safetensors`.
- This is an all-in-one file: FP8 DiT, VAE, audio VAE, vocoder and text projection.
- Also used: `ltx-2.3-spatial-upscaler-x2-1.1` and Comfy-Org's `gemma_3_12B_it_fp8_scaled`.

**Vocoder:** the LTX-2.5 template mentions `ltx-av-step-1751000_vocoder_24K.safetensors`, but only in its stale `extra.prompt` blob. The real graph uses `ltx-2.5-audio-vae-bf16`, so I did not download it.

## Commits (6e7beace14..HEAD)

| Hash | Subject |
|---|---|
| bb65ed64cf | [diffusion] fix: ComfyUI SGLD worker spawn and batched cond timesteps (**cherry-pick of 8f55f9e636** from feat/comfyui-qwen-image21) |
| 276b7e1f1c | [diffusion] feat: ComfyUICheckpointSpec.quant_markers hook for serialized Comfy quantization (**spec.py part of 95c37e50d8**; the Qwen-Image-2.1 files were dropped) |
| 9501ec4de9 | [diffusion] feat: LTX-2 DiT hooks for ComfyUI single-file checkpoints |
| 53edd3dcfa | [diffusion] feat: ComfyUI-mode worker for LTX-2.5 / LTX-2.3 (ltxav) |
| efdc55f7e7 | [diffusion] feat: ComfyUI integrated mode for LTX-2.5 / LTX-2.3 |

Files changed: 18 files, +2854/-65. All paths are under `python/sglang/multimodal_gen/`.

**DiT model**
- `runtime/models/dits/ltx_2.py`:
  - Linears now carry per-layer quantization prefixes, so Comfy marker names resolve.
  - Optional in-DiT text connectors (`embeddings_connectors_in_dit`).
  - ComfyUI-compatibility knobs: `attention_qk_norm_eps`, `v2a_video_norm_after_a2v`, the `keyframes_abs_pos_mask` kwarg, explicit `a2v_gate_timestep` / `v2a_gate_timestep`, and `dedup_timestep_embeddings`.
  - Per-frame modulation: an adaLN row of shape [B,F,D] is broadcast over each frame through a [B*F,P,D] view.
- `runtime/models/dits/ltx_2_embeddings_connector.py` (new): the text connector in ComfyUI's layout, built from the quant-aware LTX-2 attention and FF.
- `configs/models/dits/ltx_2.py`: new arch fields. All default to the existing behaviour.

**Checkpoint loading**
- `runtime/loader/comfyui_checkpoints/ltx_2.py` (new): the LTX-AV checkpoint spec.
  - Reads the arch from the safetensors `config` metadata, plus key probes.
  - Handles INT8 ConvRot and FP8 markers through `quant_markers`.
  - Filters keys for all-in-one files.
- `runtime/loader/comfyui_checkpoints/spec.py`: adds `extra_quant_formats` and a spec-level `checkpoint_key_filter` on top of the shared `quant_markers` hook.

**Worker pipeline**
- `runtime/pipelines_core/stages/model_specific_stages/ltx_2/comfyui_step.py` (new): a worker step that mirrors ComfyUI's `LTXAVModel.forward`, plus the connector operation.
- `runtime/pipelines/ltx_2_pipeline.py`: adds `create_comfyui_stages`, and skips the `model_index.json` lookup for single-file paths.

**ComfyUI plugin** (`apps/ComfyUI_SGLDiffusion/`)
- `executors/ltx_av.py` (new): adapter and executor.
- `core/generator.py`:
  - Registers the new executor.
  - Before spawning the worker, evicts models ComfyUI already holds. Otherwise, on 32 GB, the Gemma TE loaded earlier starves the worker.
- `nodes.py`:
  - Adds `ltxav` as a model type.
  - Adds an `extra_server_args` field that takes a JSON object of extra ServerArgs (ports, layerwise residency, and so on).
- Also changed: `executors/__init__.py` and `README.md`.
- New workflows: `ltx2_5_t2v_sgld.json`, `ltx2_5_i2v_sgld.json`, `ltx2_3_t2v_sgld.json`.

**Tests**
- `test/unit/test_comfyui_ltx_av.py` (new, 15 cases).

## Design

**ComfyUI side (`model_type=ltxav`).**
- Inputs: `model_base.LTXAV` unpacks the nested AV latent, so the executor receives `x=[video, audio]` and `timestep=(video, audio)`. With a denoise mask, the video timestep is per token.
- What the executor forwards:
  - `frame_rate`
  - `keyframe_idxs` and `denoise_mask` (for keyframe guides)
  - sanitized `guide_attention_entries`
  - `generated_keyframes`
- It returns `[video, audio]` velocities.
- CFG and DualCFG stay unchanged in ComfyUI, including batched cond/uncond calls.

**Text connectors.**
- ComfyUI runs the connectors (which live in the DiT checkpoint) from `preprocess_text_embeds`. The executor answers that call by sending a connector-only request (`ltxav_op=connectors`) to the worker.
- ComfyUI therefore gets the same `[B,1024,6144]` context as stock, and positive/negative prompts still batch together.
- ComfyUI never instantiates the DiT itself (`SGLDModelPatcher`, `disable_unet_model_creation`).

**Worker step.** It reproduces ComfyUI's behaviour for:
- token layout
- causal pixel-coordinate RoPE (start/end, scaled by fps) and audio coordinates in seconds
- timestep scaling, and prompt timesteps taken as the max
- the a2v / v2a gate inputs. ComfyUI uses the max of the audio timestep for the a2v gate and the max of the video timestep for v2a.
- the keyframe-guide grid mask and coordinate override
- the keyframe absolute-position marker

The conditioning is cached for the sampler run through the existing session cache.

**Places where ComfyUI differs from upstream ltx-core.** The worker follows ComfyUI in all three cases, so its output matches ComfyUI's:
1. Since ComfyUI 15989f87 ("Speedup LTX and Wan"), video-to-audio attention uses `rms_norm(vx)` taken *after* the a2v residual. ltx-core and native SGLang use the norm from before a2v.
2. ComfyUI adds `keyframes_abs_pos_embedding` to first-frame tokens even in T2V. Native SGLang allocates that weight but never uses it.
3. ComfyUI's Q/K RMSNorm eps is 1e-5, not 1e-6.

All three are off by default and only the ComfyUI checkpoint spec enables them. Point 1 is measurable: without it, the tiny fp32 equivalence test's error goes from under 2e-4 to about 6e-3. Point 3 is not measurable in fp32.

**Performance.**
- Frame-wise timesteps (I2V via `LTXVImgToVideoInplace`) are passed as one row per latent frame and broadcast through `[B*F,P,D]` views.
- Per-token timesteps are embedded once per unique value.
- Together these took the I2V stage-2 step from 1.90 to 1.67 s/it.

## Tests

```
source results/ltx25/scripts/envrc.sh; export CUDA_VISIBLE_DEVICES=0
cd wt-ltx25/python/sglang/multimodal_gen
python -m pytest test/unit/test_comfyui_adapters.py test/unit/test_comfyui_session.py test/unit/test_comfyui_h3.py \
  test/unit/test_comfyui_generator.py test/unit/test_comfyui_profile.py test/unit/test_comfyui_ltx_av.py \
  test/unit/test_ltx2_5_config.py test/unit/test_ltx2_bcg_coords.py test/unit/test_ltx2_modulate_mount.py \
  test/unit/test_ltx2_rmsnorm_modulate_fallback.py -q
# -> 133 passed, 1 skipped, 3 xfailed
```

What `test_comfyui_ltx_av.py` covers:
- **Tiny-model equivalence.** It builds a tiny ComfyUI `LTXAVModel`, copies its random weights into the SGLang DiT through the real name mapping, and compares in fp32 (rel L2 below 2e-4). Cases: connectors, t2v (batch 2), i2v (frame-wise mask), inpaint (spatial mask), and keyframe-guide steps.
- **Contracts:** layout, coordinates, gate and prompt timesteps, metadata, and key mapping.
- The ComfyUI-backed cases skip when `comfy` cannot be imported.

Other checks:
- `pre-commit run --files <changed>` passes.
- I installed pytest 9.1.1 (with pluggy and iniconfig) into `.venv`. No torch or CUDA packages changed.

## DiT parity (one forward pass on identical inputs)

Scripts:
- `ref_dit.py` runs ComfyUI's native model.
- `sgld_dit.py` runs the SGLD executor through the real worker.

Inputs:
- Real Gemma conditioning (from the mirror TE).
- Random latents: video 128x4x8x12, audio 8x26x16.
- Sigma 0.909.
- T2V uses batch 2 (positive and negative); I2V uses a mask that keeps frame 0 clean.

The "floor" column compares ComfyUI with itself: the same sample computed at batch 1 vs batch 2 (`ref_noise.py`). The floor is high because, at this sigma, the model amplifies tiny numeric differences.

| Model | Context rel L2 (pos / neg) | T2V video rel / cos | T2V audio rel / cos | I2V video / audio rel | Floor video / audio |
|---|---|---|---|---|---|
| LTX-2.5 INT8 ConvRot | 4.9e-3 / 6.6e-3 | 0.151 / 0.9886 | 0.029 / 0.99958 | 0.213 / 0.041 | 0.144 / 0.026 |
| LTX-2.5 BF16 (SGLD layerwise offload) | 3.3e-3 / 3.4e-3 | 0.135 / 0.9909 | 0.026 / 0.99966 | 0.153 / 0.021 | 0.097 / 0.016 |
| LTX-2.3 FP8 (all-in-one) | 4.6e-3 / 3.8e-3 | 0.286 / 0.9591 | 0.030 / 0.99956 | 0.279 / 0.087 | 0.226 / 0.034 |

Interpreting the INT8 numbers:
- Error grows through the stack: video rel error is 0.33% after block 0 (0.65% for audio), 1.1% after block 23 and 6.6% after block 47.
- The error sits mostly in two huge outlier channels.
- Switching the SGLang INT8 GEMM to comfy_kitchen gives the same numbers. So the difference is accumulated rounding, not semantics; the semantics are pinned by the fp32 tiny-model test.

Raw data: `logs/dit_parity*.json`.

## End-to-end (ComfyUI /prompt, official template structure)

Workflow setup:
- **Source:** `scripts/workflows.py` flattens the official subgraph templates, with the prompt enhancer off.
- **LTX-2.5:** two stages: 640x360, then the x2 latent upsampler, then 1280x704. 121 frames at 24 fps, DualCFG 1/1, euler_ancestral, 8+3 steps.
- **LTX-2.3:** the same structure at 25 fps, with CFGGuider 1 and euler.
- **Only difference between the two runs:** the model loader. Official uses `UNETLoader` (or the MODEL output of `CheckpointLoaderSimple`); integrated uses `SGLDUNETLoader`.

How outputs are compared:
- Video: mean per-frame PSNR of the mp4.
- Audio: waveform SNR.
- The saved latents are compared as well.

Reference points:
- The floor is official ComfyUI run with `--use-split-cross-attention`.
- Repeated official runs are bit-identical.

| Case (runs 0/1/2) | Integrated vs official: video PSNR / audio SNR (dB) | Floor |
|---|---|---|
| LTX-2.5 T2V, official TE | 20.2/8.1, 21.6/2.8, 21.3/5.7 | 20.6/6.2, 22.3/2.8 |
| LTX-2.5 I2V, official TE | 23.0/2.7, 22.6/4.6, 20.0/0.2 | (mirror TE) 21.6/3.5 |
| LTX-2.5 T2V, mirror TE | 19.1/3.1, 20.4/7.6, 18.0/3.2 | 18.5/4.9, 21.1/8.6, 18.2/4.5 |
| LTX-2.3 T2V FP8 | 13.4/7.4, 22.7/10.9, 15.3/10.5 | 13.3/7.4, 22.7/12.1 |

The integrated results are within ComfyUI's own implementation-noise band. The frames show the same composition (`frames/*_grid.jpg`, `videos/`). The low PSNR on LTX-2.3 run 0 comes from a small camera-framing drift that the floor run shows as well.

## Timing

How it was measured:
- Time is ComfyUI execution time.
- Each variant ran in a fresh ComfyUI process.
- The first run is cold; the warm runs use new seeds.

| Workflow | Official cold / warm (s) | Integrated cold / warm (s) | Stage 1 (official → integrated) | Stage 2 (official → integrated) |
|---|---|---|---|---|
| LTX-2.5 T2V (official TE) | 36.1 / 20.27, 20.03 | 51.3 / 19.11, 19.09 | 2.4–2.8 → 3.2–3.3 it/s | 1.60–1.62 → 1.64–1.65 s/it |
| LTX-2.5 I2V (official TE) | 31.9 / 20.59, 20.22 | 51.2 / 20.04, 20.27 | 2.6–2.7 → 3.2–3.3 it/s | 1.74–1.75 → 1.67–1.68 s/it |
| LTX-2.3 T2V FP8 | 34.6 / 17.11, 16.73 | 54.3 / 16.44, 15.84 | 2.42 → 1.79–2.01 it/s | 2.04–2.05 → 2.02–2.03 s/it |

**Warm runs.** The integrated path is equal or faster everywhere: about -5% on T2V, -1% on I2V and -5% on LTX-2.3.

**Cold runs.** The integrated path is about 15–20 s slower, because of worker spawn, the 21–25 GB DiT load and JIT warm-up. One cold run took 81 s while pre-commit was installing its environments on the same host.

**LTX-2.3 on a 32 GB card.** The 25 GB FP8 DiT plus the template's VAE decode (tile 768, temporal 4096) does not fit with the DiT fully resident: sampling succeeded, but the VAE decode then ran out of memory. The 2.3 workflow therefore keeps 75% of the layers resident and streams the rest with `dit_layerwise_offload`:
- Settings (via `extra_server_args`): `dit_layerwise_resident_layers=0.75`, `permanent` lifetime, prefetch 2.
- Effect: stage 1 is slower, while stage 2 and the end-to-end time are unaffected.
- A higher ratio (0.9) ran out of memory in ComfyUI's VAE stage.

**Allocator warnings.** Integrated LTX-2.5 runs log CUDACachingAllocator OOM warnings during the VAE decode. ComfyUI recovers through its fallback, every prompt succeeds, and the warnings' cost is included in the times above.

## Known limitations / untested

**Not supported:**
- Guide attention attenuation (`LTXVAddGuide` with strength ≠ 1, or with pixel masks). This raises NotImplementedError.
- Reference-audio (ID-LoRA) conditioning.
- Text attention masks.
- Video-only `ltxv` checkpoints.
- LTX-2.0-style caption projection.

**Untested:**
- Keyframe guides (`LTXVAddGuide` at strength 1). Covered only by the fp32 tiny-model unit test, not end to end.
- Context windows.
- `SGLDLoraLoader`.
- The prompt enhancer.
- Multi-GPU TP/SP.
- cache-dit.

**Differences from the official templates:**
- The official LTX-2.3 template uses the dev-FP8 checkpoint plus the distilled LoRA at 0.5. I used the provided distilled-FP8 checkpoint without the LoRA.
- The I2V template's input image isn't available locally, so I2V runs use a frame from the T2V output as the start image.

**Other notes:**
- The pipeline config is still detected from the file name, so `ltx-2.x` must appear in it. The DiT architecture itself comes from the checkpoint metadata.
- The quantization prefixes on LTX linears are new for every LTX user. Online `--quantization-ignored-layers` can now match LTX layers; it previously used an empty prefix and matched nothing. The GGUF lookup now uses SGLang module names, but LTX GGUF still does not load: see "Native pipeline regression" below. This line originally overstated GGUF.
- The worker-spawn fix is the cherry-picked commit shared with the Qwen branch. I dropped my own equivalent fix.

## Evidence (results/ltx25/)

- `scripts/`: environment setup, start/stop and session helpers; the DiT parity scripts; the end-to-end runners; the comparison and summary scripts.
- `logs/`:
  - `runs.jsonl` and `summary.json`
  - `cmp_*.json` and `dit_parity*.json`
  - per-session ComfyUI logs
- `frames/`: side-by-side grids. Rows are official, then the floor run where present, then integrated.
- `videos/`: the run-0 mp4s.

Cleanup: bulky intermediates (tensors and ComfyUI outputs) have been deleted. All processes are stopped and GPU 1 is empty.

Files I added to `ComfyUI-ltx25`:
- `models/` symlinks:
  - the LTX-2.3 checkpoint (in `checkpoints/` and in `diffusion_models/`)
  - the LTX-2.3 upscaler (`latent_upscale_models/`)
  - the Gemma 3 text encoder (`text_encoders/`)
  - the LTX-2.5 BF16 DiT (now pointing at the official copy)
- `input/ltx_i2v_start.png`

---

# LTX-2.3 dev-FP8 + distilled LoRA (official template as shipped)

The template as shipped: `ltx-2.3-22b-dev-fp8` + distilled LoRA 1.1 at 0.5; Gemma fp4_mixed TE + abliterated Gemma LoRA (1, 1); prompt enhancer on; stage-1 8 sigmas, stage-2 `0.85, 0.725, 0.4219, 0`; 1280x720, 121 frames, 25 fps; CFGGuider cfg 1.

- **Official path:** a fresh ComfyUI worktree, `ComfyUI-ltx2-b` (no custom nodes, port 8190, default dynamic VRAM).
- **SGLD path:** `ComfyUI-ltx25` (port 8189), using `SGLDUNETLoader(dev-fp8) -> SGLDLoraLoader(distilled, 0.5)`.
- **Hardware:** RTX 5090 32 GB. All timings below are on GPU1.

## Commits added on feat/comfyui-ltx2 (on top of the phase-1 commits)

| Commit | Subject |
|---|---|
| f219805acc | [diffusion] fix: evict ComfyUI models before each SGLD sampler run |
| 03ed84dd39 | [diffusion] feat: LoRA for ComfyUI LTX-AV via SGLDLoraLoader; reject native LoRA on SGLD models |
| 516efc6eb5 | [diffusion] fix: do not log an error for single-file checkpoints without model_index.json |
| a2a9f4e9bf | [diffusion] feat: requantizing LoRA merge for static FP8 linears |
| 896d2d7372 | [diffusion] fix: bind complete LoRA state from the sampled MODEL (cherry-pick of 0cd07c3530, niehen6174) |
| 420f695310 | [diffusion] fix: preserve parent links on SGLD model clones (cherry-pick of 772c6d5dd0, niehen6174) |
| 449f9df473 | [diffusion] fix: return the patched clone from SGLDLoraLoader (cherry-pick of fa617cfb6f, OnePunchMonk) |

- The cherry-picked subjects were reworded to upstream style.
- The old branch tip is kept as `backup/feat-comfyui-ltx2-pre-rebuild`. It can be deleted.

## How the LoRA is applied (SGLang side)

SGLang's normal LoRA merge (`LoRAPipeline.set_lora`, `merge_mode="merge"`, chosen by the LTX-AV executor) now also covers static per-tensor FP8 linears (`Fp8LinearMethod`). For each layer it:

1. dequantizes the FP8 weight;
2. adds `strength * B @ A`;
3. requantizes with a new per-tensor scale (`amax/448`);
4. writes the result back through the layerwise-offload host copy.

On unmerge, the original FP8 bytes and scale are restored.

After the merge, the denoising steps run the plain FP8 kernels with zero per-step LoRA cost. This is the fastest option and reuses the existing merge, unmerge and offload-writeback machinery.

The requantization uses stochastic rounding, seeded per layer so it is deterministic. This is not done to imitate ComfyUI. Round-to-nearest erases the LoRA. The 0.5-strength delta is about 0.8% of a weight, while one FP8 e4m3 step is 6–12%. Round-to-nearest therefore snaps most merged weights back to the base. Measured as DiT rel L2 vs ComfyUI with the LoRA on, it gives 0.525 / 0.26 (video / audio), against 0.375 / 0.043 with stochastic rounding. Stochastic rounding is unbiased, so the delta survives on average.

**Rejected alternatives:**

| Option | Why rejected |
|---|---|
| `dynamic` mode (unmerged LoRA matmuls every step) | Works, but 1.14 it/s vs 2.1 it/s |
| bf16 weight-space merge | 1.1 it/s, and doubles the weight bytes |

No ComfyUI-imitation mode is shipped. The `auto` merge mode is unchanged: FP8 still resolves to dynamic there. Only the ComfyUI LTX-AV executor asks for `merge`.

**Template's Gemma LoraLoader.** In the template, LoraLoader(model, clip) also receives the SGLD MODEL.

- `SGLDModelPatcher.add_patches` accepts patches with no `diffusion_model.` keys (the Gemma LoRA has 0 model keys), so the template runs unchanged.
- A ComfyUI-native LoRA whose keys target the served DiT is rejected loudly. Verified e2e (`F_sgld_native_lora`): "1660 LoRA / weight patches (e.g. 'diffusion_model.adaln_single...') target the diffusion model served by SGLang, which ComfyUI cannot patch. Load this LoRA with 'SGLDiffusion LoRA Loader' (SGLDLoraLoader) instead".
- Before this fix, the native LoraLoader crashed with a RecursionError (a state_dict cycle).

**Example workflow:** `apps/ComfyUI_SGLDiffusion/workflows/ltx2_3_dev_lora_t2v_sgld.json`. It is the template as shipped with the SGLD loaders, layerwise DiT offload and resident layers 0.8.

## Sanity evidence that the LoRA is applied

**DiT single forward** (dev-fp8, real Gemma conds, sigma 0.909; rel L2, video / audio):

| Comparison | Video | Audio |
|---|---|---|
| No LoRA: SGLD vs ComfyUI | 0.017 | 0.012 |
| LoRA effect, SGLD on vs off | 0.674 | 0.318 |
| LoRA effect, ComfyUI on vs off | 0.655 | 0.318 |
| LoRA on: SGLD vs ComfyUI (both merge with stochastic rounding, different draws) | 0.375–0.41 | 0.04 |
| For scale: ComfyUI's own two LoRA paths (static vs dynamic VRAM) against each other | 0.347 | 0.053 |

**End to end, no LoRA** (same template, seed 0, official vs SGLD): 37.4 dB video PSNR, audio SNR 17.6 dB (corr 0.99). Parity without the LoRA stays good.

**End to end, LoRA on vs off, seed 0:**

| Path | Video PSNR | Audio SNR (corr) |
|---|---|---|
| SGLD | 14.3 dB | -1.5 dB (0.15) |
| Official | 14.7 dB | -1.4 dB (0.17) |

With the LoRA off, the distilled 8+3-step schedule gives a dark, blurred smear with no structure on both paths. With the LoRA on, both paths give sharp, coherent video. See `frames/ltx23dev_lora_onoff_both.jpg`, with rows: official LoRA, SGLD LoRA, official no-LoRA, SGLD no-LoRA.

## End to end, 3 seeds (template as shipped, enhancer on)

The enhancer produced the identical prompt on both paths for every run (checked in `logs/runs.jsonl`). A run with the enhancer off and that prompt fixed gives bit-identical videos to the enhancer-on runs on both paths. The enhancer is therefore not a source of difference here.

| Seed | Video PSNR mean / min | Audio SNR (corr) |
|---|---|---|
| 0 | 19.3 / 15.8 dB | 6.0 dB (0.88) |
| 1 | 18.8 / 15.4 dB | 4.5 dB (0.82) |
| 2 | 16.8 / 14.8 dB | 4.2 dB (0.81) |

For scale, official dynamic VRAM vs official `--disable-dynamic-vram` (the same graph, only ComfyUI's LoRA-merge path differs):

| Seed | Video PSNR | Audio SNR |
|---|---|---|
| 0 | 19.3 dB | 5.9 dB |
| 1 | 18.1 dB | 8.1 dB |

The SGLD-vs-official gap is the same size as ComfyUI's gap with itself. Any FP8 LoRA merge with a different rounding draw lands here; it is not a sign of a misapplied LoRA.

**Frames:** `frames/ltx23dev_lora_seed{0,1,2}.jpg` (top: official, bottom: SGLD). All are sharp, with the same composition and the "LTX-2.3" title.

- Seed 2: SGLD renders "LTX-23" (the dot is missing) where official renders "LTX-2.3".
- Seed 1: the SGLD frame 0 has a faint ghosted overlay.

Both are sampling variation of the kind the dynamic-vs-static ComfyUI comparison also shows, not blur or broken structure.

**SGLD repeatability:** SGLD rerun in a fresh session gives bit-identical video (the merge is deterministic).

## Timing, VRAM, OOM (GPU1, RTX 5090 32 GB)

| | Cold (first prompt) | Warm | Stage 1 (8 steps) | Stage 2 (3 steps) | Peak VRAM (whole GPU) |
|---|---|---|---|---|---|
| Official (dynamic VRAM) | 46.5 s | 17.24 / 17.41 s | 2.42 it/s | 2.01 s/it | 31.83 GB |
| SGLD + LoRA (merge, resident 0.8) | 120.4 / 121.4 / 123.5 s | 16.49 / 15.91 / 16.27 / 16.16 / 16.30 s | 2.07–2.12 it/s | 1.99 s/it | 31.86 GB |
| SGLD, no LoRA | 83.9 s | 16.76 s | 2.12 it/s | 1.99 s/it | 31.85 GB |
| Official, enhancer off (GPU0) | 27.4 s | 17.16 s | | | |
| Official `--disable-dynamic-vram` | | 20.76 s | | | |

- **Warm:** SGLD is about 6% faster than official (16.2 s vs 17.3 s).
  - Stage 1 is a little slower per step, because 10 of 48 blocks stream from host (layerwise offload).
  - Stage 2 is the same.
  - The rest of the graph (upsampler, VAE) is shared ComfyUI code.
- **Warm run after a prompt change:** 18.0 s (the TE is re-encoded) on SGLD.
- **Cold:** about 75 s slower than official. Breakdown from the SGLD log, run 0:

| Phase | Time | Notes |
|---|---|---|
| Worker spawn and init | ~24 s | 12 s process and NCCL init, 2 s DiT load, 7 s layerwise offload setup |
| Prompt enhancer + Gemma TE | ~34 s | The same on the official path |
| LoRA wrap / bind | ~20 s | Not profiled; very likely the host snapshot of the 38 device-resident FP8 blocks (14.4 GiB) taken for unmerge |
| LoRA load + stochastic-rounding merge | ~16 s | |
| Sampling, upsample and decode | ~19 s | |

- **What would close the cold gap (not done):**
  - Start the worker while ComfyUI runs the TE/enhancer (saves up to about 24 s).
  - Snapshot lazily, or restore unmerge from the checkpoint file instead of a host copy (saves about 15–20 s).
  - Fuse dequant + delta + stochastic rounding into one kernel (saves part of the 16 s).

  All of this is one-time per ComfyUI process or per LoRA change.
- **OOM fixed by f219805acc.** Before it, the worker OOMed at its first step on 32 GB whenever ComfyUI still held the text encoder:
  - a run with no LoRA;
  - a LoRA run after a prompt change.

  ComfyUI's `free_memory(n)` only partially unloads, and it cannot see the worker's activations. The executor now unloads all other ComfyUI models before each sampler run. They reload on demand, with no measurable warm cost (16.2–16.8 s).

  After the fix: 0 OOMs and 0 allocator-OOM retries in the final sessions (2 no-LoRA runs; 4 LoRA runs including the prompt change).
- **Residency:** 0.85 and 0.9 log allocator OOM retries, so 0.8 is used (0.75 is about 10% slower in stage 1).

## The "Could not read model config ... model_index.json" error

`get_model_info()` tried to read `model_index.json` for the single-file DiT and logged it at ERROR level on every SGLD load. The diffusers fallback then resolved the right configs (`LTX23SamplingParams`), so the line was harmless but alarming.

Fixed in 516efc6eb5: for single files it is logged at debug level. A rerun showed 0 occurrences, with the same result. This replaces my earlier sampling-params-only fix, which did not cover this call site.

## Tests

`pytest` over `test/unit` comfy / ltx / lora / sampling / registry gives 304 passed, 1 skipped, 3 xfailed (`CUDA_VISIBLE_DEVICES=0`). New tests:

- stochastic FP8 rounding is unbiased and exact on the grid;
- the FP8 requant merge keeps the LoRA delta and unmerge restores the bytes and scale exactly;
- an SGLD model rejects a native DiT LoRA and accepts a TE-only one;
- the executor state_dict does not recurse;
- the eviction keeps only the sampled SGLD model.

pre-commit is clean.

## Limitations

- The FP8 merge is lossy, with stochastic-rounding noise of the same size as ComfyUI's. It is not bit-identical to ComfyUI.
- Switching LoRA at runtime (unmerge, then a different LoRA or strength) is covered by unit tests only, not e2e.
- `auto` merge mode still picks dynamic for FP8 bases outside the ComfyUI LTX-AV executor.
- The eviction before every sampler run means ComfyUI reloads its upsampler and VAE each run. That is cheap with dynamic VRAM, but not measured with `--disable-dynamic-vram`.
- Evidence files: `frames/ltx23dev_*.jpg`, `videos/ltx23dev_*.mp4`, `cmp/*.json`, `logs/runs.jsonl`, `logs/comfy*_F*.log`, `logs/vram_*.csv`.

---

# Native pipeline regression (plain `sglang generate`, no ComfyUI)

**Question:** do this branch's changes to the native LTX DiT and pipeline change SGLang's native LTX path? The changed files include `runtime/models/dits/ltx_2.py`, `configs/models/dits/ltx_2.py`, `ltx_2_pipeline.py` and `registry.py`.

**Answer:** no regression found. The default native path is bit-identical to the base. The only behaviour change, as intended, is that `--quantization-ignored-layers` now works for LTX.

## Setup

- **Base:** `/scratch/data/sgld_comfy/wt-ltx-base`, a detached worktree at 6e7beace14. **Branch:** `feat/comfyui-ltx2` at c9c5e6ad37. Both use the same venv and environment; only `PYTHONPATH` is swapped. GPU1 (RTX 5090 32 GB).
- **Model:** `Lightricks/LTX-2.5-Diffusers` (local), the default distilled DiT, `LTX2Pipeline` one-stage.
- **Prompt enhancer workaround.** The prompt-enhancer weights were not downloaded. A local Diffusers checkout that declares `prompt_enhancer` without its shards is judged "incomplete" and triggers a re-download. The native pipeline does not load the enhancer (it is used only by HTTP `enhance_prompt`). Instead of downloading 9.5 GB, I ran from `models/Lightricks--LTX-2.5-Diffusers-noenh`: symlinks to every component plus a `model_index.json` without `prompt_enhancer`. The directory name must contain `LTX-2.5` for config detection. It was removed afterwards.
- **Command:** `sglang generate --seed 42 --height 384 --width 640 --num-frames 49 --num-inference-steps 8 --dit-layerwise-offload true --dit-layerwise-residency-lifetime forward --dit-layerwise-resident-layers 0.5 --cpu-offload-components text_encoder connectors vae audio_vae vocoder`. T2V; TI2V adds `--image-path ltx_i2v_start.png`.
  - The cookbook's 32 GB options do not fit as-is. The Gemma4 TE (22.3 GB) plus GPU-resident connectors (5.9 GB) OOM when the TE is moved in, and Gemma4 does not support layerwise offload. Hence the CPU-offloaded connectors and the forward-lifetime DiT residency.
  - `--quantization fp8` (the cookbook's tight-VRAM preset) OOMs while loading the 35 GB bf16 DiT on 32 GB, on base and branch alike. So FP8 could not be run end to end here.
- Script: `scripts/native_run.sh`. Logs, perf dumps and VRAM traces: `results/ltx25/native/`.

## Results

**Outputs (mp4 with video and audio tracks, md5):**

| | Runs | md5 |
|---|---|---|
| T2V, base | 3 | f64a5918… |
| T2V, branch | 4 | f64a5918… |
| TI2V, base | 4 | c6272afa… |
| TI2V, branch | 4 | c6272afa… |

Byte-identical files. This includes TI2V, which drives per-token timesteps through the DiT, the path closest to the new per-frame modulation code. The default numerics are therefore unchanged, so PSNR/SNR are moot (inf). Frames look right: `frames/native_ltx25_t2v_ti2v.jpg`.

**DiT step time** (from `--perf-dump-path`, median of steps 1–7, two warm runs each):

| | Base | Branch |
|---|---|---|
| T2V | 804 / 793 ms | 814 / 792 ms |
| TI2V | 809 / 805 ms | 920 / 806 ms |

- The branch's first TI2V timing run was slow overall: decode took 5.7 s instead of about 1.1 s, and total time was 72 s instead of 34 s. The repeat matched the base, so I treat it as noise.
- Step 0 is 2.0–2.8 s on both trees (warm-up).

**Peak VRAM:**
- Torch peak allocated: 26318.47 MB in every run on both trees. Reserved: 29838 MB.
- `nvidia-smi` whole-GPU peak: 30890 MiB for T2V on both trees.
- With 200 ms sampling, the TI2V branch runs showed a single sample at 31638 MiB that the base did not. At 50 ms sampling the base shows the same one-sample spike (31542 MiB). It is the hand-off when the TE leaves the GPU, before any DiT code runs, so it is not a regression.

**Text-encoder-path changes on the branch:**
- `registry.py`: the debug-level log line only.
- `ltx_2_pipeline.py`: `_declares_component` returns False only for a single file, and Diffusers model paths are directories.

Neither path is reached differently for a Diffusers directory. The identical outputs agree.

## Code paths the phase-1 commits changed for native users

**`--quantization-ignored-layers`: works now; it was a silent no-op before.**

A meta-device build of a tiny LTX-2.5 DiT with `Fp8Config(ignored_layers=[...])` was run on both trees (`scripts/quant_prefix_probe.py`, `native/quant_probe_{base,branch}.json`).

| | Linears given the quant config | Linears with an empty prefix | Effect of the ignore list |
|---|---|---|---|
| Base | 38 | 38 | Skips nothing |
| Branch | 38 | 0 | Skips exactly `transformer_blocks.0.attn1.*` and every `proj_out` component |

The adaLN linears never take the quant config on either tree. The bare `proj_out` entry also matches `ff.proj_out` and `audio_ff.proj_out`, which is SGLang's dotted-component matching. Without the flag, the selection is the same as on base: no LTX defaults exist.

Regression test added (c9c5e6ad37, `test_quantization_ignored_layers_match_ltx2_module_paths` in `test_comfyui_ltx_av.py`). It fails on the base DiT code and passes on the branch.

**Other prefix-keyed formats.** The same prefixes reach any checkpoint-declared quant config that looks layers up by name, such as ModelOpt exclude lists and NVFP4 per-layer configs. These used to see "" and now see the module path. This is very likely a fix too, but it is untested: there is no such LTX checkpoint locally.

**LTX GGUF: still unsupported. The earlier claim was overstated and is corrected above.**

`GGUFConfig` looks up `f"{prefix}.weight"` verbatim in the GGUF tensor names, and the native and ComfyUI GGUF paths do not remap them for LTX.

| | Lookup result |
|---|---|
| Base | Fails at the first linear (`''`) |
| Branch | Fails at the first feed-forward linear |

I compared the 1636 linear prefixes of a real-size LTX-2.3 DiT with the real ComfyUI checkpoint names (`ltx-2.3-22b-dev-fp8.safetensors`; city96 ComfyUI-GGUF LTX files carry these names). All attention, projection and in/out layers match. The 192 feed-forward linears do not, because SGLang's `ff.proj_in`/`ff.proj_out` are ComfyUI's `ff.net.0.proj`/`ff.net.2` (48 layers × video/audio × in/out).

Making LTX GGUF work would mean remapping `tensor_meta` through the LTX arch `param_names_mapping` (`remap_gguf_tensor_meta` already keeps both aliases). The text-encoder GGUF loader does this already. It would then need an end-to-end test with a real GGUF. Not done: GGUF is a new feature, not a regression, and no LTX GGUF is local.

**LTX-2.3 native:** skipped. There are no Diffusers-format LTX-2.3 weights locally, and there is no cheap path: the local 2.3 file is a ComfyUI single file, which only the ComfyUI mode serves.

**Unit tests:** `pytest test/unit` (comfy / ltx / lora / sampling / registry) gives 305 passed, 1 skipped, 3 xfailed. pre-commit is clean.

---

# Multi-GPU (ComfyUI integrated mode): INTERIM, NOT FINAL

**Status.** Stopped on request before the end-to-end ComfyUI runs.
- The code is archived, not merged, on branch `archive/comfyui-ltx2-mgpu-wip` (8e994c9532). It contains:
  - SP video-token sharding in the LTX ComfyUI step;
  - a TP-consistent FP8 LoRA requant merge;
  - rejection of CFG-parallel;
  - unit tests.
- Only the generic cherry-pick of 0de333f7f7 (c79398dcb4: ComfyUI multi-GPU options and error handling) is on `feat/comfyui-ltx2`.
- All numbers below are single DiT steps (`scripts/mgpu_dit.py`) on 2x RTX 5090 (no P2P), at the template's shapes:
  - stage 1: 16x11x20 = 3520 video tokens;
  - stage 2: 16x22x40 = 14080 video tokens;
  - 121 audio tokens, batch 1.
- No end-to-end video/audio PSNR, cold/warm end-to-end times or end-to-end VRAM were measured.

## What the archived code does

**SP (sp_degree=2).** Each rank takes a contiguous half of the video tokens, together with their explicit RoPE coordinates, per-token or per-frame timesteps and keyframe mask. Audio is replicated (`audio_replicated_for_sp`), so its odd token count (121) is fine.
- With replicated audio, the LTX DiT gathers video K/V for self-attention and runs every attention locally. Ulysses all-to-all is therefore never used, and explicit Ulysses2 and the auto split (which resolves to `kv_gather_degree=2`) are bit-identical.
- Without this change, SP would silently compute wrong outputs: every rank fed full sequences into the SP attention.
- A video token count that is not divisible by the SP degree fails with a clear error. Example: "LTX sequence parallelism splits the 3465 video tokens (15 latent frames x 11x21) across 2 ranks, which needs a token count divisible by 2; change the frame count or resolution, or run without sp_degree".
- The template's 3520 and 14080 tokens are even.

**TP (tp_size=2).** Works unchanged for INT8 ConvRot and static-FP8 checkpoints. LTX's sharded input dimensions (2048 / 8192 / 1024) are multiples of the ConvRot group of 256.

**FP8 LoRA requant merge under TP.**
- On the current branch, each rank requantized its shard with its own amax scale and its own random draws. The merged weights then differed from the 1-GPU merge in 40–55% of their bytes (`scripts/tp_lora_merge_check.py`).
- The archived fix all-reduces the amax over TP, and keys the stochastic rounding by global element index with a counter hash. With it, the TP2 merge equals the 1-rank merge byte for byte, for both column- and row-parallel layers.

**CFG parallel** is rejected for LTX in ComfyUI mode, because ComfyUI owns CFG.

## Parity

Rel L2 against the 1-GPU worker on identical inputs. "Floor" means the same config computing the same input at batch 2 instead of batch 1, i.e. only the kernel shapes change.

| Model | Config | Stage 1 video | Stage 2 video | Stage 1 floor |
|---|---|---|---|---|
| LTX-2.5 INT8 | SP2 | 0.146 | 0.000 | 0.135 |
| LTX-2.5 INT8 | TP2 | 0.158 | 0.099 | 0.135 |
| LTX-2.3 dev-FP8, no LoRA | SP2 | 0.016 | 0.027 | 0.017 |
| LTX-2.3 dev-FP8, no LoRA | TP2 | 0.017 | 0.028 | 0.017 |
| LTX-2.3 dev-FP8 + distilled LoRA 0.5 | SP2 | 0.320 | 0.091 | 0.335 |
| LTX-2.3 dev-FP8 + distilled LoRA 0.5 | TP2 | 0.330 | 0.094 | 0.335 |

- Every multi-GPU difference is at the batch-1 vs batch-2 floor.
- The distilled models (LTX-2.5, and 2.3 with the distilled LoRA) amplify kernel-level noise at sigma 0.91. The plain dev model does not.
- The LoRA effect itself (1-GPU, LoRA on vs off) is 0.76 / 0.52 (stage 1 / stage 2).
- Against ComfyUI's own LTX-2.5 INT8 model at the stage-1 shape, 1-GPU, SP2 and TP2 score 0.129 / 0.129 / 0.122 rel L2: equally close to the reference.
- A bf16 LTX-2.5 check (1-GPU layerwise vs SP2 vs TP2) gave 0.11 / 0.11 at stage 1: the same picture.

## Warm step time and per-GPU peak VRAM (DiT only, no ComfyUI models loaded)

| Model | Config | Stage 1 s/step | Stage 2 s/step | Peak VRAM GPU0 / GPU1 |
|---|---|---|---|---|
| LTX-2.5 INT8 | 1 GPU, resident | 0.295 | 1.62 | 24.6 / – GB |
| LTX-2.5 INT8 | SP2 (Ulysses2 = auto) | 0.350 | 1.457 | 23.4 / 23.3 GB |
| LTX-2.5 INT8 | TP2 | 0.60 | 2.49 | 13.7 / 13.6 GB |
| LTX-2.3 dev-FP8 + LoRA | 1 GPU, layerwise 0.8 | 0.47 | 1.96 | 29.5 / – GB |
| LTX-2.3 dev-FP8 + LoRA | SP2, layerwise 0.8 | 0.51 | 1.74 | 27.8 / 27.6 GB |
| LTX-2.3 dev-FP8 + LoRA | SP2, fully resident | 0.41 | 1.66 | 29.5 / 29.3 GB |
| LTX-2.3 dev-FP8 + LoRA | TP2, fully resident | 0.65 | 2.67 | 18.0 / 17.9 GB |

**TP2 for LTX-2.3.** It shards the 22B FP8 DiT so it runs fully resident at about 15.4 GB per GPU (18 GB with LoRA, because the LoRA factors stay on the GPU). But it is slower than 1 GPU with offload:

| Stage | TP2 vs 1 GPU with offload |
|---|---|
| Stage 1 | 1.4x slower |
| Stage 2 | 1.36x slower |

Without NCCL peer access, the per-block all-reduces go through host memory.

**SP2.** It is the only config that speeds up the long stage 2, by 1.10–1.18x. Stage 1 is equal or slower. It keeps a full weight copy on each GPU.

**Both ranks compute.** The 200 ms `nvidia-smi` traces show both GPUs busy for a comparable number of samples in every 2-GPU run (`logs/mgpu_*.smi.csv`).

**OOM:** none in these DiT-only runs.

**Untested here:** whether SP2 fully resident fits next to ComfyUI's VAE decode on GPU0 end to end. On 1 GPU it did not.

---

# Slimmed branch (feat/comfyui-ltx2, rebuilt on upstream/main)

The goal was the smallest change to common code, with no ComfyUI numeric alignment. The full versions are kept on:
- `archive/comfyui-ltx2-full-pre-mgpu` (main);
- `archive/comfyui-ltx2-mgpu-wip` (the multi-GPU work).

**Size against upstream/main (6e7beace14):**

| | Before (full branch) | After (slimmed) |
|---|---|---|
| Overall | 26 files, +3934 / −77 | 20 files, +2255 / −30 |
| Common files only | 11 files, +717 / −63 | 7 files, +260 / −16 (including test_lora_pipeline.py, +88) |

## Commits

| Commit | Subject |
|---|---|
| ad9a8773e0 / ed151a1d0a / e5388e9328 | SGLDLoraLoader fixes by OnePunchMonk / niehen6174, cherry-picked unchanged |
| 530dd8aea6 | [diffusion] feat: LTX-2 DiT linears pass their module path to the quant config |
| cb3f12ec24 | [diffusion] feat: requantizing LoRA merge for static FP8 linears |
| 3c4aaa1d12 | [diffusion] feat: ComfyUI integrated-mode worker for LTX-2.5 / LTX-2.3 (ltxav) |
| 4db169a1f3 | [diffusion] feat: LTX-2.5 / LTX-2.3 in the ComfyUI SGLDiffusion plugin |

## Remaining common changes

| File | Lines | Why it cannot live in plugin or model-specific code |
|---|---|---|
| `runtime/models/dits/ltx_2.py` | +41 / −3 | Linear `prefix` plumbing only. Per-layer quant markers (INT8 ConvRot / FP8) are looked up by module path when the linear is built. It also fixes `--quantization-ignored-layers` for native LTX, which previously matched nothing. |
| `runtime/layers/lora/linear.py` | +70 | FP8 requantizing merge: dequantize, add, unbiased stochastic round, unmerge restore. This lives inside the LoRA layer. |
| `runtime/pipelines_core/lora/pipeline.py` | +30 / −1 | Allow merge mode for requantizable FP8 layers. Bind the layerwise host copy of `weight_scale`. Park the merged LoRA factors on the host for layerwise-offloaded modules. Without the parking, the second warm run of the LTX-2.3 template ran out of memory ("VRAM grow failed"). |
| `runtime/loader/comfyui_checkpoints/spec.py` | +22 / −8 | The serialized Comfy quantization path was hard-wired to MiniMax-H3. Adds three spec fields: `quant_markers`, `quant_formats`, `checkpoint_key_filter`. |
| `runtime/loader/comfyui_checkpoints/__init__.py` | +1 | Registers the LTX spec. |
| `runtime/pipelines/ltx_2_pipeline.py` | +12 | The `create_comfyui_stages` hook. |

Everything else is plugin code or new model-specific files:
- `comfyui_checkpoints/ltx_2.py` (+268);
- `ltx_2/comfyui_step.py` (+161);
- `executors/ltx_av.py` (+183);
- 2 workflows;
- tests.

## Dropped, and the measured effect

**In-DiT connector implementation** (`ltx_2_embeddings_connector.py`, +175, and its DiT/config fields).
- The connectors now load from the ComfyUI file into SGLang's existing `LTX2ConnectorTransformer1d`, as a separate module on the worker. The key remap and dequantization live in the LTX checkpoint spec.
- Quantized connector linears keep their stored INT8 / FP8 form and dequantize per call. A bf16 copy (+1.9 GiB on LTX-2.5) made the second I2V run OOM in ComfyUI's VAE decode.
- Context parity vs ComfyUI: 0.014 / 0.006 (LTX-2.5 pos / neg) and 0.004 (LTX-2.3).
- A unit test checks that SGLang's connector loaded with ComfyUI weights reproduces ComfyUI's `Embeddings1DConnector` to < 1e-3 rel L2; only the Q/K-norm eps differs.

**Per-frame modulation and timestep dedup.** I2V now passes per-token timesteps, which the native DiT already supports. T2V is unaffected.

| LTX-2.5 I2V (warm) | Before (per-frame) | After (per-token) |
|---|---|---|
| Stage 1 | 3.2–3.3 it/s | 2.8 it/s |
| Stage 2 | 1.67 s/it | 1.88 s/it |
| End to end | 20.0–20.3 s | 22.4–22.6 s (+11%) |

Official ComfyUI I2V warm is 20.6 s.

**ComfyUI numeric switches.** Removed: Q/K-norm eps 1e-5, v2a re-normalization after a2v, a2v/v2a gate-timestep overrides, prompt-timestep override, explicit ComfyUI RoPE coordinates, and `keyframes_abs_pos_embedding` application.
- The DiT now runs SGLang's native semantics: native coordinates, native gate and prompt timesteps.
- Parity vs ComfyUI's own model, at the stage-1 template shape (16x11x20 tokens, sigma 0.909):

| Model | T2V video / audio | I2V video / audio | Before |
|---|---|---|---|
| LTX-2.5 INT8 | 0.136 / 0.021 | 0.155 / 0.033 | 0.129 / 0.019, 0.139 / 0.020 |
| LTX-2.3 dev FP8 | 0.017 / 0.012 | 0.022 / 0.013 | 0.017 / 0.012 |

  The LTX-2.5 INT8 batch-1 vs batch-2 floor of a single config is 0.135.
- The I2V output still holds the conditioning image as its first frame and keeps the composition (`frames/slim25i_seed0.jpg`). No visible defect without the keyframe embedding.

**Keyframe guides (`LTXVAddGuide`).** These are now rejected with a clear error, as are attention masks and reference audio. The guide token filtering / coordinate code was removed.

**`registry.py` single-file log fix.** Dropped, so `[ERROR] Could not read model config ... model_index.json` is logged again: twice per worker load, harmless (the fallback resolves the right configs). Silencing it needs a common change.

**`server_args.py` auto-CFG guard** (from the Qwen agent's multi-GPU fix). Dropped with the multi-GPU work. `num_gpus=2` without explicit degrees will fail again in comfyui mode, until the Qwen branch's version lands.

**TP-consistent FP8 merge (global amax and index-keyed noise) and per-layer seeds.** These are on the archive branch. The stochastic rounding is now 7 lines.

**Workflows.** Kept 2: `ltx2_5_t2v_sgld.json` and `ltx2_3_dev_lora_t2v_sgld.json`.

## LoRA: merge (kept) vs dynamic (unmerged)

LTX-2.3 dev FP8 + distilled LoRA template, GPU1, resident 0.8:

| | Stage 1 | Stage 2 | Cold | Warm | Second warm run |
|---|---|---|---|---|---|
| Merge (requantized FP8) | 2.11 it/s | 1.98 s/it | 79 s | 16.1–16.35 s | Fine |
| Dynamic | 1.56 it/s | 2.55 s/it | 80.6 s | ≈ +3 s per run | OOM ("VRAM grow failed") |

Merge is kept because dynamic is about 19% slower warm and does not fit at the template's residency.

LoRA is applied correctly. DiT LoRA-on vs LoRA-off effect:

| | Video | Audio |
|---|---|---|
| SGLD | 0.818 | 0.498 |
| ComfyUI | 0.809 | 0.498 |

SGLD-with-LoRA vs ComfyUI-with-LoRA: 0.43 / 0.10. The two use different FP8 rounding draws, and the distilled model amplifies small differences.

## Verification (single GPU, GPU1)

**Unit tests:** comfy / ltx / lora / sampling / registry suites give 296 passed, 1 skipped, 3 xfailed. pre-commit is clean.

**End to end, official vs integrated** (same seeds; enhancer off and fixed prompt for LTX-2.3):

| Case | Video PSNR | Audio SNR | Official cold / warm | Integrated cold / warm | Peak VRAM |
|---|---|---|---|---|---|
| LTX-2.5 T2V INT8 | 20.2 / 20.3 / 23.2 dB | 3.5 / 4.0 / 2.3 dB | 35.3 / 19.9, 20.5 s | 50.3 / 20.7, 20.2 s | 30.5 GB |
| LTX-2.3 dev + LoRA | 19.0 / 20.4 / 16.2 dB | 6.7 / 5.2 / 1.5 dB | 36.5 / 17.65, 17.43 s | 79.1 / 16.1, 16.35 s | 31.7 GB |
| LTX-2.5 I2V | 19.3 / 19.2 dB | | 31.4 / 20.6 s | 54.2 / 22.6, 22.4 s | 29.1 GB |

- The same composition in every run: `frames/slim25_seed*.jpg`, `slim23_seed*.jpg`, `slim25i_seed0.jpg`.
- PSNR is at the level the full branch had (implementation noise).
- No OOM in the final sessions.
- LTX-2.5 T2V warm is now equal to official, where the full branch was about 5% faster. The integrated path evicts ComfyUI's upsampler and VAE before each sampler run, and they then reload.

**Native pipeline:** plain `sglang generate` LTX-2.5 T2V (same setup as the "Native pipeline regression" section). Base vs slimmed branch outputs are byte-identical (md5 f64a5918…). Torch peak is the same, 29836 MB.

**2-GPU smoke:** not run. GPU0 is with the Qwen agent.

---

# Rebased on the common plugin-fixes branch

`feat/comfyui-ltx2` is now `feat/comfyui-plugin-fixes` (head ede8433c83) plus 4 LTX commits. The previous slimmed branch is kept as `archive/comfyui-ltx2-slim` (4db169a1f3).

| Commit | Subject |
|---|---|
| a1afae7417 | [diffusion] feat: LTX-2 DiT linears pass their module path to the quant config |
| d131b01057 | [diffusion] feat: requantizing LoRA merge for static FP8 linears |
| 1225c6e77f | [diffusion] feat: ComfyUI integrated-mode worker for LTX-2.5 / LTX-2.3 (ltxav) |
| 2441024f1a | [diffusion] feat: LTX-2.5 / LTX-2.3 in the ComfyUI SGLDiffusion plugin |

**Removed because the common branch now provides it:**
- the three SGLDLoraLoader cherry-picks;
- evict / `state_dict` / native-LoRA rejection;
- worker-spawn isolation, error wrappers and `extra_server_args`;
- the spec hooks;
- the plugin unit tests that duplicated the common ones.

`quant_formats` is gone: format support is checked by the common branch's resolver, and LTX files are either all INT8 or all FP8.

**Diff vs the common branch:** 17 files, +1942 / −9.
- **Common components:** `models/dits/ltx_2.py` +41/−3 (linear prefixes), `layers/lora/linear.py` +70 and `lora/pipeline.py` +30/−1 (FP8 requant merge). Together +141 / −4, plus `test_lora_pipeline.py` +84 / −4.
- **Registration lines only:** spec.py discover list +1, `comfyui_checkpoints/__init__.py` +1, generator.py executor list +2 / −1, `executors/__init__.py` +3, nodes.py model_type +1, README +3.
- **Pipeline hook:** `ltx_2_pipeline.py` `create_comfyui_stages` +12.
- **Everything else** is new LTX-only files: spec, step stage, executor, 2 workflows, tests.

**`git merge-tree` with `feat/comfyui-qwen-image21`:** 2 conflicts, both registration lists:
- the README model entry (Qwen-Image-2.1 line vs LTX line);
- the generator.py executor list (`QwenImage21Executor` vs `LTXAVExecutor`).

spec.py, `comfyui_checkpoints/__init__.py`, nodes.py and `executors/__init__.py` auto-merge.

**Tests:**
- comfy / ltx / lora / sampling / registry suites: 303 passed, 1 skipped, 3 xfailed, 1 failed.
- The failure, `test_comfyui_lora_node::test_chained_lora_nodes_keep_every_lora`, comes from the common branch and has nothing to do with LTX. `test_comfyui_generator` imports `nodes` with a `folder_paths` stub that lacks `get_full_path`. The cached module then fails in the LoRA-node test when the two files run in one session. Each file passes on its own. Both files and nodes.py are unchanged from the common branch, except my one model_type line.

**GPU1 smoke** (the rebased branch vs the slimmed branch, same seed):

| Case | Result | Warm |
|---|---|---|
| LTX-2.5 T2V | Identical video and audio latents, video PSNR inf | 20.2 s |
| LTX-2.3 dev + LoRA (vs saved slimmed output) | Video PSNR inf | 17.4 s |

**`num_gpus=2` without explicit degrees:** covered by the common unit test `test_worker_start_pins_cfg_parallel_off_and_names_startup_failures` (passes). It shows `init_generator` pins `cfg_parallel_degree=1`, so the auto-CFG lookup is skipped. A full ServerArgs resolution needs two visible devices, so it was not run, since GPU0 is off limits.

---

# 2-GPU smoke on the rebased branch (feat/comfyui-ltx2)

**Bug found and fixed (new commit 2e7ea4a7d8):** "[diffusion] fix: shard LTX video by frames under sequence parallelism in ComfyUI mode".

Without the fix, sp_degree=2 ran but gave silently wrong output at about half the speed. This hit both explicit Ulysses2 and the auto split, which `num_gpus=2` alone resolves to: `kv_gather_degree=sp_degree=2`. Both ranks fed the whole token sequence into SP attention.

DiT single step vs the 1-GPU worker (rel L2), LTX-2.5 INT8:

| | Stage 1 | Stage 2 | Stage 2 speed |
|---|---|---|---|
| SP2 before the fix | 0.85 | 0.79 | 3.56 s/step |
| SP2 after the fix | 0.134 | 0.000 | 1.47 s/step |
| Batch-noise floor | 0.137 | | |

What the fix does:
- Each rank takes an equal block of whole latent frames. The DiT offsets RoPE time by the rank.
- Audio is replicated, and the video output is all-gathered.
- A frame count that the degree does not divide fails cleanly on both ranks with no hang: "LTX sequence parallelism splits the 15 latent frames across 2 ranks, which needs a frame count divisible by 2 ...".
- It adds a unit test.

TP2 needed no change: rel L2 0.149 / 0.091 against the 0.137 floor.

**End to end on 2x RTX 5090** (official two-stage templates, ComfyUI on GPU0, 3 runs for LTX-2.5 and 2 for LTX-2.3; PSNR vs the 1-GPU run with the same seed):

| Config | Stage 1 | Stage 2 | Cold | Warm | Peak VRAM GPU0 / GPU1 | PSNR vs 1 GPU |
|---|---|---|---|---|---|---|
| LTX-2.5 T2V INT8, 1 GPU | 3.2–3.4 it/s | 1.61 s/it | 50.0 s | 20.6 / 20.1 s | 30.2 / – GB | – |
| LTX-2.5 T2V INT8, Ulysses2 | 2.84 it/s | 1.46 s/it | 51.8 s | 20.4 / 20.3 s | 30.8 / 23.4 GB | 20.6 / 20.9 / 20.4 dB |
| LTX-2.5 T2V INT8, auto (`num_gpus=2` only) | 2.85 it/s | 1.46 s/it | 51.9 s | 20.0 / 20.2 s | 30.9 / 23.4 GB | identical to Ulysses2 |
| LTX-2.5 T2V INT8, TP2 | 1.67 it/s | 2.48 s/it | 56.6 s | 25.5 / 25.1 s | 20.3 / 14.7 GB | 20.0 / 21.4 / 23.1 dB |
| LTX-2.3 dev + LoRA, 1 GPU (layerwise 0.8) | 2.1 it/s | 1.95 s/it | 84.7 s | 15.8 s | 31.7 / – GB | – |
| LTX-2.3 dev + LoRA, TP2 (fully resident) | 1.51 it/s | 2.68 s/it | 61.9 s | 19.8 s | 29.6 / 19.9 GB | 19.7 / 18.1 dB |

The PSNR values are at the official-vs-integrated level (distilled models amplify small numeric differences). The frames show the same composition: `frames/mg25_seed0_1gpu_sp2_tp2.jpg` and `mg23_seed0_1gpu_tp2.jpg`.

- **Both ranks busy:** `nvidia-smi` traces show both GPUs busy in every 2-GPU run.
- **OOM:** none in the 2-GPU runs. The 1-GPU LTX-2.5 session logged one recovered allocator OOM retry in the VAE decode.

**Which configs work:**
- **Ulysses2 / auto:** correct, and the same warm time as 1 GPU. Stage 2 is faster and stage 1 slower.
- **TP2:** correct, and it halves per-GPU DiT memory (LTX-2.3 runs fully resident), but it is 20–25% slower warm without NCCL peer access.
- **TP2 + FP8 LoRA:** each rank requantizes its shard with its own scale and draws. The LoRA is applied (the output looks right), but the merged weights are not byte-equal to the 1-GPU merge. The TP-consistent merge is on `archive/comfyui-ltx2-mgpu-wip`, not on this branch.
- **Not run:** CFG parallel (ComfyUI owns CFG) and ring attention (needs an LSE backend; FA is not available on SM12.0).

**Unit tests:** 305 passed, 1 skipped, 3 xfailed.
