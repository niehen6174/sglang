# Add Wan 2.1 1.3B and Wan 2.2 5B to ComfyUI integrated mode

SGLang already implements Wan, but SGLDUNETLoader rejected ComfyUI Wan
checkpoints. This change supports Wan 2.1 T2V 1.3B and Wan 2.2 TI2V 5B
(text-to-video and image-conditioned sampling). ComfyUI retains UMT5, video
latent preparation, frame masking, sampling and VAE decoding; the existing
WanPipeline runs the DiT forward only.

The checkpoint spec derives architecture from safetensors headers, handles
original Wan names with or without the model.diffusion_model prefix, and
loads strictly. It retains native Diffusers mappings so original-format and
Diffusers-format LoRAs both resolve to runtime layers. The adapter preserves
5D latents, circular spatial padding, video geometry and Wan 2.2 per-frame
masked timesteps. These timesteps are repeated per patch token and never
stored in the condition cache. Default worker precision follows ComfyUI;
explicit component_precisions overrides remain supported.

FP32 validation also exposed FlashInfer RoPE's half-precision-only dispatch.
The existing torch fallback now handles FP32; FP16/BF16 keep their fast path.

## Validation

- RTX 5090 32 GB, one GPU, torch_sdpa, no torch.compile.
- 515 focused unit tests plus 5 existing native Wan configuration/TI2V tests passed.
- 35 selected final ComfyUI API requests succeeded with finite latent outputs.
- Both models: CFG, B=2 (row fallback), different frame counts, original ComfyUI comparisons.
- Wan 2.2: image-conditioned frame masks, automatic model detection.
- Both models: synthetic rank-2/rank-3 LoRAs A/B/AB change output; AB→A, zero strength and removing all LoRAs restore the corresponding reference exactly.
- Three complete standard API graphs: 20 UniPC steps, simple scheduler, CFG 5, shift 8, 512×320, 33 frames, VAE decode and animated WebP output. ModelSamplingSD3 is included.

FP32 two-step relative latent RMSE against native ComfyUI is 0.0074% (1.3B)
and 0.0043% (5B with an input image). FP16 two-step differences are
0.13–1.54%; full 20-step T2V sampling differs by 8.58% / 15.84% in latent
RMSE. This is functional support, not a claim of bitwise or pixel-identical
native output. No image-quality score or general performance claim is made.
The small Euler smoke tests are retained as diagnostic evidence; published
API examples use the tested UniPC graphs.

Evidence: /scratch/data/sgld_comfy/results/comfyui-wan-20261009/
(FINAL_VALIDATION.json, comparisons-*.json, lora-validation.json, workflows,
runs, logs, media and reproducible local tools). Failed exploratory loading,
FP32 and ineffective-LoRA runs remain preserved separately.

## Official template validation

A separate six-request API comparison used the published ComfyUI templates
at revision `8be1f8c4b5af2d550d70922a23b79cee599e1f3e`:
Wan 2.1 T2V 832×480/33 frames/30 steps/CFG 6 and Wan 2.2 T2V + I2V
1280×704/121 frames/20 steps/CFG 5. All native and SGLD requests passed,
including KSampler, VAE decode, CreateVideo and SaveVideo. Resolution,
frame count, prompts, seeds and sampling settings were retained. I2V
activates the template's disabled LoadImage with a supplied generated fox
first frame. API conversion adapts SaveVideo's dynamic fields for ComfyUI
0.38.0 and changes filenames for isolation. The first exploratory pair
failed only due to the converter's SaveVideo parameter encoding; the
corrected full requests passed without further runtime code changes.

Native/SGLD end-to-end seconds: 28.17/47.45 (Wan 2.1 T2V),
136.86/151.77 (Wan 2.2 T2V), 141.02/156.49 (Wan 2.2 I2V).
These include load/text/sample/VAE/save and are single requests, not
controlled performance benchmarks. No speedup is established. Output
scenes look similar but details differ; no pixel-equivalence claim.
Three reproducible official-default API graphs are included; the I2V
example requires a user image in place of `example.png`.

Evidence: /scratch/data/sgld_comfy/results/comfyui-wan-official-20261009/
(templates, sources.json, manifest.json, validation.json, runs, logs,
media, tools and CONCLUSIONS.txt).

## Scope and dependency

Branch: codex/feat-comfyui-wan. Base: codex/fix-comfyui-native-batch at
713df826c243ed47be57244b08b936640068cd86. This includes the earlier multi-LoRA
and batch fixes; this note describes only the new Wan changes.

Wan native packed batching, multi-GPU, compile, sparse attention and
quantization were not validated. Wan 2.1 I2V, VACE, Animate, camera/control,
and Wan 2.2 14B high/low-noise workflows are outside this change's scope.
Models, media and machine-local result files are not part of the commit.
