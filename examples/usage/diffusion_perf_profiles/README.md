# Opt-in diffusion performance profiles

These examples select existing SGLang delivery and memory-residency options.
They do not change runtime defaults, model forward code, or quantization kernels.
The baseline is SGLang commit `49933325cca5cfa32e03e06843774fa70fec64b5`.
Run inside its NVIDIA CUDA diffusion environment with local model components and
FFmpeg installed. The measured environments used PyTorch `2.14.1+cu130`,
diffusers `0.37.0`, transformers `5.19.0`, and Triton `3.8.0`.

`run.py` prints the complete configuration by default, without importing SGLang,
loading weights, creating output directories, or generating. Add `--run` to
generate once. This example is not a latency benchmark: initialization, native
file delivery, and encoding are not separated by its command duration.

## H3: CPU frame consumers

Use this profile only when your Python application needs CPU RGB frames as well
as the generated file. `return_frames=True` selects native GPU uint8 conversion
before returning frames, avoiding large FP32 frame transfers and subsequent
client conversion. Set it in `sampling_params_kwargs`, not engine kwargs.

```bash
python examples/usage/diffusion_perf_profiles/run.py \
  --profile h3-ref2va-cpu-frames \
  --model-path /models/h3/base-bf16 \
  --reference-image /inputs/reference.png \
  --prompt 'A small orange cat walks slowly on a sunny beach.' \
  --output-dir outputs/h3-frames
```

The measured profile uses one H200 GPU, FA, module-offloaded DiT/text encoder,
resident VAE, exact quality, 50 steps, a 5-second 768-short-edge target, and
explicit `reference_resize_mode=max`. The reference workload was 1024 x 1024
with 1024 reference tokens. Use a matching reference image for that workload.

To compare, add `--baseline`; only `return_frames` changes back to false.
Both variants explicitly set `return_file_paths_only=False`. Changing this
return contract changes raw samples from floating-point tensors to uint8 THWC
frames; consumers must use the selected contract. Audio keeps its native path.
In a Python consumer, read `result.frames`, `result.audio`, and
`result.output_file_path` from the returned `GenerationResult`; this CLI prints
only the file paths.

**File-only consumers should keep the native paths-only default.** Setting
`return_frames=True` is not needed for that path. Do not substitute
`return_raw_frames`: that transport has a different audio contract.

## MOSS SFT: repeated component offload

```bash
python examples/usage/diffusion_perf_profiles/run.py \
  --profile moss-sft-snapshot \
  --model-path /models/MOSS-Image-SFT \
  --prompt 'A small orange cat beside a blue ceramic vase, detailed photograph.' \
  --output-dir outputs/moss-snapshot
```

This selects the existing `component_residency` modes `dit=snapshot-offload` and
`text_encoder=snapshot-offload`, with `pin_cpu_memory=True`. The explicit modes
take precedence over the retained legacy CPU-offload flags. Add `--baseline`
to omit the snapshot modes while preserving all other engine/request settings.

The ordinary component strategy moves the module to and from CPU around each
use. Snapshot offload retains a CPU backing copy and can pin that snapshot,
avoiding repeated weight copies back to CPU. Merely setting
`pin_cpu_memory=True` does not select the snapshot strategy. Account for the
extra host snapshot and pinned-memory budget before choosing it.

The measured profile used one 32GB RTX 5090, SDPA, BF16 DiT, resident FP32 VAE,
1024-square output, CFG 3, 50 steps/100 DiT forwards, and the native two-pass
unconditional-skip policy. This is component offload, not layerwise offload.
It requires each complete offloaded component to fit when active.

## Evidence and limits

Each row below has one warmup and three formal samples. The H3 warmup used
2 steps with the full output shape; its formal requests used 50. MOSS used
50 steps for warmup and formal requests. Measurements include necessary native
generation, transfer and complete CPU media/image readiness; the common encoder
or PNG sink is outside the primary timing. They are not pure GPU timings.

| Profile | Baseline mean | Opt-in mean | Output checks |
| --- | ---: | ---: | --- |
| H3 Ref2VA/H200 | 304.666886 s | 302.882711 s | All three formal common MP4 files byte-identical; warmup and first-formal complete video/audio array bytes equal |
| MOSS SFT photo/5090 | 25.395641 s | 24.895931 s | Warmup and all three formal PNG files and decoded pixels identical |

Raw times and artifact hashes are in [evidence.json](evidence.json). These
observations apply to the measured fixtures and fixed source. Runs used fixed
order and unlocked clocks. MOSS retained a 2.525-second tail delay in one sample;
the snapshot strategy did not eliminate all host-side variance. Neither profile
establishes a strict equal-quality cross-engine ranking.

LTX GPU uint8 delivery, forced H3 norm fusion, and MOSS QKV packing are excluded:
whole-output numerical equivalence was not certified for those experiments.
