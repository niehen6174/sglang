# H3 CPU-frame delivery configuration

This example selects an existing SGLang frame-delivery option.
It does not change runtime defaults, model forward code, or quantization kernels.
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
python examples/usage/h3_cpu_frame_delivery/run.py \
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

## Evidence and limits

The recorded same-SGLang control used one warmup and three formal requests.
The primary endpoint is complete CPU media/image readiness, including required
generation and transfer, with the common encoder/PNG sink outside the timer.
These are instrumented deployment timings, not pure GPU timings.

The full-shape warmup used 2 steps; each formal request used 50 steps.
Baseline mean: **304.666886 s**; CPU-frame mode: **302.882711 s**;
observed reduction: **1.784175 s**. All three formal common MP4 files are
byte-identical. Warmup and first-formal complete video/audio arrays are also
byte-identical; later formal raw arrays were not separately saved.

The worker-finalize-to-client-ready window decreased from 2.913497 s to
0.909825 s, with an additional approximately 0.220541 s GPU conversion inside
finalize. The decoded FP32 payload is approximately 1.536 GB versus 0.384 GB
for uint8. These stage windows are not independent pure-IPC measurements.

This is a caller configuration correction for applications needing CPU frames,
not a new engine implementation or global-default change. The earlier reference
resize mismatch is separate; this experiment holds the reference workload fixed.

Raw samples and artifact hashes are in [evidence.json](evidence.json). Fixed
order, three formal samples and unlocked clocks limit inference. The new CLI
was checked on CPU; it has not itself been rerun on GPU. Recorded controls use
the same configuration values. No strict equal-quality cross-engine ranking is
claimed.
