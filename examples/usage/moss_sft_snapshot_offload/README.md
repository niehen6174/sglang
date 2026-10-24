# MOSS SFT snapshot-offload configuration

This example selects an existing SGLang memory-residency strategy.
It does not change runtime defaults, model forward code, or quantization kernels.
The baseline is SGLang commit `49933325cca5cfa32e03e06843774fa70fec64b5`.
Run inside its NVIDIA CUDA diffusion environment with local model components and
FFmpeg installed. The measured environments used PyTorch `2.14.1+cu130`,
diffusers `0.37.0`, transformers `5.19.0`, and Triton `3.8.0`.

`run.py` prints the complete configuration by default, without importing SGLang,
loading weights, creating output directories, or generating. Add `--run` to
generate once. This example is not a latency benchmark: initialization, native
file delivery, and encoding are not separated by its command duration.

## MOSS SFT: repeated component offload

```bash
python examples/usage/moss_sft_snapshot_offload/run.py \
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

The recorded same-SGLang control used one warmup and three formal requests.
The primary endpoint is complete CPU media/image readiness, including required
generation and transfer, with the common encoder/PNG sink outside the timer.
These are instrumented deployment timings, not pure GPU timings.

Warmup and formal requests all used 50 steps. Baseline mean:
**25.395641 s**; snapshot mean: **24.895931 s**; observed reduction:
**0.499709 s**. Warmup and all three formal PNG files and decoded pixels are
identical. The approximately 23.1314 s DiT loop is unchanged.

TE-to-DiT transition intervals were 0.528/0.528/3.234 s with component offload,
versus 0.270/0.256/0.256 s with snapshot offload. Snapshot still had a
2.525-second DiT-to-VAE delay in one sample. The three-sample mean reduction
does not establish that all latency variance is fixed or guarantee a stable
0.5-second gain. The intervention changes residency policy with pinning enabled
in both variants; it is not a pinning-only experiment.

This is an opt-in deployment strategy using existing implementation, not a
universal offload-default change. Snapshot host/pinned-memory requirements must
be considered separately from GPU capacity.

Raw samples and artifact hashes are in [evidence.json](evidence.json). Fixed
order, three formal samples and unlocked clocks limit inference. The new CLI
was checked on CPU; it has not itself been rerun on GPU. Recorded controls use
the same configuration values. No strict equal-quality cross-engine ranking is
claimed.
