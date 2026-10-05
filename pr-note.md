# PR note: H3 / VDN ComfyUI support

Base: `edee4308bcd7204d550234189d52a4cc25d93348` (upstream checkout used for the API validation). Includes the previously staged serialized INT8 ConvRot loading bridge associated with upstream PR #42121, plus the subsequent integration fixes. Compare against this base, not an unrelated fork branch.

## Changes
- Load supported H3 serialized INT8 checkpoints and validate incompatible quantization/LoRA combinations.
- Route integrated H3 through the native denoising-stage attention/cache/compile machinery; prepare sparse/cube and VDN hybrid metadata using the actual ComfyUI layout.
- Propagate and validate SGLDOptions, model identity, offload and warmup behavior; reject unsupported integrated quality modes and CUDA graph/offload combinations clearly.
- Implement real Sage3 packed-sequence execution and correct VDN configuration / example workflow options.

## Validation and limits
The complete local integration campaign recorded 125 API requests (101 successful, 22 expected rejections, 2 transport failures investigated), followed by focused regressions. Those totals describe the combined tested workspace, not an isolated rerun of this split branch. See the benchmark branch for methodology. VSA was not supported on RTX 5090 SM120. Official ComfyUI has no equivalent VDN hybrid branch. Cross-engine quality equivalence was not established.

This branch intentionally excludes allocator/RoPE/copy performance changes, model-patcher lifetime, MXFP8 staging, and returned timing fixes. The performance branch depends on this commit.

