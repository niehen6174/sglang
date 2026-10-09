# Support ComfyUI multi-image batches and opt-in native batching

Base: `codex/fix-comfyui-multi-lora` (`a443903da6`). The first additional commit imports OnePunchMonk's complete #43109 conditioning-cache fix, credited to its author.

ComfyUI sends a batch for multiple images and for concatenated CFG conditions. The existing path calls `.item()` on a timestep vector, records sequence lengths for only one sample, and passes the vector as a multi-step schedule. A two-image Flux workflow fails.

The default now sends one B=1 request per row, following #43163's fallback approach. `SGLDOptions.enable_native_batch=True` enables an actual batched request for Flux and Qwen-Image when timestep and Flux guidance are uniform. Preserve the full latent/conditioning batch, describe per-sample sequence lengths, size generators to the effective batch, and send one schedule timestep. Other input cases retain per-row execution. H3 nested inputs are unchanged.

Expose the existing `allow_bf16_reduced_precision_reduction` runtime option in SGLDOptions. It can reduce shape-dependent numerical drift but does not guarantee single-image/batched equivalence. The default keeps the upstream numerical policy.

A separate loader fix reads guidance presence from the Flux checkpoint header: Schnell has no guidance embedder and previously failed before its first forward.

Validation: 492 unit tests passed. 25 API requests against the final source passed using full Flux Schnell BF16, real text encoders, fixed per-row noise, 256x256, Euler 2 steps, single RTX 5090 and torch_sdpa. The default matches row references exactly. Native ordinary B=2 and CFG B=4 use one RPC per step. Changing another row and permuting rows preserve the expected outputs exactly. Multi-LoRA replacement/removal on B=2 matches same-batch references exactly.

Native BF16 batch output is not equivalent to independent B=1 output: this small test measured 10–15% relative latent RMSE, with similar drift in original ComfyUI. Layer diagnostics implicate shape-dependent matrix arithmetic; disabling low-precision reduction helps but does not eliminate the general issue. Default per-row execution preserves reference behavior. This does not establish image quality, all model variants, compilation configurations or multi-GPU correctness. Qwen-Image coverage is unit/interface validation only; no full Qwen GPU checkpoint was run.

Saved evidence, scripts, workflows and conclusions: `/scratch/data/sgld_comfy/results/comfyui-batch-20261009`. Early failed equivalence checks are preserved separately from the passing functional checks; no tolerance was relaxed.

Full native batch requires no additional #43163 patch on this branch: its per-row behavior is already incorporated. The branch includes the earlier multi-LoRA commits; review only commits after `a443903da6` for this change.
