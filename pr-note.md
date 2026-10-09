# Fix ComfyUI multi-LoRA state and dynamic composition

Base: PR #43186, commit `fa617cfb6f388d2bb160be38c7bdcb0887f49595`.
Branch: `codex/fix-comfyui-multi-lora`.

## Problem

Returning the cloned MODEL fixes chained loader outputs, but applying LoRAs during loader execution leaves shared SGLD runtime state dependent on ComfyUI cache hits. After generating with A+B, requesting a cached A-only MODEL can still use A+B. Removing every loader or using two MODEL branches can likewise select stale adapters. Quantized models also reject multiple dynamic LoRAs per target.

## Change

Store the complete desired adapter list on each cloned MODEL and synchronize the runtime at sampler entry, including resetting to the base model. Record the runtime configuration only after successful application. Preserve clone parent links, equivalent to the existing independent `fix/comfyui-memory` fix.

Dynamic linear wrappers now accumulate every adapter using its own rank, alpha, strength, projection slices and output offset. Expand overlapping targets before grouping; clear each layer on its first matching adapter; restore merged base weights before changing metadata. Merge-cache installation runs after all matching adapters, including when the final adapter does not cover the layer.

## Validation

- Final source: 464 unit tests passed, including 400 old/new state transitions across merged and dynamic modes, mixed ranks/alphas, runtime scales, offsets, partial targets, cache merging, failed loads and projection shard slicing.
- Native worktree source through ComfyUI `/prompt`: 13 Z-Image and 9 H3 INT8 requests. 21 successful requests and one intentionally corrupt adapter failure, followed by successful base-model recovery.
- Latent SHA-256 comparisons: A+B to cached A, all adapters removed, and zero strength exactly match their corresponding references for both models. H3 video and audio both match. Z-Image additionally checks A+B+C to cached A+B and two MODEL branches. Combined adapters differ from each single adapter, demonstrating both contribute.
- PR #43163 applies cleanly after its full #43109 prerequisite; 25 combined adapter/session tests passed. GPU batch splitting with #43163 was not exercised in this run.
- `git diff --check` passed.

GPU tests ran before the final merge-cache coverage fix; that cache-specific path is covered by the final CPU tests. Saved workflows, API histories, latent tensors, checksums, logs and runners: `/scratch/data/sgld_comfy/results/multi-lora-worktree-20261009`.

This is single-GPU correctness validation with short, fixed-seed workflows. It does not establish multi-GPU collective correctness, image quality, compilation behavior with every adapter count, or acceleration. Z-Image B/C use adapter subsets to exercise partial coverage, not independently trained styles.

## Reproduce

```bash
PYTHONPATH=/scratch/data/sgld_comfy/worktrees/comfyui-multi-lora/python:/scratch/data/sgld_comfy/ComfyUI /opt/sglang/bin/python /scratch/data/sgld_comfy/results/multi-lora-worktree-20261009/tools/run_unit.py
```

The original dirty workspace remains separate. No branch was pushed and no PR was created.

## H200 follow-up (2026-10-09)
Reviewed on 2x H200 with H3 INT8 through ComfyUI `/prompt` (same seed, latent SHA-256): base, A, A+B (INT8 dynamic, previously rejected), cached A after A+B, all removed and strength 0 match their references exactly; B alone differs from A and A+B. Added two fixes for upstream behavior also present here: reject batched MiniMax H3 latents with ComfyUI's "supports batch size 1" (the checkpoint is CFG-distilled; B=2 otherwise failed in the worker with an index_copy_ shape error), and raise the worker error before unpacking instead of a misleading noise_pred TypeError.
