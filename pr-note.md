# PR note: H3 runtime performance

Depends on `fix/comfyui-h3-support` at `801ef720efab31514f32d57d82adfda4742e0df6`. Review only the diff against that branch; merging the prerequisite first avoids unrelated feature changes in this review.

- Preserve CUDA allocator caches for noise-prediction responses while retaining existing cleanup for other response types.
- Precompute RoPE once per integrated sampler run, with the existing session-state lifecycle.
- Keep legal packed QKV views for Sage2/3 when their final stride is one; materialize unsupported strides.

Validation on the complete final workspace: 266 passed / 1 skipped / 3 expected failures in the related regression suite, and eight actual GPU Sage stride cases passed. Same-seed default H3 T2AV, Ref2AV and VDN latent comparisons were bit-identical before/after. End-to-end gains from the default-path code optimizations were modest (roughly 1–2%); do not attribute the ~30% fast-config gain to this patch alone.

The larger T2AV gain additionally uses Sage3 and `SGLANG_KITCHEN_INT8_MAX_ROWS=0`, a tested RTX 5090 deployment setting; this patch does not change that global default. Sage differs numerically from SDPA. No new eight-hour soak of this patch was performed.

