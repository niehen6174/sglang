# PR note: ComfyUI model-patcher lifetime

Base: `edee4308bcd7204d550234189d52a4cc25d93348`. Independent of the H3 feature/performance branches.

Set the cloned patcher's `parent` to the live source patcher, matching ComfyUI's model tracking contract. This allows an expired temporary clone to resolve to its live parent instead of leaving stale loaded-model weak references. Add clone-depth regression coverage (requires ComfyUI installed).

The local investigation observed 462 dead-reference warnings before the fix and zero after it; the matched 12-request check had identical latent hashes and stable USS (about 23 MiB change). The earlier decoded-video retention was also affected by ComfyUI's RAM cache policy; the benchmark launcher uses `--cache-classic`, which is a deployment setting, not a modification of ComfyUI core.

The original eight-hour soak predates these changes. Do not claim a fresh eight-hour soak for this branch.

