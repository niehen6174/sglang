# PR note: H3 / VDN API benchmark harness

Depends on `fix/comfyui-h3-support`; final optimized reproduction additionally requires the other four fix/performance branches. Keep this tooling review separate from runtime patches.

Save the actual ComfyUI prompt/history/WebSocket and native DiffGenerator benchmark harnesses, diagnostic measurement nodes, seven ordinary validated API workflows, small timing/selection evidence, and reproduction instructions. Workspace and suite roots are environment-configurable; model weights, generated media, tokens, raw machine logs and automatic model downloads are not committed.

The README explains cache/timing boundaries, matching reference sizes, cold/warm sample counts, numeric/quality limitations and branch dependencies. Existing GPU measurements belong to the combined tested workspace; only syntax/import/schema checks are rerun after packaging. No new long soak is claimed.
