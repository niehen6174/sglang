# PR note: H3 / VDN API benchmark harness

Depends on `fix/comfyui-h3-support`; final optimized reproduction additionally requires the other four fix/performance branches. Keep this tooling review separate from runtime patches.

Save the actual ComfyUI prompt/history/WebSocket and native DiffGenerator benchmark harnesses, diagnostic measurement nodes, seven ordinary validated API workflows, small timing/selection evidence, and reproduction instructions. Workspace and suite roots are environment-configurable; model weights, generated media, tokens, raw machine logs and automatic model downloads are not committed.

The README explains cache/timing boundaries, matching reference sizes, cold/warm sample counts, numeric/quality limitations and branch dependencies. Existing GPU measurements belong to the combined tested workspace; only syntax/import/schema checks are rerun after packaging. No new long soak is claimed.

## H200 follow-up (2026-10-09)
Harness extended for 2x H200: explicit multi-GPU degrees and offload toggles in the native runner (validated against ServerArgs, effective values saved), physical NVML mapping for every visible GPU plus foreign-process snapshots, configurable endpoint/log for multiple ComfyUI instances, and a media comparator. Adds a small 2x H200 summary (evidence/h200-20261009-summary.md). Only syntax/import checks are rerun for the tooling itself; measurements come from the combined validation workspace.
