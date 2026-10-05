# H3 / VDN ComfyUI API benchmarks

This is the workspace harness used for the local RTX 5090 comparison, with output/workspace roots made configurable. It is not a generic model downloader. Existing ComfyUI, model files, native pipeline metadata, and ffmpeg/ffprobe are required. Model weights and raw videos are deliberately excluded.

## Dependencies and layout

Start with `fix/comfyui-h3-support`. To reproduce final optimized results, also integrate `fix/comfyui-memory`, `perf/h3-runtime`, `fix/h3-mxfp8-loading`, and `fix/diffusion-generation-time`. Measurements in `evidence/` were collected on the combined workspace before splitting; they are not newly collected numbers for this branch alone.

Set `H3_BENCH_WORKSPACE` to a directory containing `sglang/`, `ComfyUI/`, `models/`, and `logs/comfyui-server.log`. Set `H3_BENCH_SUITE` to a fresh results directory and create it. Both the server and caller must inherit the same suite variable. Install requirements into the existing SGLang environment. Put `custom_nodes/H3Benchmark` into ComfyUI's custom_nodes directory (a symlink is sufficient). The fingerprint records this installation as well as runtime sources.

The native harness expects `models/H3-native-benchmark/{FL2VA,Ref2VA}/{text_encoder,tokenizer,processor,video_vae,audio_vae}` metadata, `models/VDN-H3-native-benchmark`, and the checkpoint filenames used by the supplied workflows under `ComfyUI/models`. Reference requests use `ComfyUI/input/h3_validation_reference.png`. Populate these from your existing local model setup; no automatic download is performed.

## Run through the public API

Start ComfyUI on localhost:8188 with `--disable-cuda-malloc --cache-classic --preview-method none`. Set both `PYTORCH_ALLOC_CONF` and `PYTORCH_CUDA_ALLOC_CONF` to `backend:native`. Disable `SGLANG_DIFFUSION_STAGE_LOGGING`, `SGLANG_DIFFUSION_SYNC_STAGE_PROFILING`, and `SGLANG_PERF_LOG_DIR` for ordinary throughput runs. Do not have concurrent requests on the single GPU.

```bash
export H3_BENCH_WORKSPACE=/path/to/workspace
export H3_BENCH_SUITE=/path/to/fresh-results
mkdir -p "$H3_BENCH_SUITE"
python bench-h3-comfy.py --case t2av --workflow workflows/t2av_sdpa.json --runs 7 --plain
python summarize-h3-performance.py
```

The harness submits public `/prompt`, subscribes to `/ws`, and reads `/history`. It records API wall time, actual workflow/seed, events, media metadata, source fingerprints and NVML samples. It refuses existing per-run directories and a busy queue. It retains first-run media, deletes only its own later output files, and requires a 100 GiB free-disk reserve. Change the case name for a new group; do not reuse result names. The supplied ordinary workflows omit measuring nodes. Diagnostic latent comparisons require workflows explicitly containing H3BenchmarkModel/Latent; do not describe ordinary throughput results as synchronized stage profiles.

For fixed-prompt T2AV, start the server/worker with `SGLANG_KITCHEN_INT8_MAX_ROWS=0` and use `workflows/t2av_fast_attention_unsplit.json`. Existing workers must restart to read this variable. This is an environment variable, not an SGLDOptions field. For changing prompts, use `t2av_fast_gpu_encoder.json --changed-prompt` with default INT8 splitting. These two combinations were separately tested and must not be stacked without another validation.

## Native full-pipeline comparison

Stop ComfyUI and release its worker GPU memory before starting the native group. Use the same allocator/backend/environment. Native initialization is recorded separately.

```bash
python bench-h3-direct.py --case t2av --label t2av_native --runs 7 --backend sage_attn_3
python bench-h3-direct.py --case vdn --label vdn_native --runs 7
python summarize-h3-performance.py
```

For the matched unsplit T2AV native run also set `SGLANG_KITCHEN_INT8_MAX_ROWS=0`. For VDN online MXFP8 add `--quantization fp8`; on the tested SM120 device this resolves to MXFP8. This path needs the loader fix. The timing harness asserts positive returned timing and thus requires the timing fix.

## Results and limits

864x480, 107 frames at 24 fps; H3 20 NFE and VDN 8 NFE. Seven requests per formal group: cold, additional warmup, five hot; use the median of the five hot requests. CUDA-synchronized diagnostic runs are separately marked. Native production stage timers are asynchronous, so use independently measured full API wall time for comparisons.

| Scenario | Official ComfyUI | Native API | Integrated API |
|---|---:|---:|---:|
| T2AV Sage3 / integrated-native unsplit INT8 | 36.671 s | 37.192 s | 36.661 s |
| Ref2AV small reference / SDPA | 52.173 s | not size-matched | 53.505 s |
| VDN BF16 | no equivalent hybrid implementation | 40.434 s | 39.857 s |
| VDN MXFP8 | no equivalent hybrid implementation | 26.962 s | 26.338 s |

ComfyUI caches unchanged conditioning while native calls re-encode, so full API deltas are not pure IPC overhead. Native Ref2AV enlarges references to 3584x2048; compare it only with the separately matched large-reference groups in summary.csv. Cross-engine trajectories differ, especially for large references, and their exact cause remains unresolved. Bit-exact same-backend unsplit checks do not establish Sage/SDPA equivalence. MXFP8 changes trajectories substantially. Finite latents and representative frames are not full motion/audio quality assessments.

The earlier eight-hour soak completed 1201 requests before the latest performance fixes; it is not an eight-hour validation of these new commits. The final ordinary workflow groups, source changes and test evidence are recorded separately. No concurrency/scaling claim is made.
