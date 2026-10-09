# CFG split for ComfyUI integrated mode — design (parked)

Status: **implemented and CPU-tested, not GPU-validated, not on any feature branch.**
- Code: branch `wip/comfyui-cfg-split` (= `archive/comfyui-qwen-image21-full`, 56ef89ae39) in the fork worktree.
- The CFG-split part is these commits:
  - `412d98f9b2` [diffusion] fix: import video_sparse_attn without a visible GPU (only needed for CPU-only tests);
  - `7e13ae036d` [diffusion] feat: ComfyUI integrated-mode CFG split across two SGLD ranks (generic plugin + worker);
  - `56ef89ae39` [diffusion] feat: Qwen-Image 2.1 CFG split in ComfyUI integrated mode.
- The slimmed `feat/comfyui-qwen-image21` does not contain it. It rejects `enable_cfg_parallel` in comfyui mode with a clear message instead.

## Problem

In integrated mode ComfyUI owns the sampler loop and does classifier-free guidance itself.
- Each step evaluates the positive and the negative cond and combines them in `cfg_function`.
- The SGLD worker only sees individual DiT calls.
- SGLang's own CFG parallelism splits the *worker's* CFG branches (`do_classifier_free_guidance`). In comfyui mode the worker has none, so `enable_cfg_parallel` makes every CFG rank recompute the same call. Before it was rejected, this was measured on 2× RTX 5090: output bit-identical to 1 GPU, same speed, twice the memory.

Goal: with 2 GPUs, run a step's cond and uncond concurrently, one per GPU, so a CFG > 1 step costs about one DiT forward instead of two. Only the velocities need to be gathered.

## ComfyUI batching facts (comfy master a4b5a045)

- `comfy/samplers.py` `_calc_cond_batch` groups the conds of a step by hooks. Within a group it concatenates conds whose `can_concat_cond` holds:
  - `input_x` shapes match;
  - the same `control` and `patches` objects;
  - every model cond's `can_concat`.
- Qwen-Image (and 2.1) `c_crossattn` is a `CONDRegular`, whose `can_concat` requires identical shapes. Positive and negative prompts with different token counts (the normal case) therefore run as **two sequential B=1 `apply_model` calls**. Prompts with equal token counts run as **one B=2 call**.
- Memory-fit batching can also cut a batch back to B=1.
- LTX pads text to a fixed length, so its cond and uncond batch as B=2.
- cfg == 1 (with the cfg1 optimization) passes `[cond, None]`: only one call.
- Hook point: `WrappersMP.CALC_COND_BATCH` (`comfy/patcher_extension.py`, invoked from `_calc_cond_batch_outer`, `comfy/samplers.py:217`). It sees all conds of one step. SGLD patcher wrappers reach it because `sampler_helpers` merges `model.wrappers` into `model_options["transformer_options"]["wrappers"]`.

## Implemented design

### Plugin: `apps/ComfyUI_SGLDiffusion/executors/cfg_split.py` and `executors/base.py`

**Enabling.**
- Enabled by `SGLDOptions(enable_cfg_parallel=True, num_gpus=2)` for executors with `supports_cfg_split = True`.
- `cfg_split_ranks_for()` validates before the worker starts: num_gpus = 2, cfg degree 2, no sp/ulysses/ring/tp.
- The generator installs a `CALC_COND_BATCH` wrapper on the SGLD patcher only in that case, so a single GPU runs the old code path unchanged.

**The wrapper reproduces `_calc_cond_batch` for the plain case only:**
- exactly two conds of one entry each;
- no `area` / `mask` / `hooks` / `control` / `gligen` / `default`;
- no `model_function_wrapper` / `multigpu_clones`;
- no `APPLY_MODEL` / `DIFFUSION_MODEL` wrappers;
- both conds inside their timestep range.

It uses ComfyUI's own `get_area_and_mult`, `cond_cat`, `apply_hooks` + `merge_nested_dicts` and `prepare_state`. It sets `cond_or_uncond` / `uuids` / `sigmas` and accumulates exactly as ComfyUI does (`zeros + out*mult`, `ones*1e-37 + mult`, divide). Anything else returns None and ComfyUI's path runs.

**Record / replay.** Each cond's real `model.apply_model` runs twice:
1. **Record.** `executor.forward` packs the DiT inputs with the model adapter and returns `adapter.placeholder_output(x)` (zeros).
2. **One request.** `execute_cfg_split` sends both calls. Each call's `Req` fields are captured by running the adapter's `fill_req` on a recorder and stored in `req.extra["comfyui_cfg_split"]`.
3. **Replay.** `executor.forward` returns call `i`'s velocity, so `calculate_input` / `calculate_denoised` remain ComfyUI's.

**Per-rank conditioning.** A cond's conditioning is sent once per run *per rank that runs it* (`_sent_cond_slots`). It is resent if a cond moves to another rank. Unsplit calls mark it as sent to all ranks.

### Worker: `runtime/pipelines_core/stages/comfyui_cfg_split.py`

**Routing.**
- Reuses SGLang's CFG-parallel group: `cfg_parallel_degree = 2`, and rank 0 already broadcasts every comfyui request over the CFG group (NCCL for tensors).
- `ComfyUICFGSplitStage` wraps the model's ComfyUI step stage. CFG rank `r` applies call `r` to its batch and runs the unchanged model stage.
- Velocities are all-gathered over the CFG group (tensor, or list of tensors such as LTX `[video, audio]`), and rank 0 replies with one per call.
- Requests without the key run unchanged on every rank.

**Per-rank cond state.** Sessions and Qwen's per-cond prefix K/V live on the rank that runs the cond.

**Collective scope fix.** Qwen's prefix-cache placement agreement had to move from the world group to the TP/SP groups. Otherwise CFG ranks, which create cond state at different times, deadlock.

**Error exchange.** Before the gather, ranks `all_gather_object` their step error on the gloo group. A failure on one rank then raises on both, instead of leaving the peer blocked in the NCCL all-gather until the timeout.

**Qwen hook.** `QwenImage21Pipeline.create_comfyui_stages` wraps its stage under `enable_cfg_parallel` and rejects sp/tp.

### Tests (CPU, `test/unit/test_comfyui_cfg_split.py`, 39 cases)

- Which steps split and which fall back.
- Option validation.
- Conditioning is resent when a cond moves to another rank.
- **Bit-equality with the real `comfy.samplers._calc_cond_batch`** (row-wise mock DiT) for:
  - different-length conds (two B=1 calls in ComfyUI);
  - equal-length conds (B=2 in ComfyUI);
  - cond strengths;
  - a latent batch of 2.
- One worker request per step; conditioning dropped on the second step.
- Worker routing for rank 0 and 1, gather, requests without the key, call-count mismatch, error exchange, list velocities.

## Why it is parked

- **It couples to ComfyUI internals.** It re-implements part of `_calc_cond_batch` (conds-by-hooks, accumulation, the `transformer_options` keys it injects) and must be kept in sync with ComfyUI. The upstream function itself carries a "keep in sync with `_calc_cond_batch_multigpu`" note.
- **Size.** About 480 non-test lines (plugin `cfg_split.py` and `base.py` record/replay; worker stage; Qwen glue), plus about 480 test lines. It touches common code (`comfyui_mode.py`, the worker stage) and the executor contract (`placeholder_output`).
- **Not GPU-validated.** Neither the speedup nor correctness on two real ranks has been measured. The GPUs were owned by the LTX agent at the time.

## Open risks

- **Equal-length prompts.** ComfyUI would batch them as B=2. The split runs B=1 per rank, so outputs can differ from single-GPU at batch-kernel rounding level. This is fine semantically but is not bit-identical.
- **Record/replay assumptions:**
  - it assumes `apply_model` is deterministic in its DiT-call order and count;
  - `APPLY_MODEL` / `DIFFUSION_MODEL` wrappers are excluded, because they would run once per pass;
  - an extension that calls the DiT more than once per `apply_model` would trip the replay order check (a clear error, not wrong output).
- **The gate excludes the common features** (area conds, ControlNet, hooks, GLIGEN). Those fall back silently to the redundant 2-rank path: correct but not faster. This is logged once per reason.
- **Communication.** Both calls' inputs are broadcast to both ranks. On host-routed NCCL (no P2P, cross-NUMA) the broadcast, the per-step gloo error exchange (`all_gather_object`) and the velocity all-gather may eat a noticeable part of the gain. Not measured.
- **Load balance.** Step 1 is uneven: each rank prefills only its own cond's prefix. A long negative vs a short positive prompt also makes the ranks finish at different times.
- **Memory.** Each rank holds the full DiT (no TP sharding) plus its cond's prefix cache. That is the same per-GPU memory as 1 GPU, but ComfyUI's text encoder and VAE stay on GPU0.
- **Reply shape.** The OutputBatch carries a *list* of velocities. Any scheduler/worker path that assumes a tensor `noise_pred` for comfyui requests would break; none was found, but it is untested on a real worker.
- **Interactions not covered:** LoRA switching between steps of a split run (each rank sets LoRA through the normal path), and `QwenImage21Cache` options.

## What is needed to finish

1. Rebase `wip/comfyui-cfg-split` onto the slimmed feature branch. The prefix-cache code it touches (`_free_memory` agreement) was simplified there; keep the agreement on the TP/SP groups.
2. GPU test plan (2× RTX 5090; GPUs 0,1):
   - **DiT parity:** the split request vs two 1-GPU calls on identical inputs (cos / rel L2), for different-length and equal-length prompts, 1-ref edit, and a latent batch of 2.
   - **End to end via `/prompt`:** official template with CFG 4 and a negative prompt (different lengths), 1024², 25 steps. Compare 1-GPU integrated vs split: PSNR plus side-by-side.
   - **Speed:** per-step time (tqdm) vs 1 GPU running cond and uncond sequentially. Target close to 2× on the DiT part. Also report warm and cold prompt time and the cost of broadcast, error exchange and gather (nsys or timers).
   - **Proof that both ranks work:** per-GPU busy and VRAM traces (`nvidia-smi -lms 100`).
   - **Fallbacks:**
     - CFG = 1: single call, both ranks redundant;
     - an area or ControlNet cond: ComfyUI path;
     - an odd-sized latent (no SP involved, so it should split fine);
     - the `QwenImage21Cache` node present.
   - **Failure:** force an error on rank 1 (e.g. drop conditioning) and check that both ranks raise and nothing hangs.
   - **LTX:** cherry-pick `7e13ae036d` onto `feat/comfyui-ltx2`, set `supports_cfg_split` on the LTX-AV executor, and wrap its step stage. Its cond and uncond are B=2, so the wrapper must still split them into two calls; it does, because it runs each cond separately.
3. Decide whether the speedup on this host-routed hardware justifies the coupling to ComfyUI internals. If it does, consider proposing a ComfyUI-side hook (e.g. an async/batched `apply_model` callback) so the plugin need not re-implement `_calc_cond_batch`.
