# [MiniMax-H3] Store only one Ulysses rank's Q/K/V/gate heads

> Backup branch note, not a PR body yet. Delete this file before opening a PR.

## Motivation

Under Ulysses sequence parallelism every rank holds the full Q/K/V and
compression-gate projections, projects all 56 heads over its own row shard,
then trades heads for sequence in an all-to-all. Each rank only ever attends
with `heads / ulysses_degree` of them. At the released shape (50 blocks,
hidden 5376, 56 x 128 heads) the replicated projections are 14.36 GiB per
rank, of which 10.77 GiB is heads this rank never uses.

## Modifications

- **`shard_ulysses_head_projections`** (`models/dits/minimax_h3.py`): keeps one
  Ulysses rank's rows of `qkv_proj` and `to_gate_compress` at
  `post_load_weights`. Declines on anything that is not a plain unquantized
  BF16/FP16/FP32 DiT block, so an unsupported layer stays on the
  replicated-weight path rather than being approximated. `float8_e4m3fn`
  reports `is_floating_point`, so the guard is an explicit dtype allowlist plus
  an `UnquantizedLinearMethod` check: block-FP8 holds one `weight_scale_inv`
  row per `weight_block_size[0]` weight rows, and slicing the weight without
  the scale mis-scales silently.
- **`project_head_sharded`**: gathers the rows once and projects them in one
  GEMM, so the output is already attention's layout.
- **`LinearBase.ulysses_head_shard`** (`layers/linear.py`) records `(rank,
  world)` for a layer narrowed this way. `BaseLayerWithLoRA._head_shard_window`
  composes that second slice onto the TP offset in `slice_lora_b_weights`, so
  an adapter at the checkpoint's full head width lands on the rows the base
  weight kept. Without it a LoRA delta is 21504 rows against a 5376-row base.
- **`SGLANG_DIFFUSION_MINIMAX_H3_HEAD_SHARD`** (`envs.py`): unset lets the
  preconditions decide, `0` keeps the replicated path. The separate
  ring/all-gather flag is gone; that choice follows from CUDA and
  graph-capture state, not from the user.

## Accuracy

`tools/bench_h3_head_sharded_ulysses.py` runs on 4 real ranks and compares
head-sharded Q/K/V/gate against the replicated path plus its input all-to-all,
once without LoRA and once with a full-width adapter. All four ranks match
bit-exactly: narrowing output rows leaves the GEMM's reduction over `hidden`
untouched. The LoRA round carries a guard that the adapter actually moves the
baseline, without which both sides would agree with LoRA silently disabled.
Removing `_head_shard_window` makes all four ranks fail with
`tensor a (5376) must match tensor b (21504)`.

## Speed and memory

4xH200, SP4, one request of 1344x768 / 5 s / 124992 tokens, seed 1001.
Per-stage times from the server log, first request dropped, mean of the rest.

| | denoise | denoise+decode | GPU used /card |
|---|---:|---:|---:|
| **FastH3 4-step, VSA 0.9** | | | |
| replicated | 3.610 s | 4.871 s | 89.6 GiB |
| head-sharded | 3.433 s | 4.697 s | 81.4 GiB |
| | **-4.9%** | **-3.6%** | **-8.2 GiB** |
| **base H3 + turbo LoRA (8 evaluations), FA** | | | |
| replicated | 10.368 s | 11.628 s | 86.8 GiB |
| head-sharded | 10.261 s | 11.521 s | 86.1 GiB |
| | **-1.0%** | **-0.9%** | **-0.7 GiB** |

Run-to-run standard deviation is 3-25 ms against deltas of 107-177 ms.

### Where the gain comes from, and why VSA gets more

Per layer the sharded path loses 0.190 ms on GEMM shape and 0.135 ms because
all-gather is slower than all-to-all at equal volume, and wins 0.664 ms by not
needing the destination-major QKV repack that all-to-all requires. Net -0.268
ms/layer, which is the FA number.

VSA adds a second win: with the compression gate active the replicated path
runs a *second* collective for it, and the sharded path computes the gate
locally from the rows it already gathered. Communication volume is

```
ulysses_degree * hidden / ((3 + gate) * heads * head_dim)
```

which for H3 at SP4 is 0.75 with the gate and exactly 1.00 without. The 0.75
depends on H3's `hidden / (heads * head_dim) = 0.75`; a model whose attention
width equals its hidden size would be at 1.00 with a gate and 1.33 without,
i.e. this moves more data, not less.

### Memory caveat

The sharding frees 10.77 GiB (VSA) and 8.08 GiB (LoRA) of weights by its own
accounting, but measured occupancy falls 8.2 GiB and only 0.7 GiB. The LoRA
gap is unexplained; `--lora-merge-mode auto` likely holds a second copy that
absorbs the saving. Anyone adopting this for memory should confirm on their
own configuration.

## What was measured and rejected

- **Ring over the hidden shards, projecting each as it arrives.** The first
  implementation. Needs a copy per shard per tensor into a shared workspace,
  16 copies per layer: 1.8% *slower* on VSA and 8.6% slower on dense FA.
- **Async gather overlapped with this rank's own rows.** The join and the
  narrower GEMMs cost more than the gather they hide: 4.2% instead of 4.9% on
  VSA, and 0.3% slower on FA.
- **Removing the destination-major QKV repack.** Timed in place at 0.21
  ms/call, 400 calls per request, 0.8% of denoise. Not worth a kernel change.

## Scope of validation

SP4, single node, one request shape, two backends. SP8 has not been run; the
volume formula says it moves 1.5x the bytes there, so the preconditions should
probably gate on it before anyone deploys at that degree. Cross-node is
untested.

`docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx` is not updated yet; this
changes per-rank resident memory and the ulysses-degree guidance.
