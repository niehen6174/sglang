Base: `edee4308bcd7204d550234189d52a4cc25d93348`. Independent of the H3 ComfyUI feature branches.

# PR note: Comfy NVFP4 block scales under tensor parallelism

Comfy serializes NVFP4 block scales swizzled in 128x4 tiles. TP loading narrowed that tensor directly: exact along output rows (column-parallel) but mixing blocks across rows for input-dim shards. On the MiniMax-H3 Qwen3-VL NVFP4 encoder, layer-10 `down_proj` shards dequantized with 43-46% relative error; every native multi-GPU H3 layout that folds the encoder (TP2, Ulysses2, auto SP) generated a coherent video unrelated to the prompt (PSNR 7.7 dB vs single GPU). BF16 encoder under the same TP2 was correct, isolating NVFP4. Fix: un-swizzle before the TP narrow and swizzle the shard back, so `apply()` and the SM100 kitchen path keep their layout. Real-layer shards now exact (rel_err 0.000000). Integrated ComfyUI mode was not affected (ComfyUI encodes text itself).
