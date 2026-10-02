"""Benefit of head-sharded Ulysses projections on a 4-GPU FastH3 layout.

Mirrors the UniServe 4-GPU latency layout (one replica, Ulysses SP4, DiT
tensor-parallel degree 1) at the padded sequence lengths from the 4x GB200
writeup. This host is 4x H200, so the times are not the blog's GB200 numbers.
The measurement is only the part that changes: Q/K/V/gate projection, QK
RMSNorm + RoPE, the input collective, and the unchanged output all-to-all.
Sparse attention and the MLP are identical on both paths and are not included.

    python -m sglang.multimodal_gen.tools.bench_h3_head_sharded_ulysses
"""

from __future__ import annotations

import os
import subprocess
import sys

import torch

# Blog padding: 10s/1K -> 78080 rows, 15s/10K -> 125696 rows.
_SHAPES = (78080, 125696)
_HEADS = 56
_HEAD_DIM = 128
_HIDDEN = 5376
_ROPE_DIM = 96
_LAYERS = 50
_STEPS = 8
_WORLD = 4


def _worker() -> int:
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    init_distributed_environment(world_size=world, rank=rank, local_rank=rank)
    initialize_model_parallel(
        tensor_parallel_degree=1,
        sequence_parallel_degree=world,
        ulysses_degree=world,
        ring_degree=1,
    )

    failures: list[str] = []
    _check_exchange(device, world, rank, failures)
    if failures:
        for item in failures:
            print(f"FAILURE rank{rank}: {item}", flush=True)
        return 1

    print(f"rank{rank}: numerical check passed", flush=True)
    if rank == 0:
        print(
            f"head-sharded Ulysses vs replicated QKV all-to-all, "
            f"SP{world}, H={_HIDDEN}, heads={_HEADS}x{_HEAD_DIM}, "
            f"{_LAYERS} layers x {_STEPS} denoise steps",
            flush=True,
        )
    for seq_len in _SHAPES:
        baseline_ms, sharded_ms = _time_shape(device, world, rank, seq_len)
        if rank == 0:
            base_step = baseline_ms * _STEPS
            shard_step = sharded_ms * _STEPS
            delta = base_step - shard_step
            pct = 100.0 * delta / base_step
            print(
                f"S={seq_len}: baseline {base_step:.1f} ms/request, "
                f"head-sharded {shard_step:.1f} ms/request, "
                f"delta {delta:.1f} ms ({pct:.1f}%)",
                flush=True,
            )
    return 0


def _check_exchange(
    device: torch.device, world: int, rank: int, failures: list[str]
) -> None:
    seq_len = 256 * world
    for with_lora in (False, True):
        label = "lora" if with_lora else "plain"
        attn = _make_attention(device)
        if with_lora:
            _attach_lora(attn)
        x, rope = _inputs(seq_len, world, rank, device)
        reference = tuple(
            None if tensor is None else tensor.detach().clone()
            for tensor in _baseline_exchange(attn, x, rope)
        )
        if with_lora and not _lora_moves_output(attn, x, rope, reference):
            failures.append("lora adapter left the baseline unchanged")
            return

        # Production order: post_load_weights shards, then LoRA wraps the
        # already-narrowed layers; adapters still arrive at full head width.
        sharded = _make_attention(device)
        released = sharded.shard_ulysses_head_projections()
        if released <= 0:
            failures.append("head shards were not installed")
            return
        if with_lora:
            _attach_lora(sharded)
        got = sharded.project_head_sharded(x, rope)
        names = ("q", "k", "v", "gate")
        for name, expected, actual in zip(names, reference, got, strict=True):
            if actual is None or expected is None:
                failures.append(f"{label} {name} missing")
                continue
            diff = (expected.float() - actual.float()).abs()
            denom = expected.float().abs().mean().clamp_min(1e-3)
            rel = diff.mean() / denom
            if rel.item() > 0.02 or diff.max().item() > 0.5:
                failures.append(
                    f"{label} {name} rel={rel.item():.4f} max={diff.max().item():.4f}"
                )


_LORA_RANK = 8


def _attach_lora(attn) -> None:
    """Wrap qkv/gate in their LoRA layers with a full-head-width adapter."""
    from sglang.multimodal_gen.runtime.layers.lora.linear import wrap_with_lora_layer

    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    for name in ("qkv_proj", "to_gate_compress"):
        base = getattr(attn, name)
        hidden = base.weight.shape[1]
        rows = attn.inner_dim * (3 if name == "qkv_proj" else 1)
        layer = wrap_with_lora_layer(
            base,
            lora_rank=_LORA_RANK,
            lora_alpha=_LORA_RANK,
            snapshot_base=False,
        )
        assert layer is not None, name

        def rand(shape, scale=1.0):
            values = torch.randn(shape, generator=generator, dtype=torch.float32)
            return (values * scale).to(
                device=base.weight.device, dtype=base.weight.dtype
            )

        layer.lora_A = rand((_LORA_RANK, hidden))
        layer.lora_B = rand((rows, _LORA_RANK), 0.05)
        layer.disable_lora = False
        layer.strength = 1.0
        setattr(attn, name, layer)


def _lora_moves_output(attn, x, rope, with_lora) -> bool:
    """Without this the sharded and replicated sides agree for the wrong reason."""
    for name in ("qkv_proj", "to_gate_compress"):
        getattr(attn, name).disable_lora = True
    without = _baseline_exchange(attn, x, rope)
    for name in ("qkv_proj", "to_gate_compress"):
        getattr(attn, name).disable_lora = False
    return all(
        (a.float() - b.float()).abs().max().item() > 1e-3
        for a, b in zip(with_lora, without, strict=True)
        if a is not None and b is not None
    )


def _time_shape(
    device: torch.device, world: int, rank: int, seq_len: int
) -> tuple[float, float]:
    import torch.distributed as dist

    attn = _make_attention(device)
    x, rope = _inputs(seq_len, world, rank, device)

    def baseline() -> None:
        for _ in range(_LAYERS):
            _baseline_exchange(attn, x, rope)

    from sglang.multimodal_gen.runtime.layers.usp import _usp_output_all_to_all

    def sharded() -> None:
        for _ in range(_LAYERS):
            q, k, v, gate = attn.project_head_sharded(x, rope)
            _usp_output_all_to_all(q[None], head_dim=2)
            del k, v, gate

    baseline_ms = _elapsed_ms(baseline)
    attn.shard_ulysses_head_projections()
    sharded_ms = _elapsed_ms(sharded)
    dist.barrier()
    return baseline_ms, sharded_ms


def _baseline_exchange(attn, x: torch.Tensor, rope):
    from sglang.multimodal_gen.runtime.layers.usp import (
        _usp_input_all_to_all,
        _usp_input_all_to_all_packed_qkv,
        _usp_output_all_to_all,
    )
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
        _norm_rope_qk,
    )

    total = x.shape[0]
    qkv, _ = attn.qkv_proj(x)
    q, k, v = qkv.split(attn.local_inner_dim, dim=-1)
    q = q.view(total, attn.num_heads, attn.head_dim)
    k = k.view(total, attn.num_heads, attn.head_dim)
    v = v.view(total, attn.num_heads, attn.head_dim)
    q, k = _norm_rope_qk(attn, q, k, rope)
    gate, _ = attn.to_gate_compress(x)
    gate = gate.view(total, attn.num_heads, attn.head_dim)
    q, k, v = _usp_input_all_to_all_packed_qkv(q, k, v)
    gate = _usp_input_all_to_all(gate[None], head_dim=2)[0]
    _usp_output_all_to_all(q[None], head_dim=2)
    return q, k, v, gate


def _make_attention(device: torch.device):
    from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
        MiniMaxH3DiTArchConfig,
    )
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
        MiniMaxH3Attention,
    )
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

    arch = MiniMaxH3DiTArchConfig(has_gate_compress=True)
    attn = MiniMaxH3Attention(arch, None, prefix="blocks.0.attn").to(device)
    attn._attention_backend_enum = AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)

    def fill(param: torch.Tensor) -> None:
        values = torch.randn(param.shape, generator=generator, dtype=torch.float32)
        param.data.copy_(values.to(device=param.device, dtype=param.dtype))

    fill(attn.qkv_proj.weight)
    fill(attn.to_gate_compress.weight)
    fill(attn.q_norm.weight)
    fill(attn.k_norm.weight)
    return attn


def _inputs(seq_len: int, world: int, rank: int, device: torch.device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1)
    hidden = torch.randn(seq_len, _HIDDEN, generator=generator, dtype=torch.float32)
    cos = torch.randn(seq_len, _ROPE_DIM, generator=generator, dtype=torch.float32)
    local = seq_len // world
    sl = slice(rank * local, (rank + 1) * local)
    x = hidden[sl].to(device=device, dtype=torch.bfloat16).contiguous()
    rope_cos = cos[sl].to(device=device, dtype=torch.bfloat16).contiguous()
    positions = torch.arange(local, device=device)
    return x, (rope_cos, positions)


def _elapsed_ms(fn) -> float:
    import torch.distributed as dist

    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(3):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 3.0


def _spawn() -> int:
    if torch.cuda.device_count() < _WORLD:
        print(f"need {_WORLD} GPUs, found {torch.cuda.device_count()}", flush=True)
        return 1
    procs = []
    for rank in range(_WORLD):
        env = os.environ.copy()
        env.update(
            {
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "WORLD_SIZE": str(_WORLD),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29671",
                "NCCL_NVLS_ENABLE": "0",
            }
        )
        procs.append(
            subprocess.Popen(
                [sys.executable, __file__],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )
    outputs = []
    codes = []
    for proc in procs:
        out, _ = proc.communicate(timeout=900)
        outputs.append(out)
        codes.append(proc.returncode)
    sys.stdout.write("\n".join(outputs))
    if any(codes):
        print("worker failed", codes, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    if "RANK" in os.environ:
        sys.exit(_worker())
    sys.exit(_spawn())
