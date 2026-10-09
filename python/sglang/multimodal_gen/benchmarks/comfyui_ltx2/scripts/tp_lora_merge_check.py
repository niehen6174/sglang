"""2-GPU check: the FP8 requant LoRA merge of TP shards equals the unsharded merge.

Column- and row-parallel static-FP8 linears are built on 2 TP ranks from one
full weight, a LoRA is merged on each rank, the merged FP8 shards and scales are
gathered and compared byte-for-byte with a 1-rank ReplicatedLinear merge.
usage: CUDA_VISIBLE_DEVICES=0,1 python tp_lora_merge_check.py
"""

import importlib.util
import os

import torch
import torch.multiprocessing as mp

OUT, IN, RANK_LORA = 256, 512, 8


def _setup(rank):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29655", RANK=str(rank),
                      LOCAL_RANK=str(rank), WORLD_SIZE="2")
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )
    import sglang.multimodal_gen as mg
    from sglang.multimodal_gen.runtime.server_args import set_global_server_args

    maybe_init_distributed_environment_and_model_parallel(tp_size=2, sp_size=1)
    spec = importlib.util.spec_from_file_location(
        "unit_conftest", os.path.join(os.path.dirname(mg.__file__), "test/unit/conftest.py"))
    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)
    set_global_server_args(conf._make_unit_server_args())


def _fp8_layer(cls, dense, scale, **kw):
    from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config

    layer = cls(IN, OUT, bias=False, quant_config=Fp8Config(
        is_checkpoint_fp8_serialized=True, activation_scheme="static"), **kw).cuda()
    layer.weight.data.copy_((dense / scale).to(torch.float8_e4m3fn))
    layer.weight_scale.data.fill_(scale)
    layer.input_scale.data.fill_(0.05)
    layer.quant_method.process_weights_after_loading(layer)
    return layer


def _merge(layer, name, A, B):
    from sglang.multimodal_gen.runtime.layers.lora.linear import wrap_with_lora_layer

    lora = wrap_with_lora_layer(layer, lora_rank=RANK_LORA, lora_alpha=RANK_LORA)
    lora.layer_name = name
    lora.set_lora_weights(A, B, strength=0.5, clear_existing=True, merge_weights=True)
    assert lora.merged
    # [in, out] storage -> [out, in] bytes, plus the per-tensor scale.
    return layer.weight.t().contiguous().view(torch.uint8), layer.weight_scale.flatten()[0]


def worker(rank):
    torch.cuda.set_device(rank)
    _setup(rank)
    from sglang.multimodal_gen.runtime.layers.linear import (
        ColumnParallelLinear,
        ReplicatedLinear,
        RowParallelLinear,
    )

    g = torch.Generator().manual_seed(0)
    dense = (torch.randn(OUT, IN, generator=g) * 0.02).cuda()
    scale = dense.abs().amax() / 448.0
    A = (torch.randn(RANK_LORA, IN, generator=g) * 0.05).cuda()
    B = (torch.randn(OUT, RANK_LORA, generator=g) * 0.02).cuda()
    ok = True
    for cls, kw, dim in ((ColumnParallelLinear, {"gather_output": True}, 0),
                         (RowParallelLinear, {"input_is_parallel": True}, 1)):
        shard = dense.chunk(2, dim=dim)[rank]
        layer = _fp8_layer(cls, shard, scale, **kw)
        bytes_, s = _merge(layer, "blocks.0.proj", A, B)
        gathered = [torch.empty_like(bytes_) for _ in range(2)]
        torch.distributed.all_gather(gathered, bytes_)
        scales = [torch.empty_like(s) for _ in range(2)]
        torch.distributed.all_gather(scales, s)
        if rank == 0:
            ref_layer = _fp8_layer(ReplicatedLinear, dense, scale)
            ref_bytes, ref_scale = _merge(ref_layer, "blocks.0.proj", A, B)
            tp_bytes = torch.cat(gathered, dim=dim)
            same = torch.equal(tp_bytes, ref_bytes) and all(
                torch.equal(x, ref_scale) for x in scales)
            frac = (tp_bytes != ref_bytes).float().mean().item()
            print(f"{cls.__name__}: TP2 merge == 1-rank merge: {same} "
                  f"(differing bytes {frac:.2%}, scales {[x.item() for x in scales]} "
                  f"vs {ref_scale.item()})", flush=True)
            ok &= same
    torch.distributed.barrier()
    if rank == 0:
        print("ALL_EQUAL" if ok else "MISMATCH", flush=True)


if __name__ == "__main__":
    mp.spawn(worker, nprocs=2)
