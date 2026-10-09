import os, sys, torch
sys.argv=[sys.argv[0]]
os.chdir("/scratch/data/sgld_comfy/ComfyUI-ltx25")
import comfy.model_management as mm, comfy.sd, comfy.utils, folder_paths, comfy.lora, comfy.quant_ops
from comfy.quant_ops import QuantizedTensor
TARGET="diffusion_model.transformer_blocks.10.attn1.to_q.weight"
orig_cw = comfy.lora.calculate_weight
def cw(patches, weight, key, intermediate_dtype=torch.float32, original_weights=None):
    out = orig_cw(patches, weight, key, intermediate_dtype=intermediate_dtype, original_weights=original_weights)
    if key == TARGET:
        print("calc_weight in", type(weight).__name__, weight.dtype, "inter", intermediate_dtype, "out", out.dtype, flush=True)
        torch.save(out.detach().cpu(), "/tmp/claude-0/-scratch-data-sgld-comfy/5b083111-d623-4dc9-9585-fbd765583e0c/scratchpad/cw_out.pt")
    return out
comfy.lora.calculate_weight = cw
orig_rq = QuantizedTensor.requantize_from_float
def rq(self, tensor, **kw):
    print("requant in", tensor.dtype, tuple(tensor.shape), kw.get("scale"), kw.get("stochastic_rounding"), flush=True)
    return orig_rq(self, tensor, **kw)
QuantizedTensor.requantize_from_float = rq
ck=folder_paths.get_full_path("diffusion_models","ltx-2.3-22b-dev-fp8.safetensors")
p=comfy.sd.load_diffusion_model(ck)
sd=comfy.utils.load_torch_file(folder_paths.get_full_path("loras","ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"), safe_load=True)
p2,_=comfy.sd.load_lora_for_models(p,None,sd,0.5,0)
import comfy.model_patcher as mp
orig_ptd = mp.ModelPatcher.patch_weight_to_device
def ptd(self, key, *a, **k):
    if key == TARGET: print("patch_weight_to_device", key, a, k, flush=True)
    return orig_ptd(self, key, *a, **k)
mp.ModelPatcher.patch_weight_to_device = ptd
mm.load_models_gpu([p2])
print("lora_compute_dtype", mm.lora_compute_dtype(mm.get_torch_device()))
