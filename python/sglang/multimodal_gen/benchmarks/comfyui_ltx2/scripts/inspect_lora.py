import os, sys, torch
sys.argv=[sys.argv[0]]
os.chdir("/scratch/data/sgld_comfy/ComfyUI-ltx25")
import comfy.model_management as mm, comfy.sd, comfy.utils, folder_paths
p=comfy.sd.load_diffusion_model(folder_paths.get_full_path("diffusion_models","ltx-2.3-22b-dev-fp8.safetensors"))
sd=comfy.utils.load_torch_file(folder_paths.get_full_path("loras","ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"), safe_load=True)
p2,_=comfy.sd.load_lora_for_models(p,None,sd,0.5,0)
mm.load_models_gpu([p2])
dm=p2.model.diffusion_model
for name in ["transformer_blocks.10.attn1.to_q","transformer_blocks.0.attn1.to_q","adaln_single.linear"]:
    m=dm.get_submodule(name)
    print(name, type(m).__name__, type(m.weight).__name__, getattr(m.weight,"dtype",None), "wf", len(getattr(m,"weight_function",[])), getattr(m,"layout_type",None), getattr(m,"_full_precision_mm",None), getattr(m,"comfy_force_cast_weights",None))
print("is_dynamic", p2.is_dynamic())
