"""What does ComfyUI's native LoraLoader do with an SGLD MODEL? (no worker)"""
import os, sys, torch
sys.argv=[sys.argv[0]]
COMFY="/scratch/data/sgld_comfy/ComfyUI-ltx25"; os.chdir(COMFY); sys.path.insert(0, os.path.join(COMFY,"custom_nodes"))
import comfy.sd, comfy.utils, folder_paths, comfy.model_management as mm
from ComfyUI_SGLDiffusion.core.generator import SGLDiffusionGenerator
from ComfyUI_SGLDiffusion.core.model_patcher import SGLDModelPatcher
from ComfyUI_SGLDiffusion.executors.ltx_av import LTXAVExecutor
gen=SGLDiffusionGenerator()
path=folder_paths.get_full_path("diffusion_models","ltx-2.3-22b-dev-fp8.safetensors")
model, cfg, mt = gen.get_comfyui_model(path, {})
ex=LTXAVExecutor(None, path, model, cfg)
model.diffusion_model=ex
patcher=SGLDModelPatcher(model, mm.get_torch_device(), mm.unet_offload_device(), model_type=mt)
for name in sys.argv[1:] or ["ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors","gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors"]:
    sd=comfy.utils.load_torch_file(folder_paths.get_full_path("loras",name), safe_load=True)
    try:
        m2,_=comfy.sd.load_lora_for_models(patcher, None, sd, 0.5, 0)
        print(name, "-> OK, patches", len(m2.patches))
    except Exception as e:
        print(name, "->", type(e).__name__, str(e)[:300])
