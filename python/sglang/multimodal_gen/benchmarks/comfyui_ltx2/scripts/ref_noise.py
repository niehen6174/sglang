"""ComfyUI self-consistency: batch-1 forward vs batch-2 slice, and tiny input perturbation."""
import os, sys, torch
sys.argv=[sys.argv[0]]
os.chdir("/scratch/data/sgld_comfy/ComfyUI-ltx25")
import comfy.model_management as mm, comfy.sd, folder_paths
RES=os.environ["LTX_RES"]
ref=torch.load(os.path.join(RES,"tensors",os.environ.get("LTX_REF_OUT","ref_dit.pt")))
p=comfy.sd.load_diffusion_model(folder_paths.get_full_path("diffusion_models", os.environ.get("LTX_UNET","ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors")))
LORA=os.environ.get("LTX_LORA")
if LORA:
    import comfy.utils
    sd=comfy.utils.load_torch_file(folder_paths.get_full_path("loras", LORA), safe_load=True)
    p,_=comfy.sd.load_lora_for_models(p, None, sd, float(os.environ.get("LTX_LORA_STRENGTH", 1.0)), 0)
mm.load_models_gpu([p]); dm=p.model.diffusion_model; dev=mm.get_torch_device(); bf=torch.bfloat16
def st(n,a,b):
    a=a.float().flatten().cpu(); b=b.float().flatten().cpu()
    print(n,"rel",round(((a-b).norm()/a.norm()).item(),5),"cos",round(torch.nn.functional.cosine_similarity(a,b,dim=0).item(),6),flush=True)
with torch.no_grad():
    v=ref["video"][:1].to(dev,bf); a=ref["audio"][:1].to(dev,bf); c=ref["ctx_pos"].to(dev,bf)
    ts=torch.full((1,),ref["sigma"],device=dev)
    o=dm([v.clone(),a.clone()],(ts,ts),context=c,attention_mask=None,frame_rate=24.0,transformer_options={})
    st("b1_vs_b2 video",ref["v_out"][:1],o[0]); st("b1_vs_b2 audio",ref["a_out"][:1],o[1])
    g=torch.Generator("cpu").manual_seed(7)
    c2=(c.float()*(1+1e-3*torch.randn(c.shape,generator=g).to(dev))).to(bf)
    o2=dm([v.clone(),a.clone()],(ts,ts),context=c2,attention_mask=None,frame_rate=24.0,transformer_options={})
    st("ctx_1e-3 video",o[0],o2[0]); st("ctx_1e-3 audio",o[1],o2[1])
    v2=(v.float()*(1+1e-3*torch.randn(v.shape,generator=g).to(dev))).to(bf)
    o3=dm([v2,a.clone()],(ts,ts),context=c,attention_mask=None,frame_rate=24.0,transformer_options={})
    st("lat_1e-3 video",o[0],o3[0]); st("lat_1e-3 audio",o[1],o3[1])
