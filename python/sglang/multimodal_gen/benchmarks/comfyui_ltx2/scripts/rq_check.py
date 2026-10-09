import torch, zlib, comfy_kitchen as ck
from safetensors import safe_open
ckp="/scratch/data/sgld_comfy/models/Lightricks--LTX-2.3-fp8/ltx-2.3-22b-dev-fp8.safetensors"
lp="/scratch/data/sgld_comfy/models/Comfy-Org--ltx-2.3/split_files/loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"
for name in ["transformer_blocks.10.attn1.to_q","transformer_blocks.10.audio_attn1.to_q","transformer_blocks.20.audio_ff.net.0.proj","transformer_blocks.20.ff.net.2"]:
    with safe_open(ckp,"pt",device="cuda") as f:
        wq=f.get_tensor("model.diffusion_model."+name+".weight"); ws=f.get_tensor("model.diffusion_model."+name+".weight_scale")
    with safe_open(lp,"pt",device="cuda") as f:
        A=f.get_tensor("diffusion_model."+name+".lora_A.weight").float(); B=f.get_tensor("diffusion_model."+name+".lora_B.weight").float(); al=f.get_tensor("diffusion_model."+name+".alpha").float().item()
    base=wq.float()*ws.float(); delta=0.5*al/A.shape[0]*(B@A); exact=base+delta
    sc=exact.abs().amax()/448; rtn=(exact/sc).clamp(-448,448).to(torch.float8_e4m3fn).float()*sc
    w16=wq.to(torch.float16)*ws.to(torch.float16); w16=w16+(0.5*(al/A.shape[0])*torch.mm(B,A)).type(torch.float16)
    s2=torch.amax(w16.abs()).float()/448; t=w16*(1.0/s2).to(torch.float16)
    g=torch.Generator(device="cuda"); g.manual_seed(zlib.crc32(("diffusion_model."+name+".weight").encode()))
    rng=torch.randint(0,256,t.size(),dtype=torch.uint8,device="cuda",generator=g)
    sr=ck.stochastic_rounding_fp8(t,rng,torch.float8_e4m3fn).float()*s2
    r=lambda a,b: ((a-b).norm()/b.norm()).item()
    print(name, "delta/base", r(exact,base)+0, "| rtn-exact", r(rtn,exact), "sr-exact", r(sr,exact), "rtn-sr", r(rtn,sr), "base-exact", r(base,exact), "orig scale", ws.item(), "new", sc.item())
