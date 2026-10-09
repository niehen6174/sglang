import os, sys, torch
sys.argv=[sys.argv[0]]
os.chdir("/scratch/data/sgld_comfy/ComfyUI-ltx25")
import comfy.model_management as mm, comfy.sd, comfy.utils, folder_paths
from safetensors import safe_open
ck=folder_paths.get_full_path("diffusion_models","ltx-2.3-22b-dev-fp8.safetensors")
lp=folder_paths.get_full_path("loras","ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors")
p=comfy.sd.load_diffusion_model(ck)
sd=comfy.utils.load_torch_file(lp, safe_load=True)
p2,_=comfy.sd.load_lora_for_models(p,None,sd,0.5,0)
mm.load_models_gpu([p2])
dm=p2.model.diffusion_model
name="transformer_blocks.10.attn1.to_q"
m=dm.get_submodule(name)
w=m.weight
print("type", type(w).__name__, "scale", w._params.scale, "dtype", w._qdata.dtype)
with safe_open(ck,"pt") as f:
    wq=f.get_tensor("model.diffusion_model."+name+".weight").cuda(); ws=f.get_tensor("model.diffusion_model."+name+".weight_scale").cuda(); ins=f.get_tensor("model.diffusion_model."+name+".input_scale").cuda()
print("orig scale", ws, "input_scale", ins)
A=sd["diffusion_model."+name+".lora_A.weight"].cuda().float(); B=sd["diffusion_model."+name+".lora_B.weight"].cuda().float(); al=sd["diffusion_model."+name+".alpha"].float().item()
delta=0.5*al/A.shape[0]*(B@A)
exact=wq.float()*ws + delta
deq=w.dequantize().float().cuda()
print("comfy weight vs exact rel", ((deq-exact).norm()/exact.norm()).item(), " base vs exact rel", ((wq.float()*ws-exact).norm()/exact.norm()).item())
print("comfy weight vs base deq rel", ((deq-wq.float()*ws).norm()/deq.norm()).item())
x=torch.randn(64,4096,device="cuda",dtype=torch.bfloat16)
with torch.no_grad():
    y=ye=yb=x.float()
    ye=torch.nn.functional.linear(x.float(), exact, m.bias.float() if m.bias is not None else None)
    yb=torch.nn.functional.linear(x, exact.bfloat16(), m.bias).float()
print("layer out vs exact fp32 rel", ((y-ye).norm()/ye.norm()).item(), " bf16-merged vs exact", ((yb-ye).norm()/ye.norm()).item())
import zlib, comfy_kitchen as ck
key="diffusion_model."+name+".weight"
w32 = wq.float()*ws.float().to(wq.device)
diff=torch.mm(B, A)
w32 += ((0.5*(al/A.shape[0]))*diff).type(torch.float32)
sc=torch.amax(w32.abs()).float()/448.0
w32=w32*(1.0/sc).to(w32.dtype)
g=torch.Generator(device=w32.device); g.manual_seed(zlib.crc32(key.encode()))
rng=torch.randint(0,256,w32.size(),dtype=torch.uint8,device=w32.device,generator=g)
q=ck.stochastic_rounding_fp8(w32,rng,torch.float8_e4m3fn)
print("replica scale", sc.item(), "comfy scale", w._params.scale.item(), "qdata equal frac", (q.view(torch.uint8)==w._qdata.cuda().view(torch.uint8)).float().mean().item())
def replica(dt_w, dt_lora):
    w = wq.to(dt_w)*ws.to(device=wq.device, dtype=dt_w)
    d = torch.mm(B.to(dt_lora), A.to(dt_lora))
    w = w + ((0.5*(al/A.shape[0]))*d).type(dt_w)
    sc = torch.amax(w.abs()).to(torch.float32)/448.0
    w = w*(1.0/sc).to(w.dtype)
    g=torch.Generator(device=w.device); g.manual_seed(zlib.crc32(key.encode()))
    rng=torch.randint(0,256,w.size(),dtype=torch.uint8,device=w.device,generator=g)
    q=ck.stochastic_rounding_fp8(w,rng,torch.float8_e4m3fn)
    return sc.item(), (q.view(torch.uint8)==w_q.view(torch.uint8)).float().mean().item()
w_q = w._qdata.cuda()
for a in (torch.float32, torch.bfloat16):
    for b in (torch.float32, torch.bfloat16):
        print("variant", a, b, replica(a, b))
def sr(wt):
    g=torch.Generator(device=wt.device); g.manual_seed(zlib.crc32(key.encode()))
    rng=torch.randint(0,256,wt.size(),dtype=torch.uint8,device=wt.device,generator=g)
    return ck.stochastic_rounding_fp8(wt,rng,torch.float8_e4m3fn)
w32 = wq.float()*ws.to(wq.device).float()
w32 = w32 + ((0.5*(al/A.shape[0]))*torch.mm(B.float(),A.float())).float()
wb = w32.to(torch.bfloat16)
sc = torch.amax(wb.abs()).float()/448.0
for nm, t in (("c_bf16scale_bf16scaling", wb*(1.0/sc).to(torch.bfloat16)), ("d_bf16scale_fp32scaling", w32*(1.0/sc)), ("e_bf16vals_fp32scaling", wb.float()*(1.0/sc))):
    q=sr(t); print("variant", nm, (q.view(torch.uint8)==w_q.view(torch.uint8)).float().mean().item())
for deq_mode in ("eager", "ck"):
    if deq_mode == "eager":
        w16 = wq.to(torch.float16)*ws.to(device=wq.device, dtype=torch.float16)
    else:
        w16 = ck.dequantize_per_tensor_fp8(wq, ws.to(wq.device).float(), torch.float16)
    d = torch.mm(B.float(), A.float())
    w16 = w16 + ((0.5*(al/A.shape[0]))*d).type(torch.float16)
    sc = torch.amax(w16.abs()).to(torch.float32)/448.0
    fi = torch.finfo(torch.float16)
    sc = 1.0/torch.clamp(1.0/sc, min=fi.min, max=fi.max)
    t = w16*(1.0/sc).to(torch.float16)
    q = sr(t)
    print("variant fp16", deq_mode, sc.item(), (q.view(torch.uint8)==w_q.view(torch.uint8)).float().mean().item())
