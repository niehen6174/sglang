import sys, torch
r=torch.load("/scratch/data/sgld_comfy/results/ltx25/tensors/ref_dit.pt")
d=torch.load(sys.argv[1])
for k in ["block0_v","block0_a","block23_v","block47_v"]:
    a=r[k].float().flatten(); b=d[k].float().flatten()
    print(sys.argv[2], k, "rel", round(((a-b).norm()/a.norm()).item(),5))
