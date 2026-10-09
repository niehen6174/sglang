"""compare_mgpu.py REF.pt OUT.pt: rel L2 of every saved output (+ stage1 b2 floor of REF)."""
import sys
import torch
ref, out = torch.load(sys.argv[1]), torch.load(sys.argv[2])
def rel(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return ((a - b).norm() / a.norm()).item()
for k in ref:
    if k == "context" or k.startswith("stage1_b2") or k.startswith("odd"):
        continue
    print(f"{k:20s} rel_l2={rel(ref[k], out[k]):.4f}" if k in out else f"{k:20s} missing")
for m in ("video", "audio"):
    if f"stage1_b2_{m}" in ref:
        print(f"floor stage1 b1-vs-b2 {m} (ref run): {rel(ref['stage1_'+m], ref['stage1_b2_'+m]):.4f}")
