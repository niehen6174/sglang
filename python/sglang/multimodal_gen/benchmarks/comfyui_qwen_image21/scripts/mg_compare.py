#!/usr/bin/env python3
"""Compare multi-GPU worker outputs with the 1-GPU worker outputs (same inputs)."""
import json
import sys

import torch

T = "/scratch/data/sgld_comfy/tmp/"
R = "/scratch/data/sgld_comfy/results/qwen_image21/multigpu/"
ref = torch.load(T + "mg_1gpu.pt")
ref_rep = json.load(open(R + "parity_1gpu.json"))
out = {}
for name in sys.argv[1:]:
    cur = torch.load(T + f"mg_{name}.pt")
    rep = json.load(open(R + f"parity_{name}.json"))
    res = {}
    for case, steps in ref.items():
        rows = []
        got = cur.get(case, [])
        for i, a in enumerate(steps):
            r = rep.get(case, [{}] * 4)
            if i >= len(got):
                rows.append({"step": i, "error": (r[i] if i < len(r) else {}).get("error", "missing")})
                continue
            a, b = a.float().flatten(), got[i].float().flatten()
            rows.append({
                "step": i,
                "cosine_vs_1gpu": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
                "rel_l2_vs_1gpu": float((a - b).norm() / a.norm()),
                "max_abs_vs_1gpu": float((a - b).abs().max()),
                "cosine_vs_comfy": r[i].get("cosine"),
                "call_ms": r[i].get("call_ms"),
                "call_ms_1gpu": ref_rep[case][i].get("call_ms"),
            })
        res[case] = rows
    out[name] = res
    print("==", name)
    for case, rows in res.items():
        for x in rows:
            if "error" in x:
                print(f"  {case:16s} step{x['step']} ERROR {x['error'][:120]}")
            else:
                print(f"  {case:16s} step{x['step']} cos={x['cosine_vs_1gpu']:.6f} rel={x['rel_l2_vs_1gpu']:.4f} max={x['max_abs_vs_1gpu']:.3f} cos_comfy={x['cosine_vs_comfy']:.6f} ms={x['call_ms'] or 0:.0f} (1gpu {x['call_ms_1gpu'] or 0:.0f})")
json.dump(out, open(R + "compare_" + "_".join(sys.argv[1:]) + ".json", "w"), indent=1)
