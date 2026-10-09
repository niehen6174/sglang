"""Aggregate timing (logs/runs.jsonl) and comparison JSONs into logs/summary.json."""

import json
import os
from collections import defaultdict

RES = "/scratch/data/sgld_comfy/results/ltx25"
runs = defaultdict(list)
for line in open(os.path.join(RES, "logs", "runs.jsonl")):
    rec = json.loads(line)
    runs[rec["name"]].append(rec["exec_s"])
timing = {
    name: {"cold_s": v[0], "warm_s": v[1:], "warm_mean_s": round(sum(v[1:]) / max(len(v) - 1, 1), 2)}
    for name, v in runs.items()
}
cmps = {}
for f in sorted(os.listdir(os.path.join(RES, "logs"))):
    if f.startswith("cmp_") and f.endswith(".json"):
        d = json.load(open(os.path.join(RES, "logs", f)))
        cmps[f[4:-5]] = {
            "video_psnr_mean": round(d["video_psnr_mean"], 2),
            "audio_snr_db": round(d.get("audio_snr_db", float("nan")), 2),
            "video_latent_rel_l2": round(d.get("video_latent", {}).get("rel_l2", float("nan")), 4),
            "audio_latent_rel_l2": round(d.get("audio_latent", {}).get("rel_l2", float("nan")), 4),
        }
dit = {}
for f in sorted(os.listdir(os.path.join(RES, "logs"))):
    if f.startswith("dit_parity") and f.endswith(".json"):
        dit[f] = json.load(open(os.path.join(RES, "logs", f)))
out = {"timing_exec_s": timing, "e2e_compare": cmps, "dit_parity": dit}
json.dump(out, open(os.path.join(RES, "logs", "summary.json"), "w"), indent=1)
print(json.dumps({"timing": timing, "cmp": cmps}, indent=1))
