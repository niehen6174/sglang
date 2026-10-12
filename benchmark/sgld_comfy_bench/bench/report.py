"""SUITE/REPORT.md and SUITE/summary.csv for one card, from whatever has run.

Track A rows: medians, speedup against the stock baseline of the same workload, GPU-s and $ per
video second (RunPod Secure on-demand prices, 2026-09-27), cold start, peak VRAM / host memory,
automatic fallbacks. Track B: SGLang vs vllm-omni side by side. Plus quality pairs and the
functional matrix. summary.csv has one row per (track, experiment, engine) for cross-card merging.

Usage: report.py --suite SUITE
"""
import argparse
import csv
import json
import pathlib

PRICE_PER_H = {"4090": 0.74, "5090": 0.99, "h200": 4.59, "b200": 6.79}  # RunPod Secure, USD
PRICE_SOURCE = "RunPod Secure Cloud on-demand, pricing page updated 2026-09-27"


def load(path):
    p = pathlib.Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def summaries(rows):
    """Last summary row per experiment (and engine)."""
    out = {}
    for r in rows:
        if r.get("phase") == "summary":
            out[(r["exp_id"], r["engine"] if r.get("track") == "B" else "")] = r
    return out


def med(r, key):
    v = r.get(key)
    return v["median"] if isinstance(v, dict) else None


def a_baseline(eid):
    """Stock reference for a track A experiment of the same workload."""
    if eid.startswith("A-H3-4"):
        return "A-H3-40"
    if eid.startswith("A-H3-"):
        return "A-H3-01" + ("-480" if eid.endswith("-480") else "")
    if eid.startswith("A-LTX-"):
        return "A-LTX-01"
    if eid.startswith("A-QW-"):
        return "A-QW-01" + eid[-5:]
    return None


def describe(r):
    c = r.get("config") or {}
    sg = c.get("sgld") or {}
    parts = []
    if r.get("engine") == "stock":
        parts.append("stock " + (c.get("flags") or "default"))
        if c.get("easycache"):
            parts.append("EasyCache")
    else:
        parts.append(f"attn={sg.get('attention_backend') or 'auto'}")
        for k in ("quantization", "model_type"):
            if sg.get(k) and sg[k] not in ("auto-detect", "none"):
                parts.append(f"{k}={sg[k]}")
        if sg.get("enable_cache_dit"):
            parts.append("cache-dit")
        if sg.get("dit_layerwise_offload"):
            parts.append("offload")
        if sg.get("num_gpus", 1) > 1:
            parts.append("TP2" if sg.get("tp_size") == 2 else "U2")
    if c.get("unet"):
        parts.append(f"file={c['unet']}")
    if c.get("env"):
        parts.append(" ".join(f"{k}={v}" for k, v in c["env"].items()))
    return ", ".join(parts)


def fmt(x, nd=2):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def vram(r):
    v = r.get("peak_vram_gib") or {}
    return "/".join(f"{x:.1f}" for x in v.values()) if v else "-"


def track_a(card, rows, csv_rows):
    S = summaries(rows)
    price = PRICE_PER_H[card]
    lines = ["## Track A: SGLD vs stock ComfyUI", "",
             "| exp | config | GPUs | E2E s | denoise s | speedup vs stock | GPU-s / video-s | $ / video-s | cold s | "
             "peak VRAM GiB | host GiB | notes |", "|" + "---|" * 12]
    for (eid, _), r in sorted(S.items()):
        if eid.startswith("P0"):
            continue
        e2e, den = med(r, "e2e_s"), med(r, "denoise_s")
        base = S.get((a_baseline(eid), ""))
        sp = (med(base, "e2e_s") / e2e) if base and e2e and med(base, "e2e_s") else None
        gps = med(r, "gpu_s_per_video_s") or med(r, "gpu_s_per_image")
        notes = [x for x in (r.get("status") != "success" and f"**{r.get('status')}**: {(r.get('error') or '')[:120]}",
                             r.get("offload_fallback") and "offload fallback",
                             r.get("attention_fallback") and f"attn fallback {r['attention_fallback']}") if x]
        lines.append(f"| {eid} | {describe(r)} | {r.get('n_gpu', 1)} | {fmt(e2e)} | {fmt(den)} | {fmt(sp)}x | {fmt(gps)} | "
                     f"{fmt(gps * price / 3600 if gps else None, 4)} | {fmt(r.get('cold_s'), 1)} | {vram(r)} | "
                     f"{fmt(r.get('peak_rss_gib'), 1)} | {'; '.join(notes)} |")
        csv_rows.append({"card": card, "track": "A", "exp_id": eid, "engine": r.get("engine"), "n_gpu": r.get("n_gpu"),
                         "config": describe(r), "status": r.get("status"), "e2e_s": e2e, "denoise_s": den,
                         "speedup_vs_stock": sp, "gpu_s_per_unit": gps, "usd_per_unit": gps * price / 3600 if gps else None,
                         "cold_s": r.get("cold_s"), "peak_vram_gib": vram(r), "peak_rss_gib": r.get("peak_rss_gib")})
    p0 = [(eid, med(r, "denoise_s")) for (eid, _), r in S.items() if eid.startswith("P0")]
    if p0:
        lines += ["", "P0 attention sweep (864x480x39, 8 steps, denoise median s): " +
                  ", ".join(f"{e[6:]}={fmt(v)}" for e, v in sorted(p0))]
    return lines


def track_b(card, rows, csv_rows):
    S = summaries(rows)
    price = PRICE_PER_H[card]
    ids = sorted({eid for eid, _ in S})
    lines = ["## Track B: SGLang native vs vllm-omni", "",
             "| exp | kind | GPUs | SGLang E2E s | vllm-omni E2E s | vllm / SGLang | SGLang VRAM | vllm VRAM | "
             "SGLang cold s | vllm cold s | notes |", "|" + "---|" * 11]
    for eid in ids:
        s, v = S.get((eid, "sglang"), {}), S.get((eid, "vllm"), {})
        es, ev = med(s, "e2e_s"), med(v, "e2e_s")
        notes = "; ".join(f"{n} {r.get('status')}: {(r.get('error') or '')[:100]}" for n, r in (("sglang", s), ("vllm", v))
                          if r and r.get("status") != "success")
        r0 = s or v
        lines.append(f"| {eid} | {r0.get('kind')} | {r0.get('n_gpu')} | {fmt(es)} | {fmt(ev)} | "
                     f"{fmt(ev / es if es and ev else None)}x | {vram(s)} | {vram(v)} | {fmt(s.get('cold_s'), 1)} | "
                     f"{fmt(v.get('cold_s'), 1)} | {notes} |")
        for eng, r in (("sglang", s), ("vllm", v)):
            if r:
                gps = med(r, "gpu_s_per_video_s") or med(r, "gpu_s_per_image")
                csv_rows.append({"card": card, "track": "B", "exp_id": eid, "engine": eng, "n_gpu": r.get("n_gpu"),
                                 "config": r.get("kind"), "status": r.get("status"), "e2e_s": med(r, "e2e_s"),
                                 "gpu_s_per_unit": gps, "usd_per_unit": gps * price / 3600 if gps else None,
                                 "cold_s": r.get("cold_s"), "peak_vram_gib": vram(r), "peak_rss_gib": r.get("peak_rss_gib")})
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    a = ap.parse_args()
    suite = pathlib.Path(a.suite).resolve()
    machine = json.loads((suite / "machine.json").read_text()) if (suite / "machine.json").exists() else {}
    card = machine.get("card") or suite.name
    csv_rows = []
    lines = [f"# {card} benchmark report", ""]
    if machine:
        m = machine["machine"]
        v = machine["versions"]
        lines += [f"- host `{m['hostname']}`, {m['cpu']} x{m['cpu_count']}, {m['ram_gb']} GB RAM",
                  f"- GPUs: `{m['nvidia_smi'].splitlines()[0]}` (x{len(m['nvidia_smi'].splitlines())})",
                  f"- SGLang `{v.get('sglang_commit')}`, ComfyUI `{v.get('comfyui_commit')}`, vllm-omni `{v.get('vllm_omni_commit')}`",
                  f"- torch {v.get('torch')} / CUDA {v.get('torch_cuda')}, sageattention {v.get('sageattention')}, "
                  f"comfy-kitchen {v.get('comfy_kitchen')}"]
    lines += [f"- price: ${PRICE_PER_H.get(card, '?')}/GPU-hour ({PRICE_SOURCE})", ""]
    lines += track_a(card, load(suite / "A" / "results.jsonl"), csv_rows) + [""]
    lines += track_b(card, load(suite / "B" / "results.jsonl"), csv_rows) + [""]
    q = load(suite / "quality.jsonl")
    if q:
        lines += ["## Quality against the reference config", "", "| track | engine | variant | reference | kind | PSNR dB | SSIM | latent cos |",
                  "|" + "---|" * 8]
        for r in q:
            lines.append(f"| {r['track']} | {r.get('engine', '')} | {r['variant']} | {r['reference']} | {r['kind']} | "
                         f"{fmt(r.get('psnr_db'))} | {fmt(r.get('ssim'), 4)} | {r.get('latent_cos') or r.get('error', '')} |")
        lines.append("")
    f = load(suite / "F" / "functional.jsonl")
    if f:
        cells = {}
        for r in f:
            cells.setdefault((r["case"], r["desc"]), {})[r["engine"]] = r["outcome"] + (
                f" ({r['error'][:80]})" if r.get("outcome") == "FAIL" and r.get("error") else "")
        lines += ["## Functional matrix", "", "| case | description | stock ComfyUI | SGLD |", "|---|---|---|---|"]
        for (cid, desc), c in sorted(cells.items()):
            lines.append(f"| {cid} | {desc} | {c.get('stock', '-')} | {c.get('sgld', '-')} |")
    (suite / "REPORT.md").write_text("\n".join(lines) + "\n")
    with open(suite / "summary.csv", "w", newline="") as fh:
        keys = sorted({k for r in csv_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"wrote {suite / 'REPORT.md'} and {suite / 'summary.csv'} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
