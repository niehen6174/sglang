"""Track A runner: SGLD integrated mode vs stock ComfyUI on this card.

Per experiment: fresh ComfyUI server, one warmup at the same shape (3 denoising steps; distilled
models and LTX run their fixed schedule), then measured runs with seed = base + i so ComfyUI
cannot replay a cached output. 5 measured runs, or 3 when the first takes >= 60 s. The first
measured run keeps its media and latents (quality and parity checks).

On 4090/5090 a SGLD experiment that runs out of memory is retried once with DiT layerwise
offload; the row records offload_fallback=true.

Usage: track_a.py --suite DIR [--gpus 1|2] [--only A-H3-01,A-QW-*] [--p0-only] [--dry-run]
"""
import argparse
import json
import os
import pathlib
import queue
import statistics
import threading
import time
import traceback

import comfyserver as comfy
import experiments_a as A
import graphs

BASE_SEED = {"h3": 150000, "ltx": 4242, "qwen": 1234}
SLOW_S = 60.0
# Worker init settles distributed ports; two lanes loading at once can race for the same port.
_start_lock = threading.Lock()
_file_lock = threading.Lock()


def _commits():
    import subprocess

    out = {}
    for key, repo in (("sglang", os.environ["SGLANG_SRC"]), ("comfyui", os.environ["COMFY_DIR"])):
        r = subprocess.run(["git", "-C", repo, "rev-parse", "--short=10", "HEAD"], capture_output=True, text=True)
        out[key] = r.stdout.strip()
    return out


def _is_oom(res):
    text = (res.get("error") or "") + " ".join(res.get("log_markers") or [])
    return "out of memory" in text.lower() or "OutOfMemory" in text


def _row(card, e, r, res, run_idx, phase, commits, extra):
    prof = res.get("profile") or {}
    calls = prof.get("apply_model_seconds") or []
    sampler = sum(prof.get("sampler_seconds") or []) or None
    seconds = r["video_seconds"]
    n = e["n_gpu"]
    return {
        "exp_id": e["id"], "track": "A", "card": card, "n_gpu": n, "engine": e["engine"], "model": e["model"],
        "commit": commits, "config": {"flags": r["flags"], "env": r["env"], "sgld": r["sgld_options"], **e["spec"]},
        "phase": phase, "run_idx": run_idx, "run_id": res["run_id"], "status": res.get("status"),
        "e2e_s": res.get("e2e_s"), "denoise_s": sampler, "dit_calls": len(calls),
        "dit_call_median_s": statistics.median(calls) if calls else None,
        "peak_vram_gib": res.get("peak_vram_gib"), "peak_device_gib": res.get("peak_gpu_gib_by_nvml_index"), "peak_rss_gib": res.get("peak_host_pss_gib"),
        "mean_util": res.get("mean_util"), "mean_power_w": res.get("mean_power_w"), "mean_sm_mhz": res.get("mean_sm_mhz"),
        "gpu_s_per_video_s": (res["e2e_s"] * n / seconds) if seconds and res.get("e2e_s") else None,
        "gpu_s_per_image": (res["e2e_s"] * n) if seconds is None and res.get("e2e_s") else None,
        "latents": res.get("latents"), "media": res.get("media"), "error": (res.get("error") or "")[:1500],
        **extra,
    }


def _append(suite, row):
    with _file_lock:
        with open(suite / "results.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")


def _done(suite, eid, retry_failed=False):
    f = suite / "results.jsonl"
    if not f.exists():
        return False
    for line in f.read_text().splitlines():
        row = json.loads(line)
        if row["exp_id"] == eid and row["phase"] == "summary" and (row["status"] == "success" or not retry_failed):
            return True
    return False


def _retry_change(res, e, card, r, extra):
    """One automatic retry per failure kind; returns a description of the change or None."""
    opts = r["sgld_options"]
    if res.get("status") == "success" or e["engine"] != "sgld" or not opts:
        return None
    text = (res.get("error") or "") + " ".join(res.get("log_markers") or [])
    if _is_oom(res) and card in ("4090", "5090") and not opts["dit_layerwise_offload"] and not extra["offload_fallback"]:
        extra["offload_fallback"] = True
        return {"dit_layerwise_offload": True}
    # cuDNN SDPA has no kernel for some H3 shapes (e.g. key/value length 1); torch SDPA picks another.
    if "No available kernel" in text and opts.get("attention_backend") == "torch_cudnn_sdpa" and not extra["attention_fallback"]:
        extra["attention_fallback"] = "torch_cudnn_sdpa->torch_sdpa"
        return {"attention_backend": "torch_sdpa"}
    return None


def run_experiment(e, card, suite, lane, timeout, commits):
    r = A.resolve(e, card, suite)
    srv, mon = comfy.Server(lane, suite), comfy.make_monitor(lane)
    extra = {"offload_fallback": False, "attention_fallback": None, "lane": lane}
    for _ in range(3):
        seed = BASE_SEED[r["model"]]
        warm_graph = graphs.with_steps(r["graph"], r["model"], r["warmup_steps"]) if r["warmup_steps"] else r["graph"]
        with _start_lock:
            srv.start(r["flags"], r["env"], monitor=mon)
            res = comfy.run_prompt(srv, graphs.set_seed(warm_graph, seed - 1), f"{e['id']}__w", mon, timeout)
        cold = res["done_at"] - srv.started_at
        change = _retry_change(res, e, card, r, extra)
        if change is None:
            break
        print(f"[{lane}] {e['id']}: retrying with {change}", flush=True)
        srv.stop()
        for node in r["graph"].values():
            if node["class_type"] == "SGLDOptions":
                node["inputs"].update(change)
        r["sgld_options"].update(change)
    extra["cold_s"] = round(cold, 2)
    _append(suite, _row(card, e, r, res, -1, "warmup", commits, extra))
    rows = []
    if res.get("status") == "success":
        n_runs = 5
        i = 0
        while i < n_runs:
            run_id = f"{e['id']}__m{i}" + ("_00" if i == 0 else "")
            res = comfy.run_prompt(srv, graphs.set_seed(r["graph"], seed + i), run_id, mon, timeout, keep_media=(i == 0))
            row = _row(card, e, r, res, i, "measured", commits, extra)
            _append(suite, row)
            rows.append(row)
            print(f"[{lane}] {e['id']:16s} m{i}: {res.get('status')} e2e={res.get('e2e_s')}s "
                  f"denoise={row['denoise_s'] and round(row['denoise_s'], 2)}s {row['error'][:120]}", flush=True)
            if res.get("status") != "success":
                break
            if i == 0 and res["e2e_s"] >= SLOW_S:
                n_runs = 3
            i += 1
    else:
        print(f"[{lane}] {e['id']:16s} warmup failed: {res.get('status')} {str(res.get('error'))[:200]}", flush=True)
    srv.stop()
    ok = [x for x in rows if x["status"] == "success"]
    summary = {"exp_id": e["id"], "track": "A", "card": card, "n_gpu": e["n_gpu"], "engine": e["engine"],
               "model": e["model"], "phase": "summary", "status": "success" if ok and len(ok) == len(rows) else "failed",
               "runs": len(ok), "commit": commits, "cold_s": extra["cold_s"], "offload_fallback": extra["offload_fallback"],
               "attention_fallback": extra["attention_fallback"],
               "config": rows[0]["config"] if rows else {"flags": r["flags"], "sgld": r["sgld_options"], **e["spec"]},
               "error": (rows[-1]["error"] if rows and rows[-1]["status"] != "success" else res.get("error") or "")[:1500]}
    for key in ("e2e_s", "denoise_s", "gpu_s_per_video_s", "gpu_s_per_image"):
        vals = [x[key] for x in ok if x[key] is not None]
        if vals:
            summary[key] = {"median": statistics.median(vals), "min": min(vals), "max": max(vals)}
    if ok:
        summary["peak_vram_gib"] = {k: max(x["peak_vram_gib"][k] for x in ok) for k in ok[0]["peak_vram_gib"]}
        summary["peak_rss_gib"] = max((x["peak_rss_gib"] or 0) for x in ok)
        summary["mean_util"], summary["mean_power_w"] = ok[0]["mean_util"], ok[0]["mean_power_w"]
    _append(suite, summary)
    return summary


def pick_best_attention(suite, card):
    rows = [json.loads(l) for l in (suite / "results.jsonl").read_text().splitlines()]
    cands = {}
    for row in rows:
        if row["phase"] == "summary" and row["exp_id"].startswith("P0-H3-") and row["status"] == "success":
            backend = row["exp_id"][len("P0-H3-"):]
            if backend in A.P0_BACKENDS:
                cands[backend] = row["denoise_s"]["median"]
    if not cands:
        print("P0: no lossless backend succeeded; keeping the default", A.DEFAULT_BEST[card], flush=True)
        return
    best = min(cands, key=cands.get)
    (suite / "best_attention.json").write_text(json.dumps({"best": best, "denoise_median_s": cands}, indent=1))
    print(f"P0: best lossless attention on {card} = {best} {cands}", flush=True)


def _worker(lane, q, card, suite, timeout, commits):
    while True:
        try:
            e = q.get_nowait()
        except queue.Empty:
            return
        try:
            run_experiment(e, card, suite, lane, timeout, commits)
        except Exception:
            print(f"[{lane}] {e['id']} crashed:\n{traceback.format_exc()}", flush=True)
            _append(suite, {"exp_id": e["id"], "track": "A", "card": card, "phase": "summary", "status": "crashed",
                            "error": traceback.format_exc()[-1500:]})


def run_many(exps, card, suite, lanes, timeout, commits, retry_failed=False):
    q = queue.Queue()
    for e in exps:
        if _done(suite, e["id"], retry_failed):
            print(f"skip {e['id']} (done)", flush=True)
        else:
            q.put(e)
    threads = []
    for lane in lanes:
        t = threading.Thread(target=_worker, args=(lane, q, card, suite, timeout, commits))
        t.start()
        threads.append(t)
        time.sleep(1)
    for t in threads:
        t.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--gpus", type=int, choices=(1, 2))
    ap.add_argument("--only", type=lambda s: s.split(","))
    ap.add_argument("--p0-only", action="store_true")
    ap.add_argument("--retry-failed", action="store_true", help="re-run experiments whose summary is not success")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    card = os.environ["CARD"]
    suite = pathlib.Path(a.suite).resolve()
    (suite / "runs").mkdir(parents=True, exist_ok=True)
    timeout = float(os.environ.get("TIMEOUT_S", "5400"))
    n_lanes = int(os.environ.get("LANES", "1"))
    lanes1 = [f"gpu{i}" for i in range(min(n_lanes, len(comfy.gpus())))]
    p0 = A.for_card(card, n_gpu=1, only=a.only, kinds=("p0",))
    exps1 = A.for_card(card, n_gpu=1, only=a.only, kinds=("perf",))
    exps2 = A.for_card(card, n_gpu=2, only=a.only, kinds=("perf",))
    if a.gpus == 1:
        exps2 = []
    if a.gpus == 2:
        p0, exps1 = [], []
    if a.p0_only:
        exps1, exps2 = [], []
    if a.dry_run:
        for e in p0 + exps1 + exps2:
            r = A.resolve(e, card, suite)
            print(f"{e['id']:16s} gpus={e['n_gpu']} flags={r['flags']!r} env={r['env']} warm={r['warmup_steps']} "
                  f"sgld={r['sgld_options']}")
        return
    commits = _commits()
    if p0:
        run_many(p0, card, suite, lanes1, timeout, commits, a.retry_failed)
        pick_best_attention(suite, card)
    run_many(exps1, card, suite, lanes1, timeout, commits, a.retry_failed)
    run_many(exps2, card, suite, ["all"], timeout, commits, a.retry_failed)
    print("TRACK_A_DONE", flush=True)


if __name__ == "__main__":
    main()
