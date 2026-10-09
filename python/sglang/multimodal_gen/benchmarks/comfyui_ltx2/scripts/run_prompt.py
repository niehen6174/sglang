"""Queue LTX-2.5 API prompts on the ComfyUI instance (port 8189) and time them.

usage: run_prompt.py NAME --sgld/--official [--repeat N] [--single-stage] [--w W --h H --sec S]
Writes logs/runs.jsonl lines {name, run, wall_s, exec_s, outputs}.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import uuid

sys.path.insert(0, os.path.dirname(__file__))
import workflows  # noqa: E402

URL = "http://127.0.0.1:8189"
RES = os.environ.get("LTX_RES", "/scratch/data/sgld_comfy/results/ltx25")


def post(path, payload):
    req = urllib.request.Request(
        URL + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def get(path):
    with urllib.request.urlopen(URL + path) as resp:
        return json.loads(resp.read())


def run(prompt, timeout=3600):
    client_id = uuid.uuid4().hex
    t0 = time.time()
    pid = post("/prompt", {"prompt": prompt, "client_id": client_id})["prompt_id"]
    while True:
        hist = get(f"/history/{pid}")
        if pid in hist and hist[pid].get("status", {}).get("completed") is not None:
            entry = hist[pid]
            if entry["status"].get("status_str") != "success" or not entry["status"]["completed"]:
                raise RuntimeError(json.dumps(entry["status"], indent=1)[:4000])
            break
        if time.time() - t0 > timeout:
            raise TimeoutError(pid)
        time.sleep(0.5)
    wall = time.time() - t0
    msgs = {m[0]: m[1] for m in entry["status"].get("messages", [])}
    exec_s = None
    if "execution_start" in msgs and "execution_success" in msgs:
        exec_s = (msgs["execution_success"]["timestamp"] - msgs["execution_start"]["timestamp"]) / 1000.0
    return pid, wall, exec_s, entry.get("outputs", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sgld", action="store_true")
    g.add_argument("--official", action="store_true")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--single-stage", action="store_true")
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--sec", type=int, default=5)
    ap.add_argument("--seed", type=int, default=558811532553686)
    ap.add_argument("--video-cfg", type=float, default=1.0)
    ap.add_argument("--audio-cfg", type=float, default=1.0)
    ap.add_argument("--sgld-options", default='{"extra_server_args": "{\\"master_port\\": 30105, \\"scheduler_port\\": 5655}"}')
    args = ap.parse_args()
    for i in range(args.repeat):
        prompt = workflows.official_t2v(
            sgld=args.sgld,
            prefix=f"ltx25/{args.name}_run{i}",
            width=args.w,
            height=args.h,
            seconds=args.sec,
            seed=args.seed + i,
            two_stage=not args.single_stage,
            sgld_options=json.loads(args.sgld_options),
        )
        workflows.dual_cfg(prompt, args.video_cfg, args.audio_cfg)
        pid, wall, exec_s, outputs = run(prompt)
        rec = {"name": args.name, "run": i, "prompt_id": pid, "wall_s": round(wall, 2),
               "exec_s": exec_s and round(exec_s, 2), "sgld": args.sgld,
               "w": args.w, "h": args.h, "sec": args.sec, "two_stage": not args.single_stage,
               "cfg": [args.video_cfg, args.audio_cfg], "outputs": outputs}
        print(json.dumps({k: v for k, v in rec.items() if k != "outputs"}), flush=True)
        with open(os.path.join(RES, "logs", "runs.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
