"""Track B runner: SGLang native serve vs vllm-omni serve, same HF weights (EXPERIMENTS.md section 6).

Per experiment and engine: fresh server, one warmup request (3 steps; distilled LTX runs its fixed
schedule), then measured requests with seed = base + i (5, or 3 when the first takes >= 60 s).
E2E is submit -> job completed on the async endpoint (video download excluded); the first
measured output is downloaded and kept.

Usage: track_b.py --suite DIR [--gpus 1|2] [--only B-H3-02,B-QW-*] [--engines sglang,vllm] [--dry-run]
"""
import argparse
import base64
import json
import os
import pathlib
import signal
import statistics
import subprocess
import time
import traceback
import urllib.error
import urllib.request
import uuid

from huggingface_hub import snapshot_download

import comfyserver
import graphs
import models
from h3_gpu_monitor import GpuMonitor, host_peaks

KIT = pathlib.Path(__file__).resolve().parents[1]
SLOW_S = 60.0
BIG, SMALL = models.BIG, models.SMALL

# Attention classes and their names in each engine.
ATTN = {"fa": ("fa", "FLASH_ATTN"), "cudnn": ("torch_cudnn_sdpa", "CUDNN_ATTN"), "sdpa": ("torch_sdpa", "TORCH_SDPA"),
        "sage": ("sage_attn", "SAGE_ATTN"), "sage3": ("sage_attn_3", "SAGE_ATTN_3")}
MATCHED_ATTN = {"h200": "fa", "b200": "cudnn", "4090": "cudnn", "5090": "cudnn"}

H3_480 = dict(width=864, height=480, frames=124)
H3_576 = dict(width=1024, height=576, frames=124)
H3_768 = dict(width=1344, height=768, frames=124)


def _b(eid, model, *, cards, n_gpu=1, kind="matched", engines=("sglang", "vllm"), **spec):
    return dict(id=eid, track="B", model=model, cards=tuple(cards), n_gpu=n_gpu, kind=kind, engines=engines, spec=spec)


EXPERIMENTS = [
    _b("B-H3-01", "h3", cards=BIG, canvas=H3_480),
    _b("B-H3-02", "h3", cards=BIG, canvas=H3_768),
    _b("B-H3-03", "h3", cards=BIG, n_gpu=2, canvas=H3_768, par="u2"),
    _b("B-H3-04", "h3", cards=BIG, n_gpu=2, canvas=H3_768, par="tp2"),
    _b("B-H3-05", "h3", cards=("4090",), n_gpu=2, canvas=H3_576, par="tp2", dlo=12),
    _b("B-H3-06", "h3", cards=("5090",), n_gpu=2, canvas=H3_768, par="tp2", dlo=20),
    _b("B-H3-10", "h3", cards=BIG, kind="lossy", canvas=H3_768, quant="fp8"),
    _b("B-H3-11", "h3", cards=BIG, kind="lossy", canvas=H3_768, quant="int8"),
    _b("B-H3-12", "h3", cards=BIG, kind="lossy", canvas=H3_768, attn="sage"),
    _b("B-H3-13", "h3", cards=BIG, kind="lossy", canvas=H3_768, cache=True),
    _b("B-H3-14", "h3", cards=BIG, kind="lossy", canvas=H3_768, variant="vdn"),
    _b("B-H3-15", "h3", cards=SMALL, n_gpu=2, kind="lossy", canvas=H3_768, par="tp2", dlo=20, quant="fp8"),
    _b("B-H3-20", "h3", cards=models.CARDS, n_gpu=2, kind="recipe", canvas=H3_768),
    _b("B-H3-30", "h3", cards=BIG, canvas=H3_768, refs=8),
    _b("B-H3-31", "h3", cards=BIG, n_gpu=2, canvas=H3_768, refs=8, par="u2"),
    _b("B-H3-32", "h3", cards=SMALL, n_gpu=2, canvas=H3_768, refs=8, par="tp2", dlo=20),
    _b("B-LTX-01", "ltx", cards=models.CARDS, stage="one"),
    _b("B-LTX-02", "ltx", cards=models.CARDS, n_gpu=2, stage="one", par="u2"),
    _b("B-LTX-03", "ltx", cards=BIG, stage="two"),
    _b("B-LTX-04", "ltx", cards=BIG, n_gpu=2, stage="two", par="u2"),
    _b("B-LTX-10", "ltx", cards=models.CARDS, kind="lossy", stage="one", quant="fp8"),
    _b("B-LTX-11", "ltx", cards=models.CARDS, kind="lossy", stage="one", attn="sage"),
    _b("B-LTX-20", "ltx", cards=models.CARDS, kind="recipe", stage="one"),
    _b("B-QW-01", "qwen", cards=models.CARDS, cfg=1.0),
    _b("B-QW-02", "qwen", cards=models.CARDS, cfg=4.0),
    _b("B-QW-03", "qwen", cards=models.CARDS, n_gpu=2, cfg=4.0, par="cfg2"),
    _b("B-QW-04", "qwen", cards=models.CARDS, n_gpu=2, cfg=1.0, par="u2"),
    _b("B-QW-05", "qwen", cards=("h200", "5090"), cfg=1.0, edit=True),
    _b("B-QW-10", "qwen", cards=models.CARDS, kind="lossy", cfg=1.0, quant="fp8"),
    _b("B-QW-11", "qwen", cards=models.CARDS, kind="lossy", cfg=1.0, quant="int8"),
    _b("B-QW-12", "qwen", cards=models.CARDS, kind="lossy", cfg=1.0, attn="sage"),
    _b("B-QW-20", "qwen", cards=models.CARDS, kind="recipe", cfg=1.0),
]


# ---- model paths: both engines load the same pinned local snapshot --------------------------
def revision(repo):
    return next(r for name, r, _ in models.TRACK_B_REPOS if name == repo)


def snapshot(repo):
    rev, patterns = next((r, pat) for name, r, pat in models.TRACK_B_REPOS if name == repo)
    return snapshot_download(repo, revision=rev, allow_patterns=patterns, local_files_only=True)


# ---- server command lines ------------------------------------------------------------------
def _small_offload(card):
    """Card-memory placement both cookbooks recommend for one 24/32 GB GPU."""
    return card in SMALL


def sglang_server(e, card, port):
    s, m, n = e["spec"], e["model"], e["n_gpu"]
    attn = ATTN[s.get("attn") or MATCHED_ATTN[card]][0]
    env = {}
    if m == "h3":
        if s.get("variant") == "vdn":
            args = ["--model-path", snapshot(models.VDN_REPO), "--attention-backend", "hybrid_window_attn_h3",
                    "--quantization", "bf16"]
        else:
            args = ["--model-path", snapshot("MiniMaxAI/MiniMax-H3"), "--model-variant", "ref2va" if s.get("refs") else "fl2va",
                    "--attention-backend", attn]
    elif m == "ltx":
        # By repo id: SGLang re-validates the local LTX snapshot and fetches any component it lacks by name.
        args = ["--model-path", "Lightricks/LTX-2.5-Diffusers", "--revision", revision("Lightricks/LTX-2.5-Diffusers"),
                "--pipeline-class-name",
                "LTX2TwoStagePipeline" if s["stage"] == "two" else "LTX2Pipeline", "--attention-backend", attn]
    else:
        args = ["--model-path", snapshot("Qwen/Qwen-Image-2.1"), "--attention-backend", attn]
    args += ["--num-gpus", str(n)]
    par = s.get("par")
    if par == "u2":
        args += ["--ulysses-degree", "2", "--tp-size", "1"]
    elif par == "tp2":
        args += ["--tp-size", "2", "--ulysses-degree", "1"]
    if m == "qwen" and n == 2:
        args += ["--enable-cfg-parallel", "true" if par == "cfg2" else "false"]
    if s.get("dlo"):
        args += ["--performance-mode", "memory", "--layerwise-offload-components", "dit,text_encoder,vae",
                 "--dit-offload-prefetch-size", "1", "--dit-layerwise-resident-layers", str(s["dlo"]),
                 "--enable-torch-compile", "false"]
    elif _small_offload(card) and n == 1 and e["kind"] != "recipe":
        args += ["--performance-mode", "manual", "--dit-layerwise-offload", "true"]
    if s.get("quant") == "fp8":
        args += ["--component-quantizations.transformer", "fp8"]
    elif s.get("quant") == "int8":
        args += ["--component-quantizations.transformer", "convrot_int8"]
    if s.get("cache"):
        env["SGLANG_CACHE_DIT_ENABLED"] = "true"
    if e["kind"] == "recipe":
        args = _sglang_recipe(e, card, args)
    # Run the CLI through the venv's python so the bench-branch checkout on PYTHONPATH is what serves.
    cli = [str(pathlib.Path(os.environ["VENV"]) / "bin/python"), "-c",
           "import sys; from sglang.cli.main import main; sys.argv[0] = 'sglang'; sys.exit(main())"]
    args += os.environ.get("KIT_EXTRA_SGLANG_ARGS", "").split()
    return [*cli, "serve", *args, "--port", str(port)], env


def _sglang_recipe(e, card, args):
    """Cookbook recommendation on this card (replaces the matched topology flags)."""
    if e["model"] == "h3" and card in SMALL:
        return args + ["--tp-size", "2", "--performance-mode", "memory", "--layerwise-offload-components",
                       "dit,text_encoder,vae", "--dit-offload-prefetch-size", "1", "--dit-layerwise-resident-layers",
                       "12" if card == "4090" else "20", "--enable-torch-compile", "false"]
    if e["model"] == "h3":
        return args + ["--ulysses-degree", "2", "--encoder-parallel", "auto", "--performance-mode", "speed"]
    if e["model"] == "qwen" and card in BIG:
        return args + ["--performance-mode", "speed"]
    if e["model"] == "ltx" and card == "5090":
        return args + ["--layerwise-offload-components", "dit", "text_encoder"]
    return args


def vllm_server(e, card, port):
    s, m, n = e["spec"], e["model"], e["n_gpu"]
    attn = ATTN[s.get("attn") or MATCHED_ATTN[card]][1]
    env = {"VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_OMNI_VIDEO_SYNC_TIMEOUT": "14400"}
    if m == "h3":
        # By repo id: a local snapshot root carries modular_model_index.json, which vllm-omni reads as a
        # FastH3 release. HF_HOME is shared, so the pinned revision resolves to the same files.
        rev = next(r for name, r, _ in models.TRACK_B_REPOS if name == "MiniMaxAI/MiniMax-H3")
        args = ["MiniMaxAI/MiniMax-H3", "--revision", rev, "--trust-remote-code", "--task-type",
                "ref2va" if s.get("refs") else "fl2va"]
        if s.get("variant") == "vdn":
            args += ["--lora-path", snapshot(models.VDN_REPO)]
            attn = "VDNH3_ATTN"
    elif m == "ltx":
        args = ["Lightricks/LTX-2.5-Diffusers", "--revision", revision("Lightricks/LTX-2.5-Diffusers"), "--model-class-name",
                "LTX2DistilledTwoStagePipeline" if s["stage"] == "two" else "LTX2DistilledOneStagePipeline"]
    else:
        args = [snapshot("Qwen/Qwen-Image-2.1")]
    args += ["--omni", "--num-gpus", str(n), "--diffusion-attention-backend", attn]
    par = s.get("par")
    if par == "u2":
        args += ["--usp", "2"]
    elif par == "tp2":
        args += ["--tensor-parallel-size", "2", "--usp", "1"]
    elif par == "cfg2":
        args += ["--cfg-parallel-size", "2"]
    if s.get("dlo"):
        args += ["--text-encoder-tp-size", "2", "--ring", "1", "--vae-patch-parallel-size", "2", "--vae-parallel-mode",
                 "tile", "--vae-use-tiling", "--enable-distributed-layerwise-offload", "--dlo-no-use-allgather",
                 "--dlo-resident-layers", str(s["dlo"]), "--enforce-eager"]
    elif _small_offload(card) and n == 1 and e["kind"] != "recipe":
        args += ["--diffusion-offload-config", '{"mode":"layer","components":["dit","text_encoder"]}', "--enforce-eager"]
    if s.get("quant") in ("fp8", "int8"):
        args += ["--diffusion-quantization-config", json.dumps({"transformer": {"method": s["quant"]}})]
    if s.get("cache"):
        args += ["--cache-backend", "cache_dit"]
    if e["kind"] == "recipe":
        args = _vllm_recipe(e, card, args)
    args += os.environ.get("KIT_EXTRA_VLLM_ARGS", "").split()
    return [str(pathlib.Path(os.environ["VENV_VLLM"]) / "bin/vllm"), "serve", *args, "--host", "127.0.0.1",
            "--port", str(port)], env


def _vllm_recipe(e, card, args):
    if e["model"] == "h3" and card in SMALL:
        return args + ["--tensor-parallel-size", "2", "--usp", "1", "--text-encoder-tp-size", "2", "--ring", "1",
                       "--vae-patch-parallel-size", "2", "--vae-parallel-mode", "tile", "--vae-use-tiling",
                       "--enable-distributed-layerwise-offload", "--dlo-no-use-allgather", "--dlo-resident-layers",
                       "12" if card == "4090" else "20", "--enforce-eager"]
    if e["model"] == "h3":
        return args + ["--usp", "2"]
    return args


# ---- requests ------------------------------------------------------------------------------
H3_PROMPT = graphs.H3_PROMPT
LTX_PROMPT = graphs.LTX_PROMPT


def _refs(n):
    """Reference images pre-resized the way SGLang encodes them (2048 px short edge, LANCZOS,
    nearest 32 px). vllm-omni keeps an uploaded image's own size, so both engines then encode
    the same pixels."""
    from PIL import Image

    src_dir = pathlib.Path(os.environ["COMFY_DIR"]) / "input"
    out_dir = pathlib.Path(os.environ["TMPDIR"]) / "ref2048"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for name in graphs.ref_images(n):
        dst = out_dir / (pathlib.Path(name).stem + ".png")
        if not dst.exists():
            img = Image.open(src_dir / name).convert("RGB")
            scale = 2048 / min(img.size)
            size = tuple(max(32, round(v * scale / 32) * 32) for v in img.size)
            img.resize(size, Image.Resampling.LANCZOS).save(dst)
        out.append(dst)
    return out


def request(engine, e, steps, seed):
    """-> (kind, path, payload) where kind is "json" or "multipart"; payload for multipart is
    (fields, files)."""
    s, m = e["spec"], e["model"]
    if m == "qwen":
        body = {"prompt": graphs.QWEN_EDIT_PROMPT if s.get("edit") else graphs.QWEN_PROMPT, "size": "1024x1024",
                "num_inference_steps": steps, "true_cfg_scale": s["cfg"], "seed": seed, "response_format": "b64_json",
                "n": 1}
        if s["cfg"] != 1.0:
            body["negative_prompt"] = "blurry, low quality, distorted"
        if s.get("edit"):
            img = pathlib.Path(os.environ["COMFY_DIR"]) / "input" / "cmp_cat_small.png"
            fields = {k: str(v) for k, v in body.items()}
            return "multipart", "/v1/images/edits", (fields, [("image", img)])
        return "json", "/v1/images/generations", body
    if m == "ltx":
        w, h = (960, 544) if s["stage"] == "one" else (1920, 1088)
        fields = {"prompt": LTX_PROMPT, "width": w, "height": h, "num_frames": 121, "fps": 24, "seed": seed}
        return "multipart", "/v1/videos", ({k: str(v) for k, v in fields.items()}, [])
    c = s["canvas"]
    refs = s.get("refs", 0)
    task = "ref2va" if refs else "t2va"
    prompt = graphs.ref_prompt(refs) if refs else H3_PROMPT
    n_steps = 8 if s.get("variant") == "vdn" else steps
    if engine == "sglang":
        # MiniMax H3 derives frames from target.duration_seconds (5.0 s -> 124 frames on its 17k+5 grid).
        body = {"prompt": prompt, "task": task, "width": c["width"], "height": c["height"],
                "target": {"short_edge": min(c["width"], c["height"]), "aspect_ratio": "16:9",
                           "duration_seconds": round((c["frames"] - 4) / 24.0, 3)},
                "num_inference_steps": n_steps, "flow_shift": 12.0, "audio_flow_shift": 3.0, "seed": seed,
                "conditions": [{"type": "image", "uri": f"file://{p}", "role": "reference"} for p in _refs(refs)]}
        return "json", "/v1/videos", body
    fields = {"prompt": prompt, "width": c["width"], "height": c["height"], "aspect_ratio": "16:9", "fps": 24,
              "num_inference_steps": n_steps,
              "flow_shift": 12.0, "seed": seed,
              "extra_params": json.dumps({"task": task, "duration": round((c["frames"] - 4) / 24.0, 3), "audio_flow_shift": 3.0})}
    # One image goes in input_reference; several must use the list field input_references
    # (a repeated single-value field binds only its last value).
    field = "input_references" if refs > 1 else "input_reference"
    return "multipart", "/v1/videos", ({k: str(v) for k, v in fields.items()}, [(field, p) for p in _refs(refs)])


def _multipart(fields, files):
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    for k, p in files:
        ctype = "image/png" if p.suffix == ".png" else "image/jpeg"
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{p.name}"\r\n'
                     f"Content-Type: {ctype}\r\n\r\n".encode() + p.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def http(port, path, data=None, ctype="application/json", timeout=120, raw=False):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers={"Content-Type": ctype} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
        return body if raw else (json.loads(body) if body else {})


def submit(port, kind, path, payload, timeout):
    """Send one request; for video jobs poll until completed. Returns (status, seconds, response, error)."""
    t0 = time.perf_counter()
    try:
        if kind == "json":
            resp = http(port, path, json.dumps(payload).encode(), timeout=timeout)
        else:
            body, ctype = _multipart(*payload)
            resp = http(port, path, body, ctype, timeout=timeout)
    except urllib.error.HTTPError as ex:
        return "rejected", time.perf_counter() - t0, None, ex.read().decode(errors="replace")[:2000]
    except Exception as ex:  # noqa: BLE001
        return "error", time.perf_counter() - t0, None, f"{type(ex).__name__}: {ex}"
    if path.startswith("/v1/videos") and resp.get("id") and resp.get("status") not in ("completed",):
        vid = resp["id"]
        while True:
            if time.perf_counter() - t0 > timeout:
                return "timeout", time.perf_counter() - t0, resp, "timeout"
            try:
                resp = http(port, f"/v1/videos/{vid}")
            except urllib.error.HTTPError as ex:
                return "failed", time.perf_counter() - t0, None, f"poll {ex.code}: {ex.read().decode(errors='replace')[:2000]}"
            except Exception as ex:  # noqa: BLE001
                return "error", time.perf_counter() - t0, None, f"poll: {type(ex).__name__}: {ex}"
            if resp.get("status") == "completed":
                break
            if resp.get("status") == "failed":
                return "failed", time.perf_counter() - t0, resp, json.dumps(resp.get("error"))[:2000]
            time.sleep(0.05)
    return "success", time.perf_counter() - t0, resp, ""


def save_output(port, resp, out):
    try:
        if resp.get("data"):
            item = resp["data"][0]
            if item.get("b64_json"):
                (out / "output.png").write_bytes(base64.b64decode(item["b64_json"]))
            return
        if resp.get("id"):
            (out / "output.mp4").write_bytes(http(port, f"/v1/videos/{resp['id']}/content", raw=True, timeout=600))
    except Exception as ex:  # noqa: BLE001
        (out / "output-error.txt").write_text(f"{type(ex).__name__}: {ex}")


# ---- server lifecycle ----------------------------------------------------------------------
class Server:
    def __init__(self, suite, gpus, port):
        self.suite, self.gpus, self.port = suite, gpus, port
        self.proc, self.log, self.baseline, self.started_at = None, None, {}, None

    def start(self, argv, env, engine, monitor, timeout=3600):
        self.baseline = {str(g["nvml_index"]): g["used_gib"] for g in monitor.sample(time.perf_counter())["gpus"]}
        full = dict(os.environ, CUDA_VISIBLE_DEVICES=self.gpus, **env)
        if engine == "vllm":
            # The vllm-omni venv must not see SGLang / ComfyUI on its import path.
            full.pop("PYTHONPATH", None)
            full["VIRTUAL_ENV"] = os.environ["VENV_VLLM"]
            full["PATH"] = f"{os.environ['VENV_VLLM']}/bin:" + full["PATH"]
        self.log = self.suite / "logs" / f"{engine}-{int(time.time())}.log"
        self.log.parent.mkdir(parents=True, exist_ok=True)
        (self.log.with_suffix(".cmd")).write_text(" ".join(argv))
        self.started_at = time.perf_counter()
        self.proc = subprocess.Popen(argv, env=full, stdout=open(self.log, "w"), stderr=subprocess.STDOUT,
                                     start_new_session=True, cwd=os.environ["TMPDIR"])
        while time.perf_counter() - self.started_at < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{engine} server exited during startup; see {self.log}")
            # /health stays 503 during the server's own warmup; /v1/models only if /health does not exist.
            try:
                http(self.port, "/health", timeout=5, raw=True)
                return
            except urllib.error.HTTPError as ex:
                if ex.code == 404:
                    try:
                        http(self.port, "/v1/models", timeout=5, raw=True)
                        return
                    except Exception:
                        pass
            except Exception:
                pass
            time.sleep(3)
        raise RuntimeError(f"{engine} server did not become healthy in {timeout}s; see {self.log}")

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=90)
            except Exception:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except Exception:
                    pass
            time.sleep(5)
        self.proc = None


def run_one(srv, monitor, engine, e, steps, seed, run_id, keep, timeout):
    out = srv.suite / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    kind, path, payload = request(engine, e, steps, seed)
    (out / "request.json").write_text(json.dumps({"kind": kind, "path": path, "payload": payload if kind == "json" else
                                                  {"fields": payload[0], "files": [str(f[1]) for f in payload[1]]}}, indent=1))
    t0 = time.perf_counter()
    _, finish = monitor.start(t0, interval=0.2, session_pid=srv.proc.pid)
    status, seconds, resp, err = submit(srv.port, kind, path, payload, timeout)
    samples, host = finish()
    res = {"run_id": run_id, "status": status, "e2e_s": round(seconds, 3), "error": err, "done_at": time.perf_counter()}
    res.update(GpuMonitor.peaks(samples))
    res.update(host_peaks(host))
    res.update(comfyserver.usage_summary(samples))
    res["peak_vram_gib"] = {k: round(v - srv.baseline.get(k, 0.0), 2) for k, v in res["peak_gpu_gib_by_nvml_index"].items()}
    comfyserver.write_usage(out / "usage.csv", samples)
    if resp:
        res["server_inference_s"] = resp.get("inference_time_s")
        res["server_peak_memory_mb"] = resp.get("peak_memory_mb")
        (out / "response.json").write_text(json.dumps({k: v for k, v in resp.items() if k != "data"}, indent=1)[:20000])
        if keep and status == "success":
            save_output(srv.port, resp, out)
    (out / "result.json").write_text(json.dumps(res, indent=1))
    return res


def video_seconds(e):
    if e["model"] == "h3":
        return e["spec"]["canvas"]["frames"] / 24.0
    if e["model"] == "ltx":
        return 121 / 24.0
    return None


def run_experiment(e, engine, card, suite, gpus, port, timeout, commits, max_runs=5, steps=20):
    eid = f"{e['id']}@{engine}"
    argv, env = (sglang_server if engine == "sglang" else vllm_server)(e, card, port)
    mon = comfyserver.make_monitor("all" if e["n_gpu"] == 2 else "gpu0")
    srv = Server(suite, gpus, port)
    base = {"exp_id": e["id"], "engine": engine, "track": "B", "card": card, "n_gpu": e["n_gpu"], "model": e["model"],
            "kind": e["kind"], "commit": commits, "config": {"argv": argv, "env": env, **e["spec"]}}
    rows = []
    try:
        srv.start(argv, env, engine, mon)
        warm_steps = 3 if e["model"] in ("h3", "qwen") and e["spec"].get("variant") != "vdn" else 20
        res = run_one(srv, mon, engine, e, warm_steps, 999, f"{eid}__w", False, timeout)
        cold = res["done_at"] - srv.started_at
        _append(suite, {**base, "phase": "warmup", **res, "cold_s": round(cold, 2)})
        if res["status"] == "success":
            n_runs, i = max_runs, 0
            while i < n_runs:
                res = run_one(srv, mon, engine, e, steps, 1000 + i, f"{eid}__m{i}", i == 0, timeout)
                vs = video_seconds(e)
                row = {**base, "phase": "measured", "run_idx": i, **res,
                       "gpu_s_per_video_s": res["e2e_s"] * e["n_gpu"] / vs if vs else None,
                       "gpu_s_per_image": res["e2e_s"] * e["n_gpu"] if vs is None else None}
                _append(suite, row)
                rows.append(row)
                print(f"{eid:22s} m{i}: {res['status']} e2e={res['e2e_s']}s server={res.get('server_inference_s')} "
                      f"{str(res['error'])[:120]}", flush=True)
                if res["status"] != "success":
                    break
                if i == 0 and res["e2e_s"] >= SLOW_S:
                    n_runs = min(n_runs, 3)
                i += 1
        else:
            print(f"{eid:22s} warmup failed: {res['status']} {str(res['error'])[:300]}", flush=True)
    except Exception:
        err = traceback.format_exc()[-1500:]
        print(f"{eid} crashed:\n{err}", flush=True)
        rows.append({"status": "crashed", "error": err})
        cold = None
    finally:
        srv.stop()
    ok = [r for r in rows if r["status"] == "success"]
    summary = {**base, "phase": "summary", "status": "success" if ok and len(ok) == len(rows) else "failed",
               "runs": len(ok), "cold_s": round(cold, 2) if cold else None,
               "error": (rows[-1]["error"] if rows and rows[-1]["status"] != "success" else "")[:1500]}
    for key in ("e2e_s", "gpu_s_per_video_s", "gpu_s_per_image"):
        vals = [r[key] for r in ok if r.get(key) is not None]
        if vals:
            summary[key] = {"median": statistics.median(vals), "min": min(vals), "max": max(vals)}
    if ok:
        summary["peak_vram_gib"] = {k: max(r["peak_vram_gib"][k] for r in ok) for k in ok[0]["peak_vram_gib"]}
        summary["peak_rss_gib"] = max((r.get("peak_host_pss_gib") or 0) for r in ok)
    _append(suite, summary)


def _append(suite, row):
    with open(suite / "results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def _done(suite, eid, engine, retry_failed=False):
    f = suite / "results.jsonl"
    if not f.exists():
        return False
    return any(r.get("exp_id") == eid and r.get("engine") == engine and r.get("phase") == "summary"
               and (r.get("status") == "success" or not retry_failed) for r in map(json.loads, f.read_text().splitlines()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--gpus", type=int, choices=(1, 2))
    ap.add_argument("--only", type=lambda s: s.split(","))
    ap.add_argument("--engines", default="sglang,vllm")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retry-failed", action="store_true", help="re-run experiments whose summary is not success")
    ap.add_argument("--runs", type=int, default=5, help="measured runs (debugging only; the protocol is 5 / 3)")
    ap.add_argument("--steps", type=int, default=20, help="denoising steps (debugging only; the protocol is 20)")
    a = ap.parse_args()
    card = os.environ["CARD"]
    suite = pathlib.Path(a.suite).resolve()
    (suite / "runs").mkdir(parents=True, exist_ok=True)
    timeout = float(os.environ.get("TIMEOUT_S", "5400"))
    g = comfyserver.gpus()
    port = int(os.environ.get("PORT_BASE", "8190")) + 50
    engines = a.engines.split(",")
    exps = [e for e in EXPERIMENTS if card in e["cards"] and (a.gpus is None or e["n_gpu"] == a.gpus)
            and (not a.only or any(e["id"] == o or (o.endswith("*") and e["id"].startswith(o[:-1])) for o in a.only))]
    exps.sort(key=lambda e: e["n_gpu"])
    commits = {k: subprocess.run(["git", "-C", os.environ[v], "rev-parse", "--short=10", "HEAD"], capture_output=True,
                                 text=True).stdout.strip() for k, v in (("sglang", "SGLANG_SRC"), ("vllm_omni", "VLLM_OMNI_SRC"))}
    for e in exps:
        for engine in engines:
            if engine not in e["engines"]:
                continue
            if a.dry_run:
                argv, env = (sglang_server if engine == "sglang" else vllm_server)(e, card, port)
                print(f"{e['id']:10s} {engine:6s} gpus={e['n_gpu']} {env} {' '.join(argv[argv.index('serve'):])}")
                print(f"{'':10s} request: {json.dumps(request(engine, e, 20, 1000), default=str)[:400]}")
                continue
            if _done(suite, e["id"], engine, a.retry_failed):
                print(f"skip {e['id']}@{engine} (done)", flush=True)
                continue
            gpus = ",".join(g) if e["n_gpu"] == 2 else g[0]
            run_experiment(e, engine, card, suite, gpus, port, timeout, commits, a.runs, a.steps)
    print("TRACK_B_DONE", flush=True)


if __name__ == "__main__":
    main()
