"""One ComfyUI server per lane, and one prompt run with GPU / host sampling.

A lane is one GPU (gpu0, gpu1) or both ("all"). Every experiment starts a fresh server with its
own flags and environment; the SGLD worker inherits that environment.
"""
import json
import os
import pathlib
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
import uuid

from h3_gpu_monitor import GpuMonitor, host_peaks

KIT = pathlib.Path(os.environ["KIT_ROOT"])
COMFY = pathlib.Path(os.environ["COMFY_DIR"])
LOG_MARKERS = ("effective backend", "convrot_int8", "comfy_kitchen", "Rejected", "Traceback", "Error",
               "rejected", "fallback", "Falling back", "cache-dit", "Cache-DiT", "offload", "OutOfMemory",
               "out of memory", "CUDA out")
MEDIA = (".mp4", ".webm", ".png")


def api(port, path, data=None, timeout=60):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
        return json.loads(body) if body else {}


def gpus():
    return [g.strip() for g in os.environ.get("GPUS", "0,1").split(",") if g.strip()]


def lane_gpus(lane):
    g = gpus()
    return ",".join(g) if lane == "all" else g[int(lane[3:])]


def lane_port(lane):
    base = int(os.environ.get("PORT_BASE", "8190"))
    return base + (len(gpus()) if lane == "all" else int(lane[3:]))


class Server:
    def __init__(self, lane, suite):
        self.lane, self.suite = lane, pathlib.Path(suite)
        self.gpus, self.port = lane_gpus(lane), lane_port(lane)
        self.proc, self.log, self.started_at, self.baseline = None, None, None, {}

    def start(self, flags, env, monitor=None):
        """monitor: sample device memory before launch, so peaks exclude memory held by others."""
        self.stop()
        if monitor is not None:
            self.baseline = {str(g["nvml_index"]): g["used_gib"] for g in monitor.sample(time.perf_counter())["gpus"]}
        try:
            api(self.port, "/system_stats", timeout=3)
        except Exception:
            pass
        else:
            raise RuntimeError(f"port {self.port} is already served by another ComfyUI; change PORT_BASE")
        full_env = dict(os.environ, H3_BENCH_SUITE=str(self.suite), COMFY_EXTRA_ARGS=flags, **env)
        self.log = self.suite / "logs" / f"comfy-{self.lane}-{int(time.time())}.log"
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.started_at = time.perf_counter()
        self.proc = subprocess.Popen([str(KIT / "scripts/start-comfyui.sh"), self.gpus, str(self.port)], env=full_env,
                                     stdout=open(self.log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(300):
            if self.proc.poll() is not None:
                raise RuntimeError(f"ComfyUI exited during startup; see {self.log}")
            try:
                api(self.port, "/object_info/SGLDOptions", timeout=5)
                return
            except Exception:
                time.sleep(2)
        raise RuntimeError(f"{self.lane}: ComfyUI did not come up; see {self.log}")

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=60)
            except Exception:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except Exception:
                    pass
            time.sleep(3)
        self.proc = None

    def log_tail(self, offset):
        if not (self.log and self.log.exists()):
            return ""
        with open(self.log, "rb") as f:
            f.seek(offset)
            return f.read().decode(errors="replace")

    def log_size(self):
        return self.log.stat().st_size if self.log and self.log.exists() else 0


def _probe_media(path):
    if path.suffix == ".png":
        return {"image": path.name}
    r = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)], capture_output=True, text=True)
    if r.returncode:
        return {"error": r.stderr[-300:]}
    s = json.loads(r.stdout)["streams"]
    v = next((x for x in s if x["codec_type"] == "video"), None)
    a = next((x for x in s if x["codec_type"] == "audio"), None)
    return {"video": v and {"w": v["width"], "h": v["height"], "frames": int(v.get("nb_frames", 0))},
            "audio": a and {"rate": a["sample_rate"], "channels": a["channels"]}}


def _collect_outputs(hist, out, keep_media):
    media = None
    for o in hist.get("outputs", {}).values():
        for item in (i for v in o.values() if isinstance(v, list) for i in v):
            if isinstance(item, dict) and item.get("type") == "output" and item.get("filename", "").endswith(MEDIA):
                src = COMFY / "output" / item.get("subfolder", "") / item["filename"]
                if not src.exists():
                    continue
                if keep_media:
                    dst = out / src.name
                    shutil.move(str(src), dst)
                    media = _probe_media(dst)
                else:
                    media = _probe_media(src)
                    src.unlink()
    return media


def run_prompt(srv, graph, run_id, monitor, timeout, keep_media=False):
    """Submit one graph and wait. Returns the result dict; artifacts go to SUITE/runs/<run_id>/."""
    out = srv.suite / "runs" / run_id
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    graph = json.loads(json.dumps(graph))
    for node in graph.values():
        if node["class_type"] in ("H3BenchmarkLatent", "H3BenchmarkModel"):
            node["inputs"]["run_id"] = run_id
    (out / "workflow.json").write_text(json.dumps(graph, indent=1))
    log_off = srv.log_size()
    res = {"run_id": run_id}
    start = time.perf_counter()
    _, finish = monitor.start(start, interval=0.2, session_pid=srv.proc.pid if srv.proc else None)
    try:
        pid = api(srv.port, "/prompt", {"prompt": graph, "client_id": uuid.uuid4().hex})["prompt_id"]
    except urllib.error.HTTPError as e:
        res.update(status="rejected_at_submit", error=e.read().decode()[:3000])
        pid = None
    while pid:
        try:
            h = api(srv.port, f"/history/{pid}")
        except Exception as e:
            res.update(status="server_lost", error=f"{type(e).__name__}: {e}")
            break
        if pid in h:
            st = h[pid]["status"]
            res["status"] = st.get("status_str")
            stamps = {m[0]: m[1].get("timestamp") for m in st.get("messages", []) if isinstance(m[1], dict)}
            if stamps.get("execution_start") and stamps.get("execution_success"):
                res["exec_s"] = (stamps["execution_success"] - stamps["execution_start"]) / 1000.0
            errs = [m[1] for m in st.get("messages", []) if m[0] == "execution_error"]
            if errs:
                e = errs[0]
                res["error"] = f"{e.get('node_type')}: {e.get('exception_type')}: {(e.get('exception_message') or '')[:2500]}"
            res["media"] = _collect_outputs(h[pid], out, keep_media)
            api(srv.port, "/history", {"delete": [pid]})
            break
        if time.perf_counter() - start > timeout:
            res["status"] = "timeout"
            srv.stop()
            break
        time.sleep(0.05)
    res["client_s"] = round(time.perf_counter() - start, 3)
    # ComfyUI's own start/success timestamps; the client wall time includes up to one poll interval.
    res["e2e_s"] = round(res.get("exec_s") or res["client_s"], 3)
    res["done_at"] = time.perf_counter()
    gpu_samples, host_samples = finish()
    res.update(GpuMonitor.peaks(gpu_samples))
    res["peak_vram_gib"] = {k: round(v - srv.baseline.get(k, 0.0), 2) for k, v in res["peak_gpu_gib_by_nvml_index"].items()}
    res.update(host_peaks(host_samples))
    res.update(usage_summary(gpu_samples))
    write_usage(out / "usage.csv", gpu_samples)
    for name, key in (("latent-info.json", "latents"), ("sampler-profile.json", "profile")):
        f = out / name
        if f.exists():
            res[key] = json.loads(f.read_text())
    seg = srv.log_tail(log_off)
    (out / "server.log").write_text(seg)
    res["log_markers"] = sorted({l.strip()[:220] for l in seg.splitlines() if any(m in l for m in LOG_MARKERS)})[:40]
    (out / "result.json").write_text(json.dumps(res, indent=1))
    return res


def usage_summary(samples):
    """Mean utilization / power / SM clock per GPU over the run."""
    acc = {}
    for s in samples:
        for g in s["gpus"]:
            a = acc.setdefault(str(g["nvml_index"]), {"util": 0.0, "power_w": 0.0, "sm_mhz": 0.0, "n": 0})
            a["util"] += g["util"]
            a["power_w"] += g["power_w"]
            a["sm_mhz"] += g["sm_mhz"]
            a["n"] += 1
    return {"mean_util": {k: round(v["util"] / v["n"], 1) for k, v in acc.items()},
            "mean_power_w": {k: round(v["power_w"] / v["n"], 1) for k, v in acc.items()},
            "mean_sm_mhz": {k: round(v["sm_mhz"] / v["n"]) for k, v in acc.items()}}


def write_usage(path, samples):
    with open(path, "w") as f:
        f.write("seconds,nvml_index,used_gib,util,power_w,sm_mhz,temperature_c\n")
        for s in samples:
            for g in s["gpus"]:
                f.write(f"{s['seconds']:.2f},{g['nvml_index']},{g['used_gib']:.3f},{g['util']},{g['power_w']:.1f},"
                        f"{g['sm_mhz']},{g['temperature_c']}\n")


def make_monitor(lane):
    """GpuMonitor reads H3_BENCH_GPUS when built; build monitors one at a time."""
    os.environ["H3_BENCH_GPUS"] = lane_gpus(lane)
    return GpuMonitor()
