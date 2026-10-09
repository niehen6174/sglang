"""Time the public ComfyUI prompt/history and websocket APIs per workflow."""

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import time
import traceback
import urllib.request
import uuid

import websocket
from h3_gpu_monitor import GpuMonitor

ROOT = pathlib.Path(os.environ["H3_BENCH_WORKSPACE"]).resolve()
SUITE = pathlib.Path(os.environ["H3_BENCH_SUITE"]).resolve()
HOST = os.environ.get("H3_BENCH_HOST", "127.0.0.1:8188")
BASE = "http://" + HOST
SERVER_LOG = pathlib.Path(
    os.environ.get("H3_BENCH_SERVER_LOG", ROOT / "logs/comfyui-server.log")
)


def api(path, data=None):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read()
        return json.loads(body) if body else {}


def source_fingerprint():
    relative = [
        "sglang/python/sglang/multimodal_gen/runtime/managers/gpu_worker.py",
        "sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/comfyui_step.py",
        "sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/sage_attn.py",
        "sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/sage_attn3.py",
        "sglang/python/sglang/multimodal_gen/runtime/loader/fsdp_load.py",
        "sglang/python/sglang/multimodal_gen/runtime/loader/weight_load_plan.py",
        "sglang/python/sglang/multimodal_gen/runtime/loader/component_loaders/transformer_loader.py",
        "sglang/python/sglang/multimodal_gen/runtime/entrypoints/diffusion_generator.py",
        "ComfyUI/comfy/model_base.py",
        "ComfyUI/comfy/ldm/minimax/model.py",
        "ComfyUI/custom_nodes/H3Benchmark/__init__.py",
    ]
    return {
        "sources": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in relative
        },
        "environment": {
            name: os.environ.get(name)
            for name in [
                "COMFY_DISABLE_CUDA_MALLOC",
                "PYTORCH_CUDA_ALLOC_CONF",
                "PYTORCH_ALLOC_CONF",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
                "SGLANG_DIFFUSION_STAGE_LOGGING",
                "SGLANG_DIFFUSION_SYNC_STAGE_PROFILING",
                "SGLANG_KITCHEN_INT8_MAX_ROWS",
                "SGLANG_KITCHEN_INT8_MIN_SPLIT_N",
            ]
        },
    }


def comparable_workflow(workflow):
    result = {}
    for key, node in workflow.items():
        inputs = dict(node["inputs"])
        if node["class_type"] in ["H3BenchmarkModel", "H3BenchmarkLatent"]:
            inputs.pop("run_id", None)
        if node["class_type"] == "SaveVideo":
            inputs.pop("filename_prefix", None)
        if node["class_type"] == "RandomNoise":
            inputs.pop("noise_seed", None)
        result[key] = {"class_type": node["class_type"], "inputs": inputs}
    return json.dumps(result, sort_keys=True)


def reuse_equivalent_profile(args, template, fingerprint):
    suffix = "_optimized_profile_native"
    if (
        args.plain
        or args.changed_prompt
        or args.runs != 3
        or not args.case.endswith(suffix)
    ):
        return False
    base = args.case[: -len(suffix)]
    if base not in ["t2av", "ref2av", "vdn"]:
        return False
    control = base + "_fixed_profile_native"
    runs = SUITE / (control + ".json")
    record = SUITE / "runs" / (control + "_00")
    needed = [
        runs,
        record / "source-sha256.json",
        record / "workflow.json",
        record / "latent-info.json",
    ]
    if not all(p.exists() for p in needed):
        return False
    data = json.loads(runs.read_text())
    if len(data) != 3 or any(
        x.get("forward_calls") != (8 if base == "vdn" else 20) for x in data
    ):
        return False
    if json.loads((record / "source-sha256.json").read_text()) != fingerprint:
        return False
    if comparable_workflow(
        json.loads((record / "workflow.json").read_text())
    ) != comparable_workflow(template):
        return False
    if not all(
        x["finite"]
        for x in json.loads((record / "latent-info.json").read_text()).values()
    ):
        return False
    proof = {
        "case": args.case,
        "reused_control_case": control,
        "note": "Same source, environment, model/sampler/conditioning workflow. No additional samples were measured or counted.",
        "control_file": str(runs),
        "source_and_environment": fingerprint,
    }
    (SUITE / (args.case + "-reuse.json")).write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof), flush=True)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--workflow", required=True)
    p.add_argument("--runs", type=int, default=7)
    p.add_argument("--changed-prompt", action="store_true")
    p.add_argument("--plain", action="store_true")
    p.add_argument("--record-final-latent", action="store_true")
    p.add_argument("--expected-nfe", type=int)
    args = p.parse_args()
    if args.record_final_latent and not args.plain:
        raise ValueError(
            "Final-only latent capture requires --plain; it must not install per-step timing wrappers"
        )
    template = json.loads(pathlib.Path(args.workflow).read_text())
    fingerprint = source_fingerprint()
    if reuse_equivalent_profile(args, template, fingerprint):
        return
    if args.plain:
        replacements = {
            key: node["inputs"][
                "model" if node["class_type"] == "H3BenchmarkModel" else "latent"
            ]
            for key, node in template.items()
            if node["class_type"] == "H3BenchmarkModel"
            or (
                node["class_type"] == "H3BenchmarkLatent"
                and not args.record_final_latent
            )
        }

        def replace(value):
            if (
                isinstance(value, list)
                and len(value) == 2
                and isinstance(value[0], str)
                and value[0] in replacements
            ):
                return replacements[value[0]]
            if isinstance(value, dict):
                return {key: replace(item) for key, item in value.items()}
            if isinstance(value, list):
                return [replace(item) for item in value]
            return value

        template = {
            key: replace(node)
            for key, node in template.items()
            if key not in replacements
        }
    summaries = []
    gpu = GpuMonitor()
    (SUITE / (args.case + "-gpus.json")).write_text(
        json.dumps({"host": HOST, "gpus": gpu.identity}, indent=2)
    )
    for i in range(args.runs):
        if shutil.disk_usage(ROOT).free < 100 * 2**30:
            raise RuntimeError("Data disk reserve exhausted")
        queue = api("/queue")
        if queue["queue_running"] or queue["queue_pending"]:
            raise RuntimeError("Service is busy")
        run_id = args.case + "_" + str(i).zfill(2)
        out = SUITE / "runs" / run_id
        out.mkdir(parents=True, exist_ok=False)
        (out / "source-sha256.json").write_text(json.dumps(fingerprint, indent=2))
        w = json.loads(json.dumps(template))
        for node in w.values():
            if node["class_type"] == "RandomNoise":
                node["inputs"]["noise_seed"] = 150000 + i
            if node["class_type"] in ["H3BenchmarkModel", "H3BenchmarkLatent"]:
                node["inputs"]["run_id"] = run_id
            if node["class_type"] == "SaveVideo":
                node["inputs"]["filename_prefix"] = "video/h3_performance/" + run_id
            if args.changed_prompt and "prompt" in node["inputs"]:
                node["inputs"][
                    "prompt"
                ] += f" A small cloud is visible in shot number {i}."
        (out / "workflow.json").write_text(json.dumps(w, indent=2))
        client = uuid.uuid4().hex
        ws = websocket.create_connection(
            "ws://" + HOST + "/ws?clientId=" + client, timeout=1800
        )
        perf_dir = os.environ.get("SGLANG_PERF_LOG_DIR")
        perf = pathlib.Path(perf_dir) / "performance.log" if perf_dir else None
        perf_offset = perf.stat().st_size if perf is not None and perf.exists() else 0
        log = SERVER_LOG
        offset = log.stat().st_size
        events = []
        processes_before = gpu.processes()
        start = time.perf_counter()
        samples, finish_samples = gpu.start(start)
        try:
            accepted = api("/prompt", {"prompt": w, "client_id": client})
            pid = accepted["prompt_id"]
            (out / "submission.json").write_text(json.dumps(accepted, indent=2))
            while True:
                message = ws.recv()
                if not isinstance(message, str):
                    continue
                data = json.loads(message)
                events.append({"seconds": time.perf_counter() - start, **data})
                if data.get("data", {}).get("prompt_id") != pid:
                    continue
                if data["type"] == "execution_error":
                    raise RuntimeError(data)
                if data["type"] == "executing" and data["data"]["node"] is None:
                    break
            wall = time.perf_counter() - start
            history = api("/history/" + pid)[pid]
            assert history["status"]["status_str"] == "success"
            profile = {"sampler_seconds": [], "apply_model_seconds": []}
            if not args.plain:
                profile = json.loads((out / "sampler-profile.json").read_text())
                assert len(profile["apply_model_seconds"]) == (
                    args.expected_nfe
                    if args.expected_nfe is not None
                    else (8 if args.case.startswith("vdn") else 20)
                ), profile
                latent = json.loads((out / "latent-info.json").read_text())
                assert all(v["finite"] for v in latent.values())
            elif args.record_final_latent:
                assert not any(
                    n["class_type"] == "H3BenchmarkModel" for n in w.values()
                )
                latent = json.loads((out / "latent-info.json").read_text())
                assert all(v["finite"] for v in latent.values())
            intervals = []
            active = None
            for e in events:
                if e["type"] == "executing" and e["data"].get("prompt_id") == pid:
                    if active is not None:
                        intervals.append(
                            {
                                "node": active[0],
                                "class_type": w[active[0]]["class_type"],
                                "seconds": e["seconds"] - active[1],
                            }
                        )
                    active = (
                        (e["data"]["node"], e["seconds"])
                        if e["data"]["node"] is not None
                        else None
                    )
            if args.plain:
                profile["sampler_seconds"] = [
                    x["seconds"]
                    for x in intervals
                    if x["class_type"] == "SamplerCustomAdvanced"
                ]
            media = []
            for outputs in history["outputs"].values():
                for key, items in outputs.items():
                    if not isinstance(items, list):
                        continue
                    for d in items:
                        if not isinstance(d, dict) or not all(
                            k in d for k in ["filename", "subfolder", "type"]
                        ):
                            continue
                        source = (
                            ROOT
                            / "ComfyUI"
                            / d["type"]
                            / d["subfolder"]
                            / d["filename"]
                        )
                        if source.suffix not in [".mp4", ".webm", ".mkv"]:
                            continue
                        probe = json.loads(
                            subprocess.run(
                                [
                                    "ffprobe",
                                    "-v",
                                    "error",
                                    "-show_streams",
                                    "-show_format",
                                    "-of",
                                    "json",
                                    str(source),
                                ],
                                capture_output=True,
                                text=True,
                                check=True,
                            ).stdout
                        )
                        assert {"video", "audio"} <= {
                            s["codec_type"] for s in probe["streams"]
                        }
                        video = next(
                            s for s in probe["streams"] if s["codec_type"] == "video"
                        )
                        assert (
                            video["width"],
                            video["height"],
                            int(video["nb_frames"]),
                        ) == (864, 480, 107), video
                        media.append({"bytes": source.stat().st_size, "probe": probe})
                        if i == 0:
                            shutil.copy2(source, out / source.name)
                        assert source.resolve().parent == (
                            ROOT / "ComfyUI/output/video/h3_performance"
                        ).resolve() and source.name.startswith(run_id)
                        source.unlink()
            assert media
            result = {
                "case": args.case,
                "run": i,
                "cold": i == 0,
                "warmup": i == 1,
                "wall_seconds": wall,
                "sampler_seconds": sum(profile["sampler_seconds"]),
                "apply_model_seconds": (
                    sum(profile["apply_model_seconds"]) if not args.plain else None
                ),
                "forward_calls": (
                    len(profile["apply_model_seconds"]) if not args.plain else None
                ),
                "instrumented": not args.plain,
                "node_intervals": intervals,
                "media": media,
                **GpuMonitor.peaks(samples),
                "gpu_processes_before": processes_before,
            }
            if perf is not None and perf.exists():
                with perf.open("rb") as stream:
                    stream.seek(perf_offset)
                    worker = [
                        json.loads(line)
                        for line in stream.read().decode().splitlines()
                        if line.strip()
                    ]
                (out / "worker-profile.json").write_text(json.dumps(worker, indent=2))
                result["worker_total_seconds"] = (
                    sum(x["total_duration_ms"] for x in worker) / 1000
                )
                result["worker_step_seconds"] = (
                    sum(
                        s["execution_time_ms"]
                        for x in worker
                        for s in x["stages"]
                        if s["name"] == "MiniMaxH3ComfyUIStepStage"
                    )
                    / 1000
                )
                result["worker_profile_calls"] = len(worker)
            (out / "result.json").write_text(json.dumps(result, indent=2))
            (out / "history.json").write_text(json.dumps(history, indent=2))
            summaries.append(result)
            (SUITE / (args.case + ".json")).write_text(json.dumps(summaries, indent=2))
            api("/history", {"delete": [pid]})
            print(json.dumps(result), flush=True)
        except Exception as error:
            (out / "failure.json").write_text(
                json.dumps(
                    {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                    indent=2,
                )
            )
            raise
        finally:
            finish_samples()
            ws.close()
            (out / "events.json").write_text(json.dumps(events, indent=2))
            (out / "gpu-samples.json").write_text(json.dumps(samples, indent=2))
            with log.open("rb") as stream:
                stream.seek(offset)
                (out / "server.log").write_bytes(stream.read())
        time.sleep(2)
    gpu.shutdown()


if __name__ == "__main__":
    main()
