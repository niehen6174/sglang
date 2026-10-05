"""Benchmark the public SGLang full H3 pipeline with existing local weights."""

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import subprocess
import threading
import time
import traceback

import torch
import pynvml
from sglang.multimodal_gen import DiffGenerator
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3_vdn import (
    VDNH3PipelineConfig,
)

ROOT = pathlib.Path(os.environ["H3_BENCH_WORKSPACE"]).resolve()
SUITE = pathlib.Path(os.environ["H3_BENCH_SUITE"]).resolve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["t2av", "ref2av", "vdn"], required=True)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--backend", default="torch_sdpa")
    parser.add_argument("--label")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--prefetch", type=int, default=0)
    parser.add_argument("--resident", type=int, default=0)
    parser.add_argument("--quantization")
    parser.add_argument("--changed-prompt", action="store_true")
    args = parser.parse_args()
    if args.profile:
        os.environ["SGLANG_DIFFUSION_STAGE_LOGGING"] = "1"
        os.environ["SGLANG_DIFFUSION_SYNC_STAGE_PROFILING"] = "1"
    out = SUITE / "direct" / (args.label or args.case)
    out.mkdir(parents=True, exist_ok=False)
    if args.profile:
        os.environ["SGLANG_PERF_LOG_DIR"] = str(out)
    torch.set_num_threads(8)
    source_names = [
        "runtime/layers/quantization/kitchen_int8.py",
        "runtime/layers/attention/backends/sage_attn.py",
        "runtime/layers/attention/backends/sage_attn3.py",
        "runtime/loader/component_loaders/transformer_loader.py",
        "runtime/loader/fsdp_load.py",
        "runtime/managers/gpu_worker.py",
        "runtime/entrypoints/diffusion_generator.py",
    ]
    prefix = ROOT / "sglang/python/sglang/multimodal_gen"
    provenance = {
        "torch_num_threads": torch.get_num_threads(),
        "sources": {
            name: hashlib.sha256((prefix / name).read_bytes()).hexdigest()
            for name in source_names
        },
        "environment": {
            name: os.environ.get(name)
            for name in [
                "PYTORCH_CUDA_ALLOC_CONF",
                "PYTORCH_ALLOC_CONF",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "SGLANG_DIFFUSION_STAGE_LOGGING",
                "SGLANG_DIFFUSION_SYNC_STAGE_PROFILING",
                "SGLANG_KITCHEN_INT8_MAX_ROWS",
                "SGLANG_KITCHEN_INT8_MIN_SPLIT_N",
            ]
        },
    }
    (out / "execution-environment.json").write_text(json.dumps(provenance, indent=2))
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    vdn = args.case == "vdn"
    ref = args.case == "ref2av"
    variant = "Ref2VA" if ref else "FL2VA"
    config = VDNH3PipelineConfig() if vdn else MiniMaxH3PipelineConfig()
    if not vdn:
        config.dit_config.arch_config.qkv_checkpoint_grouped = False
    config.text_encoder_precisions = ("fp16",)
    config.vae_precision = "fp16"
    metadata = ROOT / "models/H3-native-benchmark" / variant
    options = dict(
        model_path=str(ROOT / "models/H3-native-benchmark"),
        pipeline_class_name="MiniMaxH3Pipeline",
        model_variant="ref2va" if ref else "fl2va",
        pipeline_config=config,
        component_paths={
            name: str(metadata / name)
            for name in [
                "text_encoder",
                "tokenizer",
                "processor",
                "video_vae",
                "audio_vae",
            ]
        },
        component_weights_paths={
            "text_encoder": str(
                ROOT
                / "ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
            ),
            "video_vae": str(
                ROOT / "ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors"
            ),
            "audio_vae": str(
                ROOT / "ComfyUI/models/vae/minimax_h3_audio_vae_fp32.safetensors"
            ),
        },
        num_gpus=1,
        attention_backend=args.backend,
        dit_layerwise_offload=True,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=True,
        vae_cpu_offload=True,
        enable_torch_compile=False,
        warmup_mode="off",
    )
    if vdn:
        options.update(
            model_path=str(ROOT / "models/VDN-H3-native-benchmark"),
            pipeline_class_name="VDNH3Pipeline",
            model_variant=None,
            attention_backend="hybrid_window_attn_h3",
            quantization="none",
        )
    else:
        options["transformer_weights_path"] = str(
            ROOT
            / "ComfyUI/models/diffusion_models"
            / (
                "minimax_h3_ref2va_int8_convrot.safetensors"
                if ref
                else "minimax_h3_fl2va_int8_convrot.safetensors"
            )
        )
    if args.prefetch or args.resident:
        options.update(
            dit_offload_prefetch_size=args.prefetch,
            dit_layerwise_resident_layers=args.resident,
            dit_layerwise_residency_lifetime="permanent",
        )
    if args.quantization is not None:
        options["quantization"] = args.quantization
    prompt = (
        "Use <Picture 1> as the cat reference. The orange cat walks on a sunny beach. Audio: quiet ocean waves and soft piano, no speech."
        if ref
        else "A small orange cat walks slowly on a sunny beach. Gentle ocean waves. Audio: quiet waves and a soft piano melody, no speech."
    )
    request = dict(
        prompt=prompt,
        seed=150000,
        num_inference_steps=9 if vdn else 21,
        task="ref2va" if ref else "t2va",
        conditions=(
            [
                {
                    "type": "image",
                    "uri": "file://"
                    + str(ROOT / "ComfyUI/input/h3_validation_reference.png"),
                    "role": "reference",
                }
            ]
            if ref
            else []
        ),
        target={
            "short_edge": 480,
            "aspect_ratio": "16:9",
            "duration_seconds": 107 / 24,
        },
        output_mode="decoded_files",
        save_output=True,
        output_path=str(out),
        output_file_name="probe",
    )
    (out / "options.json").write_text(
        json.dumps(
            options,
            default=lambda x: (
                dataclasses.asdict(x) if dataclasses.is_dataclass(x) else str(x)
            ),
            indent=2,
        )
    )
    (out / "request.json").write_text(json.dumps(request, indent=2))
    gen = None
    try:
        start = time.perf_counter()
        gen = DiffGenerator.from_pretrained(**options)
        load = time.perf_counter() - start
        (out / "effective-server-args.json").write_text(
            json.dumps(vars(gen.server_args), default=str, indent=2)
        )
        records = []
        for i in range(args.runs):
            req = dict(request, seed=150000 + i, output_file_name=f"run_{i:02}")
            if args.changed_prompt:
                req["prompt"] += f" A small cloud is visible in shot number {i}."
            samples = []
            stop = threading.Event()
            start = time.perf_counter()

            def monitor():
                while not stop.is_set():
                    samples.append(
                        {
                            "seconds": time.perf_counter() - start,
                            "used_gib": pynvml.nvmlDeviceGetMemoryInfo(handle).used
                            / 2**30,
                            "util": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                            "sm_mhz": pynvml.nvmlDeviceGetClockInfo(
                                handle, pynvml.NVML_CLOCK_SM
                            ),
                            "temperature_c": pynvml.nvmlDeviceGetTemperature(
                                handle, pynvml.NVML_TEMPERATURE_GPU
                            ),
                            "power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
                        }
                    )
                    stop.wait(0.25)

            thread = threading.Thread(target=monitor, daemon=True)
            thread.start()
            try:
                result = gen.generate(req)
                elapsed = time.perf_counter() - start
            finally:
                stop.set()
                thread.join(timeout=2)
            (out / f"gpu-samples-{i:02}.json").write_text(json.dumps(samples, indent=2))
            if result is None or isinstance(result, list):
                raise RuntimeError(f"Unexpected result {result}")
            assert result.generation_time > 0, result.generation_time
            media = pathlib.Path(result.output_file_path)
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_streams",
                    "-show_format",
                    "-of",
                    "json",
                    str(media),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            streams = json.loads(probe.stdout)
            assert {"video", "audio"} <= {s["codec_type"] for s in streams["streams"]}
            video = next(s for s in streams["streams"] if s["codec_type"] == "video")
            assert (video["width"], video["height"], int(video["nb_frames"])) == (
                864,
                480,
                107,
            ), video
            row = {
                "run": i,
                "case": args.case,
                "synchronized_stage_timing": args.profile,
                "cold": i == 0,
                "warmup": i == 1,
                "load_seconds": load,
                "wall_seconds": elapsed,
                "generation_time": result.generation_time,
                "size": result.size,
                "metrics": result.metrics,
                "peak_memory_mb": result.peak_memory_mb,
                "peak_gpu_gib": max(x["used_gib"] for x in samples),
                "media": streams,
                "output_bytes": media.stat().st_size,
            }
            records.append(row)
            (out / "runs.json").write_text(json.dumps(records, default=str, indent=2))
            print(json.dumps(row, default=str), flush=True)
            if i != 0:
                media.unlink()
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
        if gen is not None:
            gen.shutdown()
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
