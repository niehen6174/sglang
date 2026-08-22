#!/usr/bin/env python3
"""Regenerate sol-attn cross-model markdown report from per-model JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs/sol_attn_eval"

TARGET_MODELS = [
    "hunyuanvideo",
    "zimage",
    "qwen",
    "ltx23-ti2v-two-stage",
    "cosmos3-super-t2v",
    "wan-t2v",
]

ENABLEMENT = {
    "hunyuanvideo": [
        "configs/models/dits/hunyuanvideo.py: add SOL_ATTN to supported backends",
        "runtime/models/dits/hunyuanvideo.py: USPAttention already passes prefix",
    ],
    "zimage": [
        "configs/models/dits/zimage.py: add SOL_ATTN",
        "runtime/models/dits/zimage.py: pass prefix to USPAttention",
    ],
    "qwen": [
        "runtime/models/dits/qwen_image.py: add SOL_ATTN + prefix on cross-attn",
    ],
    "ltx23-ti2v-two-stage": [
        "configs/models/dits/ltx_2.py: add SOL_ATTN (video self-attn only; audio 64d stays FA)",
    ],
    "cosmos3-super-t2v": [
        "configs/models/dits/cosmos3video.py: add SOL_ATTN (GEN cross-attn only; UND uses custom SDPA)",
    ],
    "wan-t2v": [
        "configs/models/dits/wanvideo.py: add SOL_ATTN (self-attn only; cross excludes sparse)",
        "Needs Morton 3D reorder for acceptable quality (not implemented)",
    ],
}

VERDICT = {
    "hunyuanvideo": "⚠️ Quality OK (~43 dB) but slower in quick mode (dense_steps>steps)",
    "zimage": "❌ Unusable quality (~8 dB); slight denoise speedup",
    "qwen": "❌ Unusable quality (~8.6 dB); slightly slower",
    "ltx23-ti2v-two-stage": "⏸ Was blocked by LTX video_to_audio layer + global transformer=fa; fixed via layer backend fallback",
    "cosmos3-super-t2v": "❌ Poor quality (~11 dB); no speedup",
    "wan-t2v": "⏸ SKIPPED_NO_GPU (needs 4 free GPUs)",
}


def verdict_from_report(report: dict) -> str:
    key = report["model_key"]
    if key in VERDICT:
        base = VERDICT[key]
    else:
        base = report.get("status", "UNKNOWN")
    status = report.get("status")
    if status == "SKIPPED_NO_GPU":
        return "⏸ SKIPPED_NO_GPU"
    if status == "ERROR":
        return "⏸ ERROR (see report.json)"
    q = report.get("quality_vs_fa") or {}
    psnr = q.get("psnr_mean_db")
    if isinstance(psnr, float) and psnr >= 30:
        speed = (report.get("speedup") or {}).get("denoise_speedup")
        if isinstance(speed, float) and speed >= 1.05:
            return "✅ Quality OK + faster"
        return "⚠️ Quality OK but limited/no speedup"
    if isinstance(psnr, float) and psnr >= 20:
        return "⚠️ Marginal quality"
    if isinstance(psnr, float):
        return "❌ Poor quality"
    return base


def load_reports() -> list[dict]:
    reports = []
    for key in TARGET_MODELS:
        path = OUTPUT_DIR / key / "report.json"
        if path.exists():
            reports.append(json.loads(path.read_text()))
        else:
            reports.append(
                {
                    "model_key": key,
                    "modality": "video" if key in {"hunyuanvideo", "ltx23-ti2v-two-stage", "cosmos3-super-t2v", "wan-t2v"} else "image",
                    "status": "NOT_RUN",
                }
            )
    return reports


def write_report(reports: list[dict]) -> None:
    lines = [
        "# Sol-Attn Cross-Model Evaluation Report",
        "",
        "Branch: `feat/sol-attn` | Mode: `--quick` (8 denoise steps) | Baseline: `transformer=fa` vs `transformer=sol_attn`",
        "",
        "Artifacts root: `outputs/sol_attn_eval/<model>/`",
        "",
        "## Summary",
        "",
        "| Model | Modality | Status | PSNR vs FA | SSIM | Denoise speedup | Verdict |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in reports:
        q = item.get("quality_vs_fa") or {}
        s = item.get("speedup") or {}
        psnr = q.get("psnr_mean_db")
        ssim = q.get("ssim_mean")
        speed = s.get("denoise_speedup")
        lines.append(
            "| {model} | {mod} | {status} | {psnr} | {ssim} | {speed} | {verdict} |".format(
                model=item["model_key"],
                mod=item.get("modality", "?"),
                status=item.get("status", "?"),
                psnr=f"{psnr:.2f} dB" if isinstance(psnr, float) else "-",
                ssim=f"{ssim:.3f}" if isinstance(ssim, float) else "-",
                speed=f"{speed:.2f}x" if isinstance(speed, float) else "-",
                verdict=verdict_from_report(item),
            )
        )

    lines.extend(
        [
            "",
            "## H3 reference (prior benchmark, full steps)",
            "",
            "- Denoise: FA 27.0s → sol_attn 23.5s (**1.15×**), PSNR **32.9 dB**, SSIM **0.91**",
            "",
            "## Per-model details",
            "",
        ]
    )

    for item in reports:
        key = item["model_key"]
        lines.append(f"### {key}")
        lines.append("")
        lines.append(f"- **Status**: {item.get('status', 'UNKNOWN')}")
        lines.append(f"- **Modality**: {item.get('modality', '?')}")
        if item.get("quick_mode"):
            lines.append("- **Quick mode**: 8 inference steps")
        gpu = item.get("gpu_check") or {}
        if gpu.get("cuda_visible_devices"):
            lines.append(f"- **GPUs used**: `{gpu['cuda_visible_devices']}`")
        lines.append("- **Enablement**:")
        for change in ENABLEMENT.get(key, ["none"]):
            lines.append(f"  - {change}")
        runs = item.get("runs") or {}
        if runs:
            lines.append("- **Artifacts**:")
            for label, run in runs.items():
                if run.get("artifact_path"):
                    lines.append(f"  - {label}: `{run['artifact_path']}`")
                if run.get("perf_path"):
                    lines.append(f"  - {label} perf: `{run['perf_path']}`")
        q = item.get("quality_vs_fa")
        if q:
            lines.append(
                f"- **Quality vs FA**: PSNR mean {q.get('psnr_mean_db', '?'):.2f} dB, "
                f"SSIM mean {q.get('ssim_mean', '?'):.3f}, frames {q.get('frames_compared', '?')}"
            )
        speed = item.get("speedup")
        if speed:
            lines.append(
                f"- **Speed**: denoise {speed.get('fa_denoise_s', '?'):.2f}s → "
                f"{speed.get('sol_attn_denoise_s', '?'):.2f}s "
                f"({speed.get('denoise_speedup', '?'):.2f}×)"
            )
        lines.append(f"- **Verdict**: {verdict_from_report(item)}")
        lines.append("")

    lines.extend(
        [
            "",
            "## Why sol_attn can look slower (not mainly GPU contention)",
            "",
            "### HunyuanVideo",
            "",
            "- Ran on **GPU 0 only** with **0% utilization** — not distorted by other jobs.",
            "- Per-step denoise: FA ~**1.5 s/step**, sol_attn ~**5.2 s/step** for all 8 quick steps.",
            "- Root cause: default `dense_steps=10` + `--quick` (8 steps) ⇒ **never enters sparse kernel**; sol_attn dense wrapper adds reshape / fake-varlen overhead on top of FlashAttention.",
            "- Quality still high (~43 dB) because both paths effectively run dense FA math.",
            "- **Fix applied**: quick benchmark now sets `dense_steps=3` so sparse path is exercised.",
            "",
            "### 2-GPU image models (Z-Image / Qwen / Cosmos)",
            "",
            "- GPU 1 often had **73–100% util** during runs — denoise timing may have minor noise.",
            "- PSNR **~8–11 dB** is dominated by **joint / cross-attn semantic mismatch**, not measurement error.",
            "",
            "### LTX-2.3 (previous ERROR)",
            "",
            "- Not a missing `.bin` weights issue — cache has `model.safetensors` (~38 GB).",
            "- Customized load failed because `transformer=fa` hit `video_to_audio_attn`, which **must stay torch_sdpa** (`force_sdpa_v2a_cross_attention=True` in HF config).",
            "- Native fallback then looked for `diffusion_pytorch_model.bin` and failed.",
            "- **Fix applied**: attention selector falls back to layer-constrained backend; `cat.png` asset downloaded.",
            "",
            "### Wan2.2",
            "",
            "- Needs **4× ~80 GB GPUs**; auto gate skipped when GPU 1 was at 100% util despite ~119 GB free.",
            "- **Fix applied**: `--gpu-ids 0,1,2,3 --max-gpu-util 100` for pinned / shared-GPU runs.",
            "",
            "## Compatibility notes",
            "",
            "- **sol_attn constraints**: head_dim=128, BF16, CUDA, `forward` or `forward_varlen`.",
            "- Use `--component-attention-backends transformer=sol_attn` (not global `--attention-backend`) so text encoders stay on `torch_sdpa`.",
            "- `dense_layers` regex only matches `blocks.N`; Hunyuan/LTX/Z-Image layer names differ.",
            "- **Ring / cross-attn / joint-attn** paths need dense backends; Wan cross-attn already filters sparse backends.",
            "",
            "## Major follow-ups",
            "",
            "| Model | Blocker | Effort |",
            "| --- | --- | --- |",
            "| Wan2.2 | Morton 3D token reorder + 4 GPUs | High |",
            "| Qwen-Image | Joint text-image varlen + replicated prefix semantics | High |",
            "| Z-Image | Joint img+caption + `num_replicated_suffix` | Medium-High |",
            "| Cosmos3 | UND path bypasses attention backend system | High |",
            "| LTX-2.3 | Audio 64d layers + v2a cross-attn torch_sdpa pin + two-stage pipeline | Medium |",
            "| HunyuanVideo | Refiner LocalAttention must stay non-sparse; tune dense_steps for short runs | Low-Medium |",
            "",
            "## How to rerun",
            "",
            "```bash",
            "pip install git+https://github.com/NVlabs/Sana.git@sol-engine#subdirectory=techniques/sparse_backends",
            "bash scripts/check_gpu_availability.sh hunyuanvideo",
            "PYTHONPATH=python python3 scripts/bench_sol_attn_cross_model.py --execute --quick --models hunyuanvideo",
            "bash scripts/run_sol_attn_eval_queue.sh",
            "PYTHONPATH=python python3 scripts/regenerate_sol_attn_report.py",
            "```",
            "",
        ]
    )

    out = OUTPUT_DIR / "SOL_ATTN_CROSS_MODEL_REPORT.md"
    out.write_text("\n".join(lines))
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(reports, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    write_report(load_reports())
