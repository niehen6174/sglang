"""Collect completed H3 performance runs without confusing cached/cold runs."""

import csv
import json
import pathlib
import os
import re
import statistics

ROOT = pathlib.Path(os.environ["H3_BENCH_SUITE"]).resolve()


def med(values):
    return statistics.median(values) if values else None


def main():
    rows = []
    for file in sorted(ROOT.glob("*.json")):
        if file.name in [
            "matrix-status.json",
            "native-metadata-manifest.json",
            "environment.json",
            "summary.json",
        ]:
            continue
        data = json.loads(file.read_text())
        if (
            not isinstance(data, list)
            or not data
            or not isinstance(data[0], dict)
            or not {"wall_seconds", "cold", "warmup", "sampler_seconds", "peak_gpu_gib"}
            <= data[0].keys()
        ):
            continue
        hot = [x for x in data if not x["cold"] and not x["warmup"]]
        row = {
            "case": file.stem,
            "path": "ComfyUI API",
            "completed": len(data),
            "hot_n": len(hot),
            "cold_wall": data[0]["wall_seconds"],
            "wall_median": med([x["wall_seconds"] for x in hot]),
            "wall_min": min((x["wall_seconds"] for x in hot), default=None),
            "wall_max": max((x["wall_seconds"] for x in hot), default=None),
            "sampler_median": med([x["sampler_seconds"] for x in hot]),
            "apply_model_median": med(
                [
                    x["apply_model_seconds"]
                    for x in hot
                    if x["apply_model_seconds"] is not None
                ]
            ),
            "gpu_peak_gib": max(x["peak_gpu_gib"] for x in data),
        }
        row["forward_calls_per_request"] = data[0].get("forward_calls")
        row["instrumented"] = data[0].get("instrumented", True)
        if hot and "worker_step_seconds" in hot[0]:
            row["worker_step_median"] = med([x["worker_step_seconds"] for x in hot])
            row["bridge_glue_median"] = med(
                [x["apply_model_seconds"] - x["worker_step_seconds"] for x in hot]
            )
            row["worker_total_median"] = med([x["worker_total_seconds"] for x in hot])
        rows.append(row)
    for file in sorted((ROOT / "direct").glob("*/runs.json")):
        data = json.loads(file.read_text())
        hot = [x for x in data if not x["cold"] and not x["warmup"]]
        if not hot:
            continue
        row = {
            "case": file.parent.name,
            "path": "SGLang Python API",
            "completed": len(data),
            "hot_n": len(hot),
            "cold_wall": data[0]["wall_seconds"],
            "load_seconds": data[0]["load_seconds"],
            "wall_median": med([x["wall_seconds"] for x in hot]),
            "wall_min": min(x["wall_seconds"] for x in hot),
            "wall_max": max(x["wall_seconds"] for x in hot),
            "gpu_reserved_peak_gib": max(x["peak_memory_mb"] for x in data) / 1024,
        }
        if "peak_gpu_gib" in data[0]:
            row["gpu_peak_gib"] = max(x["peak_gpu_gib"] for x in data)
        synced = data[0].get(
            "synchronized_stage_timing",
            file.parent.name in {"t2av_sync", "ref2av", "vdn"},
        )
        row["stage_source"] = (
            "synchronized stage timer"
            if synced
            else "asynchronous stage timer; use API wall time for comparisons"
        )
        metrics = [x["metrics"]["stages"] for x in hot]
        stage = "MiniMaxH3DenoisingStage"
        if stage in metrics[0]:
            row["sampler_median"] = med([x[stage] / 1000 for x in metrics])
        else:
            log = ROOT / ("direct-" + data[0]["case"] + ".log")
            if log.exists():
                times = [
                    float(x)
                    for x in re.findall(
                        r"\[MiniMaxH3DenoisingStage\] finished in ([\d.]+) seconds",
                        log.read_text(),
                    )
                ]
                if len(times) == len(data):
                    row["sampler_median"] = med(times[2:])
                    row["stage_source"] = (
                        "asynchronous log; validate with synchronized profile"
                    )
        rows.append(row)
    (ROOT / "summary.json").write_text(json.dumps(rows, indent=2))
    if rows:
        fields = sorted({k for r in rows for k in r})
        with (ROOT / "summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fields)
            w.writeheader()
            w.writerows(rows)
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
