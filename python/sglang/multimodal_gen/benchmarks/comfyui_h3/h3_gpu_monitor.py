"""NVML sampling for the GPUs a benchmark actually runs on.

Logical ``cuda:N`` is not physical NVML index N once CUDA_VISIBLE_DEVICES is
set, so resolve each visible entry (index or UUID) to its physical handle.
H3_BENCH_GPUS overrides CUDA_VISIBLE_DEVICES for clients that monitor a
server started with a different mask.
"""

import os
import threading
import time

import pynvml


def _text(value):
    return value.decode() if isinstance(value, bytes) else value


def visible_gpu_entries():
    raw = os.environ.get("H3_BENCH_GPUS") or os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or raw.strip() == "":
        return [str(i) for i in range(pynvml.nvmlDeviceGetCount())]
    return [x.strip() for x in raw.split(",") if x.strip()]


def _handle(entry):
    if entry.isdigit():
        return pynvml.nvmlDeviceGetHandleByIndex(int(entry))
    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        if _text(pynvml.nvmlDeviceGetUUID(handle)).startswith(entry):
            return handle
    raise ValueError(f"No NVML device matches visible entry {entry!r}")


def _processes(handle):
    try:
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    except pynvml.NVMLError:
        return None
    return [
        {"pid": p.pid, "used_gib": (p.usedGpuMemory or 0) / 2**30} for p in procs
    ]


class GpuMonitor:
    def __init__(self):
        pynvml.nvmlInit()
        self.entries = visible_gpu_entries()
        self.handles = [_handle(e) for e in self.entries]
        self.identity = [
            {
                "logical": logical,
                "visible_entry": entry,
                "nvml_index": pynvml.nvmlDeviceGetIndex(h),
                "uuid": _text(pynvml.nvmlDeviceGetUUID(h)),
                "pci_bus_id": _text(pynvml.nvmlDeviceGetPciInfo(h).busId),
                "name": _text(pynvml.nvmlDeviceGetName(h)),
            }
            for logical, (entry, h) in enumerate(zip(self.entries, self.handles))
        ]

    def processes(self):
        return {
            str(gpu["nvml_index"]): _processes(h)
            for gpu, h in zip(self.identity, self.handles)
        }

    def sample(self, start):
        gpus = []
        for gpu, h in zip(self.identity, self.handles):
            gpus.append(
                {
                    "nvml_index": gpu["nvml_index"],
                    "used_gib": pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**30,
                    "util": pynvml.nvmlDeviceGetUtilizationRates(h).gpu,
                    "sm_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                    "temperature_c": pynvml.nvmlDeviceGetTemperature(
                        h, pynvml.NVML_TEMPERATURE_GPU
                    ),
                    "power_w": pynvml.nvmlDeviceGetPowerUsage(h) / 1000,
                }
            )
        return {
            "seconds": time.perf_counter() - start,
            # Kept for older summaries: the busiest single GPU.
            "used_gib": max(g["used_gib"] for g in gpus),
            "total_used_gib": sum(g["used_gib"] for g in gpus),
            "gpus": gpus,
        }

    def start(self, start, interval=0.25):
        samples = []
        stop = threading.Event()

        def loop():
            while not stop.is_set():
                samples.append(self.sample(start))
                stop.wait(interval)

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()

        def finish():
            stop.set()
            thread.join(timeout=2)
            return samples

        return samples, finish

    @staticmethod
    def peaks(samples):
        per_gpu = {}
        for s in samples:
            for g in s["gpus"]:
                key = str(g["nvml_index"])
                per_gpu[key] = max(per_gpu.get(key, 0.0), g["used_gib"])
        return {
            "peak_gpu_gib": max((s["used_gib"] for s in samples), default=0.0),
            "peak_total_gpu_gib": max(
                (s["total_used_gib"] for s in samples), default=0.0
            ),
            "peak_gpu_gib_by_nvml_index": per_gpu,
        }

    def shutdown(self):
        pynvml.nvmlShutdown()
