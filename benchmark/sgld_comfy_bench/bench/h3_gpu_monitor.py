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

    def start(self, start, interval=0.25, session_pid=None):
        samples = []
        host_samples = []
        stop = threading.Event()

        def loop():
            # Page-table walks are expensive on a 50 GiB tree. GPU stays at
            # `interval`; host RAM is sampled every 2 s, which still covers
            # the plateaus this workload actually holds.
            next_host = 0.0
            while not stop.is_set():
                gpu = self.sample(start)
                samples.append(gpu)
                if session_pid and gpu["seconds"] >= next_host:
                    host = sample_host_tree(session_pid, gpu["seconds"])
                    host["gpu_gib"] = round(gpu["used_gib"], 3)
                    host_samples.append(host)
                    next_host = gpu["seconds"] + 2.0
                stop.wait(interval)

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()

        def finish():
            stop.set()
            thread.join(timeout=5)
            return samples, host_samples

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


_KIB_PER_GIB = 1024 * 1024
_SMAPS_KEYS = ("Rss", "Pss", "Pss_Anon", "Pss_File", "Pss_Shmem", "Anonymous", "Swap", "Locked")
_STATUS_KEYS = ("VmRSS", "VmHWM", "VmLck", "VmPin", "VmSwap")


def _gib(kib):
    return kib / _KIB_PER_GIB


def _cmdline(pid):
    try:
        raw = open(f"/proc/{pid}/cmdline", "rb").read()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode(errors="replace").strip()[:180]


def _session_pids(leader):
    """Pids in the ComfyUI session. start_new_session makes the shell its own leader."""
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            stat = open(f"/proc/{pid}/stat", "rb").read()
        except OSError:
            continue
        end = stat.rfind(b")")
        if end < 0:
            continue
        fields = stat[end + 2 :].split()
        # After comm: state, ppid, pgrp, session.
        if len(fields) >= 4 and (pid == leader or int(fields[3]) == leader):
            found.append(pid)
    return found


def _proc_mem(pid):
    smaps, status = {}, {}
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key in _SMAPS_KEYS:
                    smaps[key] = int(rest.split()[0])
    except OSError:
        pass
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key in _STATUS_KEYS:
                    status[key] = int(rest.split()[0])
    except OSError:
        pass
    return smaps, status


def sample_host_tree(leader, seconds):
    """RAM held by the ComfyUI process tree, not the whole machine.

    Pss is the share this tree is responsible for. Anonymous and CUDA-pinned
    pages have to sit in real RAM; file-backed mappings of the weights can be
    dropped when the machine is tight. A 1 TB devbox `free` reading is not the
    number a consumer machine has to provide.
    """
    totals = {k: 0 for k in ("rss", "pss", "pss_anon", "pss_file", "shmem", "anon", "swap", "locked", "pinned")}
    procs = []
    for pid in _session_pids(leader):
        smaps, status = _proc_mem(pid)
        rss = smaps.get("Rss", status.get("VmRSS", 0))
        pss = smaps.get("Pss", rss)
        row = {
            "pid": pid,
            "rss_kib": rss,
            "pss_kib": pss,
            "anon_kib": smaps.get("Anonymous", 0),
            "shmem_kib": smaps.get("Pss_Shmem", 0),
            "locked_kib": smaps.get("Locked", status.get("VmLck", 0)),
            "pinned_kib": status.get("VmPin", 0),
            "cmd": _cmdline(pid),
        }
        totals["rss"] += rss
        totals["pss"] += pss
        totals["pss_anon"] += smaps.get("Pss_Anon", 0)
        totals["pss_file"] += smaps.get("Pss_File", 0)
        totals["shmem"] += smaps.get("Pss_Shmem", 0)
        totals["anon"] += smaps.get("Anonymous", 0)
        totals["swap"] += smaps.get("Swap", status.get("VmSwap", 0))
        totals["locked"] += row["locked_kib"]
        totals["pinned"] += status.get("VmPin", 0)
        procs.append(row)
    return {
        "seconds": round(seconds, 3),
        "pss_gib": round(_gib(totals["pss"]), 3),
        "rss_gib": round(_gib(totals["rss"]), 3),
        "pss_anon_gib": round(_gib(totals["pss_anon"]), 3),
        "pss_file_gib": round(_gib(totals["pss_file"]), 3),
        "shmem_gib": round(_gib(totals["shmem"]), 3),
        "anon_gib": round(_gib(totals["anon"]), 3),
        "swap_gib": round(_gib(totals["swap"]), 3),
        "locked_gib": round(_gib(totals["locked"]), 3),
        "pinned_gib": round(_gib(totals["pinned"]), 3),
        "n_procs": len(procs),
        "procs": procs,
    }


def host_peaks(samples):
    if not samples:
        return {}
    peak = max(samples, key=lambda s: s["pss_gib"])
    top = sorted(peak["procs"], key=lambda p: p["rss_kib"], reverse=True)[:6]
    return {
        "peak_host_pss_gib": peak["pss_gib"],
        "peak_host_rss_gib": max(s["rss_gib"] for s in samples),
        "peak_host_anon_gib": max(s["anon_gib"] for s in samples),
        "peak_host_pss_anon_gib": max(s["pss_anon_gib"] for s in samples),
        "peak_host_pss_file_gib": max(s["pss_file_gib"] for s in samples),
        "peak_host_shmem_gib": max(s.get("shmem_gib", 0.0) for s in samples),
        "peak_host_locked_gib": max(s["locked_gib"] for s in samples),
        "peak_host_pinned_gib": max(s["pinned_gib"] for s in samples),
        "peak_host_swap_gib": max(s["swap_gib"] for s in samples),
        "peak_host_at_s": peak["seconds"],
        "peak_host_procs": [
            {
                "pid": p["pid"],
                "rss_gib": round(_gib(p["rss_kib"]), 3),
                "pss_gib": round(_gib(p["pss_kib"]), 3),
                "anon_gib": round(_gib(p["anon_kib"]), 3),
                "shmem_gib": round(_gib(p.get("shmem_kib", 0)), 3),
                "locked_gib": round(_gib(p["locked_kib"]), 3),
                "cmd": p["cmd"],
            }
            for p in top
            if p["rss_kib"] >= 64 * 1024
        ],
    }
