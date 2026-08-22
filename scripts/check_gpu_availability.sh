#!/usr/bin/env bash
# Check whether enough GPUs are free for a sol-attn cross-model benchmark preset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCH_PY="${REPO_ROOT}/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py"

MIN_FREE_MIB="${MIN_FREE_MIB:-80000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-90}"
GPU_IDS="${GPU_IDS:-}"
MODEL_KEY="${1:-}"

if [[ -z "${MODEL_KEY}" ]]; then
  echo "Usage: $0 <model_preset_key> [min_free_mib]" >&2
  echo "Example: $0 hunyuanvideo" >&2
  echo "Env: MAX_GPU_UTIL=100 GPU_IDS=0,1,2,3 to relax util gate or pin GPUs" >&2
  exit 2
fi

if [[ $# -ge 2 ]]; then
  MIN_FREE_MIB="$2"
fi

if [[ ! -f "${BENCH_PY}" ]]; then
  echo "Missing benchmark preset script: ${BENCH_PY}" >&2
  exit 2
fi

REQUIRED_GPUS="$(PYTHONPATH="${REPO_ROOT}/python" python3 - <<'PY' "${MODEL_KEY}" "${BENCH_PY}"
import importlib.util
import sys

model_key, bench_path = sys.argv[1:3]
spec = importlib.util.spec_from_file_location("bench_diffusion_denoise", bench_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print(mod.required_gpus_for_model(model_key))
PY
)"

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)

echo "Model preset: ${MODEL_KEY}"
echo "Required GPUs: ${REQUIRED_GPUS}"
echo "Min free memory per GPU: ${MIN_FREE_MIB} MiB"
echo "Max GPU utilization: ${MAX_GPU_UTIL}%"
if [[ -n "${GPU_IDS}" ]]; then
  echo "Pinned GPU IDs: ${GPU_IDS}"
fi
echo
printf "%-6s %-14s %-10s %s\n" "GPU" "free_MiB" "util_%" "eligible"
printf "%-6s %-14s %-10s %s\n" "---" "--------" "------" "---------"

eligible=()
eligible_util=()
for row in "${GPU_ROWS[@]}"; do
  IFS=',' read -r idx free util <<< "${row}"
  idx="${idx// /}"
  free="${free// /}"
  util="${util// /}"
  ok="no"
  if [[ -n "${GPU_IDS}" ]]; then
    IFS=',' read -ra PINNED <<< "${GPU_IDS}"
    pinned_match="no"
    for pinned in "${PINNED[@]}"; do
      if [[ "${idx}" == "${pinned// /}" ]]; then
        pinned_match="yes"
        break
      fi
    done
    if [[ "${pinned_match}" != "yes" ]]; then
      printf "%-6s %-14s %-10s %s\n" "${idx}" "${free}" "${util}" "pinned-skip"
      continue
    fi
  fi
  if (( free >= MIN_FREE_MIB && util <= MAX_GPU_UTIL )); then
    ok="yes"
    eligible+=("${idx}")
    eligible_util+=("${util}")
  fi
  printf "%-6s %-14s %-10s %s\n" "${idx}" "${free}" "${util}" "${ok}"
done

# Prefer lower-util GPUs to avoid sharing with active jobs (OOM on LTX/Wan).
if [[ -z "${GPU_IDS}" && ${#eligible[@]} -gt 1 ]]; then
  mapfile -t eligible < <(
    paste -d ' ' <(printf '%s\n' "${eligible[@]}") <(printf '%s\n' "${eligible_util[@]}") \
      | sort -k2,2n \
      | awk '{print $1}'
  )
fi

echo
if [[ -n "${GPU_IDS}" ]]; then
  IFS=',' read -ra PINNED <<< "${GPU_IDS}"
  if (( ${#PINNED[@]} >= REQUIRED_GPUS )); then
    selected="$(IFS=,; echo "${GPU_IDS}")"
    echo "STATUS=OK"
    echo "CUDA_VISIBLE_DEVICES=${selected}"
    exit 0
  fi
fi

if (( ${#eligible[@]} >= REQUIRED_GPUS )); then
  selected="$(IFS=,; echo "${eligible[*]:0:REQUIRED_GPUS}")"
  echo "STATUS=OK"
  echo "CUDA_VISIBLE_DEVICES=${selected}"
  exit 0
fi

echo "STATUS=SKIPPED_NO_GPU"
echo "Need ${REQUIRED_GPUS} eligible GPU(s), found ${#eligible[@]}."
exit 1
