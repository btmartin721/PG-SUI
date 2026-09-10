#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
BUNDLE_ROOT=${PGSUI_BENCHMARK_ROOT:-$(dirname -- "${SCRIPT_DIR}")}
WORKER_COUNT=4
PARTITION=${PGSUI_CPU_PARTITION:-shu-hpc-biocpu}
PYTHON_BIN=${PYTHON_BIN:-python}
TASK_MANIFEST="${BUNDLE_ROOT}/manifests/pgsui_gpu_tasks.tsv"
SLURM_SCRIPT="${BUNDLE_ROOT}/scripts/pgsui_canonical_gpu_array.slurm"

mkdir -p "${BUNDLE_ROOT}/logs" "${BUNDLE_ROOT}/status"

if [[ ! -f "${TASK_MANIFEST}" ]]; then
  printf 'Task manifest not found: %s\n' "${TASK_MANIFEST}" >&2
  exit 2
fi

if [[ ! -f "${SLURM_SCRIPT}" ]]; then
  printf 'SLURM script not found: %s\n' "${SLURM_SCRIPT}" >&2
  exit 2
fi

if ! command -v sbatch >/dev/null 2>&1; then
  printf '%s\n' "sbatch is not available; run this script on a SLURM login node." >&2
  exit 127
fi

TASK_COUNT=$("${PYTHON_BIN}" -c \
  'import csv, pathlib, sys; p=pathlib.Path(sys.argv[1]); print(sum(1 for _ in csv.DictReader(p.open(), delimiter="\t")))' \
  "${TASK_MANIFEST}")

if (( TASK_COUNT != 50 )); then
  printf '%s\n' "Expected 50 dataset/strategy tasks, found ${TASK_COUNT}." >&2
  exit 2
fi

INVALID_CPU_TASKS=$("${PYTHON_BIN}" -c \
  'import csv, pathlib, sys; p=pathlib.Path(sys.argv[1]); rows=csv.DictReader(p.open(), delimiter="\t"); print(sum(row.get("device", "").strip() != "cpu" or not row.get("output_prefix", "").strip().endswith("_cpu") for row in rows))' \
  "${TASK_MANIFEST}")

if (( INVALID_CPU_TASKS != 0 )); then
  printf '%s\n' \
    "Expected every task to use device=cpu and an _cpu output prefix; found ${INVALID_CPU_TASKS} invalid rows." >&2
  exit 2
fi

printf 'Submitting %s tasks across %s persistent CPU workers.\n' \
  "${TASK_COUNT}" "${WORKER_COUNT}"

sbatch \
  --chdir="${BUNDLE_ROOT}" \
  --partition="${PARTITION}" \
  --array="0-$((WORKER_COUNT - 1))%${WORKER_COUNT}" \
  --export="ALL,PGSUI_BENCHMARK_ROOT=${BUNDLE_ROOT}" \
  "${SLURM_SCRIPT}"
