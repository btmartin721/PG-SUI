#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
BUNDLE_ROOT=${PGSUI_N58_ROOT:-$(dirname -- "${SCRIPT_DIR}")}
PARTITION=${PGSUI_CPU_PARTITION:-shu-hpc-biocpu}
CONCURRENCY=${PGSUI_WORKER_CONCURRENCY:-8}
PYTHON_BIN=${PYTHON_BIN:-python}
TASK_MANIFEST="${BUNDLE_ROOT}/manifests/pgsui_n58_tasks.tsv"
WORKER_MANIFEST="${BUNDLE_ROOT}/manifests/pgsui_n58_workers.tsv"
SLURM_SCRIPT="${BUNDLE_ROOT}/scripts/pgsui_n58_cpu_workers.slurm"

mkdir -p "${BUNDLE_ROOT}/logs" "${BUNDLE_ROOT}/status"

if [[ ! -f "${TASK_MANIFEST}" ]]; then
  printf 'Task manifest not found: %s\n' "${TASK_MANIFEST}" >&2
  exit 2
fi
if [[ ! -f "${WORKER_MANIFEST}" ]]; then
  printf 'Worker manifest not found: %s\n' "${WORKER_MANIFEST}" >&2
  exit 2
fi
if [[ ! -f "${SLURM_SCRIPT}" ]]; then
  printf 'SLURM script not found: %s\n' "${SLURM_SCRIPT}" >&2
  exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
  printf '%s\n' 'sbatch is unavailable; run this on the SHU SLURM login node.' >&2
  exit 127
fi
if (( CONCURRENCY < 1 || CONCURRENCY > 8 )); then
  printf 'PGSUI_WORKER_CONCURRENCY must be between 1 and 8, found %s.\n' \
    "${CONCURRENCY}" >&2
  exit 2
fi

read -r TASK_COUNT INVALID_COUNT < <(
  "${PYTHON_BIN}" -c '
import csv, pathlib, sys
path = pathlib.Path(sys.argv[1])
with path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
invalid = sum(
    row.get("device") != "cpu"
    or row.get("n_jobs") != "1"
    or row.get("ploidy") != "2"
    or row.get("tune_n_trials") != "50"
    or row.get("sim_max_tries") != "100000"
    or row.get("preset") != "fast"
    for row in rows
)
print(len(rows), invalid)
' "${TASK_MANIFEST}"
)
if (( TASK_COUNT != 290 )); then
  printf 'Expected 290 dataset/strategy tasks, found %s.\n' "${TASK_COUNT}" >&2
  exit 2
fi
if (( INVALID_COUNT != 0 )); then
  printf 'Found %s tasks that violate the CPU/one-thread/50-trial/100000-try profile.\n' \
    "${INVALID_COUNT}" >&2
  exit 2
fi

read -r WORKER_COUNT ASSIGNED_COUNT < <(
  "${PYTHON_BIN}" -c '
import csv, pathlib, sys
path = pathlib.Path(sys.argv[1])
with path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
task_ids = [task_id for row in rows for task_id in row["task_ids"].split()]
if [int(row["worker_id"]) for row in rows] != list(range(8)):
    raise SystemExit("Worker IDs must be contiguous from 0 through 7")
if len(task_ids) != len(set(task_ids)):
    raise SystemExit("Worker assignments contain duplicate task IDs")
if sorted(map(int, task_ids)) != list(range(290)):
    raise SystemExit("Worker assignments must cover task IDs 0 through 289")
print(len(rows), len(task_ids))
' "${WORKER_MANIFEST}"
)
if (( WORKER_COUNT != 8 || ASSIGNED_COUNT != 290 )); then
  printf 'Expected 8 workers covering 290 tasks; found %s workers/%s tasks.\n' \
    "${WORKER_COUNT}" "${ASSIGNED_COUNT}" >&2
  exit 2
fi

printf 'Submitting 8 one-CPU sequential workers for %s tasks with concurrency %s on %s.\n' \
  "${TASK_COUNT}" "${CONCURRENCY}" "${PARTITION}"
sbatch \
  --chdir="${BUNDLE_ROOT}" \
  --partition="${PARTITION}" \
  --array="0-7%${CONCURRENCY}" \
  --export="ALL,PGSUI_N58_ROOT=${BUNDLE_ROOT}" \
  "${SLURM_SCRIPT}"
