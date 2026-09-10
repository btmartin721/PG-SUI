#!/bin/bash
#SBATCH --job-name=pgsui-canonical-cpu
#SBATCH --partition=shu-hpc-biocpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

set -euo pipefail

BENCHMARK_ROOT=${PGSUI_BENCHMARK_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
CONDA_ENV=${PGSUI_CONDA_ENV:-pgsui-gti-2}
PYTHON_BIN=${PYTHON_BIN:-python}
WORKER_INDEX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}
WORKER_COUNT=4
STRATEGIES_PER_DATASET=5
TASK_MANIFEST="${BENCHMARK_ROOT}/manifests/pgsui_gpu_tasks.tsv"
TASK_WORK_ROOT="${SLURM_TMPDIR:-${BENCHMARK_ROOT}/work}/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${WORKER_INDEX}"

mkdir -p "${BENCHMARK_ROOT}/logs" "${BENCHMARK_ROOT}/status"

if [[ -f "${HOME}/miniconda/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  echo "Could not locate conda.sh under ${HOME}." >&2
  exit 2
fi
conda activate "${CONDA_ENV}"

export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "${MPLCONFIGDIR}"

echo "Host: $(hostname)"
echo "Started: $(date -Iseconds)"
echo "Worker: ${WORKER_INDEX}/${WORKER_COUNT}"
echo "Benchmark root: ${BENCHMARK_ROOT}"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK:-1}"

TASK_COUNT=$("${PYTHON_BIN}" -c \
  'import csv, pathlib, sys; p=pathlib.Path(sys.argv[1]); print(sum(1 for _ in csv.DictReader(p.open(), delimiter="\t")))' \
  "${TASK_MANIFEST}")
if (( TASK_COUNT != 50 )); then
  printf 'Expected 50 manifest tasks, found %s.\n' "${TASK_COUNT}" >&2
  exit 2
fi
if (( TASK_COUNT % STRATEGIES_PER_DATASET != 0 )); then
  printf 'Task count %s is not divisible by %s strategies per dataset.\n' \
    "${TASK_COUNT}" "${STRATEGIES_PER_DATASET}" >&2
  exit 2
fi
DATASET_COUNT=$((TASK_COUNT / STRATEGIES_PER_DATASET))

TASK_FAILURES=0
TASKS_ATTEMPTED=0
for ((DATASET_INDEX = WORKER_INDEX; DATASET_INDEX < DATASET_COUNT; DATASET_INDEX += WORKER_COUNT)); do
  for ((STRATEGY_INDEX = 0; STRATEGY_INDEX < STRATEGIES_PER_DATASET; STRATEGY_INDEX++)); do
    TASK_INDEX=$((DATASET_INDEX * STRATEGIES_PER_DATASET + STRATEGY_INDEX))
    TASK_WORK_DIR="${TASK_WORK_ROOT}/task_${TASK_INDEX}"
    printf '\nWorker %s starting task %s at %s\n' \
      "${WORKER_INDEX}" "${TASK_INDEX}" "$(date -Iseconds)"
    if "${PYTHON_BIN}" "${BENCHMARK_ROOT}/scripts/run_pgsui_canonical_task.py" \
      --bundle-root "${BENCHMARK_ROOT}" \
      --task-index "${TASK_INDEX}" \
      --work-dir "${TASK_WORK_DIR}"; then
      printf 'Worker %s completed task %s at %s\n' \
        "${WORKER_INDEX}" "${TASK_INDEX}" "$(date -Iseconds)"
    else
      TASK_STATUS=$?
      TASK_FAILURES=$((TASK_FAILURES + 1))
      printf 'Worker %s: task %s failed with status %s; continuing.\n' \
        "${WORKER_INDEX}" "${TASK_INDEX}" "${TASK_STATUS}" >&2
    fi
    TASKS_ATTEMPTED=$((TASKS_ATTEMPTED + 1))
  done
done

printf '\nWorker %s finished at %s; attempted=%s failures=%s\n' \
  "${WORKER_INDEX}" "$(date -Iseconds)" "${TASKS_ATTEMPTED}" "${TASK_FAILURES}"
if (( TASK_FAILURES != 0 )); then
  exit 1
fi
