#!/usr/bin/env bash
#SBATCH --job-name=pgsui-n58
#SBATCH --partition=shu-hpc-biocpu
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00:00
#SBATCH --array=0-7%8
#SBATCH --output=logs/n58_worker_%A_%a.out
#SBATCH --error=logs/n58_worker_%A_%a.err

set -euo pipefail

BUNDLE_ROOT=${PGSUI_N58_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
CONDA_ENV=${PGSUI_CONDA_ENV:-pgsui-val}
PYTHON_BIN=${PYTHON_BIN:-python}
WORKER_INDEX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}
WORKER_ROOT="${SLURM_TMPDIR:-${BUNDLE_ROOT}/work}/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${WORKER_INDEX}"

mkdir -p "${BUNDLE_ROOT}/logs" "${BUNDLE_ROOT}/status" "${WORKER_ROOT}"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda/etc/profile.d/conda.sh"
else
  printf 'Could not locate conda.sh under %s.\n' "${HOME}" >&2
  exit 2
fi
conda activate "${CONDA_ENV}"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${WORKER_ROOT}/numba-cache"
export MPLCONFIGDIR="${WORKER_ROOT}/matplotlib"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"

printf 'Host: %s\n' "$(hostname)"
printf 'Started: %s\n' "$(date -Iseconds)"
printf 'Worker: %s\n' "${WORKER_INDEX}"
printf 'Bundle: %s\n' "${BUNDLE_ROOT}"
printf 'CPUs: %s\n' "${SLURM_CPUS_PER_TASK:-1}"

"${PYTHON_BIN}" "${BUNDLE_ROOT}/scripts/run_pgsui_n58_worker.py" \
  --bundle-root "${BUNDLE_ROOT}" \
  --worker-index "${WORKER_INDEX}" \
  --workload pgsui \
  --work-root "${WORKER_ROOT}"

printf 'Finished: %s\n' "$(date -Iseconds)"
