#!/usr/bin/env bash
#SBATCH --job-name=pgsui_n58_sumstats
#SBATCH --partition=shu-hpc-biocpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --array=0-7%8
#SBATCH --output=logs/n58_sumstats_worker_%A_%a.out
#SBATCH --error=logs/n58_sumstats_worker_%A_%a.err

set -euo pipefail

BUNDLE_ROOT=${PGSUI_N58_ROOT:?Set PGSUI_N58_ROOT}
CONDA_ENV=${PGSUI_CONDA_ENV:-pgsui-val}
PYTHON_BIN=${PYTHON_BIN:-python}
WORKER_INDEX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/pgsui-matplotlib-${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/pgsui-numba-${SLURM_JOB_ID}"
mkdir -p "$BUNDLE_ROOT/logs" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

cd "$BUNDLE_ROOT"
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
conda activate "$CONDA_ENV"

"${PYTHON_BIN}" scripts/run_pgsui_n58_worker.py \
  --bundle-root "$BUNDLE_ROOT" \
  --worker-index "$WORKER_INDEX" \
  --workload sumstats
