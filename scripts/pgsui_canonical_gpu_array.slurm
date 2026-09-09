#!/bin/bash
#SBATCH --job-name=pgsui-canonical
#SBATCH --partition=shu-hpc-biogpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-49%4
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

set -euo pipefail

BENCHMARK_ROOT=${PGSUI_BENCHMARK_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
CONDA_ENV=${PGSUI_CONDA_ENV:-pgsui-gti}
TASK_INDEX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}

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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "${MPLCONFIGDIR}"

echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Array task: ${TASK_INDEX}"
echo "Benchmark root: ${BENCHMARK_ROOT}"
nvidia-smi

python "${BENCHMARK_ROOT}/scripts/run_pgsui_canonical_task.py" \
  --bundle-root "${BENCHMARK_ROOT}" \
  --task-index "${TASK_INDEX}"

echo "Finished: $(date --iso-8601=seconds)"
