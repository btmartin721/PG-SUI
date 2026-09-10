#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_VALIDATION_ROOT=$(dirname -- "${SCRIPT_DIR}")
PARENT_VALIDATION_ROOT=$(dirname -- "${DEFAULT_VALIDATION_ROOT}")
if [[ -d "${DEFAULT_VALIDATION_ROOT}/inputs/vcfs" ]] \
  && [[ -d "${DEFAULT_VALIDATION_ROOT}/inputs/iqtree" ]] \
  && [[ -d "${DEFAULT_VALIDATION_ROOT}/masks" ]]; then
  BUNDLE_LAYOUT=1
elif [[ ! -d "${DEFAULT_VALIDATION_ROOT}/inputs/test-vcf-files" ]] \
  && [[ -d "${PARENT_VALIDATION_ROOT}/inputs/test-vcf-files" ]]; then
  DEFAULT_VALIDATION_ROOT="${PARENT_VALIDATION_ROOT}"
  BUNDLE_LAYOUT=0
else
  BUNDLE_LAYOUT=0
fi
VALIDATION_ROOT="${VALIDATION_ROOT:-${DEFAULT_VALIDATION_ROOT}}"
if (( BUNDLE_LAYOUT )); then
  INPUT_DIR="${INPUT_DIR:-${VALIDATION_ROOT}/inputs/vcfs}"
  REFERENCE_MASK_DIR="${REFERENCE_MASK_DIR:-${VALIDATION_ROOT}/masks}"
  TREE_DIR="${TREE_DIR:-${VALIDATION_ROOT}/inputs/iqtree}"
  OUTPUT_DIR="${OUTPUT_DIR:-${VALIDATION_ROOT}/regenerated_simulations}"
  DEFAULT_MASK_MODE=verify
else
  INPUT_DIR="${INPUT_DIR:-${VALIDATION_ROOT}/inputs/test-vcf-files}"
  REFERENCE_MASK_DIR="${REFERENCE_MASK_DIR:-${VALIDATION_ROOT}/outputs/gtimputation-results/zygosity_missingness_simulations}"
  TREE_DIR="${TREE_DIR:-${REFERENCE_MASK_DIR}/iqtree}"
  OUTPUT_DIR="${OUTPUT_DIR:-${VALIDATION_ROOT}/canonical_benchmark}"
  DEFAULT_MASK_MODE=regenerate
fi
DATASETS_FILE="${DATASETS_FILE:-${SCRIPT_DIR}/pgsui_gtimputation_datasets.txt}"
MASK_MODE="${MASK_MODE:-${DEFAULT_MASK_MODE}}"
SIM_PROP="${SIM_PROP:-0.30}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-0.30}"
SEED="${SEED:-42}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/pgsui-xdg-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/pgsui-mpl-cache}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${TMPDIR:-/tmp}/pgsui-numba-cache}"

command_args=(
  --input-dir "${INPUT_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --tree-dir "${TREE_DIR}"
  --reference-mask-dir "${REFERENCE_MASK_DIR}"
  --datasets-file "${DATASETS_FILE}"
  --mask-mode "${MASK_MODE}"
  --sim-prop "${SIM_PROP}"
  --validation-split "${VALIDATION_SPLIT}"
  --seed "${SEED}"
  --strategies
  random
  random_weighted
  random_weighted_inv
  nonrandom
  nonrandom_weighted
)
if [[ -n "${PGSUI_RESULTS_DIR:-}" ]]; then
  command_args+=(--pgsui-results-dir "${PGSUI_RESULTS_DIR}")
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/simulate_gtimputation_missingness.py" \
  "${command_args[@]}" \
  "$@"
