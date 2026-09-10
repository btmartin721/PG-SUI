#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_VALIDATION_ROOT=$(dirname -- "${SCRIPT_DIR}")
PARENT_VALIDATION_ROOT=$(dirname -- "${DEFAULT_VALIDATION_ROOT}")
IS_HPC_BUNDLE=false
if [[ -f "${DEFAULT_VALIDATION_ROOT}/manifests/pgsui_gpu_tasks.tsv" ]]; then
  IS_HPC_BUNDLE=true
elif [[ ! -d "${DEFAULT_VALIDATION_ROOT}/inputs/test-vcf-files" ]] \
  && [[ -d "${PARENT_VALIDATION_ROOT}/inputs/test-vcf-files" ]]; then
  DEFAULT_VALIDATION_ROOT="${PARENT_VALIDATION_ROOT}"
fi
VALIDATION_ROOT="${VALIDATION_ROOT:-${DEFAULT_VALIDATION_ROOT}}"
if [[ "${IS_HPC_BUNDLE}" == true ]]; then
  CANONICAL_ROOT="${CANONICAL_ROOT:-${VALIDATION_ROOT}}"
  PGSUI_DIR="${PGSUI_DIR:-${CANONICAL_ROOT}/results/pgsui}"
  GTI_DIR="${GTI_DIR:-${CANONICAL_ROOT}/results/gtimputation}"
  DEFAULT_OUTPUT_DIR="${CANONICAL_ROOT}/analysis/comparison"
  DEFAULT_RUNTIME_BACKENDS="cpu"
else
  CANONICAL_ROOT="${CANONICAL_ROOT:-${VALIDATION_ROOT}/canonical_benchmark}"
  PGSUI_DIR="${PGSUI_DIR:-${CANONICAL_ROOT}/results-pgsui}"
  GTI_DIR="${GTI_DIR:-${CANONICAL_ROOT}/results-gti}"
  DEFAULT_OUTPUT_DIR="${CANONICAL_ROOT}/comparison_results"
  DEFAULT_RUNTIME_BACKENDS="cpu cuda"
fi
SIM_MANIFEST="${SIM_MANIFEST:-${CANONICAL_ROOT}/manifests/simulation_manifest.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
REPORT_TYPE="${REPORT_TYPE:-zygosity}"
PGSUI_BACKEND="${PGSUI_BACKEND:-cpu}"
RUNTIME_BACKENDS_VALUE="${RUNTIME_PGSUI_BACKENDS:-${DEFAULT_RUNTIME_BACKENDS}}"
read -r -a RUNTIME_BACKENDS_ARRAY <<< "${RUNTIME_BACKENDS_VALUE}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/visualize_pgsui_gtimputation_results.py" \
  --pgsui-dir "${PGSUI_DIR}" \
  --gti-dir "${GTI_DIR}" \
  --sim-manifest "${SIM_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --report-type "${REPORT_TYPE}" \
  --pgsui-backend "${PGSUI_BACKEND}" \
  --runtime-pgsui-backends "${RUNTIME_BACKENDS_ARRAY[@]}" \
  "$@"
