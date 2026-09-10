#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(dirname -- "${SCRIPT_DIR}")
REVIEWER_PACKAGE_ROOT=${1:?Usage: build_pgsui_hpc_bundle.zsh REVIEWER_PACKAGE_ROOT [OUTPUT_DIR]}
OUTPUT_DIR=${2:-${REVIEWER_PACKAGE_ROOT}/canonical_benchmark/hpc_bundle}
PYTHON_BIN=${PYTHON_BIN:-python}

PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
"${PYTHON_BIN}" "${SCRIPT_DIR}/build_pgsui_hpc_bundle.py" \
  --reviewer-package-root "${REVIEWER_PACKAGE_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --scripts-dir "${SCRIPT_DIR}"
