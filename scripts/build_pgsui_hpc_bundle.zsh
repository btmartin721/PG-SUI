#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
REVIEWER_PACKAGE_ROOT=${1:?Usage: build_pgsui_hpc_bundle.zsh REVIEWER_PACKAGE_ROOT [OUTPUT_DIR]}
OUTPUT_DIR=${2:-${REVIEWER_PACKAGE_ROOT}/canonical_benchmark/hpc_bundle}

PYTHONPATH="${SCRIPT_DIR:h}${PYTHONPATH:+:${PYTHONPATH}}" \
python "${SCRIPT_DIR}/build_pgsui_hpc_bundle.py" \
  --reviewer-package-root "${REVIEWER_PACKAGE_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --scripts-dir "${SCRIPT_DIR}"
