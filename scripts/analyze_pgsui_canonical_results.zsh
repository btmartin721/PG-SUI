#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
BUNDLE_ROOT=${PGSUI_BENCHMARK_ROOT:-${SCRIPT_DIR:h}}

python "${SCRIPT_DIR}/analyze_pgsui_canonical_results.py" \
  --bundle-root "${BUNDLE_ROOT}" \
  "$@"
