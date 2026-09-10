#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
BUNDLE_ROOT=${PGSUI_BENCHMARK_ROOT:-$(dirname -- "${SCRIPT_DIR}")}
PYTHON_BIN=${PYTHON_BIN:-python}

"${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_pgsui_canonical_results.py" \
  --bundle-root "${BUNDLE_ROOT}" \
  "$@"
