#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
BUNDLE_ROOT=${PGSUI_BENCHMARK_ROOT:-$(dirname -- "${SCRIPT_DIR}")}
TASK_INDEX=${1:?Usage: run_pgsui_canonical_task.zsh TASK_INDEX [additional Python arguments]}
PYTHON_BIN=${PYTHON_BIN:-python}
shift

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_pgsui_canonical_task.py" \
  --bundle-root "${BUNDLE_ROOT}" \
  --task-index "${TASK_INDEX}" \
  "$@"
