#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
BUNDLE_ROOT=${PGSUI_BENCHMARK_ROOT:-${SCRIPT_DIR:h}}
TASK_INDEX=${1:?Usage: run_pgsui_canonical_task.zsh TASK_INDEX [additional Python arguments]}
shift

python "${SCRIPT_DIR}/run_pgsui_canonical_task.py" \
  --bundle-root "${BUNDLE_ROOT}" \
  --task-index "${TASK_INDEX}" \
  "$@"
