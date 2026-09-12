#!/usr/bin/env zsh

set -euo pipefail

if (( $# != 1 )); then
  print -u2 "Usage: $0 BUNDLE_ROOT"
  exit 64
fi

BUNDLE_ROOT=${1:A}
PYTHON_BIN=${PGSUI_PYTHON:-python}
SCRIPTS="$BUNDLE_ROOT/scripts"
METRICS="$BUNDLE_ROOT/analysis/tables/n58_metrics_long.tsv"
POPGEN_ROOT="$BUNDLE_ROOT/analysis/popgen_stats/by_dataset"
POPGEN_COMPARISON="$BUNDLE_ROOT/analysis/popgen_stats/comparison"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MPLCONFIGDIR="$BUNDLE_ROOT/work/posthoc/matplotlib"
export NUMBA_CACHE_DIR="$BUNDLE_ROOT/work/posthoc/numba"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$POPGEN_COMPARISON"

"$PYTHON_BIN" "$SCRIPTS/audit_pgsui_n58_results.py" \
  --bundle-root "$BUNDLE_ROOT"

"$PYTHON_BIN" "$SCRIPTS/audit_pgsui_n58_sumstats.py" \
  --bundle-root "$BUNDLE_ROOT"

"$PYTHON_BIN" "$SCRIPTS/analyze_pgsui_n58_results.py" \
  --bundle-root "$BUNDLE_ROOT" \
  --dpi 300

"$PYTHON_BIN" "$SCRIPTS/compare_dataset_stats.py" \
  --root "$POPGEN_ROOT" \
  --output-dir "$POPGEN_COMPARISON" \
  --out-prefix n58_popgen \
  --plots \
  --completed-metrics-long "$METRICS" \
  --strict-strategies \
  --expected-datasets 58

"$PYTHON_BIN" "$SCRIPTS/analyze_pgsui_n58_feature_effects.py" \
  --bundle-root "$BUNDLE_ROOT" \
  --dpi 300

print "N=58 post-hoc analysis complete: $BUNDLE_ROOT/analysis"
