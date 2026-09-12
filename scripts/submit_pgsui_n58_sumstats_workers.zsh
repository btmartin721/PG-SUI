#!/usr/bin/env zsh

set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  print -u2 "Usage: $0 BUNDLE_ROOT [MAX_CONCURRENT=8]"
  exit 64
fi

BUNDLE_ROOT=${1:A}
MAX_CONCURRENT=${2:-8}
PARTITION=${PGSUI_CPU_PARTITION:-shu-hpc-biocpu}
MANIFEST="$BUNDLE_ROOT/manifests/pgsui_n58_sumstats_tasks.tsv"
WORKER_MANIFEST="$BUNDLE_ROOT/manifests/pgsui_n58_sumstats_workers.tsv"
SBATCH_SCRIPT="$BUNDLE_ROOT/scripts/pgsui_n58_sumstats_workers.slurm"

if [[ ! -f "$MANIFEST" ]]; then
  print -u2 "Missing summary-statistics task manifest: $MANIFEST"
  exit 66
fi
if [[ ! -f "$WORKER_MANIFEST" ]]; then
  print -u2 "Missing summary-statistics worker manifest: $WORKER_MANIFEST"
  exit 66
fi
if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  print -u2 "Missing SLURM script: $SBATCH_SCRIPT"
  exit 66
fi
if [[ "$MAX_CONCURRENT" != <-> ]] || (( MAX_CONCURRENT < 1 || MAX_CONCURRENT > 8 )); then
  print -u2 "MAX_CONCURRENT must be an integer between 1 and 8"
  exit 64
fi

python - "$MANIFEST" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
if len(rows) != 58:
    raise SystemExit(f"Expected 58 tasks, found {len(rows)}")
if [int(row["task_id"]) for row in rows] != list(range(58)):
    raise SystemExit("Task IDs must be contiguous from 0 through 57")
if any(row["n_jobs"] != "1" for row in rows):
    raise SystemExit("Every summary-statistics task must use one thread")
if any(row["ploidy"] != "2" for row in rows):
    raise SystemExit("Every N=58 summary-statistics task must be diploid")
PY

python - "$WORKER_MANIFEST" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
task_ids = [task_id for row in rows for task_id in row["task_ids"].split()]
if [int(row["worker_id"]) for row in rows] != list(range(8)):
    raise SystemExit("Worker IDs must be contiguous from 0 through 7")
if len(task_ids) != 58 or len(task_ids) != len(set(task_ids)):
    raise SystemExit("Eight workers must cover all 58 tasks exactly once")
if sorted(map(int, task_ids)) != list(range(58)):
    raise SystemExit("Worker assignments must cover task IDs 0 through 57")
PY

export PGSUI_N58_ROOT="$BUNDLE_ROOT"
sbatch \
  --partition="$PARTITION" \
  --array="0-7%${MAX_CONCURRENT}" \
  --export="ALL,PGSUI_N58_ROOT=${BUNDLE_ROOT}" \
  "$SBATCH_SCRIPT"
