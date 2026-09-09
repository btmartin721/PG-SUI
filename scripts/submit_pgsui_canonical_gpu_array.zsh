#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
BUNDLE_ROOT=${PGSUI_BENCHMARK_ROOT:-${SCRIPT_DIR:h}}
MAX_CONCURRENT=${MAX_CONCURRENT:-4}
PARTITION=${PGSUI_GPU_PARTITION:-shu-hpc-biogpu}

if [[ ! "${MAX_CONCURRENT}" =~ '^[1-9][0-9]*$' ]]; then
  print -u2 "MAX_CONCURRENT must be a positive integer."
  exit 2
fi

mkdir -p "${BUNDLE_ROOT}/logs" "${BUNDLE_ROOT}/status"

TASK_COUNT=$(python -c \
  'import csv, pathlib, sys; p=pathlib.Path(sys.argv[1]); print(sum(1 for _ in csv.DictReader(p.open(), delimiter="\t")))' \
  "${BUNDLE_ROOT}/manifests/pgsui_gpu_tasks.tsv")

if (( TASK_COUNT != 50 )); then
  print -u2 "Expected 50 dataset/strategy tasks, found ${TASK_COUNT}."
  exit 2
fi

sbatch \
  --partition="${PARTITION}" \
  --array="0-$((TASK_COUNT - 1))%${MAX_CONCURRENT}" \
  --export="ALL,PGSUI_BENCHMARK_ROOT=${BUNDLE_ROOT}" \
  "${BUNDLE_ROOT}/scripts/pgsui_canonical_gpu_array.slurm"
