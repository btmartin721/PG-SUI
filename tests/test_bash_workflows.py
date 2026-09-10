"""Regression tests for Bash-compatible reviewer workflow entrypoints."""

from __future__ import annotations

import csv
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASH_ENTRYPOINTS = (
    PROJECT_ROOT / "scripts" / "analyze_pgsui_canonical_results.zsh",
    PROJECT_ROOT / "scripts" / "build_pgsui_hpc_bundle.zsh",
    PROJECT_ROOT / "scripts" / "run_pgsui_canonical_task.zsh",
    PROJECT_ROOT / "scripts" / "simulate_gtimputation_missingness.zsh",
    PROJECT_ROOT / "scripts" / "submit_pgsui_canonical_gpu_array.zsh",
    PROJECT_ROOT / "scripts" / "visualize_pgsui_gtimputation_results.zsh",
    PROJECT_ROOT / "scripts" / "pgsui_canonical_gpu_array.slurm",
)


@pytest.mark.parametrize("script_path", BASH_ENTRYPOINTS, ids=lambda path: path.name)
def test_reviewer_workflow_entrypoint_is_valid_bash(script_path: Path) -> None:
    """Require each reviewer workflow entrypoint to parse under Bash."""
    source = script_path.read_text(encoding="utf-8")

    assert source.startswith(("#!/usr/bin/env bash\n", "#!/bin/bash\n"))
    subprocess.run(
        ["bash", "-n", str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_cpu_array_resources_and_concurrency() -> None:
    """Require four persistent CPU workers without an explicit memory limit."""
    submit_source = (
        PROJECT_ROOT / "scripts" / "submit_pgsui_canonical_gpu_array.zsh"
    ).read_text(encoding="utf-8")
    slurm_source = (
        PROJECT_ROOT / "scripts" / "pgsui_canonical_gpu_array.slurm"
    ).read_text(encoding="utf-8")

    assert "WORKER_COUNT=4" in submit_source
    assert "PGSUI_CPU_PARTITION" in submit_source
    assert "PGSUI_GPU_PARTITION" not in submit_source
    assert '--array="0-$((WORKER_COUNT - 1))%${WORKER_COUNT}"' in submit_source
    assert '--chdir="${BUNDLE_ROOT}"' in submit_source
    assert "#SBATCH --array=0-3%4" in slurm_source
    assert "#SBATCH --partition=shu-hpc-biocpu" in slurm_source
    assert "#SBATCH --cpus-per-task=4" in slurm_source
    assert "#SBATCH --time=72:00:00" in slurm_source
    assert "#SBATCH --mem" not in slurm_source
    assert "#SBATCH --gres" not in slurm_source
    assert "nvidia-smi" not in slurm_source
    assert "CONDA_ENV=${PGSUI_CONDA_ENV:-pgsui-gti-2}" in slurm_source
    assert 'export CUDA_VISIBLE_DEVICES=""' in slurm_source
    assert "DATASET_INDEX += WORKER_COUNT" in slurm_source
    assert "STRATEGIES_PER_DATASET=5" in slurm_source
    assert '--work-dir "${TASK_WORK_DIR}"' in slurm_source
    assert "continuing" in slurm_source


def test_cpu_submission_wrapper_builds_expected_sbatch_command(
    tmp_path: Path,
) -> None:
    """Exercise manifest validation and the complete CPU submission command."""
    bundle_root = tmp_path / "bundle"
    scripts_dir = bundle_root / "scripts"
    manifest_dir = bundle_root / "manifests"
    executable_dir = tmp_path / "bin"
    scripts_dir.mkdir(parents=True)
    manifest_dir.mkdir()
    executable_dir.mkdir()

    for name in (
        "submit_pgsui_canonical_gpu_array.zsh",
        "pgsui_canonical_gpu_array.slurm",
    ):
        shutil.copy2(PROJECT_ROOT / "scripts" / name, scripts_dir / name)

    manifest_path = manifest_dir / "pgsui_gpu_tasks.tsv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("task_id", "device", "output_prefix"),
            delimiter="\t",
        )
        writer.writeheader()
        for task_id in range(50):
            writer.writerow(
                {
                    "task_id": task_id,
                    "device": "cpu",
                    "output_prefix": f"results/task_{task_id:02d}_cpu",
                }
            )

    capture_path = tmp_path / "sbatch-arguments.txt"
    sbatch_path = executable_dir / "sbatch"
    sbatch_path.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "${SBATCH_CAPTURE}"\n',
        encoding="utf-8",
    )
    sbatch_path.chmod(0o755)

    completed = subprocess.run(
        ["bash", str(scripts_dir / "submit_pgsui_canonical_gpu_array.zsh")],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{executable_dir}{os.pathsep}{os.environ['PATH']}",
            "PGSUI_BENCHMARK_ROOT": str(bundle_root),
            "PGSUI_CPU_PARTITION": "test-cpu",
            "PYTHON_BIN": sys.executable,
            "SBATCH_CAPTURE": str(capture_path),
        },
    )

    assert completed.returncode == 0, completed.stderr
    arguments = capture_path.read_text(encoding="utf-8").splitlines()
    assert f"--chdir={bundle_root}" in arguments
    assert "--partition=test-cpu" in arguments
    assert "--array=0-3%4" in arguments
    assert str(scripts_dir / "pgsui_canonical_gpu_array.slurm") in arguments


def test_cpu_worker_continues_after_task_failure(tmp_path: Path) -> None:
    """Keep a worker moving through its lane after an individual task fails."""
    bundle_root = tmp_path / "bundle"
    scripts_dir = bundle_root / "scripts"
    manifest_dir = bundle_root / "manifests"
    conda_dir = tmp_path / "home" / "miniconda3" / "etc" / "profile.d"
    scripts_dir.mkdir(parents=True)
    manifest_dir.mkdir()
    conda_dir.mkdir(parents=True)

    slurm_path = scripts_dir / "pgsui_canonical_gpu_array.slurm"
    shutil.copy2(PROJECT_ROOT / "scripts" / slurm_path.name, slurm_path)
    (manifest_dir / "pgsui_gpu_tasks.tsv").write_text(
        "task_id\n" + "".join(f"{task_id}\n" for task_id in range(50)),
        encoding="utf-8",
    )
    (conda_dir / "conda.sh").write_text(
        "conda() { return 0; }\n",
        encoding="utf-8",
    )

    task_log = tmp_path / "task-indices.txt"
    python_stub = tmp_path / "python-stub"
    python_stub.write_text(
        """#!/usr/bin/env bash
set -u
if [[ "${1:-}" == "-c" ]]; then
  printf '50\\n'
  exit 0
fi
task_index=""
while (( $# )); do
  case "$1" in
    --task-index)
      task_index=$2
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
printf '%s\\n' "${task_index}" >> "${FAKE_TASK_LOG}"
if [[ "${task_index}" == "4" ]]; then
  exit 7
fi
""",
        encoding="utf-8",
    )
    python_stub.chmod(0o755)

    completed = subprocess.run(
        ["bash", str(slurm_path)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "HOME": str(tmp_path / "home"),
            "PGSUI_BENCHMARK_ROOT": str(bundle_root),
            "PYTHON_BIN": str(python_stub),
            "FAKE_TASK_LOG": str(task_log),
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_CPUS_PER_TASK": "4",
            "SLURM_JOB_ID": "123",
            "SLURM_SUBMIT_DIR": str(bundle_root),
        },
    )

    assert completed.returncode == 1
    assert task_log.read_text(encoding="utf-8").splitlines() == [
        "0",
        "1",
        "2",
        "3",
        "4",
        "20",
        "21",
        "22",
        "23",
        "24",
        "40",
        "41",
        "42",
        "43",
        "44",
    ]
    assert "task 4 failed with status 7; continuing" in completed.stderr


def test_task_runner_stages_vcf_and_index_privately(tmp_path: Path) -> None:
    """Keep SNPio input and sidecar creation inside one task directory."""
    runner_path = PROJECT_ROOT / "scripts" / "run_pgsui_canonical_task.py"
    spec = importlib.util.spec_from_file_location("canonical_task_runner", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    source = tmp_path / "inputs" / "dataset.vcf.gz"
    source.parent.mkdir()
    source.write_bytes(b"vcf-data")
    Path(f"{source}.tbi").write_bytes(b"index-data")
    work_dir = tmp_path / "task-work"

    staged = runner.stage_vcf(source, work_dir)

    assert staged == work_dir / source.name
    assert staged.read_bytes() == b"vcf-data"
    assert Path(f"{staged}.tbi").read_bytes() == b"index-data"


def test_bundle_runner_prefers_portable_cpu_support(tmp_path: Path) -> None:
    """Prefer the bundle's support helper over an older installed release."""
    bundle_root = tmp_path / "bundle"
    scripts_dir = bundle_root / "scripts"
    manifest_dir = bundle_root / "manifests"
    scripts_dir.mkdir(parents=True)
    manifest_dir.mkdir()

    runner_path = scripts_dir / "run_pgsui_canonical_task.py"
    shutil.copy2(PROJECT_ROOT / "scripts" / runner_path.name, runner_path)
    shutil.copy2(
        PROJECT_ROOT / "pgsui" / "utils" / "canonical_benchmark.py",
        scripts_dir / "_canonical_benchmark_support.py",
    )

    required_paths = (
        bundle_root / "inputs" / "example.vcf",
        bundle_root / "masks" / "example.evaluation.tsv",
        bundle_root / "masks" / "example.mask.npz",
        bundle_root / "splits" / "example.tsv",
    )
    for path in required_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    row = {
        "task_id": 0,
        "dataset_id": "example",
        "strategy": "random",
        "input_vcf": "inputs/example.vcf",
        "evaluation_mask_tsv": "masks/example.evaluation.tsv",
        "mask_npz": "masks/example.mask.npz",
        "split_tsv": "splits/example.tsv",
        "output_prefix": "results/pgsui/example_cpu",
        "device": "cpu",
        "models": (
            "ImputeAutoencoder ImputeVAE ImputeNLPCA ImputeUBP "
            "ImputeMostFrequent ImputeRefAllele"
        ),
        "tune_n_trials": 100,
        "tune_metrics": "f1 mcc average_precision",
        "expected_pgsui_version": "1.8.5",
    }
    with (manifest_dir / "pgsui_gpu_tasks.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t")
        writer.writeheader()
        writer.writerow(row)

    fake_package = tmp_path / "fake-site" / "pgsui"
    fake_package.mkdir(parents=True)
    (fake_package / "__init__.py").write_text(
        'raise RuntimeError("stale installed PG-SUI helper was imported")\n',
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(runner_path),
            "--bundle-root",
            str(bundle_root),
            "--task-index",
            "0",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(tmp_path / "fake-site")},
    )

    assert completed.returncode == 0, completed.stderr
    assert "--device cpu" in completed.stdout


def test_simulation_wrapper_handles_no_optional_arguments() -> None:
    """Keep the wrapper compatible with Bash 3.2 and ``set -u``."""
    script_path = PROJECT_ROOT / "scripts" / "simulate_gtimputation_missingness.zsh"
    completed = subprocess.run(
        ["bash", str(script_path), "--verbose"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHON_BIN": "/bin/echo"},
    )

    assert "--mask-mode regenerate" in completed.stdout
    assert "--verbose" in completed.stdout
