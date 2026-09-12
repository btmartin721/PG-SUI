#!/usr/bin/env python3
"""Build a portable AWS PCS bundle for the canonical benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pgsui.utils.canonical_benchmark import (
    MODEL_ORDER,
    SIMULATION_STRATEGIES,
    CanonicalBenchmarkTask,
    read_task_manifest,
)

EXPECTED_VERSION = "1.8.6"
PORTABLE_SUPPORT_NAME = "_canonical_benchmark_support.py"
SCRIPT_NAMES: tuple[str, ...] = (
    "build_pgsui_hpc_bundle.py",
    "build_pgsui_hpc_bundle.zsh",
    "run_pgsui_canonical_task.py",
    "run_pgsui_canonical_task.zsh",
    "pgsui_canonical_gpu_array.slurm",
    "submit_pgsui_canonical_gpu_array.zsh",
    "analyze_pgsui_canonical_results.py",
    "analyze_pgsui_canonical_results.zsh",
    "visualize_pgsui_gtimputation_results.py",
    "visualize_pgsui_gtimputation_results.zsh",
    "simulate_gtimputation_missingness.py",
    "simulate_gtimputation_missingness.zsh",
    "PGSUI_GTImputation_REPRODUCIBILITY.md",
    "pgsui_gtimputation_datasets.txt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reviewer-package-root",
        type=Path,
        required=True,
        help="05_pgsui_gtimputation_validation directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to canonical_benchmark/hpc_bundle under the package root.",
    )
    parser.add_argument(
        "--scripts-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    return parser.parse_args()


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def portable_mask_path(value: str, canonical_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (canonical_root / "manifests" / path).resolve()


def task_rows(simulation_manifest: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered = simulation_manifest.assign(
        strategy_order=simulation_manifest["strategy"].map(
            {name: index for index, name in enumerate(SIMULATION_STRATEGIES)}
        )
    ).sort_values(["dataset_id", "strategy_order"])
    for task_id, row in enumerate(ordered.to_dict("records")):
        dataset = str(row["dataset_id"])
        strategy = str(row["strategy"])
        needs_tree = strategy.startswith("nonrandom")
        input_name = Path(str(row["input_vcf"])).name
        rows.append(
            {
                "task_id": task_id,
                "dataset_id": dataset,
                "strategy": strategy,
                "seed": int(row["seed"]),
                "sim_prop": float(row["sim_prop"]),
                "validation_split": float(row["validation_split"]),
                "input_vcf": f"inputs/vcfs/{input_name}",
                "treefile": f"inputs/iqtree/{dataset}.treefile" if needs_tree else "",
                "qmatrix": f"inputs/iqtree/{dataset}.iqtree" if needs_tree else "",
                "siterates": f"inputs/iqtree/{dataset}.rate" if needs_tree else "",
                "mask_npz": f"masks/{dataset}/{strategy}/{Path(row['mask_npz']).name}",
                "evaluation_mask_tsv": (
                    f"masks/{dataset}/{strategy}/"
                    f"{Path(row['evaluation_mask_tsv']).name}"
                ),
                "split_tsv": f"splits/{dataset}/{Path(row['split_tsv']).name}",
                "output_prefix": f"results/pgsui/{dataset}_{strategy}_cpu",
                "device": "cpu",
                "preset": "balanced",
                "models": " ".join(MODEL_ORDER),
                "n_jobs": 1,
                "tune_n_trials": 100,
                "tune_metrics": "f1 mcc average_precision",
                "batch_size": 128,
                "expected_pgsui_version": EXPECTED_VERSION,
            }
        )
    return rows


def write_tsv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def validate_portable_tasks(
    manifest_path: Path, bundle_root: Path
) -> list[CanonicalBenchmarkTask]:
    """Validate the portable task grid and every required input path."""
    tasks = read_task_manifest(manifest_path)
    if len(tasks) != 50:
        raise ValueError(f"Expected 50 PG-SUI tasks, found {len(tasks)}")

    for task in tasks:
        if task.device != "cpu":
            raise ValueError(
                f"Task {task.task_id} must use device=cpu, found {task.device!r}"
            )
        if not task.output_prefix.endswith("_cpu"):
            raise ValueError(
                f"Task {task.task_id} output prefix must end in '_cpu': "
                f"{task.output_prefix}"
            )

        required_paths = task.required_input_paths(bundle_root)
        missing = [path for path in required_paths if not path.is_file()]
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(
                f"Required inputs are missing for task {task.task_id}:\n{formatted}"
            )

        if task.strategy.startswith("nonrandom"):
            expected_treefile = f"inputs/iqtree/{task.dataset_id}.treefile"
            expected_iqtree = f"inputs/iqtree/{task.dataset_id}.iqtree"
            expected_siterates = f"inputs/iqtree/{task.dataset_id}.rate"
            observed = (task.treefile, task.qmatrix, task.siterates)
            expected = (expected_treefile, expected_iqtree, expected_siterates)
            if observed != expected:
                raise ValueError(
                    f"Incorrect phylogenetic inputs for task {task.task_id}: "
                    f"expected {expected}, found {observed}"
                )

    return tasks


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(bundle_root: Path) -> int:
    excluded = {"results", "analysis", "logs", "status"}
    paths = [
        path
        for path in bundle_root.rglob("*")
        if path.is_file()
        and not excluded.intersection(path.relative_to(bundle_root).parts)
        and path.name != "SHA256SUMS"
    ]
    lines = [
        f"{sha256(path)}  {path.relative_to(bundle_root)}" for path in sorted(paths)
    ]
    checksum_path = bundle_root / "provenance" / "SHA256SUMS"
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def reviewer_readme() -> str:
    return """# PG-SUI canonical CPU benchmark bundle

This portable directory contains 10 SNPio-canonical VCF datasets, five
missingness masks per dataset, inputs for phylogeny-aware simulations, a
50-task CPU manifest, SLURM orchestration, strict post-hoc validation, and empty
result destinations.

All workflow shell entrypoints use Bash. Their existing `.zsh` suffixes are
retained only for backward-compatible filenames. The two orchestration
filenames containing `gpu` are also retained for compatibility; their contents
and submitted jobs are CPU-only.

## Software

Create or activate a Python 3.12 `pgsui-gti-2` environment containing PG-SUI
1.8.6. The bundle includes a portable canonical benchmark helper, and the task
runner refuses to execute a different PG-SUI version. PG-SUI installs its
compatible SNPio dependency.

## Transfer to AWS PCS

From a local ZSH session, set the login hostname and run:

```zsh
export AWS_PCS_LOGIN_HOST="your-login-host"
rsync -avh --partial --info=progress2 \\
  "/path/to/hpc_bundle/" \\
  "martinb6@${AWS_PCS_LOGIN_HOST}:/home/martinb6/pgsui_gti/"
```

The recommended remote destination is `/home/martinb6/pgsui_gti/`. The SLURM
submission wrapper exports the selected bundle root, so the workflow remains
portable if a different remote directory is preferred. No hostname is embedded
because AWS PCS login endpoints differ between deployments.

## Submit four persistent CPU workers

The task manifest uses `device=cpu`, `_cpu` output prefixes, and `n_jobs=1`.
The SLURM template requests the `shu-hpc-biocpu` partition, four CPUs per
worker, no explicit memory limit, and 72 hours per worker. The default Conda
environment is `pgsui-gti-2`. Override these site-specific defaults with
`PGSUI_CPU_PARTITION` or `PGSUI_CONDA_ENV` when needed. From this directory on
the AWS PCS login node:

```bash
export PGSUI_BENCHMARK_ROOT="$PWD"
bash scripts/submit_pgsui_canonical_gpu_array.zsh
```

The submission creates only four array elements. Each persistent worker runs
all five strategies for one dataset sequentially before taking another dataset.
Therefore no dataset is read by multiple SNPio processes at once, no more than
four PG-SUI tasks run concurrently, and only four jobs appear in the scheduler.
Every task also stages its VCF in private node-local storage. Workers continue
after individual failures; resubmission skips successful tasks and retries
incomplete or failed tasks.

Each task uses 100 Optuna trials with `f1`, `mcc`, and `average_precision`, seed
42, the balanced preset, and verbose logging. A success marker is written only
after all six model reports match the canonical REF/HET/ALT test-mask support.

For every `nonrandom` and `nonrandom_weighted` task, the runner passes the
dataset-specific files under `inputs/iqtree/` as follows:

- `--treefile inputs/iqtree/<dataset>.treefile`
- `--qmatrix inputs/iqtree/<dataset>.iqtree`
- `--siterates inputs/iqtree/<dataset>.rate`

The task runner stops before launching PG-SUI if any required file is missing.
The three options are omitted for the other simulation strategies.

## Regenerate and verify all five simulation strategies

The bundle includes the original VCFs, IQ-TREE inputs, and canonical mask
archives. From the bundle root, activate PG-SUI 1.8.6 and run:

```bash
bash scripts/simulate_gtimputation_missingness.zsh --verbose
```

The bundle-aware defaults rerun all 10 datasets by all five strategies and
require exact agreement with the included masks. Regenerated files are written
under `regenerated_simulations/`; use `--force` for a deliberate repeat run.
The portable simulation manifest points its reference fields to these included
canonical archives. Its `legacy_reference_*` columns retain the comparison
against the superseded pre-correction masks for provenance.

## Validate and summarize PG-SUI results

```bash
bash scripts/analyze_pgsui_canonical_results.zsh
```

Tables are written to `analysis/tables`, publication PNGs to `analysis/plots`,
and provenance to `analysis/analysis_manifest.json`.

## GTImputation

Use `manifests/gtimputation_manual_tasks.tsv` to run Naive and SOM manually in
the GUI. Each row names the exact masked VCF and expected output destination.
After those VCFs are returned, use the supplied comparison visualizer for the
combined PG-SUI/GTImputation figures and inferential tables.

## Return results to the reviewer package

From the local `hpc_bundle` directory:

```zsh
export AWS_PCS_LOGIN_HOST="your-login-host"
rsync -avh --partial --info=progress2 \\
  "martinb6@${AWS_PCS_LOGIN_HOST}:/home/martinb6/pgsui_gti/results/" \\
  "./results/"
rsync -avh --partial --info=progress2 \\
  "martinb6@${AWS_PCS_LOGIN_HOST}:/home/martinb6/pgsui_gti/status/" \\
  "./status/"
rsync -avh --partial --info=progress2 \\
  "martinb6@${AWS_PCS_LOGIN_HOST}:/home/martinb6/pgsui_gti/logs/" \\
  "./logs/"
```

## Integrity

Run this from the bundle root before and after transfer:

```bash
sha256sum --check provenance/SHA256SUMS
```
"""


def main() -> int:
    args = parse_args()
    package_root = args.reviewer_package_root.resolve()
    canonical_root = package_root / "canonical_benchmark"
    output_dir = (
        args.output_dir.resolve() if args.output_dir else canonical_root / "hpc_bundle"
    )
    scripts_dir = args.scripts_dir.resolve()

    simulation_manifest_path = canonical_root / "manifests" / "simulation_manifest.csv"
    simulation_manifest = pd.read_csv(simulation_manifest_path)
    if len(simulation_manifest) != 50:
        raise ValueError(
            f"Expected 50 simulation rows, found {len(simulation_manifest)}"
        )
    pairs = simulation_manifest[["dataset_id", "strategy"]].drop_duplicates()
    if len(pairs) != 50:
        raise ValueError("Simulation manifest does not contain 50 unique pairs")

    datasets = sorted(simulation_manifest["dataset_id"].unique())
    iqtree_root = (
        package_root
        / "outputs"
        / "gtimputation-results"
        / "zygosity_missingness_simulations"
        / "iqtree"
    )
    for dataset in datasets:
        dataset_rows = simulation_manifest.loc[
            simulation_manifest["dataset_id"].astype(str).eq(dataset)
        ]
        source_vcfs = {
            portable_mask_path(str(value), canonical_root)
            for value in dataset_rows["input_vcf"]
        }
        if len(source_vcfs) != 1:
            raise ValueError(
                f"Expected one canonical input VCF for {dataset}, found "
                f"{sorted(str(path) for path in source_vcfs)}"
            )
        source_vcf = source_vcfs.pop()
        bundled_vcf = output_dir / "inputs" / "vcfs" / source_vcf.name
        copy_file(source_vcf, bundled_vcf)
        source_index = Path(f"{source_vcf}.tbi")
        if source_index.is_file():
            copy_file(source_index, Path(f"{bundled_vcf}.tbi"))
        copy_file(
            iqtree_root / f"{dataset}.treefile",
            output_dir / "inputs" / "iqtree" / f"{dataset}.treefile",
        )
        copy_file(
            iqtree_root / f"{dataset}.iqtree",
            output_dir / "inputs" / "iqtree" / f"{dataset}.iqtree",
        )
        copy_file(
            iqtree_root / f"{dataset}.rate",
            output_dir / "inputs" / "iqtree" / f"{dataset}.rate",
        )

    for row in simulation_manifest.to_dict("records"):
        dataset = str(row["dataset_id"])
        strategy = str(row["strategy"])
        for column in ("mask_npz", "evaluation_mask_tsv"):
            source = portable_mask_path(str(row[column]), canonical_root)
            copy_file(source, output_dir / "masks" / dataset / strategy / source.name)
        split_source = portable_mask_path(str(row["split_tsv"]), canonical_root)
        copy_file(split_source, output_dir / "splits" / dataset / split_source.name)
        masked_source = portable_mask_path(str(row["masked_vcf"]), canonical_root)
        copy_file(
            masked_source,
            output_dir
            / "inputs"
            / "masked_vcfs"
            / dataset
            / strategy
            / masked_source.name,
        )

    task_row_data = task_rows(simulation_manifest)
    task_manifest_path = output_dir / "manifests" / "pgsui_gpu_tasks.tsv"
    write_tsv(task_row_data, task_manifest_path)
    tasks = validate_portable_tasks(task_manifest_path, output_dir)

    gti_rows: list[dict[str, Any]] = []
    gti_result_rows: list[dict[str, Any]] = []
    for task_data in task_row_data:
        for method in ("naive", "som"):
            dataset = task_data["dataset_id"]
            strategy = task_data["strategy"]
            masked_name = next(
                (output_dir / "inputs" / "masked_vcfs" / dataset / strategy).glob(
                    "*.vcf"
                )
            ).name
            gti_rows.append(
                {
                    "task_id": len(gti_rows),
                    "dataset_id": dataset,
                    "strategy": strategy,
                    "method": method,
                    "input_masked_vcf": (
                        f"inputs/masked_vcfs/{dataset}/{strategy}/{masked_name}"
                    ),
                    "output_vcf": (
                        f"results/gtimputation/vcfs/{method}/"
                        f"{dataset}__sim30__{strategy}__seed42__{method}.vcf"
                    ),
                    "evaluation_mask_tsv": task_data["evaluation_mask_tsv"],
                }
            )
            output_name = f"{dataset}__sim30__{strategy}__seed42__{method}.vcf"
            gti_result_rows.append(
                {
                    "run_id": output_name.removesuffix(".vcf"),
                    "method": method,
                    "dataset_name": dataset,
                    "simulation_strategy": strategy,
                    "seed": task_data["seed"],
                    "source_masked_vcf": (
                        f"../../inputs/masked_vcfs/{dataset}/{strategy}/{masked_name}"
                    ),
                    "copied_vcf": f"vcfs/{method}/{output_name}",
                    "candidate_vcf_filename": output_name,
                    "runtime_seconds": "",
                }
            )
    write_tsv(gti_rows, output_dir / "manifests" / "gtimputation_manual_tasks.tsv")
    portable_simulation = simulation_manifest.copy()
    for column in (
        "reference_mask_checked",
        "reference_mask_match",
        "n_reference_simulated_mask_differences",
        "n_reference_original_mask_differences",
    ):
        portable_simulation[f"legacy_{column}"] = portable_simulation[column]
    portable_simulation["reference_mask_checked"] = True
    portable_simulation["reference_mask_match"] = True
    portable_simulation["n_reference_simulated_mask_differences"] = 0
    portable_simulation["n_reference_original_mask_differences"] = 0
    for index, row in portable_simulation.iterrows():
        dataset = str(row["dataset_id"])
        strategy = str(row["strategy"])
        portable_simulation.at[index, "input_vcf"] = f"../inputs/vcfs/{dataset}.vcf"
        portable_simulation.at[index, "masked_vcf"] = (
            f"../inputs/masked_vcfs/{dataset}/{strategy}/{Path(row['masked_vcf']).name}"
        )
        portable_simulation.at[index, "mask_npz"] = (
            f"../masks/{dataset}/{strategy}/{Path(row['mask_npz']).name}"
        )
        portable_simulation.at[index, "reference_mask"] = (
            f"../masks/{dataset}/{strategy}/{Path(row['mask_npz']).name}"
        )
        portable_simulation.at[index, "evaluation_mask_tsv"] = (
            f"../masks/{dataset}/{strategy}/{Path(row['evaluation_mask_tsv']).name}"
        )
        portable_simulation.at[index, "full_mask_tsv"] = ""
        portable_simulation.at[index, "split_tsv"] = (
            f"../splits/{dataset}/{Path(row['split_tsv']).name}"
        )
        portable_simulation.at[index, "treefile"] = (
            f"../inputs/iqtree/{dataset}.treefile"
            if strategy.startswith("nonrandom")
            else ""
        )
        portable_simulation.at[index, "qmatrix"] = (
            f"../inputs/iqtree/{dataset}.iqtree"
            if strategy.startswith("nonrandom")
            else ""
        )
        portable_simulation.at[index, "siterates"] = (
            f"../inputs/iqtree/{dataset}.rate"
            if strategy.startswith("nonrandom")
            else ""
        )
    portable_simulation.to_csv(
        output_dir / "manifests" / "simulation_manifest.csv", index=False
    )

    for script_name in SCRIPT_NAMES:
        copy_file(scripts_dir / script_name, output_dir / "scripts" / script_name)
    copy_file(
        scripts_dir.parent / "pgsui" / "utils" / "canonical_benchmark.py",
        output_dir / "scripts" / PORTABLE_SUPPORT_NAME,
    )

    for relative in (
        "results/pgsui",
        "results/gtimputation/vcfs/naive",
        "results/gtimputation/vcfs/som",
        "results/gtimputation/metadata",
        "analysis/tables",
        "analysis/plots",
        "logs",
        "status",
        "provenance",
    ):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)

    pd.DataFrame(gti_result_rows).to_csv(
        output_dir / "results" / "gtimputation" / "metadata" / "manifest.csv",
        index=False,
    )

    (output_dir / "README.md").write_text(reviewer_readme(), encoding="utf-8")
    provenance = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_simulation_manifest": str(simulation_manifest_path),
        "expected_pgsui_version": EXPECTED_VERSION,
        "n_datasets": len(datasets),
        "n_strategies": len(SIMULATION_STRATEGIES),
        "n_pgsui_tasks": len(tasks),
        "pgsui_device": tasks[0].device,
        "portable_canonical_support": f"scripts/{PORTABLE_SUPPORT_NAME}",
        "slurm_worker_count": 4,
        "n_tree_enabled_pgsui_tasks": sum(
            task.strategy.startswith("nonrandom") for task in tasks
        ),
        "n_gtimputation_manual_tasks": len(gti_rows),
    }
    (output_dir / "provenance" / "bundle_manifest.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    provenance["n_checksummed_files"] = write_checksums(output_dir)
    (output_dir / "provenance" / "bundle_manifest.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_checksums(output_dir)
    print(f"Created canonical HPC bundle: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
