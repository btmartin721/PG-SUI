# PG-SUI versus GTImputation reproducibility workflow

This workflow creates one coordinate-identical benchmark for PG-SUI and
GTImputation across ten datasets and five missingness strategies. It does not
modify the source VCFs or the legacy result directories.

## Requirements

- Python 3.12 or newer
- PG-SUI installed from the accompanying source tree
- the PG-SUI runtime dependencies, including SNPio and scikit-learn
- the original VCFs in `inputs/test-vcf-files/`
- the IQ-TREE files retained in the legacy simulation tree

Run commands from the `scripts/` directory in this reviewer-package section.
Set `PYTHON_BIN` when PG-SUI is installed in a non-default environment.
All workflow shell entrypoints use Bash. Their existing `.zsh` suffixes are
retained only for backward-compatible filenames.
The orchestration filenames containing `gpu` are also retained for
compatibility; their contents and submitted jobs are CPU-only.

## 1. Generate the canonical masks and GTImputation inputs

```bash
PYTHON_BIN=/path/to/python bash simulate_gtimputation_missingness.zsh
```

The wrapper regenerates all five strategies with seed 42, 30% simulated
missingness, and a 30% validation/test partition. The Python program:

1. loads the original VCF with SNPio and obtains 0/1/2 truth from
   `GenotypeEncoder.genotypes_012` without filtering samples or loci;
2. reconstructs PG-SUI's deterministic train/validation/test sample split;
3. regenerates `random`, `random_weighted`, `random_weighted_inv`,
   `nonrandom`, and `nonrandom_weighted` masks;
4. writes missing `GT` calls directly into a copy of each VCF while preserving
   all other FORMAT fields;
5. requires exact coordinate equality between the intended mask and the
   written VCF; and
6. records arguments, software versions, hashes, class labels, and mask
   differences from the legacy saved masks.

The outputs are placed in `canonical_benchmark/`. Existing files are not
overwritten unless `--force` is explicitly passed.

The default `regenerate` mode is the primary workflow. Two audit modes are
also available:

```bash
MASK_MODE=reuse bash simulate_gtimputation_missingness.zsh
MASK_MODE=verify OUTPUT_DIR=/tmp/pgsui-mask-verification \
  REFERENCE_MASK_DIR=../canonical_benchmark \
  bash simulate_gtimputation_missingness.zsh
```

`reuse` copies an existing mask into the corrected VCF writer. `verify`
regenerates each mask and stops unless every coordinate matches the reference.

## 2. Build and transfer the AWS PCS bundle

From the PG-SUI repository root, build the portable bundle:

```bash
bash scripts/build_pgsui_hpc_bundle.zsh \
  /path/to/05_pgsui_gtimputation_validation
```

Transfer the resulting `canonical_benchmark/hpc_bundle/` directory to
`/home/martinb6/pgsui_gti/` on the AWS PCS login host. Exact outbound and
return `rsync` commands are included in the bundle README.

## 3. Rerun PG-SUI on CPU nodes

The canonical masks and class labels use SNPio's REF/HET/ALT 0/1/2 encoder.
Therefore, legacy PG-SUI reports must not be combined with the canonical
GTImputation rerun. The task manifest uses `device=cpu`, `_cpu` output
prefixes, and `n_jobs=1`. Install PG-SUI 1.8.6 in the `pgsui-gti-2`
environment.

The SLURM template requests `shu-hpc-biocpu`, four CPUs per worker, no explicit
memory limit, and 72 hours per worker. Override the partition with
`PGSUI_CPU_PARTITION` or the environment with `PGSUI_CONDA_ENV` if necessary.
Submit from the transferred bundle root:

```bash
export PGSUI_BENCHMARK_ROOT="$PWD"
bash scripts/submit_pgsui_canonical_gpu_array.zsh
```

The submission creates four persistent array workers rather than 50 array
elements. Each worker runs all five strategies for one dataset sequentially
before taking another dataset. Thus, SNPio never reads one dataset in multiple
concurrent tasks, the scheduler sees only four jobs, and at most four PG-SUI
tasks run at once. Every task stages its VCF and SNPio outputs in a private
working directory. A failed task is recorded and does not stop that worker from
attempting its remaining tasks. Resubmission skips successful tasks and retries
incomplete or failed tasks.

Every command uses the balanced preset, seed 42, 30% missingness, a 30%
combined validation/test split, batch size 128, 100 multi-objective Optuna
trials, `f1`, `mcc`, and `average_precision`, and `--verbose`. `n_jobs=1`
prevents nested training parallelism from contending for a worker's four CPUs.

For every `nonrandom` and `nonrandom_weighted` task, the runner supplies the
dataset-specific phylogenetic inputs from `inputs/iqtree/`:

- `--treefile inputs/iqtree/<dataset>.treefile`
- `--qmatrix inputs/iqtree/<dataset>.iqtree`
- `--siterates inputs/iqtree/<dataset>.rate`

The runner verifies all required files before launching PG-SUI. Tree-related
options are omitted for `random`, `random_weighted`, and `random_weighted_inv`.

Each model writes `evaluation_mask_test.npz`. The task runner compares exact
test indices and every evaluated coordinate with the canonical mask before it
writes a success marker.

## 4. Validate PG-SUI results

After the array completes:

```bash
bash scripts/analyze_pgsui_canonical_results.zsh
```

The strict analyzer writes completeness and coordinate-audit tables, per-class
metrics, alternate-genotype summaries, runtime metadata, and publication PNGs.

## 5. Run GTImputation manually

For every masked VCF listed in
`manifests/gtimputation_manual_tasks.tsv`, run both the Naive
and SOM methods. For SOM, build/use the GTDB from that same masked input.

Copy each imputed VCF to the exact `output_vcf` destination under
`results/gtimputation/`. The ready-to-use scoring manifest is
`results/gtimputation/metadata/manifest.csv`. Its optional `runtime_seconds`
column may be filled with the measured wall-clock runtime.

Do not run either program on only a subset of samples or loci. The comparison
program rejects missing mask coordinates and mismatched class support.

## 6. Score and reproduce combined tables and figures

After both result grids are complete:

```bash
PYTHON_BIN=/path/to/python bash scripts/visualize_pgsui_gtimputation_results.zsh
```

The scoring program uses only the exported PG-SUI test coordinates, encodes
both truth and GTImputation VCFs with SNPio's `GenotypeEncoder`, and validates
total and REF/HET/ALT support before producing the comparison tables and plots.

## Output map

- `canonical_benchmark/masked_vcfs/`: GTImputation input VCFs
- `canonical_benchmark/canonical_vcfs/`: SNPio-sorted/indexed truth VCFs used
  by both programs
- `canonical_benchmark/masks/`: full and test-only masks in NPZ and TSV form
- `canonical_benchmark/splits/`: sample split indices and sample-level TSVs
- `canonical_benchmark/manifests/simulation_manifest.csv`: one row per
  dataset-strategy input
- `canonical_benchmark/manifests/gtimputation_run_sheet.csv`: 100 manual GUI
  runs and expected output filenames
- `canonical_benchmark/manifests/pgsui_run_sheet.csv`: PG-SUI rerun settings
- `canonical_benchmark/provenance/`: arguments, software versions, and SHA-256
  hashes
- `canonical_benchmark/comparison_results/`: regenerated tables and figures
- `canonical_benchmark/hpc_bundle/`: self-contained AWS PCS inputs, manifests,
  scripts, results destinations, post-hoc analysis, logs, and checksums
