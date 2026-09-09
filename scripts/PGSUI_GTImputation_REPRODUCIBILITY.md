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

## 1. Generate the canonical masks and GTImputation inputs

```zsh
PYTHON_BIN=/path/to/python ./simulate_gtimputation_missingness.zsh
```

The wrapper regenerates all five strategies with seed 42, 30% simulated
missingness, and a 30% validation/test partition. The Python program:

1. reads the original VCF `GT` fields without filtering samples or loci;
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

```zsh
MASK_MODE=reuse ./simulate_gtimputation_missingness.zsh
MASK_MODE=verify OUTPUT_DIR=/tmp/pgsui-mask-verification \
  REFERENCE_MASK_DIR=../canonical_benchmark \
  ./simulate_gtimputation_missingness.zsh
```

`reuse` copies an existing mask into the corrected VCF writer. `verify`
regenerates each mask and stops unless every coordinate matches the reference.

## 2. Build and transfer the AWS PCS bundle

From the PG-SUI repository root, build the portable bundle:

```zsh
zsh scripts/build_pgsui_hpc_bundle.zsh \
  /path/to/05_pgsui_gtimputation_validation
```

Transfer the resulting `canonical_benchmark/hpc_bundle/` directory to
`/home/martinb6/pgsui_gti/` on the AWS PCS login host. Exact outbound and
return `rsync` commands are included in the bundle README.

## 3. Rerun PG-SUI on GPU nodes

The canonical masks use the unmodified VCF REF/HET/ALT genotypes as 0/1/2.
Therefore, legacy PG-SUI reports must not be combined with the canonical
GTImputation rerun. Install PG-SUI 1.8.4 in the `pgsui-gti` environment, then
submit from the transferred bundle root:

```zsh
export PGSUI_BENCHMARK_ROOT="$PWD"
export MAX_CONCURRENT=4
zsh scripts/submit_pgsui_canonical_gpu_array.zsh
```

The 50-row array runs one dataset/strategy combination per GPU. Every command
uses the balanced preset, seed 42, 30% missingness, a 30% combined
validation/test split, batch size 128, 100 multi-objective Optuna trials,
`f1`, `mcc`, and `average_precision`, and `--verbose`. `n_jobs=1` prevents
multiple training trials from contending for one GPU; array concurrency
provides scheduler-level parallelism.

Each model writes `evaluation_mask_test.npz`. The task runner compares exact
test indices and every evaluated coordinate with the canonical mask before it
writes a success marker.

## 4. Validate PG-SUI results

After the array completes:

```zsh
zsh scripts/analyze_pgsui_canonical_results.zsh
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

```zsh
PYTHON_BIN=/path/to/python zsh scripts/visualize_pgsui_gtimputation_results.zsh
```

The scoring program uses only the exported PG-SUI test coordinates, maps both
programs to the same PG-SUI 0/1/2 class semantics, and validates total and
REF/HET/ALT support before producing the comparison tables and plots.

## Output map

- `canonical_benchmark/masked_vcfs/`: GTImputation input VCFs
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
