# PG-SUI N=58 manuscript validation

This workflow runs PG-SUI 1.8.6 with SNPio 1.7.4 on the 58 diploid empirical
datasets, five missingness strategies, and six manuscript models. Every
dataset/strategy task runs all six models against the same seeded split and
simulated mask. Exact evaluation coordinates are audited after execution.

## Fixed execution profile

- Models: ImputeAutoencoder, ImputeVAE, ImputeNLPCA, ImputeUBP,
  ImputeMostFrequent, and ImputeRefAllele.
- Missingness strategies: random, random_weighted, random_weighted_inv,
  nonrandom, and nonrandom_weighted.
- Seed: 42.
- Simulated missingness: 30%.
- Nonrandom search limit: 100,000 clade-placement attempts, followed when
  necessary by recorded seeded tip-clade completion whose marginal tip weights
  follow the same uniform-node or branch-length-weighted tree distribution.
  Every prepared strategy masks exactly
  `round(0.30 * n_pre_simulation_called_cells)` genotypes while retaining at
  least one observed genotype per locus.
- Preset: fast.
- Tuning: 50 multi-objective trials.
- Device: CPU.
- Threads per task: one.
- HPC layout: eight worker jobs; each worker processes seven or eight complete
  datasets and all five strategy combinations sequentially. Dataset-preserving
  largest-processing-time assignment balances matrix cell counts across jobs.
- SLURM partition: shu-hpc-biocpu.
- Conda environment: `pgsui-val`.

Create the validation environment from the released packages before staging
the bundle on the cluster:

```bash
conda create -n pgsui-val \
  -c btmartin721 -c conda-forge -c bioconda \
  python=3.12 pg-sui=1.8.6 snpio=1.7.4
conda activate pgsui-val
python -c 'import pgsui, snpio; print(pgsui.__version__, snpio.__version__)'
```

The version check must print `1.8.6 1.7.4` before submission.

## Recover original VCF inputs

Set a Dryad API token to enable individual-file downloads, particularly for
four deposits whose public whole-dataset archives exceed 1 GiB (`results105`,
`results117`, `results204`, and `results256`):

```bash
printf 'Dryad API token: '
IFS= read -rs DRYAD_API_TOKEN
printf '\n'
export DRYAD_API_TOKEN
python scripts/prepare_pgsui_n60_inputs.py \
  --source-manifest /path/to/final_dataset_source_manifest.csv \
  --phylip-dir /path/to/validation_inputs_phylip/final_60 \
  --iqtree-dir /path/to/iqtree_results_GTR_merged.nosync \
  --output-dir /path/to/original_vcfs.nosync \
  --download
```

The input-preparation command recovers and verifies the full 60-dataset source
catalog: repository checksums, sample/locus counts, sample order, consistent
source GT ploidy (haploid, diploid, or tetraploid), exact SNPio nucleotide
content, and SNPio 0/1/2 equivalence at biallelic loci while allowing the
expected per-locus REF/ALT inversion. This preserves multiallelic and haploid
source data without forcing it through a diploid PHYLIP interpretation. It
never guesses among ambiguous VCF candidates. Without a token, the bounded
public archive fallback can recover most smaller deposits; do not paste a
Dryad token into a manifest, script, log, or shell history.

Raw checksum-matched downloads remain under `original_vcfs.nosync/vcfs`.
SNPio-ready derivatives are written to `original_vcfs.nosync/benchmark_vcfs`;
historical gap alleles are converted deterministically without dropping loci,
and only gap-containing calls are marked missing. Omit `--download` to
revalidate cached VCFs without attempting network downloads.

The runnable benchmark excludes `results131` (haploid) and `results576`
(tetraploid). Their verified source files remain in the 60-dataset source
catalog, but neither dataset, its masks, nor its population statistics enter
the N=58 manifests or reviewer package.

IQ-TREE selected uniform-rate models for `results125` and `results180` and
emitted header-only `.rate` files. The preparation step retains those original
files and derives explicit one-rate-per-locus tables containing 1.0, which are
the exact site-rate representations of the reported uniform models.
Header-only rate files from any non-uniform model are rejected.

## Generate canonical masks

```bash
python scripts/simulate_gtimputation_missingness.py \
  --input-dir /path/to/original_vcfs.nosync/benchmark_vcfs \
  --tree-dir /path/to/original_vcfs.nosync/iqtree \
  --output-dir /path/to/pgsui_n58_run.nosync \
  --datasets-file scripts/pgsui_n58_dataset_ids.txt \
  --mask-mode regenerate \
  --seed 42 \
  --sim-prop 0.30 \
  --sim-max-tries 100000 \
  --validation-split 0.30 \
  --ploidy auto \
  --siterates-suffix .rate \
  --skip-comparison-run-sheets
```

## Build and submit the CPU workers

```bash
python scripts/build_pgsui_n58_run.py \
  --simulation-root /path/to/pgsui_n58_run.nosync \
  --vcf-source-manifest \
    /path/to/original_vcfs.nosync/n60_vcf_source_manifest.tsv

cd /path/to/pgsui_n58_run.nosync
bash scripts/submit_pgsui_n58_cpu_workers.zsh
```

The builder stages each checksum-verified raw Dryad VCF under
`inputs/original_vcfs` and records the SNPio-normalized run input under
`canonical_vcfs`. The portable source manifest retains checksums for both
layers. It also stages the exact PG-SUI Python source tree under
`inputs/software/pgsui`; each task verifies that source digest and prepends the
snapshot to `PYTHONPATH`. Consequently, the versioned environment supplies the
declared PG-SUI 1.8.6 dependencies while the run uses precisely the audited
source that generated the canonical masks, even when the repository was dirty
at bundle-creation time. The Git revision, dirty-tree state, and source digest
are recorded in provenance. SNPio may sort an unindexed VCF into the exact
order used by the manuscript PHYLIP/IQ-TREE analysis. Mask generation records
that permutation and aborts unless the complete locus identities, alleles,
original missingness, and called genotypes agree after reordering; sample order
may never change.

The submission creates eight one-CPU worker jobs. Each worker owns seven or
eight whole datasets and processes their five simulation strategies
sequentially. The builder balances total sample-by-locus matrix cells across
workers while never splitting a dataset. `PGSUI_WORKER_CONCURRENCY` may lower
concurrency but cannot exceed eight. A combination is complete only after all
six reports pass exact mask-coordinate and class-support audits.

Treat a bundle built from a dirty or unreleased PG-SUI tree as a dry-run
candidate only. Rebuild the manifests, staged source snapshot, and provenance
from the clean release commit before submission; the reviewer packager rejects
dirty-tree provenance.

The run provenance lists every task that reaches the clade-attempt cap, its
completion count, completion fraction, and phylogenetic tip-weighting mode.
Review this list before submission; it must never be inferred from log text
alone.

## Generate SNPio population-genetic covariates

The run manifest includes one raw-source-VCF summary task for each of the 58
diploid datasets. Ho, He, nucleotide diversity, and Ragsdale-Gravel unbiased
unphased LD are obtained from SNPio 1.7.4. The LD point estimate uses at most
100,000 deterministically sampled SNP pairs, zero bootstraps, and one thread;
the validation SNPs are explicitly treated as unlinked. Custom calculations
retain explicit names:
genotype-category diversity is not called haplotype diversity, and Tajima's D
is calculated once as a multi-site statistic over complete-call loci rather
than per SNP.

```bash
cd /path/to/pgsui_n58_run.nosync
zsh scripts/submit_pgsui_n58_sumstats_workers.zsh \
  /path/to/pgsui_n58_run.nosync 8
```

The sumstats submission likewise uses eight sequential workers on
`shu-hpc-biocpu`, with one CPU/thread per job.

## Audit and regenerate post-hoc outputs

Run this only after all 290 combinations and 58 SNPio tasks have completed:

```bash
cd /path/to/pgsui_n58_run.nosync
zsh scripts/run_pgsui_n58_posthoc.zsh /path/to/pgsui_n58_run.nosync
```

The strict audit requires 290 success records, 1,740 model reports, and exact
coordinate equality between every model's evaluation artifact and its canonical
test mask. The post-hoc workflow writes distribution summaries, dataset-blocked
Friedman tests, paired Wilcoxon tests with Holm correction, model ranks,
dataset/strategy winners, population-genetic comparisons, and HC3-adjusted
feature-effect tables and plots. Model inference averages the five strategies
within each of the 58 independent dataset blocks; strategy inference is paired
within dataset.

## Build the reviewer package

The packager refuses incomplete audits and existing destinations. It copies
inputs, manifests, provenance, raw PG-SUI outputs, post-hoc outputs, logs,
status records, and scripts without hard links, then verifies SHA-256 checksums.

```bash
python scripts/build_pgsui_n58_reviewer_package.py \
  --bundle-root /path/to/pgsui_n58_run.nosync \
  --output /path/to/pgsui_n58_reviewer_package \
  --source-metadata-dir /path/to/pgsui_osf_reviewer_package/02_inputs_metadata \
  --tar-gz
```
