# PG-SUI v1.8.6

PG-SUI v1.8.6 provides the scientifically corrected and reproducible N=58
diploid validation workflow used for the PG-SUI manuscript.

## Scientific correctness

- All five simulation strategies now generate an exact, seeded global mask
  target while retaining at least one observed genotype per locus and never
  overlapping naturally missing calls.
- Phylogenetic simulations batch clade placements and record any exact-target
  completion required after the configured attempt cap. Completion preserves
  the marginal uniform-node or branch-length-weighted tree distribution.
- The validation catalog is reconstructed from checksum-verified source VCFs
  and checked against the manuscript PHYLIP and IQ-TREE inputs using SNPio's
  peer-reviewed genotype encoder.
- The haploid and tetraploid datasets are excluded, leaving 58 consistently
  diploid datasets. RF and HGB are outside this manuscript validation.
- SNPioSP now obtains 0/1/2 encoding, Ho, He, nucleotide diversity, and
  Ragsdale-Gravel unbiased unphased LD from SNPio. Custom calculations use
  explicit terminology and a corrected multi-site Tajima's D implementation.

## Reproducibility and HPC execution

- The fixed benchmark contains 290 dataset-by-strategy tasks and six models.
  Every model is audited against the same seeded test rows and exact simulated
  evaluation coordinates.
- Eight CPU-only workers process all five strategies for each assigned dataset
  sequentially with one thread per job. Dataset-preserving matrix-cell load
  balancing minimizes the slowest worker's estimated workload.
- The fixed profile uses 50 multi-objective trials, the `fast` preset, the
  `shu-hpc-biocpu` partition, and the `pgsui-val` Conda environment.
- Run manifests record input hashes, source and analysis ploidy, software
  versions, Git revision, source-tree digest, simulation details, worker
  assignments, and phylogenetic completion provenance.
- Strict audits, regenerated post-hoc statistics and plots, and a
  checksum-complete reviewer-package builder are included.

## Distribution

- PG-SUI now requires SNPio 1.7.4 or newer.
- PyPI, Conda, GitHub Release, and Docker publication remain tag-driven.
