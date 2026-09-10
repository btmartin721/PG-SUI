# PG-SUI v1.8.5

PG-SUI v1.8.5 corrects genotype encoding and filesystem isolation in the
canonical PG-SUI/GTImputation benchmark workflow.

## Scientific correctness

- PG-SUI's neural and deterministic imputers now delegate 0/1/2 decoding to
  SNPio's peer-reviewed `GenotypeEncoder` instead of maintaining duplicated
  decoder implementations.
- Canonical mask generation and post-hoc GTImputation scoring now use SNPio's
  REF/HET/ALT 0/1/2 encoder, including its collapsed ALT-dosage behavior for
  multiallelic VCF loci.
- Family-specific report discovery now recognizes the neural
  `zygosity_report.json` filename while retaining the deterministic report
  filename.

## Reproducibility and HPC execution

- Every PG-SUI task stages its VCF and SNPio artifacts in a private working
  directory, preventing HDF5 and bgzip/tabix sidecar collisions.
- Four persistent CPU workers now process all five strategies for a dataset
  sequentially, so the same dataset is never opened by concurrent tasks.
- The strict report audit still requires all six models to match the canonical
  SNPio-derived test rows, evaluation coordinates, and class support before a
  task is marked successful.

## Distribution

- PyPI, Conda, GitHub Release, and documentation publication remain tag-driven.
- The Docker workflow is initiated by the release tag but is not required for
  the CPU benchmark environment.
