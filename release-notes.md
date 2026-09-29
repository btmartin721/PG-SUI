# PG-SUI v1.8.7

PG-SUI v1.8.7 corrects the masked-data contract for UBP and NLPCA genotype
imputation.

## Scientific correctness

- UBP and NLPCA now initialize sample embeddings from the corrupted genotype
  matrix. PCA is fitted on training rows, with missing cells filled using
  training-locus means, so simulated-missing truth is unavailable to the warm
  start in all splits.
- NLPCA now optimizes its decoder and sample embeddings against observed
  genotypes only. Originally missing and simulated-missing cells remain masked
  throughout training; predicted values are no longer fed back as targets.
- Validation and test metrics continue to score only simulated-missing calls
  with known truth. Final imputation fills naturally missing calls by projecting
  samples against their observed genotypes.

## Verification

- Added focused regression coverage for masked PCA initialization in UBP and
  NLPCA, observed-only NLPCA training targets, and end-to-end imputation.
- Updated the NLPCA animation and method descriptions, including the displayed
  sign of focal cross-entropy.
