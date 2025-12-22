Simulated Missingness and Evaluation
====================================

.. _simulated_missingness:

.. _simulate_missingness_eval:

Overview
--------

PG-SUI evaluates imputers by masking a subset of observed genotypes and
checking whether each model can recover those values. This is called
"simulated missingness" and it is essential because true missing genotypes
have no known ground truth. The same strategy can be applied across models
so their evaluation masks are aligned.

When simulated missingness is used
----------------------------------

- Unsupervised models: simulated masking is required for training and
  evaluation. The masked entries define the evaluation set; original missing
  values are never scored.
- Deterministic and supervised models: simulated masking is optional. If it
  is disabled, evaluation is performed on all originally observed entries
  within the test split.

Simulation process
------------------

1. Start from the encoded genotype matrix (0/1/2 with negative values for
   missing).
2. Build an "original missing" mask for any pre-existing missing entries.
3. Use :class:`pgsui.data_processing.transformers.SimMissingTransformer` to
   select a subset of observed cells to mask based on ``sim_strategy`` and
   ``sim_prop``.
4. Produce a masked matrix for model training/inference plus three boolean
   masks:

   - ``original_missing_mask_``: missing in the input data.
   - ``sim_missing_mask_``: simulated missing on observed cells.
   - ``all_missing_mask_``: union of original and simulated missing.

The simulated mask is always applied to observed cells only, so model
performance is measured against known truth.

Simulation strategies
---------------------

- ``random``: uniform masking across eligible cells until the target
  proportion is reached.
- ``random_weighted``: probability proportional to genotype frequency in each
  column (common genotypes are masked more often).
- ``random_weighted_inv``: probability inversely proportional to genotype
  frequency in each column (rare genotypes are masked more often).
- ``nonrandom``: phylogenetically clustered masking based on a genotype tree.
- ``nonrandom_weighted``: like ``nonrandom`` but clades are sampled in
  proportion to branch length.

``nonrandom`` and ``nonrandom_weighted`` require a tree parser; provide
``--treefile``, ``--qmatrix``, and ``--siterates`` on the CLI.

Simulation Strategy Summary Table
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Summary of missing data simulation strategies
   :header-rows: 1
   :widths: auto

   * - Strategy
     - Selection logic
     - Biologically mimics
     - Expected difficulty
   * - Random
     - Uniform coin flip per cell
     - Random sequencing errors; read-depth fluctuations
     - Easy
   * - Random weighted
     - Probability proportional to genotype frequency (masks common)
     - Reference bias; over-representation of common alleles
     - Moderate
   * - Random weighted inv
     - Probability inversely proportional to genotype frequency (masks rare)
     - Allelic dropout; minor-allele loss; ascertainment bias
     - Hard
   * - Nonrandom
     - Phylogenetically clustered masking
     - Clade-specific dropout; sample batch effects
     - Hard
   * - Nonrandom weighted
     - Clustered masking weighted by branch length
     - Divergence-linked dropout; lineage-specific failures
     - Very hard

Evaluation workflow
-------------------

PG-SUI evaluates models only on cells with known truth:

- Unsupervised models: the simulated mask defines the evaluation set. Metrics
  are computed by comparing predictions to the pre-mask ground truth values.
- Deterministic and supervised models: a train/test split is created. If
  simulated masking is enabled, the simulated mask is restricted to the test
  rows; if disabled, all observed test cells are scored.

Outputs include zygosity (REF/HET/ALT) and 10-class IUPAC reports, confusion
matrices, PR curves, and summary metrics. These files are written under
``{prefix}_output/<Family>/metrics/<Model>/`` with associated plots under the
corresponding ``plots`` directory.

Configuration and CLI controls
------------------------------

YAML configuration (applies per model config):

.. code-block:: yaml

   sim:
     simulate_missing: true
     sim_strategy: random_weighted_inv
     sim_prop: 0.30
     sim_kwargs: {}

CLI overrides (apply to all selected models):

.. code-block:: bash

   pg-sui \
     --input data.vcf.gz \
     --sim-strategy random_weighted_inv \
     --sim-prop 0.30

Use ``--disable-simulate-missing`` to turn off simulated masking for
supervised/deterministic runs. Unsupervised models require simulated
masking and will error if it is disabled.

Notes and best practices
------------------------

- Keep ``sim_prop`` small enough to avoid removing too much signal; 0.1 to
  0.3 is typical for evaluation.
- Use the same strategy and proportion across models when comparing
  performance.
- For ``nonrandom`` strategies, ensure the tree inputs correspond to the
  dataset you are imputing; mismatched trees can bias the mask placement.
