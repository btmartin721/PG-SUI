ImputeUBP
=========

.. _impute_ubp:

Overview
--------

``ImputeUBP`` implements Unsupervised Backpropagation (UBP) for genotype
imputation. UBP learns a latent embedding for each sample and a decoder network
that maps those embeddings to genotype logits. Unlike autoencoders, there is no
encoder; the latent vectors are optimized directly with the decoder.

PG-SUI follows a modified UBP schedule inspired by Gashler et al. (2014): latent
vectors are initialized with PCA, the decoder is refined with the embeddings
frozen, and then both embeddings and decoder weights are jointly optimized.

Missingness is simulated once on the full matrix and reused across train/val/test
splits; evaluation metrics are computed on simulated-missing positions only.

Model formulation
-----------------

Let :math:`X \in \mathbb{R}^{N \times L}` be the genotype matrix encoded as
0/1/2 (missing = -1). Each sample has a latent embedding
:math:`v_i \in \mathbb{R}^{K}` and a shared decoder :math:`f_W`:

.. math::

   \hat{X}_i = f_W(v_i)

Training minimizes a masked focal cross-entropy loss with optional class
weights and L1 regularization on decoder weights:

.. math::

   \mathcal{L} =
   \frac{1}{|M|} \sum_{(i, j) \in M}
   w_{y_{ij}} (1 - p_{ij})^{\gamma} \log(p_{ij})
   + \lambda \lVert W \rVert_1

where :math:`M` indexes observed entries, :math:`p_{ij}` is the probability
assigned to the true genotype class, and :math:`\gamma` is the focal-loss
gamma.

Training phases
---------------

1. **Initialization (PCA warm start):** latent embeddings are initialized by
   PCA on observed training genotypes.
2. **Phase 2 (decoder refinement):** freeze the embeddings and optimize decoder
   weights to reconstruct genotypes.
3. **Phase 3 (joint refinement):** jointly optimize embeddings and decoder
   weights, allowing embeddings to move off the linear PCA manifold.

Validation and inference use projection steps that refine embeddings with the
decoder frozen, improving reconstruction on observed entries before scoring and
final imputation.

Configuration highlights
------------------------

``ImputeUBP`` uses :class:`pgsui.data_processing.containers.UBPConfig` with the
standard ``io``, ``model``, ``train``, ``tune``, ``plot``, and ``sim`` blocks.
The ``ubp`` section adds projection controls:

- ``ubp.projection_lr``: learning rate for latent projection.
- ``ubp.projection_epochs``: number of projection steps per evaluation.
- ``train.gamma_schedule``: optionally anneal focal-loss gamma during training.

See :doc:`optuna_tuning` for Optuna-driven tuning details.

Usage
-----

.. code-block:: python

   from snpio import VCFReader
   from pgsui import ImputeUBP
   from pgsui.data_processing.containers import UBPConfig

   gdata = VCFReader("cohort.vcf.gz", popmapfile="pops.popmap")
   cfg = UBPConfig.from_preset("balanced")
   cfg.ubp.projection_lr = 0.03

   model = ImputeUBP(genotype_data=gdata, config=cfg)
   model.fit()
   genotypes_iupac = model.transform()

References
----------

Gashler, M. S., Smith, M. R., Morris, R., & Martinez, T. R. (2014). Missing Value Imputation with Unsupervised Backpropagation. Computational Intelligence, 32(2), 196-215.
