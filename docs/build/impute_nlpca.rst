ImputeNLPCA
===========

.. _impute_nlpca:

Overview
--------

``ImputeNLPCA`` implements non-linear PCA (NLPCA) for genotype imputation. The
model learns a low-dimensional latent vector for each sample and a decoder
network that maps those latent vectors to per-locus genotype logits. Unlike
autoencoders, there is no explicit encoder; the latent vectors are optimized
directly by backpropagation.

PG-SUI uses NLPCA as a decoder-only model with latent refinement during
evaluation and inference. It is equivalent to the joint-optimization phase of
UBP, with an additional input-refinement step that updates originally missing
entries while keeping simulated-missing cells masked to avoid leakage.

Model formulation
-----------------

Let :math:`X \in \mathbb{R}^{N \times L}` be the genotype matrix encoded as
0/1/2 (missing = -1). Each sample :math:`i` has a latent embedding
:math:`v_i \in \mathbb{R}^{K}` and a shared decoder :math:`f_W` that predicts
class logits for every locus:

.. math::

   \hat{X}_i = f_W(v_i)

Training minimizes a masked focal cross-entropy loss over observed entries
with optional class weights and L1 regularization on decoder weights:

.. math::

   \mathcal{L} =
   \frac{1}{|M|} \sum_{(i, j) \in M}
   w_{y_{ij}} (1 - p_{ij})^{\gamma} \log(p_{ij})
   + \lambda \lVert W \rVert_1

where :math:`M` indexes non-missing entries, :math:`p_{ij}` is the probability
assigned to the true genotype class, and :math:`\gamma` is the focal-loss
gamma.

Algorithm summary
-----------------

1. Encode genotypes as 0/1/2, simulate missingness once on the full matrix, and
   build masks for original and simulated missingness (reused across splits).
2. Initialize latent embeddings with PCA on observed training data.
3. Initialize the working matrix by filling originally missing entries with
   per-locus mode values; simulated-missing positions stay masked.
4. Jointly optimize latent embeddings :math:`V` and decoder weights :math:`W`
   using focal cross-entropy on observed entries.
5. Input refinement (EM-like): after selected epochs, replace **only**
   originally missing entries in the working matrix with current reconstructions
   while keeping simulated-missing positions masked.
6. During validation/inference, refine embeddings by projection (optimize
   :math:`V` with :math:`W` frozen) to improve reconstruction quality.

Configuration highlights
------------------------

``ImputeNLPCA`` uses :class:`pgsui.data_processing.containers.NLPCAConfig` with
the standard ``io``, ``model``, ``train``, ``tune``, ``plot``, and ``sim``
sections. The ``nlpca`` block adds projection controls:

- ``nlpca.projection_lr``: learning rate for latent projection.
- ``nlpca.projection_epochs``: number of projection steps per evaluation.

See :doc:`optuna_tuning` for how Optuna-driven tuning is applied to NLPCA.

Usage
-----

.. code-block:: python

   from snpio import VCFReader
   from pgsui import ImputeNLPCA
   from pgsui.data_processing.containers import NLPCAConfig

   gdata = VCFReader("cohort.vcf.gz", popmapfile="pops.popmap")
   cfg = NLPCAConfig.from_preset("balanced")
   cfg.nlpca.projection_epochs = 150

   model = ImputeNLPCA(genotype_data=gdata, config=cfg)
   model.fit()
   genotypes_iupac = model.transform()
