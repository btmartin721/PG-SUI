ImputeAutoencoder
=================

.. _impute_autoencoder:

Overview
--------

``ImputeAutoencoder`` implements a standard encoder-decoder autoencoder for
genotype imputation. The model maps genotype vectors into a low-dimensional
latent space and reconstructs per-locus genotype logits. It uses masked focal
cross-entropy to ignore missing entries and handle class imbalance.

Model formulation
-----------------

Let :math:`X \in \mathbb{R}^{N \times L}` be the genotype matrix encoded as
0/1/2 (missing = -1). The autoencoder learns an encoder :math:`f_{\phi}` and
decoder :math:`f_{\theta}`:

.. math::

   z_i = f_{\phi}(x_i)
   \qquad
   \hat{x}_i = f_{\theta}(z_i)

Training minimizes a masked focal cross-entropy loss over observed entries,
with optional class weights and L1 regularization:

.. math::

   \mathcal{L} =
   \frac{1}{|M|} \sum_{(i, j) \in M}
   w_{y_{ij}} (1 - p_{ij})^{\gamma} \log(p_{ij})
   + \lambda \lVert \theta \rVert_1

where :math:`M` indexes non-missing entries, :math:`p_{ij}` is the probability
assigned to the true genotype class, and :math:`\gamma` is the focal-loss
parameter.

Algorithm summary
-----------------

1. Encode genotypes to 0/1/2, simulate missingness once on the full matrix, and
   build masks for original and simulated missingness (reused across splits).
2. Train the encoder-decoder network on observed entries using masked focal
   loss with class weighting; optional gamma scheduling is supported.
3. Optimize with AdamW and a warmup-to-cosine learning rate schedule, while
   monitoring validation loss for early stopping; metrics are scored on
   simulated-missing entries only.
4. ``transform()`` predicts genotype logits, fills only originally missing
   entries, and decodes to IUPAC outputs.

Configuration highlights
------------------------

``ImputeAutoencoder`` uses :class:`pgsui.data_processing.containers.AutoencoderConfig`
with the standard ``io``, ``model``, ``train``, ``tune``, ``plot``, and ``sim``
sections.

- ``model.latent_dim`` and ``model.layer_schedule`` control architecture.
- ``train.gamma`` and ``train.weights_*`` control focal loss and class weights.
- ``train.gamma_schedule`` optionally anneals focal-loss gamma during training.
- ``train.early_stop_gen`` / ``train.min_epochs`` gate early stopping.

See :doc:`optuna_tuning` for Optuna-driven tuning details.

Usage
-----

.. code-block:: python

   from snpio import VCFReader
   from pgsui import ImputeAutoencoder
   from pgsui.data_processing.containers import AutoencoderConfig

   gdata = VCFReader("cohort.vcf.gz", popmapfile="pops.popmap")
   cfg = AutoencoderConfig.from_preset("balanced")
   cfg.model.latent_dim = 12

   model = ImputeAutoencoder(genotype_data=gdata, config=cfg)
   model.fit()
   genotypes_iupac = model.transform()

References
----------

Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. *Science*, 313(5786), 504-507.
