ImputeVAE
=========

.. _impute_vae:

Overview
--------

``ImputeVAE`` implements a variational autoencoder (VAE) for genotype
imputation. The encoder predicts a latent distribution (mean and log-variance),
samples latent vectors via the reparameterization trick, and the decoder
reconstructs genotype logits. Training combines masked focal reconstruction
loss with a KL divergence penalty.

Model formulation
-----------------

Let :math:`X \in \mathbb{R}^{N \times L}` be the genotype matrix encoded as
0/1/2 (missing = -1). The encoder outputs a Gaussian distribution in latent
space:

.. math::

   \mu = f_{\mu}(x), \qquad \log \sigma^2 = f_{\sigma}(x)

Sampling uses the reparameterization trick:

.. math::

   z = \mu + \epsilon \cdot \sigma, \qquad \epsilon \sim \mathcal{N}(0, I)

The decoder predicts logits :math:`\hat{x}` and the total loss is:

.. math::

   \mathcal{L} =
   \mathcal{L}_{\text{recon}} + \beta \, D_{\text{KL}}(q(z \mid x)\,\|\,p(z))

where :math:`\mathcal{L}_{\text{recon}}` is masked focal cross-entropy over
observed entries, and :math:`\beta` is the KL weight.

Algorithm summary
-----------------

1. Encode genotypes to 0/1/2 and build masks for original and simulated
   missingness.
2. Train the encoder-decoder with masked focal reconstruction loss plus KL
   divergence (weighted by ``vae.kl_beta``).
3. Validate using reconstruction metrics on simulated-missing entries.
4. ``transform()`` predicts genotype probabilities and fills missing entries
   with MAP labels before decoding to IUPAC outputs.

Configuration highlights
------------------------

``ImputeVAE`` uses :class:`pgsui.data_processing.containers.VAEConfig`, which
extends the autoencoder config with a ``vae`` section:

- ``vae.kl_beta`` controls the KL divergence weight.
- ``vae.kl_beta_schedule`` enables optional KL annealing.
- ``train.gamma`` and ``train.weights_*`` control reconstruction loss behavior.

See :doc:`optuna_tuning` for Optuna-driven tuning details.

Usage
-----

.. code-block:: python

   from snpio import VCFReader
   from pgsui import ImputeVAE
   from pgsui.data_processing.containers import VAEConfig

   gdata = VCFReader("cohort.vcf.gz", popmapfile="pops.popmap")
   cfg = VAEConfig.from_preset("balanced")
   cfg.vae.kl_beta = 1.25

   model = ImputeVAE(genotype_data=gdata, config=cfg)
   model.fit()
   genotypes_iupac = model.transform()

References
----------

Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. *arXiv preprint* arXiv:1312.6114.
