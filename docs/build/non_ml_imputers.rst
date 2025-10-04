Deterministic (non-Machine Learning) Imputers
=============================================

Overview
--------

The deterministic imputers provide fast, interpretable baselines that mirror the
**fit/transform** contract used across PG-SUI:

- You **instantiate** with a `GenotypeData` and a **dataclass config** (or YAML path).
- Call :py:meth:`fit()` with **no arguments** to set up evaluation (TRAIN/TEST split, masking).
- Call :py:meth:`transform()` with **no arguments** to impute and write plots/metrics.

Both imputers operate on SNPio’s 0/1/2 working encoding (with ``-1`` or ``-9`` as missing),
and produce the same evaluation artifacts as the deep models (zygosity reports,
IUPAC-10 reports, confusion matrices, and distribution plots).

What’s included
---------------

- **ImputeMostFrequent** — per-locus mode imputation. Supports global modes and
  population-aware modes when a popmap is available.
- **ImputeRefAllele** — replaces all missing values with REF genotype (0).

Shared behavior & outputs
-------------------------

- **Evaluation protocol:** single TRAIN/TEST split by samples; on TEST rows, *all originally
  observed* cells are masked for unbiased reconstruction scoring; metrics are computed on
  those masked cells.
- **Metrics & figures:** macro F1/PR/accuracy at zygosity level (REF/HET/ALT, with
  haploid folding), and IUPAC-10 classification plus confusion matrices; genotype
  distribution plots pre/post imputation.
- **I/O layout:** artifacts are written under ``{prefix}_output/Deterministic/{plots,metrics,models,optimize}/{Model}/``.

Quick start (Python)
--------------------

.. code-block:: python

   from snpio import VCFReader
   from pgsui.data_processing.containers import MostFrequentConfig, RefAlleleConfig
   from pgsui.impute.deterministic.imputers.mode import ImputeMostFrequent
   from pgsui.impute.deterministic.imputers.ref_allele import ImputeRefAllele

   gd = VCFReader(
       filename="data.vcf.gz",
       popmapfile="pops.popmap",   # optional but recommended
       prefix="demo"
   )

   # Most-frequent (global)
   mf_cfg = MostFrequentConfig.from_preset("fast")
   mf_cfg.io.prefix = "mf_demo"
   mf = ImputeMostFrequent(genotype_data=gd, config=mf_cfg)
   mf.fit()
   X_mf = mf.transform()   # IUPAC array (n_samples, n_loci)

   # Most-frequent (population-aware)
   mf_pop = MostFrequentConfig.from_preset("balanced")
   mf_pop.io.prefix = "mf_perpop"
   mf_pop.algo.by_populations = True
   mf2 = ImputeMostFrequent(genotype_data=gd, config=mf_pop)
   mf2.fit()
   X_mf_perpop = mf2.transform()

   # Reference-allele filler
   ra_cfg = RefAlleleConfig.from_preset("fast")
   ra_cfg.io.prefix = "ref_demo"
   ra = ImputeRefAllele(genotype_data=gd, config=ra_cfg)
   ra.fit()
   X_ref = ra.transform()

YAML configuration
------------------

Both imputers accept a YAML file (merged with an optional ``preset``) and support
dot-path overrides in the CLI (and in Python via helpers). Minimal examples:

**MostFrequent (``mostfrequent.yaml``)**

.. code-block:: yaml

   preset: balanced
   io:
     prefix: "mf_yaml"
   split:
     test_size: 0.2
   algo:
     by_populations: true
     missing: -1
     default: 0
   plot:
     fmt: "pdf"
     dpi: 300
     show: false

**RefAllele (``refallele.yaml``)**

.. code-block:: yaml

   preset: fast
   io:
     prefix: "ref_yaml"
   split:
     test_size: 0.25
   algo:
     missing: -1
   plot:
     fmt: "png"
     dpi: 150
     show: false

CLI usage (concept)
-------------------

These deterministic models follow the same CLI precedence model as the rest of PG-SUI
(code defaults < preset < YAML < explicit flags < ``--set k=v``). Example:

.. code-block:: bash

   # Most-frequent, population-aware, YAML + a final override
   pg-sui \
     --vcf data.vcf.gz \
     --popmap pops.popmap \
     --models ImputeMostFrequent \
     --preset balanced \
     --config mostfrequent.yaml \
     --set io.prefix=mf_cli \
     --set algo.by_populations=true

   # REF-allele baseline
   pg-sui \
     --vcf data.vcf.gz \
     --models ImputeRefAllele \
     --preset fast \
     --set io.prefix=ref_cli

Configuration dataclasses
-------------------------

.. autoclass:: pgsui.data_processing.containers.MostFrequentConfig
   :members:
   :show-inheritance:
   :noindex:

.. autoclass:: pgsui.data_processing.containers.RefAlleleConfig
   :members:
   :show-inheritance:
   :noindex:

API reference
-------------

ImputeMostFrequent
^^^^^^^^^^^^^^^^^^

.. automodule:: pgsui.impute.deterministic.imputers.mode
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

ImputeRefAllele
^^^^^^^^^^^^^^^^

.. automodule:: pgsui.impute.deterministic.imputers.ref_allele
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
