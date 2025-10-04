Supervised Imputers
===================

Overview
--------

PG-SUI's supervised imputers frame genotype imputation as **multiclass prediction** (0 = REF, 1 = HET, 2 = ALT for diploids; haploids collapse to two classes). They use typed ``*Config`` dataclasses with presets (``fast``, ``balanced``, ``thorough``), optional YAML, and a consistent **instantiate → fit() → transform()** workflow:

.. code-block:: python

   from snpio import VCFReader
   from pgsui import ImputeRandomForest, RFConfig 

   gd = VCFReader("data.vcf.gz", popmapfile="pops.popmap", prefix="demo")

   cfg = RFConfig.from_preset("balanced")
   cfg.io.prefix = "rf_demo"

   model = ImputeRandomForest(genotype_data=gd, config=cfg)
   model.fit()
   X012_imputed = model.transform()

Shared Arguments
----------------

The following options are common across supervised imputers (see the specific
``*Config`` classes below for full details and defaults).

- **I/O (``io``)**:
  - ``prefix``: run name used for output directories and artifacts.
  - ``plot_format``, ``verbose``, ``debug``, ``seed``: logging/replicability.

- **Training (``training``)**:
  - ``validation_split``: fraction of samples for validation.
  - ``device``: kept for API consistency; tree models run on CPU.
  - ``seed``: controls sklearn and any internal shuffles.

- **Tuning (``tuning``)**:
  - Tuning is not available for supervised imputers, but the dataclass fields are retained for API consistency.

- **Plotting (``plot``)**:
  - ``show``, ``dpi``: control display and export quality.

- **Evaluation**:
  - Metrics reported per-zygosity (REF/HET/ALT) and macro-averaged; confusion matrices, PR curves, zygosity bars, and a cross-model radar summary are written to ``{prefix}_output/Supervised/plots/{Model}/`` with matching CSV/JSON metrics.

Notes
^^^^^

- Inputs are taken from SNPio's ``GenotypeData`` attached at **instantiation**.
- ``fit()`` trains and writes reports; ``transform()`` returns an imputed 0/1/2 array.
- Class imbalance is handled via macro metrics and (optionally) class-weight shaping.

Config Dataclasses (API)
------------------------

Random Forest (Config)
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pgsui.data_processing.containers.RFConfig
   :members:
   :show-inheritance:
   :inherited-members:
   :noindex:

HistGradientBoosting (Config)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pgsui.data_processing.containers.HGBConfig
   :members:
   :show-inheritance:
   :inherited-members:
   :noindex:

Supervised Imputer Models
-------------------------

Random Forest
^^^^^^^^^^^^^

.. autoclass:: pgsui.impute.supervised.imputers.random_forest.ImputeRandomForest
   :members:
   :show-inheritance:
   :inherited-members:
   :noindex:

HistGradientBoosting
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pgsui.impute.supervised.imputers.hist_gradient_boosting.ImputeHistGradientBoosting
   :members:
   :show-inheritance:
   :inherited-members:
   :noindex:

CLI Examples
------------

Run both supervised models with a preset and Optuna tuning:

.. code-block:: bash

   pg-sui \
     --vcf data.vcf.gz \
     --popmap pops.popmap \
     --models ImputeRandomForest ImputeHistGradientBoosting \
     --preset balanced \
     --set io.prefix=supervised_demo \

YAML + overrides:

.. code-block:: bash

   pg-sui \
     --vcf data.vcf.gz \
     --popmap pops.popmap \
     --models ImputeHistGradientBoosting \
     --preset thorough \
     --config hgb.yaml \
     --set io.prefix=hgb_thorough

Outputs
-------

- Plots: ``{prefix}_output/Supervised/plots/{Model}/``
- Metrics (CSV/JSON): ``{prefix}_output/Supervised/metrics/{Model}/``
- Cross-model radar summary compares macro-F1, macro-PR, accuracy, and HET-F1.
