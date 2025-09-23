Supervised Imputers
====================

Shared Arguments
-----------------

Included here is the documentation shared among all supervised imputers that are not specific to a given model. This includes arguments pertaining to the number of nearest neighbors to use for the IterativeImputer, as well as parameters for grid searches or validation and various other settings such as the number of CPUs to use.

Supervised Imputer Models
--------------------------

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