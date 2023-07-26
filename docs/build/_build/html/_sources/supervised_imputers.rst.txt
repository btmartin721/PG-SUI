Supervised Imputers
====================

Shared Arguments
-----------------

Included here is the documentation shared among all supervised imputers that are not specific to a given model. This includes arguments pertaining to the number of nearest neighbors to use for the IterativeImputer, as well as parameters for grid searches or validation and various other settings such as the number of CPUs to use.

.. autoclass:: pgsui.impute.estimators.SupervisedImputer
   :members:
   :show-inheritance:
   :noindex:


Supervised Imputer Models
--------------------------

K Nearest Neighbors
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pgsui.impute.estimators.ImputeKNN
   :members:
   :show-inheritance:
   :inherited-members:
   :noindex:

Random Forest (and Extra Trees)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pgsui.impute.estimators.ImputeRandomForest
   :members:
   :show-inheritance:
   :inherited-members:
   :noindex:

XGBoost
^^^^^^^^

.. autoclass:: pgsui.impute.estimators.ImputeXGBoost
   :members:
   :show-inheritance:
   :inherited-members:
   :noindex: