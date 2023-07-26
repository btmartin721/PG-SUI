Unsupervised Imputers
======================

Shared Arguments
-----------------

Included here is the documentation shared among all unsupervised imputers that are not specific to a given model. This includes arguments pertaining to the number of epochs, batch size, number of components to reduce the input to, learning rate, etc., as well as parameters for grid searches or validation and various other settings.

.. autoclass:: pgsui.impute.estimators.UnsupervisedImputer
   :members:
   :show-inheritance:
   :noindex:

Unsupervised Imputer Models (Neural Networks)
---------------------------------------------

Non-linear Principal Component Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pgsui.impute.estimators.ImputeNLPCA
   :members:
   :show-inheritance:
   :noindex:

Unsupervised Backpropagation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pgsui.impute.estimators.ImputeUBP
   :members:
   :show-inheritance:
   :inherited-members:
   :noindex:

Standard AutoEncoder
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pgsui.impute.estimators.ImputeStandardAutoEncoder
   :members:
   :show-inheritance:
   :inherited-members:
   :noindex:

Variational AutoEncoder
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pgsui.impute.estimators.ImputeVAE
   :members:
   :show-inheritance:
   :inherited-members:
   :noindex:
