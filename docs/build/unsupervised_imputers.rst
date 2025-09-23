Unsupervised Imputers
======================

Included here is the documentation for each unsupervised imputers that are not specific to a given model. This includes arguments pertaining to the number of epochs, batch size, number of components to reduce the input to, learning rate, etc., as well as parameters for tuning, validation, and various other settings.

Unsupervised Imputer Models (Neural Networks)
---------------------------------------------

Non-linear Principal Component Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
