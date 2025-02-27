==========
Changelog
==========

Overview of the changes made to PG-SUI for each release.

Version 1.5.1 - 2025-02-07
--------------------------

Bug Fixes
^^^^^^^^^

- Fixed bug where `ImputeAutoencoder` would not run due to missing `self` argument.

Features
^^^^^^^^

- Added new simulation strategies for simulating missing data in genotype calls.

Changes
^^^^^^^

- Updated the `SimGenotypeDataTransformer` class to support additional simulation strategies.
- Improved documentation and tutorials.

Version 1.5 - 2025-01-28
------------------------

Features
^^^^^^^^

- Added Optuna parameter optimization for all deep learning models.
- More efficient code for all deep learning models.
- Added new deep learning models for imputation.
- Modularity for developing and adjusting deep learning models.

Changed
^^^^^^^

- Refactored the entire codebase to use the `GenotypeData` class for all data manipulation and imputation.
- Refactored to use PyTorch instead of Tensorflow for all deep learning models.
- GridSearchCV and GASearchCV have been replaced with Optuna for all deep learning models.

Version 1.0.2.1 - 2023-09-11
----------------------------

Bug Fixes
^^^^^^^^^

- Fixed bug where supervised imputers would fail due to duplicated `self` argument.

- Fixed bug where `ImputeNLPCA` would run `ImputeUBP` instead.

- Fixed gt_probability heatmap plot. It works correctly now.

- Fixed issues where plot directories were not being created.

- Fixed bugs where non-ML imputers would not decode the integer genotypes.

- Renamed gt probability plot to `simulated_genotypes`.

- Fixed default `prefix` argument for supervised imputers. It now aligns with the unsupervised imputers as `imputer`.

- Fixed bugs where `ImputeKNN` and `ImputeRandomForest` would not run.

- Set max pandas version to prevent future warnings.

- Added `warnings.simplefilter` to each module to catch `FutureWarning`

Changed
^^^^^^^

- Implemented new plotting for test.py

Version 1.0.2 - 2023-08-28
--------------------------

Bug Fixes
^^^^^^^^^

- Use GenotypeData copy method internally due to Cython pysam VariantHeader.

Version 1.0 - 2023-07-29
------------------------

Changed
^^^^^^^

- No longer in beta stages. Full release v1.0.

Version 0.3.0 - 2023-07-26
--------------------------

Features
^^^^^^^^
- Unsupervised models now impute on nucleotide-encoded multilabel data instead of 012 data.

- Much higher metric scores due to the new multilabel encodings (less class imbalance)

- Unsupervised grid searches are faster due to less redundant code and removig unnecessary calculations with the scoring functions.

Changed
^^^^^^^

- Documentation updates to make it easier to understand all the available arguments and what they are for.

- Refactored ``estimators.py`` and ``scorers.py`` for better modularity and easier code maintenance.

Removed
^^^^^^^

- 012-encoded inputs for unsupervised imputers. Now uses multilabel nucleotide (4 class) inputs.

Version 0.2.4 - 2023-07-24
--------------------------

Features
^^^^^^^^

- Initial release with four unsupervised neural network models, three supervised IterativeImputer models, and four non-machine-learning imputers.

Version 1.5.1 - 2025-02-07
--------------------------

Bug Fixes
^^^^^^^^^

- Fixed bug where `ImputeAutoencoder` would not run due to missing `self` argument.

Features
^^^^^^^^

- Added new simulation strategies for simulating missing data in genotype calls.

Changes
^^^^^^^

- Updated the `SimGenotypeDataTransformer` class to support additional simulation strategies.
- Improved documentation and tutorials.

Version 1.5 - 2025-01-28
------------------------

Features
^^^^^^^^

- Added Optuna parameter optimization for all deep learning models.
- More efficient code for all deep learning models.
- Added new deep learning models for imputation.
- Modularity for developing and adjusting deep learning models.

Changed
^^^^^^^

- Refactored the entire codebase to use the `GenotypeData` class for all data manipulation and imputation.
- Refactored to use PyTorch instead of Tensorflow for all deep learning models.
- GridSearchCV and GASearchCV have been replaced with Optuna for all deep learning models.

Version 1.0.2.1 - 2023-09-11
----------------------------

Bug Fixes
^^^^^^^^^

- Fixed bug where supervised imputers would fail due to duplicated `self` argument.

- Fixed bug where `ImputeNLPCA` would run `ImputeUBP` instead.

- Fixed gt_probability heatmap plot. It works correctly now.

- Fixed issues where plot directories were not being created.

- Fixed bugs where non-ML imputers would not decode the integer genotypes.

- Renamed gt probability plot to `simulated_genotypes`.

- Fixed default `prefix` argument for supervised imputers. It now aligns with the unsupervised imputers as `imputer`.

- Fixed bugs where `ImputeKNN` and `ImputeRandomForest` would not run.

- Set max pandas version to prevent future warnings.

- Added `warnings.simplefilter` to each module to catch `FutureWarning`

Changed
^^^^^^^

- Implemented new plotting for test.py

Version 1.0.2 - 2023-08-28
--------------------------

Bug Fixes
^^^^^^^^^

- Use GenotypeData copy method internally due to Cython pysam VariantHeader.

Version 1.0 - 2023-07-29
------------------------

Changed
^^^^^^^

- No longer in beta stages. Full release v1.0.

Version 0.3.0 - 2023-07-26
--------------------------

Features
^^^^^^^^

- Unsupervised models now impute on nucleotide-encoded multilabel data instead of 012 data.

- Much higher metric scores due to the new multilabel encodings (less class imbalance)

- Unsupervised grid searches are faster due to less redundant code and removig unnecessary calculations with the scoring functions.

Changed
^^^^^^^

- Documentation updates to make it easier to understand all the available arguments and what they are for.

- Refactored ``estimators.py`` and ``scorers.py`` for better modularity and easier code maintenance.

Removed
^^^^^^^

- 012-encoded inputs for unsupervised imputers. Now uses multilabel nucleotide (4 class) inputs.

Version 0.2.4 - 2023-07-24
--------------------------

Features
^^^^^^^^

- Initial release with four unsupervised neural network models, three supervised IterativeImputer models, and four non-machine-learning imputers.

Version 1.5.2 - 2025-03-01
--------------------------

Features
^^^^^^^^

- Added `ImputeAutoencoder` class for imputing missing values in genotype data using an Autoencoder model.
- Added `ImputeNLPCA` class for imputing missing values in genotype data using Non-linear Principal Component Analysis (NLPCA).
- Added `ImputeUBP` class for imputing missing values in genotype data using Unsupervised Backpropagation (UBP).
- Added `ImputeVAE` class for imputing missing values in genotype data using a Variational Autoencoder (VAE).

Changes
^^^^^^^

- Updated `BaseNNImputer` class to support new imputation models.
- Improved logging and error handling across all imputation models.
- Enhanced documentation with new tutorials and examples for implementing new imputation models.