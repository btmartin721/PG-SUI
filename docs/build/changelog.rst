================
Changelog
================

This document provides a high-level view of the changes made to PG-SUI for each release.


Unreleased
----------
- Make HTML report with all plots and logs.

Version 1.0.2.1 - 2023-09-11
-----------------------------

Bug Fixes
~~~~~~~~~~

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
~~~~~~~~~

- Implemented new plotting for test.py

Version 1.0.2 - 2023-08-28
---------------------------

Bug Fixes
~~~~~~~~~~

- Use GenotypeData copy method internally due to Cython pysam VariantHeader.

Version 1.0 - 2023-07-29
--------------------------

Changed
^^^^^^^^
- No longer in beta stages. Full release v1.0.

Version 0.3.0 - 2023-07-26
--------------------------

Features
^^^^^^^^
- Unsupervised models now impute on nucleotide-encoded multilabel data instead of 012 data.

- Much higher metric scores due to the new multilabel encodings (less class imbalance)

- Unsupervised grid searches are faster due to less redundant code and removig unnecessary calculations with the scoring functions.

Changed
^^^^^^^^
- Documentation updates to make it easier to understand all the available arguments and what they are for.

- Refactored ``estimators.py`` and ``scorers.py`` for better modularity and easier code maintenance.

Removed
^^^^^^^^
- 012-encoded inputs for unsupervised imputers. Now uses multilabel nucleotide (4 class) inputs.

Version 0.2.4 - 2023-07-24
--------------------------

Features
^^^^^^^^
- Initial release with four unsupervised neural network models, three supervised IterativeImputer models, and four non-machine-learning imputers.
