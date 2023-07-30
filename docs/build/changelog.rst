================
Changelog
================

This document provides a high-level view of the changes made to PG-SUI for each release.

Unreleased
----------
- Make HTML report with all plots and logs.

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
