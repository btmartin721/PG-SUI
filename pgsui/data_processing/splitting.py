"""Deterministic dataset-splitting utilities shared by PG-SUI imputers."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def train_validation_test_indices(
    n_samples: int,
    validation_split: float,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the canonical PG-SUI train, validation, and test row indices.

    ``validation_split`` is the combined fraction reserved from training. The
    reserved rows are then divided equally between validation and test sets.
    This preserves the split sequence historically used by the neural
    imputers while making it reusable by deterministic baselines.

    Args:
        n_samples: Number of sample rows in the genotype matrix.
        validation_split: Combined validation/test fraction in ``(0, 1)``.
        seed: Random state supplied to both scikit-learn splits.

    Returns:
        Train, validation, and test row-index arrays, in that order.

    Raises:
        ValueError: If the dataset or validation fraction cannot support the
            canonical three-way split.
    """
    if n_samples < 3:
        raise ValueError(
            f"Not enough samples ({n_samples}) for train/validation/test split."
        )

    split_fraction = float(validation_split)
    if not 0.0 < split_fraction < 1.0:
        raise ValueError(
            f"validation_split must be in (0.0, 1.0), but got {validation_split}."
        )

    indices = np.arange(n_samples, dtype=int)
    train_idx, validation_test_idx = train_test_split(
        indices,
        test_size=split_fraction,
        random_state=seed,
    )

    if validation_test_idx.size < 4:
        raise ValueError(
            "Not enough samples "
            f"({validation_test_idx.size}) for validation/test split."
        )

    validation_idx, test_idx = train_test_split(
        validation_test_idx,
        test_size=0.5,
        random_state=seed,
    )
    return train_idx, validation_idx, test_idx
