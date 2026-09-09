"""Write compact, coordinate-verifiable evaluation-mask artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_test_evaluation_mask(
    output_dir: Path,
    *,
    test_indices: np.ndarray,
    evaluation_mask: np.ndarray,
    n_samples: int,
    n_loci: int,
    seed: int | None,
    strategy: str,
    validation_split: float,
) -> Path:
    """Save exact test rows and evaluated coordinates for post-hoc auditing.

    Args:
        output_dir: Model-specific metrics directory.
        test_indices: Original full-matrix row indices used for testing.
        evaluation_mask: Boolean mask with either full-matrix rows or test rows.
        n_samples: Number of rows in the full genotype matrix.
        n_loci: Number of loci in the full genotype matrix.
        seed: Configured random seed.
        strategy: Missingness simulation strategy.
        validation_split: Combined validation/test fraction.

    Returns:
        Path to the compressed NumPy artifact.

    Raises:
        ValueError: If indices or mask shapes are inconsistent.
    """
    test_idx = np.asarray(test_indices, dtype=np.int64)
    mask = np.asarray(evaluation_mask, dtype=bool)
    if test_idx.ndim != 1:
        raise ValueError("test_indices must be one-dimensional")
    if np.any((test_idx < 0) | (test_idx >= n_samples)):
        raise ValueError("test_indices contain out-of-bounds rows")
    if mask.ndim != 2 or mask.shape[1] != n_loci:
        raise ValueError(
            f"evaluation_mask must have {n_loci} loci; received shape {mask.shape}"
        )
    if mask.shape[0] == n_samples:
        test_mask = mask[test_idx]
    elif mask.shape[0] == test_idx.size:
        test_mask = mask
    else:
        raise ValueError(
            "evaluation_mask rows must match either the full dataset or test set: "
            f"mask={mask.shape[0]}, full={n_samples}, test={test_idx.size}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "evaluation_mask_test.npz"
    np.savez_compressed(
        path,
        test_idx=test_idx,
        evaluation_mask_test=test_mask,
        full_shape=np.asarray((n_samples, n_loci), dtype=np.int64),
        seed=np.asarray(-1 if seed is None else seed, dtype=np.int64),
        strategy=np.asarray(strategy),
        validation_split=np.asarray(validation_split, dtype=np.float64),
    )
    return path
