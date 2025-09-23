from dataclasses import dataclass
from typing import Literal


@dataclass
class _SimParams:
    """Container for simulation hyperparameters.

    Attributes:
        prop_missing (float): Proportion of genotypes to simulate as missing.
        strategy (str): Strategy for simulating missingness.
        missing_val (int | float): Value representing missing genotypes.
        het_boost (float): Factor to boost heterozygous missingness.
        seed (int | None): Random seed for reproducibility.
    """

    prop_missing: float = 0.3
    strategy: Literal["random", "random_inv_genotype"] = "random_inv_genotype"
    missing_val: int | float = -1
    het_boost: float = 2.0
    seed: int | None = None

    def to_dict(self) -> dict:
        """Convert the dataclass to a dictionary.

        Returns:
            dict: Dictionary representation of the dataclass.
        """
        return {
            "prop_missing": self.prop_missing,
            "strategy": self.strategy,
            "missing_val": self.missing_val,
            "het_boost": self.het_boost,
            "seed": self.seed,
        }


@dataclass
class _ImputerParams:
    """Container for imputer hyperparameters.

    Attributes:
        n_nearest_features (int | None): Number of nearest features for IterativeImputer.
        max_iter (int): Maximum iterations for IterativeImputer.
        keep_empty_features (bool): Whether to keep features that are entirely missing.
        initial_strategy (str): Strategy for initial imputation in IterativeImputer.
        random_state (int | None): Random seed for reproducibility.
        verbose (bool): Verbose logging.
    """

    n_nearest_features: int | None = 10
    max_iter: int = 10
    initial_strategy: Literal["mean", "median", "most_frequent", "constant"] = (
        "most_frequent"
    )
    keep_empty_features: bool = True
    random_state: int | None = None
    verbose: bool = False

    def to_dict(self) -> dict:
        """Convert the dataclass to a dictionary.

        Returns:
            dict: Dictionary representation of the dataclass.
        """
        return {
            "n_nearest_features": self.n_nearest_features,
            "max_iter": self.max_iter,
            "initial_strategy": self.initial_strategy,
            "keep_empty_features": self.keep_empty_features,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }


@dataclass
class _RFParams:
    """Container for RandomForest hyperparameters.

    Attributes:
        n_estimators (int): Number of trees in the forest.
        max_depth (int | None): Maximum depth of the tree.
        min_samples_split (int): Minimum samples required to split an internal node.
        min_samples_leaf (int): Minimum samples required to be at a leaf node.
        max_features (str | float | None): Number of features to consider at each split.
        criterion (str): Function to measure the quality of a split.
        class_weight (str | None): Weights associated with classes.
    """

    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Literal["sqrt", "log2"] | float | None = "sqrt"
    criterion: Literal["gini", "entropy", "log_loss"] = "gini"
    class_weight: Literal["balanced", "balanced_subsample", None] = "balanced"

    def to_dict(self) -> dict:
        """Convert the dataclass to a dictionary.

        Returns:
            dict: Dictionary representation of the dataclass.
        """
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "criterion": self.criterion,
            "class_weight": self.class_weight,
        }


@dataclass
class _HGBParams:
    """Container for HistGradientBoosting hyperparameters.

    Attributes:
        max_iter (int): Maximum number of iterations.
        learning_rate (float): Learning rate shrinks the contribution of each tree.
        max_depth (int | None): Maximum depth of the tree.
        min_samples_leaf (int): Minimum samples required to be at a leaf node.
        n_iter_no_change (int): Number of iterations with no improvement to wait before stopping.
        tol (float): Tolerance for the early stopping.
        max_features (str | float | None): Number of features to consider at each split.
        class_weight (str | None): Weights associated with classes.
        random_state (int | None): Random seed for reproducibility.
    """

    max_iter: int = 100
    learning_rate: float = 0.1
    max_depth: int | None = None
    min_samples_leaf: int = 1
    n_iter_no_change: int = 10
    tol: float = 1e-7
    max_features: float = 1.0
    class_weight: Literal["balanced", "balanced_subsample", None] = "balanced"
    random_state: int | None = None
    verbose: bool = False

    def to_dict(self) -> dict:
        """Convert the dataclass to a dictionary.

        Returns:
            dict: Dictionary representation of the dataclass.
        """
        return {
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "n_iter_no_change": self.n_iter_no_change,
            "tol": self.tol,
        }
