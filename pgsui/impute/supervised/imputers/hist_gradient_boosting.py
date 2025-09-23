# Standard library
from typing import TYPE_CHECKING, Generator, List, Literal

# Third-party
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split

# Project
from snpio.analysis.genotype_encoder import GenotypeEncoder
from snpio.utils.logging import LoggerManager

from pgsui.data_processing.containers import _HGBParams, _ImputerParams, _SimParams
from pgsui.data_processing.transformers import SimGenotypeDataTransformer
from pgsui.impute.supervised.base import BaseImputer
from pgsui.utils.plotting import Plotting
from pgsui.utils.scorers import Scorer

if TYPE_CHECKING:
    from snpio.read_input.genotype_data import GenotypeData


class ImputeHistGradientBoosting(BaseImputer):
    """Supervised imputation with Histogram-based Gradient Boosting on 0/1/2 genotypes.

    This implementation uses the following workflow:
        - Uses 0/1/2 encoded genotypes (with -9 for missing).
        - Splits by samples into train/test once (no per-locus CV).
        - Trains one HistGradientBoosting per locus using all *other* loci as predictors.
        - Computes the same metrics (accuracy, F1, PR-macro, etc.) and renders the same plots/classification reports (zygosity and IUPAC).

    Compared with allele-wise/iterative imputers, this is simpler and scales well with parallelization across loci.

    Attributes:
        model_name (str): Name used in logs/paths.
        genotypes_012_ (np.ndarray): 0/1/2 matrix with -9 for missing.
        models_ (list[HistGradientBoostingClassifier | None]): One fitted model per locus; None where model couldn't be trained.
    """

    def __init__(
        self,
        genotype_data: "GenotypeData",
        *,
        # General parameters
        prefix: str = "pgsui",
        seed: int | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
        debug: bool = False,
        # Model hyperparameters
        model_n_estimators: int = 300,
        model_learning_rate: float = 0.1,
        model_n_iter_no_change: int = 10,
        model_tol: float = 1e-7,
        model_max_depth: int | None = None,
        model_min_samples_leaf: int = 1,
        model_max_features: float = 1.0,
        model_validation_split: float = 0.2,
        model_n_nearest_features: int | None = 10,
        model_max_iter: int = 10,
        # Simulation parameters
        sim_prop_missing: float = 0.5,
        sim_strategy: Literal["random", "random_inv_genotype"] = "random_inv_genotype",
        sim_het_boost: float = 2.0,
        # Plotting parameters
        plot_format: Literal["pdf", "png", "jpg", "jpeg"] = "pdf",
        plot_fontsize: int = 18,
        plot_despine: bool = True,
        plot_dpi: int = 300,
        plot_show_plots: bool = False,
    ):
        """Initialize the HistGradientBoostingClassifier imputer.

        Args:
            genotype_data (GenotypeData): SNPio genotype container.
            prefix (str): Output prefix.
            seed (int | None): RNG seed.
            n_jobs (int): Parallel jobs for per-locus training/prediction.
            verbose (bool): Verbose logging.
            debug (bool): Debug logging.
            model_n_estimators (int): HGB boosted trees.
            model_learning_rate (float): HGB learning rate.
            model_n_iter_no_change (int): Early stopping rounds.
            model_tol (float): Early stopping tolerance.
            model_max_depth (int | None): HGB depth.
            model_min_samples_split (int): Min split.
            model_min_samples_leaf (int): Min leaf.
            model_max_features (str|float|None): max_features.
            model_criterion (str): Split criterion.
            model_validation_split (float): Test fraction by samples.
            model_n_nearest_features (int | None): Number of Nearest Neighbors for IterativeImputer. Defaults to 10.
            model_max_iter (int): Max IterativeImputer iterations.
            sim_prop_missing (float): Proportion of genotypes to simulate as missing.
            sim_strategy (str): Strategy for simulating missingness.
            sim_het_boost (float): Factor to boost heterozygous missingness.
            plot_format (Literal["pdf", "png", "jpg", "jpeg"]): Plot format.
            plot_fontsize (int): Font size for plots.
            plot_despine (bool): Whether to despine plots.
            plot_dpi (int): DPI for plots.
            plot_show_plots (bool): Whether to show plots.
        """
        self.model_name = "ImputeHistGradientBoosting"
        self.Model = HistGradientBoostingClassifier

        self.genotype_data = genotype_data
        self.pgenc = GenotypeEncoder(genotype_data)
        self.prefix = prefix
        self.seed = seed
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.debug = debug

        super().__init__(verbose=verbose, debug=debug)

        logman = LoggerManager(__name__, prefix=prefix, verbose=verbose, debug=debug)
        self.logger = logman.get_logger()

        # Ensure supervised-family directories (models/plots/metrics/optimize)
        self._create_model_directories(
            prefix, ["models", "plots", "metrics", "optimize"]
        )

        # Plotting config
        self.plot_format = plot_format
        self.plot_dpi = plot_dpi
        self.plot_fontsize = plot_fontsize
        self.title_fontsize = plot_fontsize
        self.despine = plot_despine
        self.show_plots = plot_show_plots

        # Data/task config
        self.validation_split = model_validation_split

        # HGB params
        self.params = _HGBParams(
            max_iter=model_n_estimators,
            learning_rate=model_learning_rate,
            max_depth=model_max_depth,
            min_samples_leaf=model_min_samples_leaf,
            max_features=model_max_features,
            random_state=seed,
            verbose=debug,
            n_iter_no_change=model_n_iter_no_change,
            tol=model_tol,
        )

        # Imputer params
        self.imputer_params = _ImputerParams(
            n_nearest_features=model_n_nearest_features,
            max_iter=model_max_iter,
            random_state=seed,
            verbose=verbose,
        )

        self.sim_params = _SimParams(
            prop_missing=sim_prop_missing,
            strategy=sim_strategy,
            missing_val=-1,
            het_boost=sim_het_boost,
            seed=seed,
        )

        self.max_iter = model_max_iter
        self.n_nearest_features = model_n_nearest_features

        # Will be set in fit()
        self.is_haploid_: bool | None = None
        self.num_classes_: int | None = None
        self.num_features_: int | None = None
        self.models_: List[HistGradientBoostingClassifier | None] | None = None
        self.is_fit_: bool = False

    def fit(self) -> "BaseImputer":
        """Fit the imputer using self.genotype_data with no arguments.

        Steps:
            1) Encode to 0/1/2 with -9/-1 as missing.
            2) Split samples into train/test.
            3) Train IterativeImputer on train (convert missing -> NaN).
            4) Evaluate on test **non-missing positions** (reconstruction metrics)
            and call your original plotting stack via _make_class_reports().

        Returns:
            BaseImputer: self.
        """
        # Prepare utilities & metadata
        self.scorers_ = Scorer(
            prefix=self.prefix, average="macro", verbose=self.verbose, debug=self.debug
        )

        self.plotter_ = Plotting(
            self.model_name,
            prefix=self.prefix,
            plot_format=self.plot_format,
            plot_dpi=self.plot_dpi,
            plot_fontsize=self.plot_fontsize,
            title_fontsize=self.title_fontsize,
            despine=self.despine,
            show_plots=self.show_plots,
            verbose=self.verbose,
            debug=self.debug,
        )

        X_int = self.pgenc.genotypes_012
        self.X012_ = X_int.astype(float)
        self.X012_[self.X012_ < 0] = np.nan  # Ensure missing are NaN
        self.is_haploid_ = np.count_nonzero(self.X012_ == 1) == 0
        self.num_classes_ = 2 if self.is_haploid_ else 3
        self.n_samples_, self.n_features_ = X_int.shape

        # Split
        X_train, X_test = train_test_split(
            self.X012_,
            test_size=self.validation_split,
            random_state=self.seed,
            shuffle=True,
        )

        # Simulate missing values on test set.
        sim_transformer = SimGenotypeDataTransformer(**self.sim_params.to_dict())

        X_test = np.nan_to_num(X_test, nan=-1)  # ensure missing are -1
        sim_transformer.fit(X_test)
        X_test_sim, missing_masks = sim_transformer.transform(X_test)
        sim_mask = missing_masks["simulated"]
        X_test_sim[X_test_sim < 0] = np.nan  # ensure missing are NaN

        self.model_params_ = self.params.to_dict()
        self.model_params_["random_state"] = self.seed

        # Train IterativeImputer
        est = self.Model(**self.model_params_)

        self.imputer_ = IterativeImputer(estimator=est, **self.imputer_params.to_dict())

        self.imputer_.fit(X_train)
        self.is_fit_ = True

        X_test_imputed = self.imputer_.transform(X_test_sim)

        # Predict on simulated test set
        y_true_flat = X_test[sim_mask].copy()
        y_pred_flat = X_test_imputed[sim_mask].copy()

        # Round and clip predictions to valid {0,1,2} or {0,1} if haploid.
        if self.is_haploid_:
            y_pred_flat = np.clip(np.rint(y_pred_flat), 0, 1).astype(int, copy=False)
            y_true_flat = np.clip(np.rint(y_true_flat), 0, 1).astype(int, copy=False)
        else:
            y_pred_flat = np.clip(np.rint(y_pred_flat), 0, 2).astype(int, copy=False)
            y_true_flat = np.clip(np.rint(y_true_flat), 0, 2).astype(int, copy=False)

        # Evaluate (012 / zygosity)
        self._evaluate_012_and_plot(y_true_flat.copy(), y_pred_flat.copy())

        # Evaluate (IUPAC)
        encodings_dict = {
            "A": 0,
            "C": 1,
            "G": 2,
            "T": 3,
            "W": 4,
            "R": 5,
            "M": 6,
            "K": 7,
            "Y": 8,
            "S": 9,
            "N": -1,
        }

        y_true_iupac_tmp = self.pgenc.decode_012(y_true_flat)
        y_pred_iupac_tmp = self.pgenc.decode_012(y_pred_flat)
        y_true_iupac = self.pgenc.convert_int_iupac(
            y_true_iupac_tmp, encodings_dict=encodings_dict
        )
        y_pred_iupac = self.pgenc.convert_int_iupac(
            y_pred_iupac_tmp, encodings_dict=encodings_dict
        )
        self._evaluate_iupac10_and_plot(y_true_iupac, y_pred_iupac)

        return self

    def transform(self) -> np.ndarray:
        """Impute all samples and return imputed genotypes.

        Returns:
            np.ndarray: (n_samples, n_loci) integers with no -9/-1/NaN.
        """
        if not self.is_fit_:
            msg = "Imputer has not been fit; call fit() before transform()."
            self.logger.error(msg)
            raise RuntimeError(msg)

        X = self.X012_.copy()
        X_imp = self.imputer_.transform(X)

        if np.any(X_imp < 0) or np.isnan(X_imp).any():
            self.logger.warning("Some imputed values are still missing; setting to -9.")
            X_imp[X_imp < 0] = -9
            X_imp[np.isnan(X_imp)] = -9

        return self.pgenc.decode_012(X_imp)
