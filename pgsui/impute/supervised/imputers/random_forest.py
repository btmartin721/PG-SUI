# Standard library
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal

# Third-party
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split

# Project
from snpio.analysis.genotype_encoder import GenotypeEncoder
from snpio.utils.logging import LoggerManager

from pgsui.data_processing.config import apply_dot_overrides, load_yaml_to_dataclass
from pgsui.data_processing.containers import (
    RFConfig,
    _ImputerParams,
    _RFParams,
    _SimParams,
)
from pgsui.data_processing.transformers import SimGenotypeDataTransformer
from pgsui.impute.supervised.base import BaseImputer
from pgsui.utils.logging_utils import configure_logger
from pgsui.utils.plotting import Plotting
from pgsui.utils.scorers import Scorer

if TYPE_CHECKING:
    from snpio.read_input.genotype_data import GenotypeData


def ensure_rf_config(config: RFConfig | Dict | str | None) -> RFConfig:
    """Resolve RF configuration from dataclass, mapping, or YAML path."""

    if config is None:
        return RFConfig()
    if isinstance(config, RFConfig):
        return config
    if isinstance(config, str):
        return load_yaml_to_dataclass(config, RFConfig)
    if isinstance(config, dict):
        payload = dict(config)
        preset = payload.pop("preset", None)
        base = RFConfig.from_preset(preset) if preset else RFConfig()

        def _flatten(prefix: str, data: Dict[str, Any], out: Dict[str, Any]) -> None:
            for key, value in data.items():
                dotted = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    _flatten(dotted, value, out)
                else:
                    out[dotted] = value

        flat: Dict[str, Any] = {}
        _flatten("", payload, flat)
        return apply_dot_overrides(base, flat)

    raise TypeError("config must be an RFConfig, dict, YAML path, or None.")


class ImputeRandomForest(BaseImputer):
    """Supervised RF imputer driven by :class:`RFConfig`."""

    def __init__(
        self,
        genotype_data: "GenotypeData",
        *,
        config: RFConfig | Dict | str | None = None,
        overrides: Dict | None = None,
    ) -> None:
        self.model_name = "ImputeRandomForest"
        self.Model = RandomForestClassifier

        cfg = ensure_rf_config(config)
        if overrides:
            cfg = cfg.apply_overrides(overrides)
        self.cfg = cfg

        self.genotype_data = genotype_data
        self.pgenc = GenotypeEncoder(genotype_data)

        self.prefix = cfg.io.prefix
        self.seed = cfg.io.seed
        self.n_jobs = cfg.io.n_jobs
        self.verbose = cfg.io.verbose
        self.debug = cfg.io.debug

        super().__init__(verbose=self.verbose, debug=self.debug)

        logman = LoggerManager(
            __name__, prefix=self.prefix, verbose=self.verbose, debug=self.debug
        )
        self.logger = configure_logger(
            logman.get_logger(), verbose=self.verbose, debug=self.debug
        )

        self._create_model_directories(
            self.prefix, ["models", "plots", "metrics", "optimize", "parameters"]
        )

        self.plot_format: Literal["png", "pdf", "svg", "jpg", "jpeg"] = cfg.plot.fmt

        self.plot_fontsize = cfg.plot.fontsize
        self.title_fontsize = cfg.plot.fontsize
        self.plot_dpi = cfg.plot.dpi
        self.despine = cfg.plot.despine
        self.show_plots = cfg.plot.show

        self.validation_split = cfg.train.validation_split

        self.params = _RFParams(
            n_estimators=cfg.model.n_estimators,
            max_depth=cfg.model.max_depth,
            min_samples_split=cfg.model.min_samples_split,
            min_samples_leaf=cfg.model.min_samples_leaf,
            max_features=cfg.model.max_features,
            criterion=cfg.model.criterion,
            class_weight=cfg.model.class_weight,
        )

        self.imputer_params = _ImputerParams(
            n_nearest_features=cfg.imputer.n_nearest_features,
            max_iter=cfg.imputer.max_iter,
            random_state=self.seed,
            verbose=self.verbose,
        )

        self.sim_params = _SimParams(
            prop_missing=cfg.sim.prop_missing,
            strategy=cfg.sim.strategy,
            missing_val=cfg.sim.missing_val,
            het_boost=cfg.sim.het_boost,
            seed=self.seed,
        )

        self.max_iter = cfg.imputer.max_iter
        self.n_nearest_features = cfg.imputer.n_nearest_features

        # Will be set in fit()
        self.is_haploid_: bool | None = None
        self.num_classes_: int | None = None
        self.num_features_: int | None = None
        self.rf_models_: List[RandomForestClassifier | None] | None = None
        self.is_fit_: bool = False

    def fit(self) -> "BaseImputer":
        """Fit the imputer using self.genotype_data with no arguments.

        This method trains the imputer on the provided genotype data.

        Steps:
            1) Encode to 0/1/2 with -9/-1 as missing.
            2) Split samples into train/test.
            3) Train IterativeImputer on train (convert missing -> NaN).
            4) Evaluate on test **non-missing positions** (reconstruction metrics) and call your original plotting stack via _make_class_reports().

        Returns:
            BaseImputer: self.
        """
        # Prepare utilities & metadata
        self.scorers_ = Scorer(
            prefix=self.prefix, average="macro", verbose=self.verbose, debug=self.debug
        )

        pf: Literal["png", "pdf", "svg", "jpg", "jpeg"] = self.plot_format

        self.plotter_ = Plotting(
            self.model_name,
            prefix=self.prefix,
            plot_format=pf,
            plot_dpi=self.plot_dpi,
            plot_fontsize=self.plot_fontsize,
            title_fontsize=self.title_fontsize,
            despine=self.despine,
            show_plots=self.show_plots,
            verbose=self.verbose,
            debug=self.debug,
            multiqc=True,
            multiqc_section=f"PG-SUI: {self.model_name} Model Imputation",
        )

        X_int = self.pgenc.genotypes_012
        self.X012_ = X_int.astype(float)
        self.X012_[self.X012_ < 0] = np.nan  # Ensure missing are NaN
        self.ploidy = self.cfg.io.ploidy
        self.is_haploid = self.ploidy == 1
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
        self.model_params_["n_jobs"] = self.n_jobs
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

        self.best_params_ = self.model_params_
        self.best_params_.update(self.imputer_params.to_dict())
        self.best_params_.update(self.sim_params.to_dict())
        self._save_best_params(self.best_params_)

        return self

    def transform(self) -> np.ndarray:
        """Impute all samples and return imputed genotypes.

        This method applies the trained imputer to the entire dataset, filling in missing genotype values. It ensures that any remaining missing values after imputation are set to -9, and decodes the imputed 0/1/2 genotypes back to their original format.

        Returns:
            np.ndarray: (n_samples, n_loci) IUPAC strings (single-character codes).
        """
        if not self.is_fit_:
            msg = "Imputer has not been fit; call fit() before transform()."
            self.logger.error(msg)
            raise NotFittedError(msg)

        X = self.X012_.copy()
        X_imp = self.imputer_.transform(X)

        if np.any(X_imp < 0) or np.isnan(X_imp).any():
            self.logger.warning("Some imputed values are still missing; setting to -9.")
            X_imp[X_imp < 0] = -9
            X_imp[np.isnan(X_imp)] = -9

        return self.pgenc.decode_012(X_imp)
