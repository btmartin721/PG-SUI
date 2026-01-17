Tutorial: Implementing New Imputation Models
============================================

This guide shows how to add a new imputation model that conforms to PG-SUI's
refactored architecture:

- **Typed dataclass configuration** (``*Config``) with presets and optional dot-key overrides
- **Instantiate → fit() → transform()** contract (no arguments to ``fit``/``transform``)
- **BaseNNImputer** wrapper that manages IO, encoding/decoding, training loops, tuning, and plotting
- **Objective function** compatible with Optuna tuning
- **Top-level API imports** (``from pgsui import ...``), consistent with existing models (see ``ImputeAutoencoder`` and ``AutoencoderModel``)

Prerequisites
-------------

- PyTorch fundamentals (``torch.nn.Module``, optimizers, schedulers)
- SNP genotype encodings (0/1/2 with -9 or -1 for missing; IUPAC decode/encode via SNPio)
- Familiarity with ``BaseNNImputer`` life-cycle and helper methods

Model Development Overview
--------------------------

A new model requires **three parts**:

1. **A config dataclass**: ``MyNewModelConfig`` (typed, preset-aware)
2. **A PyTorch module**: ``MyNewAutoencoder`` (or encoder-decoder) with a clearly shaped output
3. **An imputer wrapper**: ``ImputeMyNewModel(BaseNNImputer)`` that plugs your module into PG-SUI's training/tuning/eval pipeline

.. note::

    Follow this contract strictly:
    ``imputer = ImputeX(genotype_data=gd, config=SomeConfig.from_preset("balanced")); imputer.fit(); X_imp = imputer.transform()``

Step 0 — Create a Config Dataclass
----------------------------------

Define a typed config that mirrors the structure used by existing models (``io``, ``model``, ``train``, ``tune``, ``plot``, ``sim``) and provides ``from_preset`` plus optional dot-key overrides.

.. code-block:: python

    from __future__ import annotations
    from dataclasses import dataclass, field, asdict
    from typing import Any, Dict, Literal

    from pgsui.data_processing.containers import (
        IOConfig, TrainConfig, TuneConfig, PlotConfig, ModelConfig, SimConfig
    )
    from pgsui.data_processing.config import apply_dot_overrides

    @dataclass
    class MyNewModelConfig:
        """Top-level configuration for ImputeMyNewModel."""
        io: IOConfig = field(default_factory=IOConfig)
        model: ModelConfig = field(default_factory=ModelConfig)
        train: TrainConfig = field(default_factory=TrainConfig)
        tune: TuneConfig = field(default_factory=TuneConfig)
        plot: PlotConfig = field(default_factory=PlotConfig)
        sim: SimConfig = field(default_factory=SimConfig)

        @classmethod
        def from_preset(
            cls, preset: Literal["fast", "balanced", "thorough"] = "balanced"
        ) -> "MyNewModelConfig":
            if preset not in {"fast", "balanced", "thorough"}:
                raise ValueError(f"Unknown preset: {preset}")
            cfg = cls()

            # Common defaults
            cfg.io.verbose = True
            cfg.train.validation_split = 0.2
            cfg.model.activation = "relu"
            cfg.model.layer_schedule = "pyramid"
            cfg.model.latent_init = "random"

            if preset == "fast":
                cfg.model.latent_dim = 4
                cfg.model.num_hidden_layers = 1
                cfg.model.dropout_rate = 0.10
                cfg.train.batch_size = 128
                cfg.train.learning_rate = 1e-3
                cfg.tune.enabled = False
            elif preset == "balanced":
                cfg.model.latent_dim = 8
                cfg.model.num_hidden_layers = 2
                cfg.model.dropout_rate = 0.20
                cfg.train.batch_size = 128
                cfg.train.learning_rate = 8e-4
                cfg.tune.enabled = True
                cfg.tune.n_trials = 100
            else:  # thorough
                cfg.model.latent_dim = 16
                cfg.model.num_hidden_layers = 3
                cfg.model.dropout_rate = 0.30
                cfg.train.batch_size = 64
                cfg.train.learning_rate = 6e-4
                cfg.tune.enabled = True
                cfg.tune.n_trials = 250

            return cfg

        def apply_overrides(self, overrides: Dict[str, Any] | None) -> "MyNewModelConfig":
            if not overrides:
                return self
            return apply_dot_overrides(self, overrides)

        def to_dict(self) -> Dict[str, Any]:
            return asdict(self)

Step 1 — Implement the PyTorch Module
-------------------------------------

The module predicts per-SNP class logits. Match shapes used by your wrapper (e.g., ``(batch, n_features, n_classes)``). Keep the API similar to ``AutoencoderModel``.

.. code-block:: python

    from typing import List, Literal
    import numpy as np
    import torch
    import torch.nn as nn

    class MyNewDecoder(nn.Module):
        """Minimal decoder predicting per-SNP logits."""
        def __init__(
            self,
            n_features: int,
            num_classes: int = 3,
            latent_dim: int = 8,
            hidden_layer_sizes: List[int] | np.ndarray = (128, 64),
            dropout_rate: float = 0.2,
            activation: Literal["relu", "elu", "selu", "leaky_relu"] = "relu",
        ):
            super().__init__()
            if isinstance(hidden_layer_sizes, np.ndarray):
                hidden_layer_sizes = hidden_layer_sizes.tolist()
            act = self._resolve_activation(activation)

            layers: list[nn.Module] = []
            in_dim = latent_dim
            for h in hidden_layer_sizes:
                layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.Dropout(dropout_rate), act]
                in_dim = h

            layers += [nn.Linear(in_dim, n_features * num_classes)]
            self.decoder = nn.Sequential(*layers)
            self.n_features = n_features
            self.num_classes = num_classes

        def _resolve_activation(self, name: str) -> nn.Module:
            return {"relu": nn.ReLU(), "elu": nn.ELU(), "selu": nn.SELU(), "leaky_relu": nn.LeakyReLU()}[name]

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            # Output shape: (batch, n_features, n_classes)
            logits = self.decoder(z)
            return logits.view(-1, self.n_features, self.num_classes)

Step 2 — Write the Imputer Wrapper
----------------------------------

Mirror the pattern in ``ImputeAutoencoder``:

- Normalize a config (dataclass, dict, or YAML path) → concrete config
- Initialize logging via ``LoggerManager``
- Prepare data via ``GenotypeEncoder`` (0/1/2; -1 for missing)
- Build/train/evaluate the model
- Provide ``fit(self)`` and ``transform(self)`` with **no arguments**

.. code-block:: python

    from __future__ import annotations
    from typing import Any, Dict
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.exceptions import NotFittedError
    from sklearn.model_selection import train_test_split

    from snpio.analysis.genotype_encoder import GenotypeEncoder
    from snpio.utils.logging import LoggerManager
    from pgsui.impute.unsupervised.base import BaseNNImputer
    from pgsui.data_processing.config import load_yaml_to_dataclass, apply_dot_overrides

    # -- Config normalization helper -------------------------------------------
    def ensure_my_config(config: MyNewModelConfig | dict | str | None) -> MyNewModelConfig:
        if config is None:
            return MyNewModelConfig.from_preset("balanced")
        if isinstance(config, MyNewModelConfig):
            return config
        if isinstance(config, str):
            return load_yaml_to_dataclass(
                config, MyNewModelConfig, preset_builder=MyNewModelConfig.from_preset
            )
        if isinstance(config, dict):
            base = MyNewModelConfig.from_preset(config.get("preset", "balanced"))
            return apply_dot_overrides(base, _flatten_dict(config))
        raise TypeError("config must be a MyNewModelConfig, dict, YAML path, or None.")

    def _flatten_dict(d: dict, prefix: str = "", out: dict | None = None) -> dict:
        out = out or {}
        for k, v in d.items():
            kk = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _flatten_dict(v, kk, out)
            else:
                out[kk] = v
        return out

    class ImputeMyNewModel(BaseNNImputer):
        """Impute missing 0/1/2 genotypes using MyNewDecoder."""

        def __init__(
            self,
            genotype_data,
            *,
            config: MyNewModelConfig | dict | str | None = None,
            overrides: dict | None = None,
        ):
            self.model_name = "ImputeMyNewModel"
            self.genotype_data = genotype_data

            cfg = ensure_my_config(config)
            if overrides:
                cfg = apply_dot_overrides(cfg, overrides)
            self.cfg = cfg

            logman = LoggerManager(
                __name__,
                prefix=self.cfg.io.prefix,
                debug=self.cfg.io.debug,
                verbose=self.cfg.io.verbose,
            )
            self.logger = logman.get_logger()

            super().__init__(
                model_name=self.model_name,
                genotype_data=self.genotype_data,
                prefix=self.cfg.io.prefix,
                device=self.cfg.train.device,
                verbose=self.cfg.io.verbose,
                debug=self.cfg.io.debug,
            )

            self.Model = MyNewDecoder
            self.pgenc = GenotypeEncoder(genotype_data)
            self.seed = self.cfg.io.seed
            self.rng = np.random.default_rng(self.seed)

            # Cache common attrs from config
            self.latent_dim = self.cfg.model.latent_dim
            self.dropout_rate = self.cfg.model.dropout_rate
            self.num_hidden_layers = self.cfg.model.num_hidden_layers
            self.activation = self.cfg.model.activation
            self.batch_size = self.cfg.train.batch_size
            self.learning_rate = self.cfg.train.learning_rate
            self.validation_split = self.cfg.train.validation_split
            self.epochs = self.cfg.train.max_epochs
            self.early_stop_gen = self.cfg.train.early_stop_gen
            self.min_epochs = self.cfg.train.min_epochs

            # Tuning flags
            self.tune = self.cfg.tune.enabled
            self.n_trials = self.cfg.tune.n_trials
            self.tune_metric = self.cfg.tune.metrics

            # Plotting
            self.plot_dpi = self.cfg.plot.dpi
            self.show_plots = self.cfg.plot.show

            # Filled in at fit()
            self.num_classes_ = None
            self.num_features_ = None
            self.class_weights_ = None
            self.best_params_: Dict[str, Any] = {}
            self.is_fit_ = False

        def fit(self) -> "ImputeMyNewModel":
            """Train the model on 0/1/2 data (with -1 for missing)."""
            self.logger.info(f"Fitting {self.model_name}...")

            # Prepare 0/1/2 matrix and mark missing as -1
            X = self.pgenc.genotypes_012.astype(np.int64, copy=True)
            X[X < 0] = -1
            n_samples, self.num_features_ = X.shape

            # Determine classes (diploid: 3; haploid collapses to 2)
            is_haploid = self.pgenc.is_haploid
            self.num_classes_ = 2 if is_haploid else 3
            if is_haploid:
                X[X == 2] = 1  # map {0,2} -> {0,1}

            # Use base class helper for splitting
            self.train_idx_, self.val_idx_, self.test_idx_ = self._train_val_test_split(X)
            X_train = X[self.train_idx_]
            X_val = X[self.val_idx_]

            # Hidden sizes from config helper on Base
            input_dim = self.num_features_ * self.num_classes_
            hidden = self._compute_hidden_layer_sizes(
                n_inputs=input_dim,
                n_outputs=self.num_classes_,
                n_samples=len(self.train_idx_),
                n_hidden=self.num_hidden_layers,
                latent_dim=self.latent_dim,
                alpha=getattr(self.cfg.model, "layer_scaling_factor", 4.0),
                schedule=self.cfg.model.layer_schedule,
            )

            self.best_params_ = {
                "n_features": self.num_features_,
                "num_classes": self.num_classes_,
                "latent_dim": self.latent_dim,
                "hidden_layer_sizes": hidden,
                "dropout_rate": self.dropout_rate,
                "activation": self.activation,
            }

            # Build model and train
            model = self.build_model(self.Model, self.best_params_)
            model.apply(self.initialize_weights)

            # NOTE: Use appropriate data loader logic here (masked vs full)
            train_loader = self._get_data_loaders(X_train, X_train, mask=np.ones_like(X_train, dtype=bool), batch_size=self.batch_size, shuffle=True)
            val_loader = self._get_data_loaders(X_val, X_val, mask=np.ones_like(X_val, dtype=bool), batch_size=self.batch_size, shuffle=False)

            loss, self.model_, self.history_ = self._train_and_validate_model(
                model=model,
                X_train=X_train,
                y_train=X_train,
                val_loader=val_loader,
                lr=self.learning_rate,
                l1_penalty=getattr(self.cfg.train, "l1_penalty", 0.0),
                params=self.best_params_,
            )

            self.is_fit_ = True
            return self

        def transform(self) -> np.ndarray:
            """Impute the full dataset and return IUPAC strings."""
            if not self.is_fit_:
                raise NotFittedError("Call fit() before transform().")

            X_all = self.pgenc.genotypes_012.astype(np.int64, copy=True)
            X_all[X_all < 0] = -1

            labels, _ = self._predict(self.model_, X_all)

            miss = X_all == -1
            X_imp = X_all.copy()
            X_imp[miss] = labels[miss]

            return self.pgenc.decode_012(X_imp)

Step 3 — (Optional) Hyperparameter Tuning
-----------------------------------------

If your model supports Optuna tuning, mirror the ``ImputeAutoencoder`` pattern:

- ``_objective(self, trial)`` samples hyperparameters → trains quickly → returns a scalar metric (e.g., ``f1``)
- ``_sample_hyperparameters(self, trial)`` returns a dictionary with both raw choices and a ``model_params`` payload
- ``_set_best_params(self, best_params)`` converts the winning trial into the final ``model_params``

Keep the validation logic inside the wrapper so you can reuse PG-SUI's scorers/plotters.

Registering the Model (Top-Level API & CLI)
-------------------------------------------

Expose your model and config in the top-level package so users can import them cleanly:

.. code-block:: python

    # pgsui/__init__.py
    from .impute.unsupervised.imputers.my_new_model import ImputeMyNewModel
    from .data_processing.containers import MyNewModelConfig
    __all__ = [..., "ImputeMyNewModel", "MyNewModelConfig"]

Add the class name to the CLI's ``--models`` registry (mirroring how the other models are discovered).

Usage Examples
--------------

**Python**

.. code-block:: python

    from snpio import VCFReader
    from pgsui import ImputeMyNewModel, MyNewModelConfig

    gd = VCFReader("example.vcf.gz", popmapfile="example.popmap", prefix="demo")
    cfg = MyNewModelConfig.from_preset("balanced").apply_overrides({"io.prefix": "mymodel_demo"})
    model = ImputeMyNewModel(genotype_data=gd, config=cfg)
    model.fit()
    X_imp = model.transform()

**CLI**

.. code-block:: bash

    pg-sui \
        --input example.vcf.gz \
        --popmap example.popmap \
        --models ImputeMyNewModel \
        --preset balanced \
        --sim-strategy random_weighted \
        --sim-prop 0.25 \
        --set io.prefix=mymodel_demo

The CLI propagates ``--sim-strategy``/``--sim-prop`` choices to every selected model, so new implementations participate automatically; ``--disable-simulate-missing`` is intended for supervised/deterministic runs (unsupervised models require simulated masking).

Design Notes & Best Practices
-----------------------------

- **Shapes**: Keep module output ``(batch, n_features, n_classes)`` to simplify loss computation and downstream reports.
- **Missing values**: Standardize to **-1** before loss; use ``ignore_index=-1`` in CE/focal variants.
- **Class imbalance**: Use PG-SUI's weighting helpers (temperature/alpha/normalize) or your own, but store weights on device.
- **Reproducibility**: Read seeds from ``io.seed``; pass to NumPy, PyTorch, and Optuna where applicable.
- **Presets**: Ensure ``from_preset`` tunes depth/width/epochs proportionally so ``fast`` is actually fast and ``thorough`` explores more.
- **Plots/metrics**: Reuse the Base helper methods so your model automatically participates in radar/PR/confusion outputs.

FAQ
---

**Q: Do I pass arrays to ``fit`` or ``transform``?**
A: No. Like ``ImputeAutoencoder``, you pass ``genotype_data`` at construction; then call ``fit()`` and ``transform()`` with **no arguments**.

**Q: Can my module use a full autoencoder (encoder+decoder)?**
A: Yes. Expose a consistent forward that returns per-SNP logits and adapt the wrapper's latent handling accordingly.

**Q: How do I add Optuna tuning quickly?**
A: Implement ``_objective``, ``_sample_hyperparameters``, and ``_set_best_params`` following the other models.
