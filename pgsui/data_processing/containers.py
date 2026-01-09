from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional, Sequence

from pgsui.data_processing.config import apply_dot_overrides, load_yaml_to_dataclass


@dataclass
class _SimParams:
    """Container for simulation hyperparameters.

    Attributes:
        prop_missing (float): Proportion of missing values to simulate.
        strategy (Literal["random", "random_inv_genotype"]): Strategy for simulating missing values.
        missing_val (int | float): Value to represent missing data.
        het_boost (float): Boost factor for heterozygous genotypes.
        seed (int | None): Random seed for reproducibility.
    """

    prop_missing: float = 0.3
    strategy: Literal["random", "random_inv_genotype"] = "random_inv_genotype"
    missing_val: int | float = -1
    het_boost: float = 2.0
    seed: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _ImputerParams:
    """Container for imputer hyperparameters.

    Attributes:
        n_nearest_features (int | None): Number of nearest features to consider for imputation.
        max_iter (int): Maximum number of iterations for the imputation algorithm.
        initial_strategy (Literal["mean", "median", "most_frequent", "constant"]): Strategy for initial imputation.
        keep_empty_features (bool): Whether to keep features that are entirely missing.
        random_state (int | None): Random seed for reproducibility.
        verbose (bool): If True, enables verbose logging during imputation.
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
        return asdict(self)


@dataclass
class _RFParams:
    """Container for RandomForest hyperparameters.

    Attributes:
        n_estimators (int): Number of trees in the forest.
        max_depth (int | None): Maximum depth of the trees.
        min_samples_split (int): Minimum number of samples required to split an internal node.
        min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
        max_features (Literal["sqrt", "log2"] | float | None): Number of features to consider for split.
        criterion (Literal["gini", "entropy", "log_loss"]): Function to measure the quality of a split.
        class_weight (Literal["balanced", "balanced_subsample", None]): Weights associated with classes.
    """

    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Literal["sqrt", "log2"] | float | None = "sqrt"
    criterion: Literal["gini", "entropy", "log_loss"] = "gini"
    class_weight: Literal["balanced", "balanced_subsample", None] = "balanced"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _HGBParams:
    """Container for HistGradientBoosting hyperparameters.

    Attributes:
        max_iter (int): Maximum number of iterations.
        learning_rate (float): Learning rate shrinks the contribution of each tree.
        max_depth (int | None): Maximum depth of the individual regression estimators.
        min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
        n_iter_no_change (int): Number of iterations with no improvement to wait before early stopping.
        tol (float): Tolerance for the early stopping.
        max_features (float | None): The fraction of features to consider when looking for the best split.
        class_weight (Literal["balanced", "balanced_subsample", None]): Weights associated with classes.
        random_state (int | None): Random seed for reproducibility.
        verbose (bool): If True, enables verbose logging during training.
    """

    max_iter: int = 100
    learning_rate: float = 0.1
    max_depth: int | None = None
    min_samples_leaf: int = 1
    n_iter_no_change: int = 10
    tol: float = 1e-7
    max_features: float | None = 1.0
    class_weight: Literal["balanced", "balanced_subsample", None] = "balanced"
    random_state: int | None = None
    verbose: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelConfig:
    """Model architecture configuration.

    Attributes:
        latent_init (Literal["random", "pca"]): Method for initializing the latent space.
        latent_dim (int): Dimensionality of the latent space.
        dropout_rate (float): Dropout rate for regularization.
        num_hidden_layers (int): Number of hidden layers in the neural network.
        activation (Literal["relu", "elu", "selu", "leaky_relu"]): Activation function.
        layer_scaling_factor (float): Scaling factor for the number of neurons in hidden layers.
        layer_schedule (Literal["pyramid", "linear"]): Schedule for scaling hidden layer sizes.
    """

    latent_init: Literal["random", "pca"] = "random"
    latent_dim: int = 2
    dropout_rate: float = 0.2
    num_hidden_layers: int = 2
    activation: Literal["relu", "elu", "selu", "leaky_relu"] = "relu"
    layer_scaling_factor: float = 5.0
    layer_schedule: Literal["pyramid", "linear"] = "pyramid"


@dataclass
class TrainConfig:
    """Training procedure configuration.

    Attributes:
        batch_size (int): Number of samples per training batch.
        learning_rate (float): Learning rate for the optimizer.
        l1_penalty (float): L1 regularization penalty.
        early_stop_gen (int): Number of generations with no improvement to wait before early stopping.
        min_epochs (int): Minimum number of epochs to train.
        max_epochs (int): Maximum number of epochs to train.
        validation_split (float): Proportion of data to use for validation.
        weights_max_ratio (float | None): Maximum ratio for class weights to prevent extreme values.
        gamma (float): Focusing parameter for focal loss.
        device (Literal["gpu", "cpu", "mps"]): Device to use for computation.
    """

    batch_size: int = 64
    learning_rate: float = 1e-3
    l1_penalty: float = 0.0
    early_stop_gen: int = 25
    min_epochs: int = 100
    max_epochs: int = 2000
    validation_split: float = 0.2
    device: Literal["gpu", "cpu", "mps"] = "cpu"
    weights_max_ratio: Optional[float] = None
    weights_power: float = 1.0
    weights_normalize: bool = True
    weights_inverse: bool = False
    gamma: float = 0.0
    gamma_schedule: bool = False


def _default_train_config() -> TrainConfig:
    """Typed default factory for TrainConfig (helps some type checkers).

    Using the class object directly (default_factory=TrainConfig) is valid at runtime but certain type checkers can fail to match dataclasses.field overloads.
    """

    return TrainConfig()


@dataclass
class TuneConfig:
    """Hyperparameter tuning configuration.

    Attributes:
        enabled (bool): If True, enables hyperparameter tuning.
        metric (Literal["f1", "accuracy", "pr_macro", "average_precision", "roc_auc", "precision", "recall", "mcc", "jaccard"]): Metric to optimize during tuning.
        n_trials (int): Number of hyperparameter trials to run.
        resume (bool): If True, resumes tuning from a previous state.
        save_db (bool): If True, saves the tuning results to a database.
        epochs (int): Number of epochs to train each trial.
        batch_size (int): Batch size for training during tuning.
        patience (int): Number of evaluations with no improvement before stopping early.
    """

    enabled: bool = False
    metric: Literal[
        "f1",
        "accuracy",
        "pr_macro",
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "mcc",
        "jaccard",
    ] = "f1"
    n_trials: int = 100
    resume: bool = False
    save_db: bool = False
    epochs: int = 500
    batch_size: int = 64
    patience: int = 10


@dataclass
class PlotConfig:
    """Plotting configuration.

    Attributes:
        fmt (Literal["pdf", "png", "jpg", "jpeg", "svg"]): Output file format.
        dpi (int): Dots per inch for the output figure.
        fontsize (int): Font size for text in the plots.
        despine (bool): If True, removes the top and right spines from plots.
        show (bool): If True, displays the plot interactively.
    """

    fmt: Literal["pdf", "png", "jpg", "jpeg", "svg"] = "pdf"
    dpi: int = 300
    fontsize: int = 18
    despine: bool = True
    show: bool = True


@dataclass
class IOConfig:
    """I/O configuration.

    Dataclass that includes configuration settings for file naming, logging verbosity, random seed, and parallelism.

    Attributes:
        prefix (str): Prefix for output files. Default is "pgsui".
        ploidy (int): Ploidy level of the organism. Default is 2.
        verbose (bool): If True, enables verbose logging. Default is False.
        debug (bool): If True, enables debug mode. Default is False.
        seed (int | None): Random seed for reproducibility. Default is None.
        n_jobs (int): Number of parallel jobs to run. Default is 1.
        scoring_averaging (Literal["macro", "micro", "weighted"]): Averaging method.
    """

    prefix: str = "pgsui"
    ploidy: int = 2
    verbose: bool = False
    debug: bool = False
    seed: int | None = None
    n_jobs: int = 1
    scoring_averaging: Literal["macro", "micro", "weighted"] = "macro"


@dataclass
class SimConfig:
    """Top-level configuration for data simulation and imputation.

    Attributes:
        simulate_missing (bool): If True, simulates missing data.
        sim_strategy (Literal["random", ...]): Strategy for simulating missing data.
        sim_prop (float): Proportion of data to simulate as missing.
        sim_kwargs (dict | None): Additional keyword arguments for simulation.
    """

    simulate_missing: bool = False
    sim_strategy: Literal[
        "random",
        "random_weighted",
        "random_weighted_inv",
        "nonrandom",
        "nonrandom_weighted",
    ] = "random"
    sim_prop: float = 0.20
    sim_kwargs: dict | None = None


@dataclass
class AutoencoderConfig:
    """Top-level configuration for ImputeAutoencoder.

    This configuration class encapsulates all settings required for the
    ImputeAutoencoder model, including I/O, model architecture, training,
    hyperparameter tuning, plotting, and simulated-missing configuration.

    Attributes:
        io (IOConfig): I/O configuration.
        model (ModelConfig): Model architecture configuration.
        train (TrainConfig): Training procedure configuration.
        tune (TuneConfig): Hyperparameter tuning configuration.
        plot (PlotConfig): Plotting configuration.
        sim (SimConfig): Simulated-missing configuration.
    """

    io: IOConfig = field(default_factory=IOConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=_default_train_config)
    tune: TuneConfig = field(default_factory=TuneConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    sim: SimConfig = field(default_factory=SimConfig)

    @classmethod
    def from_preset(
        cls, preset: Literal["fast", "balanced", "thorough"] = "balanced"
    ) -> "AutoencoderConfig":
        """Build a AutoencoderConfig from a named preset.

        Args:
            preset (Literal["fast", "balanced", "thorough"]): Preset name.

        Returns:
            AutoencoderConfig: Configuration instance corresponding to the preset.
        """
        if preset not in {"fast", "balanced", "thorough"}:
            raise ValueError(f"Unknown preset: {preset}")

        cfg = cls()

        # Common baselines
        cfg.io.verbose = False
        cfg.io.ploidy = 2
        cfg.train.validation_split = 0.2
        cfg.model.activation = "relu"
        cfg.model.layer_schedule = "pyramid"
        cfg.model.layer_scaling_factor = 2.0
        cfg.sim.sim_strategy = "random"
        cfg.sim.sim_prop = 0.2
        cfg.plot.show = True

        # Train settings
        cfg.train.weights_max_ratio = None
        cfg.train.weights_power = 1.0
        cfg.train.weights_normalize = True
        cfg.train.weights_inverse = False
        cfg.train.gamma = 0.0
        cfg.train.gamma_schedule = False
        cfg.train.min_epochs = 100

        # Tune
        cfg.tune.enabled = False
        cfg.tune.n_trials = 100

        if preset == "fast":
            # Model
            cfg.model.latent_dim = 4
            cfg.model.num_hidden_layers = 1
            cfg.model.dropout_rate = 0.10

            # Train
            cfg.train.batch_size = 128
            cfg.train.learning_rate = 2e-3
            cfg.train.early_stop_gen = 15
            cfg.train.max_epochs = 200
            cfg.train.weights_max_ratio = None

            # Tune
            cfg.tune.patience = 15

        elif preset == "balanced":
            # Model
            cfg.model.latent_dim = 8
            cfg.model.num_hidden_layers = 2
            cfg.model.dropout_rate = 0.20

            # Train
            cfg.train.batch_size = 64
            cfg.train.learning_rate = 1e-3
            cfg.train.early_stop_gen = 25
            cfg.train.max_epochs = 500
            cfg.train.weights_max_ratio = None

            # Tune
            cfg.tune.patience = 25

        else:  # thorough
            # Model
            cfg.model.latent_dim = 16
            cfg.model.num_hidden_layers = 3
            cfg.model.dropout_rate = 0.30

            # Train
            cfg.train.batch_size = 64
            cfg.train.learning_rate = 5e-4
            cfg.train.early_stop_gen = 50
            cfg.train.max_epochs = 1000
            cfg.train.weights_max_ratio = None

            # Tune
            cfg.tune.patience = 50

        return cfg

    def apply_overrides(self, overrides: Dict[str, Any] | None) -> "AutoencoderConfig":
        """Apply flat dot-key overrides.

        Args:
            overrides (Dict[str, Any] | None): Dictionary of overrides with dot-separated keys.

        Returns:
            AutoencoderConfig: New configuration instance with overrides applied.
        """
        if not overrides:
            return self
        for k, v in overrides.items():
            node = self
            parts = k.split(".")
            for p in parts[:-1]:
                node = getattr(node, p)
            last = parts[-1]
            if hasattr(node, last):
                setattr(node, last, v)
            else:
                raise KeyError(f"Unknown config key: {k}")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VAEExtraConfig:
    kl_beta: float = 1.0
    kl_beta_schedule: bool = False


@dataclass
class VAEConfig:
    """Top-level configuration for ImputeVAE (AE-parity + VAE extras).

    Mirrors AutoencoderConfig sections and adds a ``vae`` block with KL-beta
    controls for the VAE loss.

    Attributes:
        io (IOConfig): I/O configuration.
        model (ModelConfig): Model architecture configuration.
        train (TrainConfig): Training procedure configuration.
        tune (TuneConfig): Hyperparameter tuning configuration.
        plot (PlotConfig): Plotting configuration.
        vae (VAEExtraConfig): VAE-specific configuration.
        sim (SimConfig): Simulated-missing configuration.
    """

    io: IOConfig = field(default_factory=IOConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=_default_train_config)
    tune: TuneConfig = field(default_factory=TuneConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    vae: VAEExtraConfig = field(default_factory=VAEExtraConfig)
    sim: SimConfig = field(default_factory=SimConfig)

    @classmethod
    def from_preset(
        cls, preset: Literal["fast", "balanced", "thorough"] = "balanced"
    ) -> "VAEConfig":
        """Build a VAEConfig from a named preset.

        Args:
            preset (Literal["fast", "balanced", "thorough"]): Preset name.

        Returns:
            VAEConfig: Configuration instance corresponding to the preset.
        """
        if preset not in {"fast", "balanced", "thorough"}:
            raise ValueError(f"Unknown preset: {preset}")

        cfg = cls()

        # General settings
        cfg.io.verbose = False
        cfg.io.ploidy = 2
        cfg.train.validation_split = 0.2
        cfg.model.activation = "relu"
        cfg.model.layer_schedule = "pyramid"
        cfg.model.layer_scaling_factor = 2.0
        cfg.sim.simulate_missing = True
        cfg.sim.sim_strategy = "random"
        cfg.sim.sim_prop = 0.2
        cfg.plot.show = True

        # Train settings
        cfg.train.weights_max_ratio = None
        cfg.train.weights_power = 1.0
        cfg.train.weights_normalize = True
        cfg.train.weights_inverse = False
        cfg.train.gamma = 0.0
        cfg.train.gamma_schedule = False
        cfg.train.min_epochs = 100

        # VAE-specific
        cfg.vae.kl_beta = 1.0
        cfg.vae.kl_beta_schedule = False

        # Tune
        cfg.tune.enabled = False
        cfg.tune.n_trials = 100

        if preset == "fast":
            # Model
            cfg.model.latent_dim = 4
            cfg.model.num_hidden_layers = 2
            cfg.model.dropout_rate = 0.10

            # Train
            cfg.train.batch_size = 128
            cfg.train.learning_rate = 2e-3
            cfg.train.early_stop_gen = 15
            cfg.train.max_epochs = 200

            # Tune
            cfg.tune.patience = 15

        elif preset == "balanced":
            # Model
            cfg.model.latent_dim = 8
            cfg.model.num_hidden_layers = 4
            cfg.model.dropout_rate = 0.20

            # Train
            cfg.train.batch_size = 64
            cfg.train.learning_rate = 1e-3
            cfg.train.early_stop_gen = 25
            cfg.train.max_epochs = 500

            # Tune
            cfg.tune.patience = 25

        else:  # thorough
            # Model
            cfg.model.latent_dim = 16
            cfg.model.num_hidden_layers = 8
            cfg.model.dropout_rate = 0.30

            # Train
            cfg.train.batch_size = 64
            cfg.train.learning_rate = 5e-4
            cfg.train.early_stop_gen = 50
            cfg.train.max_epochs = 1000

            # Tune
            cfg.tune.patience = 50

        return cfg

    def apply_overrides(self, overrides: Dict[str, Any] | None) -> "VAEConfig":
        """Apply flat dot-key overrides."""
        if not overrides:
            return self
        for k, v in overrides.items():
            node = self
            parts = k.split(".")
            for p in parts[:-1]:
                node = getattr(node, p)
            last = parts[-1]
            if hasattr(node, last):
                setattr(node, last, v)
            else:
                raise KeyError(f"Unknown config key: {k}")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MostFrequentAlgoConfig:
    """Algorithmic knobs for ImputeMostFrequent.

    Attributes:
        by_populations (bool): Whether to compute per-population modes. Default is False.
        default (int): Fallback mode if no valid entries in a locus. Default is 0.
        missing (int): Code for missing genotypes in 0/1/2. Default is -1.
    """

    by_populations: bool = False
    default: int = 0
    missing: int = -1


@dataclass
class DeterministicSplitConfig:
    """Evaluation split configuration shared by deterministic imputers.

    Attributes:
        test_size (float): Proportion of data to use as the test set. Default is 0.2.
        test_indices (Optional[Sequence[int]]): Specific indices to use as the test set. Default is None.
    """

    test_size: float = 0.2
    test_indices: Optional[Sequence[int]] = None


@dataclass
class MostFrequentConfig:
    """Top-level configuration for ImputeMostFrequent.

    Deterministic imputers primarily use ``io``, ``plot``, ``split``, ``algo``,
    and ``sim``. The ``train`` and ``tune`` sections are retained for schema
    parity with NN models but are not currently used by ImputeMostFrequent.

    Attributes:
        io (IOConfig): I/O configuration.
        plot (PlotConfig): Plotting configuration.
        split (DeterministicSplitConfig): Data splitting configuration.
        algo (MostFrequentAlgoConfig): Algorithmic configuration.
        sim (SimConfig): Simulation configuration.
        tune (TuneConfig): Hyperparameter tuning configuration.
        train (TrainConfig): Training configuration.
    """

    io: IOConfig = field(default_factory=IOConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    split: DeterministicSplitConfig = field(default_factory=DeterministicSplitConfig)
    algo: MostFrequentAlgoConfig = field(default_factory=MostFrequentAlgoConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    tune: TuneConfig = field(default_factory=TuneConfig)
    train: TrainConfig = field(default_factory=_default_train_config)

    @classmethod
    def from_preset(
        cls,
        preset: Literal["fast", "balanced", "thorough"] = "balanced",
    ) -> "MostFrequentConfig":
        """Construct a preset configuration.

        Args:
            preset (Literal["fast", "balanced", "thorough"]): Preset name.

        Returns:
            MostFrequentConfig: Configuration instance corresponding to the preset.
        """
        if preset not in {"fast", "balanced", "thorough"}:
            raise ValueError(f"Unknown preset: {preset}")

        cfg = cls()
        cfg.io.verbose = False
        cfg.io.ploidy = 2
        cfg.split.test_size = 0.2
        cfg.sim.simulate_missing = True
        cfg.sim.sim_strategy = "random"
        cfg.sim.sim_prop = 0.2

        return cfg

    def apply_overrides(self, overrides: Dict[str, Any] | None) -> "MostFrequentConfig":
        """Apply dot-key overrides."""
        if not overrides:
            return self
        for k, v in overrides.items():
            node = self
            parts = k.split(".")
            for p in parts[:-1]:
                node = getattr(node, p)
            last = parts[-1]
            if hasattr(node, last):
                setattr(node, last, v)
            else:
                pass
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RefAlleleAlgoConfig:
    """Algorithmic knobs for ImputeRefAllele.

    Attributes:
        missing (int): Code for missing genotypes in 0/1/2.
    """

    missing: int = -1


@dataclass
class RefAlleleConfig:
    """Top-level configuration for ImputeRefAllele.

    Deterministic imputers primarily use ``io``, ``plot``, ``split``, ``algo``,
    and ``sim``. The ``train`` and ``tune`` sections are retained for schema
    parity with NN models but are not currently used by ImputeRefAllele.

    Attributes:
        io (IOConfig): I/O configuration.
        plot (PlotConfig): Plotting configuration.
        split (DeterministicSplitConfig): Data splitting configuration.
        algo (RefAlleleAlgoConfig): Algorithmic configuration.
        sim (SimConfig): Simulation configuration.
        tune (TuneConfig): Hyperparameter tuning configuration.
        train (TrainConfig): Training configuration.
    """

    io: IOConfig = field(default_factory=IOConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    split: DeterministicSplitConfig = field(default_factory=DeterministicSplitConfig)
    algo: RefAlleleAlgoConfig = field(default_factory=RefAlleleAlgoConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    tune: TuneConfig = field(default_factory=TuneConfig)
    train: TrainConfig = field(default_factory=_default_train_config)

    @classmethod
    def from_preset(
        cls, preset: Literal["fast", "balanced", "thorough"] = "balanced"
    ) -> "RefAlleleConfig":
        """Presets mainly keep parity with logging/IO and split test_size.

        Args:
            preset (Literal["fast", "balanced", "thorough"]): Preset name.

        Returns:
            RefAlleleConfig: Configuration instance corresponding to the preset.
        """
        if preset not in {"fast", "balanced", "thorough"}:
            raise ValueError(f"Unknown preset: {preset}")

        cfg = cls()
        cfg.io.verbose = False
        cfg.io.ploidy = 2
        cfg.split.test_size = 0.2
        cfg.sim.simulate_missing = True
        cfg.sim.sim_strategy = "random"
        cfg.sim.sim_prop = 0.2
        return cfg

    def apply_overrides(self, overrides: Dict[str, Any] | None) -> "RefAlleleConfig":
        """Apply dot-key overrides."""
        if not overrides:
            return self
        for k, v in overrides.items():
            node = self
            parts = k.split(".")
            for p in parts[:-1]:
                node = getattr(node, p)
            last = parts[-1]
            if hasattr(node, last):
                setattr(node, last, v)
            else:
                pass
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _flatten_dict(
    d: Dict[str, Any], prefix: str = "", out: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Flatten a nested dictionary into dot-key format."""
    out = out or {}
    for k, v in d.items():
        kk = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten_dict(v, kk, out)
        else:
            out[kk] = v
    return out


@dataclass
class IOConfigSupervised:
    """I/O, logging, and run identity.

    Attributes:
        prefix (str): Prefix for output files and logs.
        seed (Optional[int]): Random seed for reproducibility.
        n_jobs (int): Number of parallel jobs to use.
        verbose (bool): Whether to enable verbose logging.
        debug (bool): Whether to enable debug mode.
    """

    prefix: str = "pgsui"
    seed: Optional[int] = None
    n_jobs: int = 1
    verbose: bool = False
    debug: bool = False


@dataclass
class PlotConfigSupervised:
    """Plot/figure styling.

    Attributes:
        fmt (Literal["pdf", "png", "jpg", "jpeg"]): File format.
        dpi (int): Resolution in dots per inch.
        fontsize (int): Base font size for plot text.
        despine (bool): Whether to remove top/right spines.
        show (bool): Whether to display plots interactively.
    """

    fmt: Literal["pdf", "png", "jpg", "jpeg"] = "pdf"
    dpi: int = 300
    fontsize: int = 18
    despine: bool = True
    show: bool = False


@dataclass
class TrainConfigSupervised:
    """Training/evaluation split (by samples).

    Attributes:
        validation_split (float): Proportion of data to use for validation.
    """

    validation_split: float = 0.20

    def __post_init__(self):
        if not (0.0 < self.validation_split < 1.0):
            raise ValueError("validation_split must be between 0.0 and 1.0")


@dataclass
class ImputerConfigSupervised:
    """IterativeImputer-like scaffolding used by current supervised wrappers.

    Attributes:
        n_nearest_features (Optional[int]): Number of nearest features to use.
        max_iter (int): Maximum number of imputation iterations to perform.
    """

    n_nearest_features: Optional[int] = 10
    max_iter: int = 10


@dataclass
class SimConfigSupervised:
    """Simulation of missingness for evaluation.

    Attributes:
        prop_missing (float): Proportion of features to set as missing.
        strategy (Literal["random", "random_inv_genotype"]): Strategy.
        het_boost (float): Boosting factor for heterogeneity.
        missing_val (int): Internal code for missing genotypes.
    """

    prop_missing: float = 0.5
    strategy: Literal["random", "random_inv_genotype"] = "random_inv_genotype"
    het_boost: float = 2.0
    missing_val: int = -1


@dataclass
class TuningConfigSupervised:
    """Optuna tuning envelope."""

    enabled: bool = True
    n_trials: int = 100
    metric: str = "pr_macro"
    n_jobs: int = 8
    fast: bool = True


@dataclass
class RFModelConfig:
    """Random Forest hyperparameters.

    Attributes:
        n_estimators (int): Number of trees in the forest.
        max_depth (Optional[int]): Maximum depth of the trees.
        min_samples_split (int): Minimum number of samples required to split.
        min_samples_leaf (int): Minimum number of samples required at a leaf.
        max_features (Literal["sqrt", "log2"] | float | None): Features to consider.
        criterion (Literal["gini", "entropy", "log_loss"]): Split quality metric.
        class_weight (Literal["balanced", "balanced_subsample", None]): Class weights.
    """

    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Literal["sqrt", "log2"] | float | None = "sqrt"
    criterion: Literal["gini", "entropy", "log_loss"] = "gini"
    class_weight: Literal["balanced", "balanced_subsample", None] = "balanced"


@dataclass
class HGBModelConfig:
    """Histogram-based Gradient Boosting hyperparameters.

    Attributes:
        n_estimators (int): Number of boosting iterations (max_iter).
        learning_rate (float): Step size for each boosting iteration.
        max_depth (Optional[int]): Maximum depth of each tree.
        min_samples_leaf (int): Minimum number of samples required at a leaf.
        max_features (float | None): Proportion of features to consider.
        n_iter_no_change (int): Iterations to wait for early stopping.
        tol (float): Minimum improvement in the loss.
    """

    n_estimators: int = 100  # maps to max_iter
    learning_rate: float = 0.1
    max_depth: Optional[int] = None
    min_samples_leaf: int = 1
    max_features: float | None = 1.0
    n_iter_no_change: int = 10
    tol: float = 1e-7

    def __post_init__(self) -> None:
        if isinstance(self.max_features, float):
            if not (0.0 < self.max_features <= 1.0):
                raise ValueError("max_features as float must be in (0.0, 1.0]")

        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer")


@dataclass
class RFConfig:
    """Configuration for ImputeRandomForest.

    Attributes:
        io (IOConfigSupervised): Run identity, logging, and seeds.
        model (RFModelConfig): RandomForest hyperparameters.
        train (TrainConfigSupervised): Sample split for validation.
        imputer (ImputerConfigSupervised): IterativeImputer scaffolding.
        sim (SimConfigSupervised): Simulated missingness.
        plot (PlotConfigSupervised): Plot styling.
        tune (TuningConfigSupervised): Optuna knobs.
    """

    io: IOConfigSupervised = field(default_factory=IOConfigSupervised)
    model: RFModelConfig = field(default_factory=RFModelConfig)
    train: TrainConfigSupervised = field(default_factory=TrainConfigSupervised)
    imputer: ImputerConfigSupervised = field(default_factory=ImputerConfigSupervised)
    sim: SimConfigSupervised = field(default_factory=SimConfigSupervised)
    plot: PlotConfigSupervised = field(default_factory=PlotConfigSupervised)
    tune: TuningConfigSupervised = field(default_factory=TuningConfigSupervised)

    @classmethod
    def from_preset(cls, preset: str = "balanced") -> "RFConfig":
        """Build a config from a named preset.

        Args:
            preset (str): Preset name.

        Returns:
            RFConfig: Configuration instance corresponding to the preset.
        """
        cfg = cls()
        if preset == "fast":
            cfg.model.n_estimators = 50
            cfg.model.max_depth = None
            cfg.imputer.max_iter = 5
            cfg.io.n_jobs = 1
            cfg.tune.enabled = False
        elif preset == "balanced":
            cfg.model.n_estimators = 200
            cfg.model.max_depth = None
            cfg.imputer.max_iter = 10
            cfg.io.n_jobs = 1
            cfg.tune.enabled = False
            cfg.tune.n_trials = 100
        elif preset == "thorough":
            cfg.model.n_estimators = 500
            cfg.model.max_depth = 50  # Added safety cap
            cfg.imputer.max_iter = 20
            cfg.io.n_jobs = 1
            cfg.tune.enabled = False
            cfg.tune.n_trials = 250
        else:
            raise ValueError(f"Unknown preset: {preset}")

        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "RFConfig":
        """Load from YAML; honors optional top-level 'preset'."""
        return load_yaml_to_dataclass(path, cls)

    def apply_overrides(self, overrides: Dict[str, Any] | None) -> "RFConfig":
        """Apply flat dot-key overrides."""
        if overrides:
            apply_dot_overrides(self, overrides)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_imputer_kwargs(self) -> Dict[str, Any]:
        return {
            "prefix": self.io.prefix,
            "seed": self.io.seed,
            "n_jobs": self.io.n_jobs,
            "verbose": self.io.verbose,
            "debug": self.io.debug,
            "model_n_estimators": self.model.n_estimators,
            "model_max_depth": self.model.max_depth,
            "model_min_samples_split": self.model.min_samples_split,
            "model_min_samples_leaf": self.model.min_samples_leaf,
            "model_max_features": self.model.max_features,
            "model_criterion": self.model.criterion,
            "model_validation_split": self.train.validation_split,
            "model_n_nearest_features": self.imputer.n_nearest_features,
            "model_max_iter": self.imputer.max_iter,
            "sim_prop_missing": self.sim.prop_missing,
            "sim_strategy": self.sim.strategy,
            "sim_het_boost": self.sim.het_boost,
            "plot_format": self.plot.fmt,
            "plot_fontsize": self.plot.fontsize,
            "plot_despine": self.plot.despine,
            "plot_dpi": self.plot.dpi,
            "plot_show_plots": self.plot.show,
        }


@dataclass
class HGBConfig:
    """Configuration for ImputeHistGradientBoosting.

    Attributes:
        io (IOConfigSupervised): Run identity, logging, and seeds.
        model (HGBModelConfig): HistGradientBoosting hyperparameters.
        train (TrainConfigSupervised): Sample split for validation.
        imputer (ImputerConfigSupervised): IterativeImputer scaffolding.
        sim (SimConfigSupervised): Simulated missingness.
        plot (PlotConfigSupervised): Plot styling.
        tune (TuningConfigSupervised): Optuna knobs.
    """

    io: IOConfigSupervised = field(default_factory=IOConfigSupervised)
    model: HGBModelConfig = field(default_factory=HGBModelConfig)
    train: TrainConfigSupervised = field(default_factory=TrainConfigSupervised)
    imputer: ImputerConfigSupervised = field(default_factory=ImputerConfigSupervised)
    sim: SimConfigSupervised = field(default_factory=SimConfigSupervised)
    plot: PlotConfigSupervised = field(default_factory=PlotConfigSupervised)
    tune: TuningConfigSupervised = field(default_factory=TuningConfigSupervised)

    @classmethod
    def from_preset(cls, preset: str = "balanced") -> "HGBConfig":
        """Build a config from a named preset.

        Args:
            preset (str): Preset name.

        Returns:
            HGBConfig: Configuration instance corresponding to the preset.
        """
        cfg = cls()
        if preset == "fast":
            cfg.model.n_estimators = 50
            cfg.model.learning_rate = 0.2
            cfg.model.max_depth = None
            cfg.imputer.max_iter = 5
            cfg.io.n_jobs = 1
            cfg.tune.enabled = False
            cfg.tune.n_trials = 50
        elif preset == "balanced":
            cfg.model.n_estimators = 150
            cfg.model.learning_rate = 0.1
            cfg.model.max_depth = None
            cfg.imputer.max_iter = 10
            cfg.io.n_jobs = 1
            cfg.tune.enabled = False
            cfg.tune.n_trials = 100
        elif preset == "thorough":
            cfg.model.n_estimators = 500
            cfg.model.learning_rate = 0.05
            cfg.model.n_iter_no_change = 20  # Increased patience
            cfg.model.max_depth = None
            cfg.imputer.max_iter = 20
            cfg.io.n_jobs = 1
            cfg.tune.enabled = False
            cfg.tune.n_trials = 250
        else:
            raise ValueError(f"Unknown preset: {preset}")
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "HGBConfig":
        return load_yaml_to_dataclass(path, cls)

    def apply_overrides(self, overrides: Dict[str, Any] | None) -> "HGBConfig":
        if overrides:
            apply_dot_overrides(self, overrides)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_imputer_kwargs(self) -> Dict[str, Any]:
        return {
            "prefix": self.io.prefix,
            "seed": self.io.seed,
            "n_jobs": self.io.n_jobs,
            "verbose": self.io.verbose,
            "debug": self.io.debug,
            "model_n_estimators": self.model.n_estimators,
            "model_learning_rate": self.model.learning_rate,
            "model_n_iter_no_change": self.model.n_iter_no_change,
            "model_tol": self.model.tol,
            "model_max_depth": self.model.max_depth,
            "model_min_samples_leaf": self.model.min_samples_leaf,
            "model_max_features": self.model.max_features,
            "model_validation_split": self.train.validation_split,
            "model_n_nearest_features": self.imputer.n_nearest_features,
            "model_max_iter": self.imputer.max_iter,
            "sim_prop_missing": self.sim.prop_missing,
            "sim_strategy": self.sim.strategy,
            "sim_het_boost": self.sim.het_boost,
            "plot_format": self.plot.fmt,
            "plot_fontsize": self.plot.fontsize,
            "plot_despine": self.plot.despine,
            "plot_dpi": self.plot.dpi,
            "plot_show_plots": self.plot.show,
        }
