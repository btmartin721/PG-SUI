from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional, Sequence


@dataclass
class _SimParams:
    """Container for simulation hyperparameters.

    This class holds the hyperparameters for the simulation process, including the proportion of missing values, the imputation strategy, and other relevant settings.

    Attributes:
        prop_missing (float): Proportion of missing values to simulate.
        strategy (Literal["random", "random_inv_genotype"]): Strategy for simulating missing values.
        missing_val (int | float): Value to represent missing data.
        het_boost (float): Boost factor for heterozygous genotypes.
        seed (int | None): Random seed for reproducibility.

    Notes:
        - The `strategy` attribute determines how missing values are simulated.
            "random" selects missing values uniformly at random, while "random_inv_genotype" selects missing values based on the inverse of the genotype distribution.
    """

    prop_missing: float = 0.3
    strategy: Literal["random", "random_inv_genotype"] = "random_inv_genotype"
    missing_val: int | float = -1
    het_boost: float = 2.0
    seed: int | None = None

    def to_dict(self) -> dict:
        """Convert the simulation parameters to a dictionary.

        Uses `asdict` from the `dataclasses` module to convert the dataclass instance into a dictionary.

        Returns:
            dict: A dictionary representation of the simulation parameters.
        """
        return asdict(self)


@dataclass
class _ImputerParams:
    """Container for imputer hyperparameters.

    This class holds the hyperparameters for the imputation process, including the number of nearest features to consider, the maximum number of iterations, and other relevant settings.

    Attributes:
        n_nearest_features (int | None): Number of nearest features to consider for imputation
        max_iter (int): Maximum number of iterations for the imputation algorithm.
        initial_strategy (Literal["mean", "median", "most_frequent", "constant"]): Strategy for initial imputation of missing values.
        keep_empty_features (bool): Whether to keep features that are entirely missing.
        random_state (int | None): Random seed for reproducibility.
        verbose (bool): If True, enables verbose logging during imputation.

    Notes:
        - The `initial_strategy` attribute determines how initial missing values are imputed before the iterative process begins.
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
        """Convert the imputer parameters to a dictionary.

        Uses `asdict` from the `dataclasses` module to convert the dataclass instance into a dictionary.

        Returns:
            dict: A dictionary representation of the imputer parameters.
        """

        return asdict(self)


@dataclass
class _RFParams:
    """Container for RandomForest hyperparameters.

    This class holds the hyperparameters for the RandomForest classifier, including the number of estimators, maximum depth, and other relevant settings.

    Attributes:
        n_estimators (int): Number of trees in the forest.
        max_depth (int | None): Maximum depth of the trees.
        min_samples_split (int): Minimum number of samples required to split an internal node.
        min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
        max_features (Literal["sqrt", "log2"] | float | None): Number of
            features to consider when looking for the best split.
        criterion (Literal["gini", "entropy", "log_loss"]): Function to measure
            the quality of a split.
        class_weight (Literal["balanced", "balanced_subsample", None]): Weights
            associated with classes.
    """

    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Literal["sqrt", "log2"] | float | None = "sqrt"
    criterion: Literal["gini", "entropy", "log_loss"] = "gini"
    class_weight: Literal["balanced", "balanced_subsample", None] = "balanced"

    def to_dict(self) -> dict:
        """Convert the RandomForest parameters to a dictionary.

        Uses `asdict` from the `dataclasses` module to convert the dataclass instance into a dictionary.

        Returns:
            dict: A dictionary representation of the RandomForest parameters.
        """
        return asdict(self)


@dataclass
class _HGBParams:
    """Container for HistGradientBoosting hyperparameters.

    This class holds the hyperparameters for the HistGradientBoosting classifier, including the number of iterations, learning rate, and other relevant settings.

    Attributes:
        max_iter (int): Maximum number of iterations.
        learning_rate (float): Learning rate shrinks the contribution of each tree.
        max_depth (int | None): Maximum depth of the individual regression estimators.
        min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
        n_iter_no_change (int): Number of iterations with no improvement to wait before early stopping
        tol (float): Tolerance for the early stopping.
        max_features (float): The fraction of features to consider when looking for the best split.
        class_weight (Literal["balanced", "balanced_subsample", None]): Weights associated with classes.
        random_state (int | None): Random seed for reproducibility.
        verbose (bool): If True, enables verbose logging during training.

    Notes:
        - The `class_weight` attribute helps to handle class imbalance by adjusting the weights associated with classes.
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
        """Convert the HistGradientBoosting parameters to a dictionary.

        Uses `asdict` from the `dataclasses` module to convert the dataclass instance into a dictionary.

        Returns:
            dict: A dictionary representation of the HistGradientBoosting parameters.
        """
        return asdict(self)


@dataclass
class ModelConfig:
    """Model architecture configuration.

    Attributes:
        latent_init (Literal["random", "pca"]): Method for initializing the latent space.
        latent_dim (int): Dimensionality of the latent space.
        dropout_rate (float): Dropout rate for regularization.
        num_hidden_layers (int): Number of hidden layers in the neural network.
        hidden_activation (Literal["relu", "elu", "selu", "leaky_relu"]): Activation function for hidden layers.
        layer_scaling_factor (float): Scaling factor for the number of neurons in hidden layers.
        layer_schedule (Literal["pyramid", "constant", "linear"]): Schedule for scaling hidden layer sizes.
        gamma (float): Parameter for the loss function.

    Notes:
        - The `layer_schedule` attribute determines how the size of hidden layers changes across the network (e.g., "pyramid" means decreasing size).
        - The `latent_init` attribute specifies how the latent space is initialized, either randomly or using PCA.
    """

    latent_init: Literal["random", "pca"] = "random"
    latent_dim: int = 2
    dropout_rate: float = 0.2
    num_hidden_layers: int = 2
    hidden_activation: Literal["relu", "elu", "selu", "leaky_relu"] = "relu"
    layer_scaling_factor: float = 5.0
    layer_schedule: Literal["pyramid", "constant", "linear"] = "pyramid"
    gamma: float = 2.0


@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    lr_input_factor: float = 1.0
    l1_penalty: float = 0.0
    early_stop_gen: int = 20
    min_epochs: int = 100
    max_epochs: int = 5000
    validation_split: float = 0.2
    weights_beta: float = 0.9999
    weights_max_ratio: float = 1.0
    device: Literal["gpu", "cpu", "mps"] = "cpu"


@dataclass
class TuneConfig:
    """Hyperparameter tuning configuration.

    Attributes:
        enabled (bool): If True, enables hyperparameter tuning.
        metric (Literal["f1", "accuracy", "pr_macro"]): Metric to optimize
        n_trials (int): Number of hyperparameter trials to run.
        resume (bool): If True, resumes from previous tuning results.
        save_db (bool): If True, saves tuning results to a database.
        fast (bool): If True, uses a faster tuning strategy.
        max_samples (int): Maximum number of samples to use for tuning.
        max_loci (int): Maximum number of loci to use for tuning (0 = all
            loci).
        epochs (int): Number of epochs for each trial.
        batch_size (int): Batch size for training during tuning.
    """

    enabled: bool = False
    metric: Literal["f1", "accuracy", "pr_macro"] = "f1"
    n_trials: int = 100
    resume: bool = False
    save_db: bool = False
    fast: bool = True
    max_samples: int = 512
    max_loci: int = 0  # 0 = all
    epochs: int = 500
    batch_size: int = 64
    eval_interval: int = 1
    infer_epochs: int = 100
    patience: int = 10
    proxy_metric_batch: int = 0


@dataclass
class EvalConfig:
    """Evaluation configuration.

    Attributes:
        batch_size (int): Batch size for evaluation.
        eval_interval (int): Interval (in epochs) for evaluation during training.
        infer_epochs (int): Number of epochs for inference during evaluation.
        patience (int): Patience for early stopping during evaluation.
        proxy_metric_batch (int): Batch size for proxy metric calculation.
        eval_latent_steps (int): Number of optimization steps for latent space evaluation.
        eval_latent_lr (float): Learning rate for latent space optimization.
        eval_latent_weight_decay (float): Weight decay for latent space optimization.
    """

    eval_latent_steps: int = 50
    eval_latent_lr: float = 1e-2
    eval_latent_weight_decay: float = 0.0


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
    show: bool = False


@dataclass
class IOConfig:
    """I/O configuration.

    This class contains configuration options for input/output operations.

    Attributes:
        prefix (str): Prefix for output files. Default is "pgsui".
        verbose (bool): If True, enables verbose logging. Default is False.
        debug (bool): If True, enables debug mode. Default is False.
        seed (int | None): Random seed for reproducibility. Default is None.
        n_jobs (int): Number of parallel jobs to run. Default is 1.
        scoring_averaging (Literal["macro", "micro", "weighted"]): Averaging
            method for scoring metrics. Default is "macro".
    """

    prefix: str = "pgsui"
    verbose: bool = False
    debug: bool = False
    seed: int | None = None
    n_jobs: int = 1
    scoring_averaging: Literal["macro", "micro", "weighted"] = "macro"


@dataclass
class NLPCAConfig:
    """Top-level configuration for ImputeNLPCA.

    This class contains all the configuration options for the ImputeNLPCA model.

    Attributes:
        io (IOConfig): I/O configuration.
        model (ModelConfig): Model architecture configuration.
        train (TrainConfig): Training procedure configuration.
        tune (TuneConfig): Hyperparameter tuning configuration.
        evaluate (EvalConfig): Evaluation configuration.
        plot (PlotConfig): Plotting configuration.

    Notes:
        - Presets: "fast": Prioritizes speed over thoroughness. "balanced": A balance between speed and thoroughness. "thorough": Prioritizes thoroughness over speed.
        - Overrides: Overrides are applied after presets and can be used to fine-tune specific parameters.
    """

    io: IOConfig = field(default_factory=IOConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tune: TuneConfig = field(default_factory=TuneConfig)
    evaluate: EvalConfig = field(default_factory=EvalConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)

    @classmethod
    def from_preset(
        cls, preset: Literal["fast", "balanced", "thorough"] = "balanced"
    ) -> "NLPCAConfig":
        """Build a config from a named preset.

        The presets are intended to be practical, not theoretical:
        - fast: Very quick sanity runs; minimal capacity; NO tuning by default.
        - balanced: Good default for most datasets; moderate tuning, moderate depth.
        - thorough: Maximum quality; deeper nets, longer training, more trials.

        Args:
            preset: One of {"fast", "balanced", "thorough"}.

        Returns:
            NLPCAConfig: Configuration instance with preset values applied.
        """
        if preset not in {"fast", "balanced", "thorough"}:
            raise ValueError(f"Unknown preset: {preset}")

        cfg = cls()  # start from dataclass defaults

        # Common sensible baselines
        cfg.io.verbose = True
        cfg.train.validation_split = 0.2
        cfg.evaluate.eval_latent_steps = 50
        cfg.evaluate.eval_latent_lr = 1e-2
        cfg.evaluate.eval_latent_weight_decay = 0.0
        cfg.model.hidden_activation = "relu"
        cfg.model.layer_schedule = "pyramid"
        cfg.model.latent_init = "random"

        if preset == "fast":
            # Model
            cfg.model.latent_dim = 4
            cfg.model.num_hidden_layers = 1
            cfg.model.layer_scaling_factor = 2.0
            cfg.model.dropout_rate = 0.10
            cfg.model.gamma = 1.5

            # Train
            cfg.train.batch_size = 128
            cfg.train.learning_rate = 1e-3
            cfg.train.early_stop_gen = 5
            cfg.train.min_epochs = 10
            cfg.train.max_epochs = 100
            cfg.train.weights_beta = 0.9999
            cfg.train.weights_max_ratio = 1.0  # no rebalancing pressure

            # Tuning (off for true "fast")
            cfg.tune.enabled = False
            cfg.tune.fast = True
            cfg.tune.n_trials = 50
            cfg.tune.epochs = 100
            cfg.tune.batch_size = 128
            cfg.tune.max_samples = 512  # cap data for speed
            cfg.tune.max_loci = 0
            cfg.tune.eval_interval = 1
            cfg.tune.infer_epochs = 50
            cfg.tune.patience = 5
            cfg.tune.proxy_metric_batch = 0

        elif preset == "balanced":
            # Model
            cfg.model.latent_dim = 8
            cfg.model.num_hidden_layers = 2
            cfg.model.layer_scaling_factor = 4.0
            cfg.model.dropout_rate = 0.20
            cfg.model.gamma = 2.0

            # Train
            cfg.train.batch_size = 128
            cfg.train.learning_rate = 8e-4
            cfg.train.early_stop_gen = 15
            cfg.train.min_epochs = 50
            cfg.train.max_epochs = 1000
            cfg.train.weights_beta = 0.9999
            cfg.train.weights_max_ratio = 1.0

            # Tuning
            cfg.tune.enabled = True
            cfg.tune.fast = True  # favor speed with good coverage
            cfg.tune.n_trials = 100  # more trials
            cfg.tune.epochs = 250
            cfg.tune.batch_size = 128
            cfg.tune.max_samples = 1024
            cfg.tune.max_loci = 0
            cfg.tune.eval_interval = 1
            cfg.tune.infer_epochs = 80
            cfg.tune.patience = 10
            cfg.tune.proxy_metric_batch = 0

        else:  # thorough
            # Model
            cfg.model.latent_dim = 16
            cfg.model.num_hidden_layers = 3
            cfg.model.layer_scaling_factor = 6.0
            cfg.model.dropout_rate = 0.30
            cfg.model.gamma = 2.5

            # Train
            cfg.train.batch_size = 64
            cfg.train.learning_rate = 6e-4
            cfg.train.early_stop_gen = 30
            cfg.train.min_epochs = 100
            cfg.train.max_epochs = 3000
            cfg.train.weights_beta = 0.9999
            cfg.train.weights_max_ratio = 1.0

            # Tuning
            cfg.tune.enabled = True
            cfg.tune.fast = False
            cfg.tune.n_trials = 250
            cfg.tune.epochs = 1000
            cfg.tune.batch_size = 64
            cfg.tune.max_samples = 0  # use all samples
            cfg.tune.max_loci = 0  # use all loci
            cfg.tune.eval_interval = 1
            cfg.tune.infer_epochs = 120
            cfg.tune.patience = 20
            cfg.tune.proxy_metric_batch = 0

        return cfg

    def apply_overrides(self, overrides: Dict[str, Any] | None) -> "NLPCAConfig":
        """Apply flat dot-key overrides (e.g. {'model.latent_dim': 4}).

        Args:
            overrides (Dict[str, Any] | None): A mapping of dot-key paths to values to override.

        Returns:
            NLPCAConfig: The updated config instance (same as `self`).
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
        """Return the config as a nested dictionary.

        This method uses `asdict` from the `dataclasses` module to convert the dataclass instance into a dictionary.

        Returns:
            Dict[str, Any]: The config as a nested dictionary.
        """
        return asdict(self)


@dataclass
class UBPConfig:
    """Top-level configuration for ImputeUBP.

    This class contains all the configuration options for the ImputeUBP model. The configuration is organized into several sections, each represented by a dataclass.

    Attributes:
        io (IOConfig): I/O configuration.
        model (ModelConfig): Model architecture configuration.
        train (TrainConfig): Training procedure configuration.
        tune (TuneConfig): Hyperparameter tuning configuration.
        evaluate (EvalConfig): Evaluation configuration.
        plot (PlotConfig): Plotting configuration.

    Notes:
        - Presets:
            fast:       Prioritizes speed.
            balanced:   Balanced speed vs thoroughness.
            thorough:   Prioritizes thoroughness.
        - Overrides: flat dot-keys like {"model.latent_dim": 8}.
    """

    io: IOConfig = field(default_factory=IOConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tune: TuneConfig = field(default_factory=TuneConfig)
    evaluate: EvalConfig = field(default_factory=EvalConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)

    @classmethod
    def from_preset(
        cls, preset: Literal["fast", "balanced", "thorough"] = "balanced"
    ) -> "UBPConfig":
        """Build a UBPConfig from a named preset.

        UBP is often used when classes (genotype states) are imbalanced. Presets adjust both capacity and weighting behavior across speed/quality tradeoffs:
        - fast:     Quick baseline; tiny net; NO tuning by default.
        - balanced: Practical default; moderate tuning; mild class weighting cap.
        - thorough: Highest quality; deeper nets; extensive tuning; stronger focus.

        Args:
            preset (Literal["fast", "balanced", "thorough"]): One of {"fast","balanced","thorough"}.

        Returns:
            UBPConfig: Populated config instance.

        Raises:
            ValueError: If an unknown preset is provided.
        """
        if preset not in {"fast", "balanced", "thorough"}:
            raise ValueError(f"Unknown preset: {preset}")

        cfg = cls()

        # Shared baselines
        cfg.io.verbose = True
        cfg.model.hidden_activation = "relu"
        cfg.model.layer_schedule = "pyramid"
        cfg.model.latent_init = "random"

        if preset == "fast":
            # Model (slightly smaller than NLPCA fast)
            cfg.model.latent_dim = 3
            cfg.model.num_hidden_layers = 1
            cfg.model.layer_scaling_factor = 2.0
            cfg.model.dropout_rate = 0.10
            cfg.model.gamma = 1.5  # lighter focusing

            # Train
            cfg.train.batch_size = 128
            cfg.train.learning_rate = 1e-3
            cfg.train.early_stop_gen = 5
            cfg.train.min_epochs = 10
            cfg.train.max_epochs = 100
            cfg.train.weights_beta = 0.9999
            cfg.train.weights_max_ratio = 2.0  # allow mild rebalancing

            # Tuning (off for true "fast")
            cfg.tune.enabled = False
            cfg.tune.fast = True
            cfg.tune.n_trials = 50
            cfg.tune.epochs = 100
            cfg.tune.batch_size = 128
            cfg.tune.max_samples = 512
            cfg.tune.max_loci = 0
            cfg.tune.eval_interval = 1
            cfg.tune.infer_epochs = 50
            cfg.tune.patience = 5
            cfg.tune.proxy_metric_batch = 0

        elif preset == "balanced":
            # Model
            cfg.model.latent_dim = 6
            cfg.model.num_hidden_layers = 2
            cfg.model.layer_scaling_factor = 3.0
            cfg.model.dropout_rate = 0.20
            cfg.model.gamma = 2.0

            # Train
            cfg.train.batch_size = 128
            cfg.train.learning_rate = 8e-4
            cfg.train.early_stop_gen = 15
            cfg.train.min_epochs = 50
            cfg.train.max_epochs = 1000
            cfg.train.weights_beta = 0.9999
            cfg.train.weights_max_ratio = 3.0  # moderate cap for imbalance

            # Tuning
            cfg.tune.enabled = True
            cfg.tune.fast = True
            cfg.tune.n_trials = 100
            cfg.tune.epochs = 250
            cfg.tune.batch_size = 128
            cfg.tune.max_samples = 1024
            cfg.tune.max_loci = 0
            cfg.tune.eval_interval = 1
            cfg.tune.infer_epochs = 80
            cfg.tune.patience = 10
            cfg.tune.proxy_metric_batch = 0

        else:  # thorough
            # Model
            cfg.model.latent_dim = 12
            cfg.model.num_hidden_layers = 3
            cfg.model.layer_scaling_factor = 5.0
            cfg.model.dropout_rate = 0.30
            cfg.model.gamma = 2.5  # stronger focusing for harder imbalance

            # Train
            cfg.train.batch_size = 64
            cfg.train.learning_rate = 6e-4
            cfg.train.early_stop_gen = 30
            cfg.train.min_epochs = 100
            cfg.train.max_epochs = 3000
            cfg.train.weights_beta = 0.9999
            cfg.train.weights_max_ratio = 5.0  # allow stronger class weighting

            # Tuning
            cfg.tune.enabled = True
            cfg.tune.fast = False
            cfg.tune.n_trials = 250
            cfg.tune.epochs = 1000
            cfg.tune.batch_size = 64
            cfg.tune.max_samples = 0  # all samples
            cfg.tune.max_loci = 0  # all loci
            cfg.tune.eval_interval = 1
            cfg.tune.infer_epochs = 120
            cfg.tune.patience = 20
            cfg.tune.proxy_metric_batch = 0

        return cfg

    def apply_overrides(self, overrides: Dict[str, Any] | None) -> "UBPConfig":
        """Apply flat dot-key overrides (e.g. {'model.latent_dim': 4}).

        Args:
            overrides: Mapping of dot-key paths to values to override.

        Returns:
            UBPConfig: This instance after applying overrides.
        """
        if overrides is None or not overrides:
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
        """Return the config as a nested dictionary.

        Returns:
            Dict[str, Any]: Nested dictionary.
        """
        return asdict(self)


@dataclass
class AutoencoderConfig:
    """Top-level configuration for ImputeAutoencoder.

    This class contains all the configuration options for the ImputeAutoencoder model. The configuration is organized into several sections, each represented by a dataclass.

    Attributes:
        io (IOConfig): I/O configuration.
        model (ModelConfig): Model architecture configuration.
        train (TrainConfig): Training procedure configuration.
        tune (TuneConfig): Hyperparameter tuning configuration.
        evaluate (EvalConfig): Evaluation configuration.
        plot (PlotConfig): Plotting configuration.

    Notes:
        - Presets:
            fast:       Prioritizes speed.
            balanced:   Balanced speed vs thoroughness.
            thorough:   Prioritizes thoroughness.
        - Overrides: flat dot-keys like {"model.latent_dim": 8}.
    """

    io: IOConfig = field(default_factory=IOConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tune: TuneConfig = field(default_factory=TuneConfig)
    evaluate: EvalConfig = field(default_factory=EvalConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)

    @classmethod
    def from_preset(
        cls, preset: Literal["fast", "balanced", "thorough"] = "balanced"
    ) -> "AutoencoderConfig":
        if preset not in {"fast", "balanced", "thorough"}:
            raise ValueError(f"Unknown preset: {preset}")

        cfg = cls()

        # Common sensible baselines (aligned with NLPCA)
        cfg.io.verbose = True
        cfg.train.validation_split = 0.2
        cfg.model.hidden_activation = "relu"
        cfg.model.layer_schedule = "pyramid"

        # AE difference: no latent refinement during eval
        cfg.evaluate.eval_latent_steps = 0
        cfg.evaluate.eval_latent_lr = 0.0
        cfg.evaluate.eval_latent_weight_decay = 0.0

        if preset == "fast":
            # Model
            cfg.model.latent_dim = 4
            cfg.model.num_hidden_layers = 1
            cfg.model.layer_scaling_factor = 2.0
            cfg.model.dropout_rate = 0.10
            cfg.model.gamma = 1.5
            # Train
            cfg.train.batch_size = 128
            cfg.train.learning_rate = 1e-3
            cfg.train.early_stop_gen = 5
            cfg.train.min_epochs = 10
            cfg.train.max_epochs = 100
            cfg.train.weights_beta = 0.9999
            cfg.train.weights_max_ratio = 1.0
            # Tuning (off for true fast)
            cfg.tune.enabled = False
            cfg.tune.fast = True
            cfg.tune.n_trials = 50
            cfg.tune.epochs = 100
            cfg.tune.batch_size = 128
            cfg.tune.max_samples = 512
            cfg.tune.max_loci = 0
            cfg.tune.eval_interval = 1
            cfg.tune.patience = 5
            cfg.tune.proxy_metric_batch = 0
            if hasattr(cfg.tune, "infer_epochs"):
                cfg.tune.infer_epochs = 0

        elif preset == "balanced":
            # Model
            cfg.model.latent_dim = 8
            cfg.model.num_hidden_layers = 2
            cfg.model.layer_scaling_factor = 4.0
            cfg.model.dropout_rate = 0.20
            cfg.model.gamma = 2.0
            # Train
            cfg.train.batch_size = 128
            cfg.train.learning_rate = 8e-4
            cfg.train.early_stop_gen = 15
            cfg.train.min_epochs = 50
            cfg.train.max_epochs = 1000
            cfg.train.weights_beta = 0.9999
            cfg.train.weights_max_ratio = 1.0
            # Tuning
            cfg.tune.enabled = True
            cfg.tune.fast = True
            cfg.tune.n_trials = 100
            cfg.tune.epochs = 250
            cfg.tune.batch_size = 128
            cfg.tune.max_samples = 1024
            cfg.tune.max_loci = 0
            cfg.tune.eval_interval = 1
            cfg.tune.patience = 10
            cfg.tune.proxy_metric_batch = 0
            if hasattr(cfg.tune, "infer_epochs"):
                cfg.tune.infer_epochs = 0

        else:  # thorough
            # Model
            cfg.model.latent_dim = 16
            cfg.model.num_hidden_layers = 3
            cfg.model.layer_scaling_factor = 6.0
            cfg.model.dropout_rate = 0.30
            cfg.model.gamma = 2.5
            # Train
            cfg.train.batch_size = 64
            cfg.train.learning_rate = 6e-4
            cfg.train.early_stop_gen = 30
            cfg.train.min_epochs = 100
            cfg.train.max_epochs = 3000
            cfg.train.weights_beta = 0.9999
            cfg.train.weights_max_ratio = 1.0
            # Tuning
            cfg.tune.enabled = True
            cfg.tune.fast = False
            cfg.tune.n_trials = 250
            cfg.tune.epochs = 1000
            cfg.tune.batch_size = 64
            cfg.tune.max_samples = 0  # use all samples
            cfg.tune.max_loci = 0  # use all loci
            cfg.tune.eval_interval = 1
            cfg.tune.patience = 20
            cfg.tune.proxy_metric_batch = 0
            if hasattr(cfg.tune, "infer_epochs"):
                cfg.tune.infer_epochs = 0

        return cfg

    def apply_overrides(self, overrides: Dict[str, Any] | None) -> "AutoencoderConfig":
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
    """VAE-specific knobs."""

    kl_beta: float = 1.0  # final β for KL
    kl_warmup: int = 50  # epochs with β=0
    kl_ramp: int = 200  # linear ramp to β


@dataclass
class VAEConfig:
    """Top-level configuration for ImputeVAE (AE-parity + VAE extras)."""

    io: IOConfig = field(default_factory=IOConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tune: TuneConfig = field(default_factory=TuneConfig)
    evaluate: EvalConfig = field(default_factory=EvalConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    vae: VAEExtraConfig = field(default_factory=VAEExtraConfig)

    @classmethod
    def from_preset(
        cls, preset: Literal["fast", "balanced", "thorough"] = "balanced"
    ) -> "VAEConfig":
        """Mirror AutoencoderConfig presets and add VAE defaults."""
        if preset not in {"fast", "balanced", "thorough"}:
            raise ValueError(f"Unknown preset: {preset}")

        cfg = cls()

        # Common sensible baselines (match AE/NLPCA style)
        cfg.io.verbose = True
        cfg.train.validation_split = 0.2
        cfg.model.hidden_activation = "relu"
        cfg.model.layer_schedule = "pyramid"

        # Like AE, no latent refinement during eval
        cfg.evaluate.eval_latent_steps = 0
        cfg.evaluate.eval_latent_lr = 0.0
        cfg.evaluate.eval_latent_weight_decay = 0.0

        # VAE-specific schedule defaults (can be overridden)
        cfg.vae.kl_beta = 1.0
        cfg.vae.kl_warmup = 50
        cfg.vae.kl_ramp = 200

        if preset == "fast":
            cfg.model.latent_dim = 4
            cfg.model.num_hidden_layers = 1
            cfg.model.layer_scaling_factor = 2.0
            cfg.model.dropout_rate = 0.10
            cfg.model.gamma = 1.5

            cfg.train.batch_size = 128
            cfg.train.learning_rate = 1e-3
            cfg.train.early_stop_gen = 5
            cfg.train.min_epochs = 10
            cfg.train.max_epochs = 100
            cfg.train.weights_beta = 0.9999
            cfg.train.weights_max_ratio = 1.0

            cfg.tune.enabled = False
            cfg.tune.fast = True
            cfg.tune.n_trials = 50
            cfg.tune.epochs = 100
            cfg.tune.batch_size = 128
            cfg.tune.max_samples = 512
            cfg.tune.max_loci = 0
            cfg.tune.eval_interval = 1
            cfg.tune.patience = 5

            if hasattr(cfg.tune, "infer_epochs"):
                cfg.tune.infer_epochs = 0

        elif preset == "balanced":
            cfg.model.latent_dim = 8
            cfg.model.num_hidden_layers = 2
            cfg.model.layer_scaling_factor = 4.0
            cfg.model.dropout_rate = 0.20
            cfg.model.gamma = 2.0

            cfg.train.batch_size = 128
            cfg.train.learning_rate = 8e-4
            cfg.train.early_stop_gen = 15
            cfg.train.min_epochs = 50
            cfg.train.max_epochs = 1000
            cfg.train.weights_beta = 0.9999
            cfg.train.weights_max_ratio = 1.0

            cfg.tune.enabled = True
            cfg.tune.fast = True
            cfg.tune.n_trials = 100
            cfg.tune.epochs = 250
            cfg.tune.batch_size = 128
            cfg.tune.max_samples = 1024
            cfg.tune.max_loci = 0
            cfg.tune.eval_interval = 1
            cfg.tune.patience = 10

            if hasattr(cfg.tune, "infer_epochs"):
                cfg.tune.infer_epochs = 0

        else:  # thorough
            cfg.model.latent_dim = 16
            cfg.model.num_hidden_layers = 3
            cfg.model.layer_scaling_factor = 6.0
            cfg.model.dropout_rate = 0.30
            cfg.model.gamma = 2.5

            cfg.train.batch_size = 64
            cfg.train.learning_rate = 6e-4
            cfg.train.early_stop_gen = 30
            cfg.train.min_epochs = 100
            cfg.train.max_epochs = 3000
            cfg.train.weights_beta = 0.9999
            cfg.train.weights_max_ratio = 1.0

            cfg.tune.enabled = True
            cfg.tune.fast = False
            cfg.tune.n_trials = 250
            cfg.tune.epochs = 1000
            cfg.tune.batch_size = 64
            cfg.tune.max_samples = 0
            cfg.tune.max_loci = 0
            cfg.tune.eval_interval = 1
            cfg.tune.patience = 20

            if hasattr(cfg.tune, "infer_epochs"):
                cfg.tune.infer_epochs = 0

        return cfg

    def apply_overrides(self, overrides: Dict[str, Any] | None) -> "VAEConfig":
        """Apply flat dot-key overrides (e.g., {'vae.kl_beta': 2.0})."""
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
    """Algorithmic knobs for ImputeMostFrequent."""

    by_populations: bool = False  # compute per-pop modes when populations are available
    default: int = 0  # fallback mode if no valid entries in a locus
    missing: int = -1  # code for missing genotypes in 0/1/2


@dataclass
class DeterministicSplitConfig:
    """Evaluation split configuration shared by deterministic imputers."""

    test_size: float = 0.2
    # If provided, overrides test_size.
    test_indices: Optional[Sequence[int]] = None


@dataclass
class MostFrequentConfig:
    """Top-level configuration for ImputeMostFrequent.

    Sections mirror other configs for alignment:
        - io (IOConfig)
        - plot (PlotConfig)
        - split (DeterministicSplitConfig)
        - algo (MostFrequentAlgoConfig)
    """

    io: IOConfig = field(default_factory=IOConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    split: DeterministicSplitConfig = field(default_factory=DeterministicSplitConfig)
    algo: MostFrequentAlgoConfig = field(default_factory=MostFrequentAlgoConfig)

    @classmethod
    def from_preset(
        cls, preset: Literal["fast", "balanced", "thorough"] = "balanced"
    ) -> "MostFrequentConfig":
        """Presets mainly keep parity with logging/IO and split test_size.

        Deterministic imputers don't have model/train knobs; presets exist
        for interface symmetry and minor UX defaults.
        """
        if preset not in {"fast", "balanced", "thorough"}:
            raise ValueError(f"Unknown preset: {preset}")

        cfg = cls()
        cfg.io.verbose = True
        cfg.split.test_size = 0.2  # keep stable across presets
        return cfg

    def apply_overrides(self, overrides: Dict[str, Any] | None) -> "MostFrequentConfig":
        """Apply dot-key overrides (e.g., {'algo.by_populations': True})."""
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
class RefAlleleAlgoConfig:
    """Algorithmic knobs for ImputeRefAllele."""

    missing: int = -1


@dataclass
class RefAlleleConfig:
    """Top-level configuration for ImputeRefAllele.

    Sections for alignment:
        - io (IOConfig)
        - plot (PlotConfig)
        - split (DeterministicSplitConfig)
        - algo (RefAlleleAlgoConfig)
    """

    io: IOConfig = field(default_factory=IOConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    split: DeterministicSplitConfig = field(default_factory=DeterministicSplitConfig)
    algo: RefAlleleAlgoConfig = field(default_factory=RefAlleleAlgoConfig)

    @classmethod
    def from_preset(
        cls, preset: Literal["fast", "balanced", "thorough"] = "balanced"
    ) -> "RefAlleleConfig":
        if preset not in {"fast", "balanced", "thorough"}:
            raise ValueError(f"Unknown preset: {preset}")

        cfg = cls()
        cfg.io.verbose = True
        cfg.split.test_size = 0.2
        return cfg

    def apply_overrides(self, overrides: Dict[str, Any] | None) -> "RefAlleleConfig":
        """Apply dot-key overrides (e.g., {'split.test_size': 0.3})."""
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
