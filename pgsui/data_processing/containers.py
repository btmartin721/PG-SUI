from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional, Sequence

from pgsui.data_processing.config import apply_dot_overrides, load_yaml_to_dataclass


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
        max_features (float | None): The fraction of features to consider when looking for the best split.
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
    max_features: float | None = 1.0
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

    This class contains configuration options for the model architecture, including latent space initialization, dimensionality, dropout rate, and other relevant settings.

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
    """Training procedure configuration.

    This class contains configuration options for the training procedure, including batch size, learning rate, early stopping criteria, and other relevant settings.

    Attributes:
        batch_size (int): Number of samples per training batch.
        learning_rate (float): Learning rate for the optimizer.
        lr_input_factor (float): Factor to scale the learning rate for input layer.
        l1_penalty (float): L1 regularization penalty.
        early_stop_gen (int): Number of generations with no improvement to wait before early stopping.
        min_epochs (int): Minimum number of epochs to train.
        max_epochs (int): Maximum number of epochs to train.
        validation_split (float): Proportion of data to use for validation.
        weights_beta (float): Smoothing factor for class weights.
        weights_max_ratio (float): Maximum ratio for class weights to prevent extreme values.
        device (Literal["gpu", "cpu", "mps"]): Device to use for computation.

    Notes:
        - The `device` attribute specifies the computation device to use, such as "gpu", "cpu", or "mps" (for Apple Silicon).
    """

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

    This class contains configuration options for hyperparameter tuning, including the number of trials, evaluation metrics, and other relevant settings.

    Attributes:
        enabled (bool): If True, enables hyperparameter tuning.
        metric (Literal["f1", "accuracy", "pr_macro"]): Metric to optimize during tuning.
        n_trials (int): Number of hyperparameter trials to run.
        resume (bool): If True, resumes tuning from a previous state.
        save_db (bool): If True, saves the tuning results to a database.
        fast (bool): If True, uses a faster but less thorough tuning approach.
        max_samples (int): Maximum number of samples to use for tuning. 0 means all samples.
        max_loci (int): Maximum number of loci to use for tuning. 0 means all loci.
        epochs (int): Number of epochs to train each trial.
        batch_size (int): Batch size for training during tuning.
        eval_interval (int): Interval (in epochs) at which to evaluate the model during tuning.
        infer_epochs (int): Number of epochs for inference during tuning.
        patience (int): Number of evaluations with no improvement before stopping early.
        proxy_metric_batch (int): If > 0, uses a subset of data for proxy metric evaluation.
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
    ] = "f1"
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

    This class contains configuration options for the evaluation process, including batch size, evaluation intervals, and other relevant settings.

    Attributes:
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

    This class contains configuration options for plotting, including file format, resolution, and other relevant settings.

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

    This class contains configuration options for input/output operations, including file prefixes, verbosity, random seed, and other relevant settings.

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
class SimConfig:
    """Top-level configuration for data simulation and imputation.

    This class contains all the configuration options for simulating missing data and performing imputation. The configuration is organized into several sections, each represented by a dataclass.

    Attributes:
        simulate_missing (bool): If True, simulates missing data according to the specified strategy.
        sim_strategy (Literal["random", "random_weighted", "random_weighted_inv", "nonrandom", "nonrandom_weighted"]): Strategy for simulating missing data.
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
    sim_prop: float = 0.10
    sim_kwargs: dict | None = None


@dataclass
class NLPCAConfig:
    """Top-level configuration for ImputeNLPCA.

    This class contains all the configuration options for the ImputeNLPCA model. The configuration is organized into several sections, each represented by a dataclass.

    Attributes:
        io (IOConfig): I/O configuration.
        model (ModelConfig): Model architecture configuration.
        train (TrainConfig): Training procedure configuration.
        tune (TuneConfig): Hyperparameter tuning configuration.
        evaluate (EvalConfig): Evaluation configuration.
        plot (PlotConfig): Plotting configuration.
        sim (SimConfig): Simulation configuration.

    Notes:
        - fast:     Quick baseline; tiny net; NO tuning by default.
        - balanced: Practical default balancing speed and model performance; moderate tuning.
        - thorough: Prioritizes model performance; deeper nets; extensive tuning.
        - Overrides: Overrides are applied after presets and can be used to fine-tune specific parameters. Specifically uses flat dot-keys like {"model.latent_dim": 8}.
    """

    io: IOConfig = field(default_factory=IOConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tune: TuneConfig = field(default_factory=TuneConfig)
    evaluate: EvalConfig = field(default_factory=EvalConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    sim: SimConfig = field(default_factory=SimConfig)

    @classmethod
    def from_preset(
        cls, preset: Literal["fast", "balanced", "thorough"] = "balanced"
    ) -> "NLPCAConfig":
        """Build a NLPCAConfig from a named preset.

        This method allows for easy construction of a NLPCAConfig instance with sensible defaults based on the chosen preset. NLPCA is often used when classes (genotype states) are imbalanced. Presets adjust both capacity and weighting behavior across speed/quality tradeoffs.

        Args:
            preset (Literal["fast", "balanced", "thorough"]): One of {"fast","balanced","thorough"}.

        Returns:
            NLPCAConfig: Populated config instance.
        """
        if preset not in {"fast", "balanced", "thorough"}:
            raise ValueError(f"Unknown preset: {preset}")

        cfg = cls()

        # Common baselines
        cfg.io.verbose = False
        cfg.train.validation_split = 0.20
        cfg.model.hidden_activation = "relu"
        cfg.model.layer_schedule = "pyramid"
        cfg.model.latent_init = "random"
        # Eval uses latent refinement for NLPCA
        cfg.evaluate.eval_latent_lr = 1e-2
        cfg.evaluate.eval_latent_weight_decay = 0.0
        cfg.sim.simulate_missing = True
        cfg.sim.sim_strategy = "random"
        cfg.sim.sim_prop = 0.2

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
            cfg.train.max_epochs = 120
            cfg.train.weights_beta = 0.9999
            cfg.train.weights_max_ratio = 2.0
            # Tuning (enabled but light)
            cfg.tune.enabled = True
            cfg.tune.fast = True
            cfg.tune.n_trials = 25
            cfg.tune.epochs = 120
            cfg.tune.batch_size = 128
            cfg.tune.max_samples = 512
            cfg.tune.max_loci = 0
            cfg.tune.eval_interval = 5
            cfg.tune.infer_epochs = 20
            cfg.tune.patience = 5
            cfg.tune.proxy_metric_batch = 0
            # Eval
            cfg.evaluate.eval_latent_steps = 20

        elif preset == "balanced":
            # Model
            cfg.model.latent_dim = 8
            cfg.model.num_hidden_layers = 2
            cfg.model.layer_scaling_factor = 3.0
            cfg.model.dropout_rate = 0.20
            cfg.model.gamma = 2.0
            # Train
            cfg.train.batch_size = 128
            cfg.train.learning_rate = 8e-4
            cfg.train.early_stop_gen = 15
            cfg.train.min_epochs = 50
            cfg.train.max_epochs = 600
            cfg.train.weights_beta = 0.9999
            cfg.train.weights_max_ratio = 2.0
            # Tuning
            cfg.tune.enabled = True
            cfg.tune.fast = True
            cfg.tune.n_trials = 75
            cfg.tune.epochs = 300
            cfg.tune.batch_size = 128
            cfg.tune.max_samples = 2048
            cfg.tune.max_loci = 0
            cfg.tune.eval_interval = 5
            cfg.tune.infer_epochs = 40
            cfg.tune.patience = 10
            cfg.tune.proxy_metric_batch = 0
            # Eval
            cfg.evaluate.eval_latent_steps = 30

        else:  # thorough
            # Model
            cfg.model.latent_dim = 16
            cfg.model.num_hidden_layers = 3
            cfg.model.layer_scaling_factor = 5.0
            cfg.model.dropout_rate = 0.30
            cfg.model.gamma = 2.5
            # Train
            cfg.train.batch_size = 64
            cfg.train.learning_rate = 6e-4
            cfg.train.early_stop_gen = 30
            cfg.train.min_epochs = 100
            cfg.train.max_epochs = 1200
            cfg.train.weights_beta = 0.9999
            cfg.train.weights_max_ratio = 2.0
            # Tuning
            cfg.tune.enabled = True
            cfg.tune.fast = False
            cfg.tune.n_trials = 150
            cfg.tune.epochs = 600
            cfg.tune.batch_size = 64
            cfg.tune.max_samples = 0
            cfg.tune.max_loci = 0
            cfg.tune.eval_interval = 5
            cfg.tune.infer_epochs = 80
            cfg.tune.patience = 20
            cfg.tune.proxy_metric_batch = 0
            # Eval
            cfg.evaluate.eval_latent_steps = 50

        return cfg

    def apply_overrides(self, overrides: Dict[str, Any] | None) -> "NLPCAConfig":
        """Apply flat dot-key overrides (e.g. {'model.latent_dim': 4}).

        This method allows for easy modification of the configuration by specifying the keys to change in a flat dictionary format.

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
        sim (SimConfig): Simulated-missing configuration.

    Notes:
        - fast:     Quick baseline; tiny net; NO tuning by default.
        - balanced: Practical default balancing speed and model performance; moderate tuning.
        - thorough: Prioritizes model performance; deeper nets; extensive tuning.
        - Overrides: Overrides are applied after presets and can be used to fine-tune specific parameters. Specifically uses flat dot-keys like {"model.latent_dim": 8}.
    """

    io: IOConfig = field(default_factory=IOConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tune: TuneConfig = field(default_factory=TuneConfig)
    evaluate: EvalConfig = field(default_factory=EvalConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    sim: SimConfig = field(default_factory=SimConfig)

    @classmethod
    def from_preset(
        cls, preset: Literal["fast", "balanced", "thorough"] = "balanced"
    ) -> "UBPConfig":
        """Build a UBPConfig from a named preset.

        This method allows for easy construction of a UBPConfig instance with sensible defaults based on the chosen preset. UBP is often used when classes (genotype states) are imbalanced. Presets adjust both capacity and weighting behavior across speed/quality tradeoffs.

        Args:
            preset (Literal["fast", "balanced", "thorough"]): One of {"fast","balanced","thorough"}.

        Returns:
            UBPConfig: Populated config instance.
        """
        if preset not in {"fast", "balanced", "thorough"}:
            raise ValueError(f"Unknown preset: {preset}")

        cfg = cls()

        # Common baselines
        cfg.io.verbose = False
        cfg.model.hidden_activation = "relu"
        cfg.model.layer_schedule = "pyramid"
        cfg.model.latent_init = "random"
        cfg.sim.simulate_missing = True
        cfg.sim.sim_strategy = "random"
        cfg.sim.sim_prop = 0.2

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
            cfg.train.max_epochs = 120
            cfg.train.weights_beta = 0.9999
            cfg.train.weights_max_ratio = 2.0
            # Tuning
            cfg.tune.enabled = True
            cfg.tune.fast = True
            cfg.tune.n_trials = 25
            cfg.tune.epochs = 120
            cfg.tune.batch_size = 128
            cfg.tune.max_samples = 512
            cfg.tune.max_loci = 0
            cfg.tune.eval_interval = 5
            cfg.tune.infer_epochs = 20
            cfg.tune.patience = 5
            cfg.tune.proxy_metric_batch = 0
            # Eval
            cfg.evaluate.eval_latent_steps = 20
            cfg.evaluate.eval_latent_lr = 1e-2
            cfg.evaluate.eval_latent_weight_decay = 0.0

        elif preset == "balanced":
            # Model
            cfg.model.latent_dim = 8
            cfg.model.num_hidden_layers = 2
            cfg.model.layer_scaling_factor = 3.0
            cfg.model.dropout_rate = 0.20
            cfg.model.gamma = 2.0
            # Train
            cfg.train.batch_size = 128
            cfg.train.learning_rate = 8e-4
            cfg.train.early_stop_gen = 15
            cfg.train.min_epochs = 50
            cfg.train.max_epochs = 600
            cfg.train.weights_beta = 0.9999
            cfg.train.weights_max_ratio = 2.0
            # Tuning
            cfg.tune.enabled = True
            cfg.tune.fast = True
            cfg.tune.n_trials = 75
            cfg.tune.epochs = 300
            cfg.tune.batch_size = 128
            cfg.tune.max_samples = 2048
            cfg.tune.max_loci = 0
            cfg.tune.eval_interval = 5
            cfg.tune.infer_epochs = 40
            cfg.tune.patience = 10
            cfg.tune.proxy_metric_batch = 0
            # Eval
            cfg.evaluate.eval_latent_steps = 30
            cfg.evaluate.eval_latent_lr = 1e-2
            cfg.evaluate.eval_latent_weight_decay = 0.0

        else:  # thorough
            # Model
            cfg.model.latent_dim = 16
            cfg.model.num_hidden_layers = 3
            cfg.model.layer_scaling_factor = 5.0
            cfg.model.dropout_rate = 0.30
            cfg.model.gamma = 2.5
            # Train
            cfg.train.batch_size = 64
            cfg.train.learning_rate = 6e-4
            cfg.train.early_stop_gen = 30
            cfg.train.min_epochs = 100
            cfg.train.max_epochs = 1200
            cfg.train.weights_beta = 0.9999
            cfg.train.weights_max_ratio = 2.0
            # Tuning
            cfg.tune.enabled = True
            cfg.tune.fast = False
            cfg.tune.n_trials = 150
            cfg.tune.epochs = 600
            cfg.tune.batch_size = 64
            cfg.tune.max_samples = 0
            cfg.tune.max_loci = 0
            cfg.tune.eval_interval = 5
            cfg.tune.infer_epochs = 80
            cfg.tune.patience = 20
            cfg.tune.proxy_metric_batch = 0
            # Eval
            cfg.evaluate.eval_latent_steps = 50
            cfg.evaluate.eval_latent_lr = 1e-2
            cfg.evaluate.eval_latent_weight_decay = 0.0

        return cfg

    def apply_overrides(self, overrides: Dict[str, Any] | None) -> "UBPConfig":
        """Apply flat dot-key overrides (e.g. {'model.latent_dim': 4}).

        Args:
            overrides (Dict[str, Any] | None): Mapping of dot-key paths to values to override.

        Returns:
            UBPConfig: This instance after applying overrides.
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
        sim (SimConfig): Simulated-missing configuration.

    Notes:
        - fast:     Quick baseline; tiny net; NO tuning by default.
        - balanced: Practical default; moderate tuning.
        - thorough: Prioritizes model performance; deeper nets; extensive tuning.
        - Overrides: flat dot-keys like {"model.latent_dim": 8}.
    """

    io: IOConfig = field(default_factory=IOConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tune: TuneConfig = field(default_factory=TuneConfig)
    evaluate: EvalConfig = field(default_factory=EvalConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    sim: SimConfig = field(default_factory=SimConfig)

    @classmethod
    def from_preset(
        cls, preset: Literal["fast", "balanced", "thorough"] = "balanced"
    ) -> "AutoencoderConfig":
        """Build a AutoencoderConfig from a named preset.

        This method allows for easy construction of a AutoencoderConfig instance with sensible defaults based on the chosen preset. Presets adjust both capacity and weighting behavior across speed/quality tradeoffs.

        Args:
            preset (Literal["fast", "balanced", "thorough"]): One of {"fast","balanced","thorough"}.

        Returns:
            AutoencoderConfig: Populated config instance.
        """
        if preset not in {"fast", "balanced", "thorough"}:
            raise ValueError(f"Unknown preset: {preset}")

        cfg = cls()

        # Common baselines (no latent refinement at eval)
        cfg.io.verbose = False
        cfg.train.validation_split = 0.20
        cfg.model.hidden_activation = "relu"
        cfg.model.layer_schedule = "pyramid"
        cfg.evaluate.eval_latent_steps = 0
        cfg.evaluate.eval_latent_lr = 0.0
        cfg.evaluate.eval_latent_weight_decay = 0.0
        cfg.sim.simulate_missing = True
        cfg.sim.sim_strategy = "random"
        cfg.sim.sim_prop = 0.2

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
            cfg.train.max_epochs = 120
            cfg.train.weights_beta = 0.9999
            cfg.train.weights_max_ratio = 2.0
            cfg.tune.enabled = True
            cfg.tune.fast = True
            cfg.tune.n_trials = 25
            cfg.tune.epochs = 120
            cfg.tune.batch_size = 128
            cfg.tune.max_samples = 512
            cfg.tune.max_loci = 0
            cfg.tune.eval_interval = 5
            cfg.tune.patience = 5
            cfg.tune.proxy_metric_batch = 0
            if hasattr(cfg.tune, "infer_epochs"):
                cfg.tune.infer_epochs = 0

        elif preset == "balanced":
            cfg.model.latent_dim = 8
            cfg.model.num_hidden_layers = 2
            cfg.model.layer_scaling_factor = 3.0
            cfg.model.dropout_rate = 0.20
            cfg.model.gamma = 2.0
            cfg.train.batch_size = 128
            cfg.train.learning_rate = 8e-4
            cfg.train.early_stop_gen = 15
            cfg.train.min_epochs = 50
            cfg.train.max_epochs = 600
            cfg.train.weights_beta = 0.9999
            cfg.train.weights_max_ratio = 2.0
            cfg.tune.enabled = True
            cfg.tune.fast = True
            cfg.tune.n_trials = 75
            cfg.tune.epochs = 300
            cfg.tune.batch_size = 128
            cfg.tune.max_samples = 2048
            cfg.tune.max_loci = 0
            cfg.tune.eval_interval = 5
            cfg.tune.patience = 10
            cfg.tune.proxy_metric_batch = 0
            if hasattr(cfg.tune, "infer_epochs"):
                cfg.tune.infer_epochs = 0

        else:  # thorough
            cfg.model.latent_dim = 16
            cfg.model.num_hidden_layers = 3
            cfg.model.layer_scaling_factor = 5.0
            cfg.model.dropout_rate = 0.30
            cfg.model.gamma = 2.5
            cfg.train.batch_size = 64
            cfg.train.learning_rate = 6e-4
            cfg.train.early_stop_gen = 30
            cfg.train.min_epochs = 100
            cfg.train.max_epochs = 1200
            cfg.train.weights_beta = 0.9999
            cfg.train.weights_max_ratio = 2.0
            cfg.tune.enabled = True
            cfg.tune.fast = False
            cfg.tune.n_trials = 150
            cfg.tune.epochs = 600
            cfg.tune.batch_size = 64
            cfg.tune.max_samples = 0
            cfg.tune.max_loci = 0
            cfg.tune.eval_interval = 5
            cfg.tune.patience = 20
            cfg.tune.proxy_metric_batch = 0
            if hasattr(cfg.tune, "infer_epochs"):
                cfg.tune.infer_epochs = 0

        return cfg

    def apply_overrides(self, overrides: Dict[str, Any] | None) -> "AutoencoderConfig":
        """Apply flat dot-key overrides (e.g. {'model.latent_dim': 4}).

        Args:
            overrides (Dict[str, Any] | None): Mapping of dot-key paths to values to override.

        Returns:
            AutoencoderConfig: This instance after applying overrides.
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
    """VAE-specific knobs.

    This class contains additional configuration options specific to Variational Autoencoders (VAEs), particularly for controlling the KL divergence term in the loss function.

    Attributes:
        kl_beta (float): Final β for KL divergence term.
        kl_warmup (int): Number of epochs with β=0 (warm-up period
            to stabilize training).
        kl_ramp (int): Number of epochs for linear ramp to final β.

    Notes:
        - These parameters control the behavior of the KL divergence term in the VAE loss function.
        - The warm-up period helps to stabilize training by gradually introducing the KL term.
        - The ramp period defines how quickly the KL term reaches its final value.
    """

    kl_beta: float = 1.0  # final β for KL
    kl_warmup: int = 50  # epochs with β=0
    kl_ramp: int = 200  # linear ramp to β


@dataclass
class VAEConfig:
    """Top-level configuration for ImputeVAE (AE-parity + VAE extras).

    This class contains all the configuration options for the ImputeVAE model. The configuration is organized into several sections, each represented by a dataclass.

    Attributes:
        io (IOConfig): I/O configuration.
        model (ModelConfig): Model architecture configuration.
        train (TrainConfig): Training procedure configuration.
        tune (TuneConfig): Hyperparameter tuning configuration.
        evaluate (EvalConfig): Evaluation configuration.
        plot (PlotConfig): Plotting configuration.
        vae (VAEExtraConfig): VAE-specific configuration.
        sim (SimConfig): Simulated-missing configuration.

    Notes:
        - fast:     Quick baseline; tiny net; NO tuning by default.
        - balanced: Practical default; moderate tuning.
        - thorough: Prioritizes model performance; deeper nets; extensive tuning.
        - Overrides: flat dot-keys like {"model.latent_dim": 8}.
    """

    io: IOConfig = field(default_factory=IOConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tune: TuneConfig = field(default_factory=TuneConfig)
    evaluate: EvalConfig = field(default_factory=EvalConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    vae: VAEExtraConfig = field(default_factory=VAEExtraConfig)
    sim: SimConfig = field(default_factory=SimConfig)

    @classmethod
    def from_preset(
        cls, preset: Literal["fast", "balanced", "thorough"] = "balanced"
    ) -> "VAEConfig":
        """Build a VAEConfig from a named preset.

        This method allows for easy construction of a VAEConfig instance with sensible defaults based on the chosen preset. Presets adjust both capacity and weighting behavior across speed/quality tradeoffs.

        Args:
            preset (Literal["fast", "balanced", "thorough"]): One of {"fast","balanced","thorough"}.

        Returns:
            VAEConfig: Populated config instance.
        """
        if preset not in {"fast", "balanced", "thorough"}:
            raise ValueError(f"Unknown preset: {preset}")

        cfg = cls()

        # Common baselines (match AE; no latent refinement at eval)
        cfg.io.verbose = False
        cfg.train.validation_split = 0.20
        cfg.model.hidden_activation = "relu"
        cfg.model.layer_schedule = "pyramid"
        cfg.evaluate.eval_latent_steps = 0
        cfg.evaluate.eval_latent_lr = 0.0
        cfg.evaluate.eval_latent_weight_decay = 0.0
        cfg.sim.simulate_missing = True
        cfg.sim.sim_strategy = "random"
        cfg.sim.sim_prop = 0.2

        # VAE KL schedules, shortened for speed
        cfg.vae.kl_beta = 1.0
        cfg.vae.kl_warmup = 25
        cfg.vae.kl_ramp = 100

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
            cfg.train.max_epochs = 120
            cfg.train.weights_beta = 0.9999
            cfg.train.weights_max_ratio = 2.0
            cfg.tune.enabled = True
            cfg.tune.fast = True
            cfg.tune.n_trials = 25
            cfg.tune.epochs = 120
            cfg.tune.batch_size = 128
            cfg.tune.max_samples = 512
            cfg.tune.max_loci = 0
            cfg.tune.eval_interval = 5
            cfg.tune.patience = 5
            cfg.tune.proxy_metric_batch = 0
            if hasattr(cfg.tune, "infer_epochs"):
                cfg.tune.infer_epochs = 0

        elif preset == "balanced":
            cfg.model.latent_dim = 8
            cfg.model.num_hidden_layers = 2
            cfg.model.layer_scaling_factor = 3.0
            cfg.model.dropout_rate = 0.20
            cfg.model.gamma = 2.0
            cfg.train.batch_size = 128
            cfg.train.learning_rate = 8e-4
            cfg.train.early_stop_gen = 15
            cfg.train.min_epochs = 50
            cfg.train.max_epochs = 600
            cfg.train.weights_beta = 0.9999
            cfg.train.weights_max_ratio = 2.0
            cfg.tune.enabled = True
            cfg.tune.fast = True
            cfg.tune.n_trials = 75
            cfg.tune.epochs = 300
            cfg.tune.batch_size = 128
            cfg.tune.max_samples = 2048
            cfg.tune.max_loci = 0
            cfg.tune.eval_interval = 5
            cfg.tune.patience = 10
            cfg.tune.proxy_metric_batch = 0
            if hasattr(cfg.tune, "infer_epochs"):
                cfg.tune.infer_epochs = 0

        else:  # thorough
            cfg.model.latent_dim = 16
            cfg.model.num_hidden_layers = 3
            cfg.model.layer_scaling_factor = 5.0
            cfg.model.dropout_rate = 0.30
            cfg.model.gamma = 2.5
            cfg.train.batch_size = 64
            cfg.train.learning_rate = 6e-4
            cfg.train.early_stop_gen = 30
            cfg.train.min_epochs = 100
            cfg.train.max_epochs = 1200
            cfg.train.weights_beta = 0.9999
            cfg.train.weights_max_ratio = 2.0
            cfg.tune.enabled = True
            cfg.tune.fast = False
            cfg.tune.n_trials = 150
            cfg.tune.epochs = 600
            cfg.tune.batch_size = 64
            cfg.tune.max_samples = 0
            cfg.tune.max_loci = 0
            cfg.tune.eval_interval = 5
            cfg.tune.patience = 20
            cfg.tune.proxy_metric_batch = 0
            if hasattr(cfg.tune, "infer_epochs"):
                cfg.tune.infer_epochs = 0

        return cfg

    def apply_overrides(self, overrides: Dict[str, Any] | None) -> "VAEConfig":
        """Apply flat dot-key overrides (e.g., {'vae.kl_beta': 2.0}).

        Args:
            overrides (Dict[str, Any] | None): Mapping of dot-key paths to values to override.

        Returns:
            VAEConfig: This instance after applying overrides.
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
class MostFrequentAlgoConfig:
    """Algorithmic knobs for ImputeMostFrequent.

    This class contains configuration options specific to the most frequent genotype imputation algorithm.

    Attributes:
        by_populations (bool): Whether to compute per-population modes when populations are available.
        default (int): Fallback mode if no valid entries in a locus.
        missing (int): Code for missing genotypes in 0/1/2.
    """

    by_populations: bool = False  # per-pop modes if pops available
    default: int = 0  # fallback mode if no valid entries in a locus
    missing: int = -1  # code for missing genotypes in 0/1/2


@dataclass
class DeterministicSplitConfig:
    """Evaluation split configuration shared by deterministic imputers.

    This class contains configuration options for splitting data into training and testing sets for deterministic imputation algorithms. The split can be defined by a proportion of the data or by specific indices.

    Attributes:
        test_size (float): Proportion of data to use as the test set.
        test_indices (Optional[Sequence[int]]): Specific indices to use as the test set. If provided, this overrides the `test_size` parameter.
    """

    test_size: float = 0.2

    # If provided, overrides test_size.
    test_indices: Optional[Sequence[int]] = None


@dataclass
class MostFrequentConfig:
    """Top-level configuration for ImputeMostFrequent.

    This class contains all the configuration options for the
    ImputeMostFrequent model. The configuration is organized into several
    sections, each represented by a dataclass.

    Attributes:
        io (IOConfig): I/O configuration.
        plot (PlotConfig): Plotting configuration.
        split (DeterministicSplitConfig): Data splitting configuration.
        algo (MostFrequentAlgoConfig): Algorithmic configuration.
        sim (SimConfig): Simulation configuration controlling how
            missing values are synthetically introduced for evaluation.
        tune (TuneConfig): Hyperparameter tuning configuration. For
            compatibility only. Ignored for deterministic imputers.
        train (TrainConfig): Training configuration. Present for interface
            symmetry with NN-based imputers; typically unused here.
    """

    io: IOConfig = field(default_factory=IOConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    split: DeterministicSplitConfig = field(default_factory=DeterministicSplitConfig)
    algo: MostFrequentAlgoConfig = field(default_factory=MostFrequentAlgoConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    tune: TuneConfig = field(default_factory=TuneConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_preset(
        cls,
        preset: Literal["fast", "balanced", "thorough"] = "balanced",
    ) -> "MostFrequentConfig":
        """Construct a preset configuration.

        Deterministic imputers don't have model/train knobs; presets exist for interface symmetry and minor UX defaults.

        Args:
            preset (Literal["fast", "balanced", "thorough"]): One of
                {"fast", "balanced", "thorough"}.

        Returns:
            MostFrequentConfig: Populated config instance.
        """
        if preset not in {"fast", "balanced", "thorough"}:
            raise ValueError(f"Unknown preset: {preset}")

        cfg = cls()
        cfg.io.verbose = False
        cfg.split.test_size = 0.2  # keep stable across presets
        cfg.sim.simulate_missing = True  # simulate for evaluation
        cfg.sim.sim_strategy = "random"
        cfg.sim.sim_prop = 0.2

        return cfg

    def apply_overrides(self, overrides: Dict[str, Any] | None) -> "MostFrequentConfig":
        """Apply dot-key overrides (e.g., {'algo.by_populations': True}).

        This method allows for easy modification of the configuration by specifying the keys to change in a flat dictionary format.

        Args:
            overrides (Dict[str, Any] | None): Mapping of dot-key paths to
                values to override.

        Returns:
            MostFrequentConfig: This instance after applying overrides.
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
                # Unknown override field; silently ignore for now.
                pass
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Return the config as a dictionary.

        Returns:
            Dict[str, Any]: The config as a nested dictionary.
        """
        return asdict(self)


@dataclass
class RefAlleleAlgoConfig:
    """Algorithmic knobs for ImputeRefAllele.

    This class contains configuration options specific to the reference allele imputation algorithm.

    Attributes:
        missing (int): Code for missing genotypes in 0/1/2.
    """

    missing: int = -1


@dataclass
class RefAlleleConfig:
    """Top-level configuration for ImputeRefAllele.

    This class contains all the configuration options for the ImputeRefAllele model. The configuration is organized into several sections, each represented by a dataclass.

    Attributes:
        io (IOConfig): I/O configuration.
        plot (PlotConfig): Plotting configuration.
        split (DeterministicSplitConfig): Data splitting configuration.
        algo (RefAlleleAlgoConfig): Algorithmic configuration.
        sim (SimConfig): Simulation configuration controlling how missing
            values are synthetically introduced for evaluation.
        tune (TuneConfig): Hyperparameter tuning configuration. For
            compatibility only. Ignored for deterministic imputers.
        train (TrainConfig): Training configuration. Present for interface
            symmetry with NN-based imputers; typically unused here.
    """

    io: IOConfig = field(default_factory=IOConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    split: DeterministicSplitConfig = field(default_factory=DeterministicSplitConfig)
    algo: RefAlleleAlgoConfig = field(default_factory=RefAlleleAlgoConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    tune: TuneConfig = field(default_factory=TuneConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_preset(
        cls, preset: Literal["fast", "balanced", "thorough"] = "balanced"
    ) -> "RefAlleleConfig":
        """Presets mainly keep parity with logging/IO and split test_size.

        Deterministic imputers don't have model/train knobs; presets exist
        for interface symmetry and minor UX defaults.

        Args:
            preset (Literal["fast", "balanced", "thorough"]): One of
                {"fast", "balanced", "thorough"}.

        Returns:
            RefAlleleConfig: Populated config instance.
        """
        if preset not in {"fast", "balanced", "thorough"}:
            raise ValueError(f"Unknown preset: {preset}")

        cfg = cls()
        cfg.io.verbose = False
        cfg.split.test_size = 0.2
        cfg.sim.simulate_missing = True  # simulate for evaluation
        cfg.sim.sim_strategy = "random"
        cfg.sim.sim_prop = 0.2
        return cfg

    def apply_overrides(self, overrides: Dict[str, Any] | None) -> "RefAlleleConfig":
        """Apply dot-key overrides (e.g., {'split.test_size': 0.3}).

        This method allows for easy modification of the configuration by specifying the keys to change in a flat dictionary format.

        Args:
            overrides (Dict[str, Any] | None): A mapping of dot-key paths
                to values to override.

        Returns:
            RefAlleleConfig: The updated config instance (same as `self`).
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
                # Unknown override; ignore for forward compatibility.
                pass
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary.

        Returns:
            Dict[str, Any]: The config as a nested dictionary.
        """
        return asdict(self)


def _flatten_dict(
    d: Dict[str, Any], prefix: str = "", out: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Flatten a nested dictionary into dot-key format.

    Args:
        d (Dict[str, Any]): The nested dictionary to flatten.
        prefix (str): The prefix to use for keys (used in recursion).
        out (Optional[Dict[str, Any]]): The output dictionary to populate.

    Returns:
        Dict[str, Any]: The flattened dictionary with dot-key format.
    """
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

    This class contains configuration options for input/output operations, logging, and run identification.

    Attributes:
        prefix (str): Prefix for output files and logs.
        seed (Optional[int]): Random seed for reproducibility.
        n_jobs (int): Number of parallel jobs to use. -1 uses all available cores.
        verbose (bool): Whether to enable verbose logging.
        debug (bool): Whether to enable debug mode with more detailed logs.

    Notes:
        - The prefix is used to name output files and logs, helping to organize results from different runs.
        - Setting a random seed ensures that results are reproducible across different runs.
        - The number of jobs can be adjusted based on the available computational resources.
        - Verbose and debug modes provide additional logging information, which can be useful for troubleshooting.
    """

    prefix: str = "pgsui"
    seed: Optional[int] = None
    n_jobs: int = 1
    verbose: bool = False
    debug: bool = False


@dataclass
class PlotConfigSupervised:
    """Plot/figure styling.

    This class contains parameters for controlling the appearance of plots generated during the imputation process.

    Attributes:
        fmt (Literal["pdf", "png", "jpg", "jpeg"]): File format
            for saving plots.
        dpi (int): Resolution in dots per inch for raster formats.
        fontsize (int): Base font size for plot text.
        despine (bool): Whether to remove top/right spines from plots.
        show (bool): Whether to display plots interactively.

    Notes:
        - Supported formats: "pdf", "png", "jpg", "jpeg".
        - Higher DPI values yield better quality in raster images.
        - Despining is a common aesthetic choice for cleaner plots.
    """

    fmt: Literal["pdf", "png", "jpg", "jpeg"] = "pdf"
    dpi: int = 300
    fontsize: int = 18
    despine: bool = True
    show: bool = False


@dataclass
class TrainConfigSupervised:
    """Training/evaluation split (by samples).

    This class contains configuration options for splitting the dataset into training and validation sets during the training process.

    Attributes:
        validation_split (float): Proportion of data to use for validation.

    Notes:
        - Value should be between 0.0 and 1.0.
    """

    validation_split: float = 0.20

    def __post_init__(self):
        """Validate that validation_split is between 0.0 and 1.0."""
        if not (0.0 < self.validation_split < 1.0):
            raise ValueError("validation_split must be between 0.0 and 1.0")


@dataclass
class ImputerConfigSupervised:
    """IterativeImputer-like scaffolding used by current supervised wrappers.

    This class contains configuration options for the imputation process, specifically for iterative imputation methods.

    Attributes:
        n_nearest_features (Optional[int]): Number of nearest features to use
            for imputation. If None, all features are used.
        max_iter (int): Maximum number of imputation iterations to perform.

    Notes:
        - n_nearest_features can help speed up imputation by limiting the number of features considered.
        - max_iter controls how many times the imputation process is repeated to refine estimates.
        - If n_nearest_features is None, the imputer will consider all features for each missing value.
        - Default max_iter is set to 10, which is typically sufficient for convergence.
        - Iterative imputation can be computationally intensive; consider adjusting n_nearest_features for large datasets.
    """

    n_nearest_features: Optional[int] = 10
    max_iter: int = 10


@dataclass
class SimConfigSupervised:
    """Simulation of missingness for evaluation.

    This class contains configuration options for simulating missing data during the evaluation process.

    Attributes:
        prop_missing (float): Proportion of features to randomly set as missing.
        strategy (Literal["random", "random_inv_genotype"]): Strategy for generating missingness.
        het_boost (float): Boosting factor for heterogeneity in missingness.
        missing_val (int): Internal code for missing genotypes (e.g., -1).

    Notes:
        - The choice of strategy can affect the realism of the missing data simulation.
        - Heterogeneous missingness can be useful for testing model robustness.
    """

    prop_missing: float = 0.5
    strategy: Literal["random", "random_inv_genotype"] = "random_inv_genotype"
    het_boost: float = 2.0
    missing_val: int = -1  # internal use; your wrappers expect -1


@dataclass
class TuningConfigSupervised:
    """Optuna tuning envelope (kept for parity with unsupervised)."""

    enabled: bool = True
    n_trials: int = 100
    metric: str = "pr_macro"
    n_jobs: int = 8  # for parallel eval (model-dependent)
    fast: bool = True  # placeholder—trees don't need it but kept for consistency


@dataclass
class RFModelConfig:
    """Random Forest hyperparameters.

    This class contains configuration options for the Random Forest model used in imputation.

    Attributes:
        n_estimators (int): Number of trees in the forest.
        max_depth (Optional[int]): Maximum depth of the trees. If None, nodes are expanded until all leaves are pure or contain less than min_samples_leaf samples.
        min_samples_split (int): Minimum number of samples required to split an internal node.
        min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
        max_features (Literal["sqrt", "log2"] | float | None): Number of features to consider when looking for the best split.
        criterion (Literal["gini", "entropy", "log_loss"]): Function to measure the quality of a split.
        class_weight (Literal["balanced", "balanced_subsample", None]): Weights associated with classes. If "balanced", the class weights will be adjusted inversely proportional to class frequencies in the input data. If "balanced_subsample", the weights will be adjusted based on the bootstrap sample for each tree. If None, all classes will have weight of 1.0.
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

    This class contains configuration options for the Histogram-based Gradient Boosting (HGB) model used in imputation.

    Attributes:
        n_estimators (int): Number of boosting iterations.
        learning_rate (float): Step size for each boosting iteration.
        max_depth (Optional[int]): Maximum depth of each tree. If None, nodes are expanded until all leaves are pure or contain less than min_samples_leaf samples.
        min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
        max_features (float | None): Proportion of features to consider when looking for the best split. If None, all features are considered.
        n_iter_no_change (int): Number of iterations with no improvement to wait before early stopping.
        tol (float): Minimum improvement in the loss to qualify as an improvement.

    Notes:
        - These parameters control the complexity and learning behavior of the HGB model.
        - Early stopping is implemented to prevent overfitting.
        - The choice of criterion affects how the quality of a split is measured.
        - The model is sensitive to the learning_rate; smaller values require more estimators.
        - max_features can be set to a float between 0.0 and 1.0 to use a proportion of features.
        - Early stopping is driven by ``n_iter_no_change / tol``; sklearn controls randomness via random_state.
    """

    # sklearn.HistGradientBoostingClassifier uses 'max_iter'
    # as number of boosting iterations
    # instead of 'n_estimators'.
    n_estimators: int = 100  # maps to max_iter
    learning_rate: float = 0.1
    max_depth: Optional[int] = None
    min_samples_leaf: int = 1
    max_features: float | None = 1.0
    n_iter_no_change: int = 10
    tol: float = 1e-7

    def __post_init__(self) -> None:
        """Validate max_features if it's a float.

        This method checks if the `max_features` attribute is a float and ensures that it falls within the valid range (0.0, 1.0]. It also validates that `n_estimators` is a positive integer.
        """
        if isinstance(self.max_features, float):
            if not (0.0 < self.max_features <= 1.0):
                raise ValueError("max_features as float must be in (0.0, 1.0]")

        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer")


@dataclass
class RFConfig:
    """Configuration for ImputeRandomForest.

    This dataclass mirrors the legacy ``__init__`` signature while supporting presets, YAML loading, and dot-key overrides. Use ``to_imputer_kwargs()`` to call the current constructor, or refactor the imputer to accept ``config: RFConfig``.

    Attributes:
        io (IOConfigSupervised): Run identity, logging, and seeds.
        model (RFModelConfig): RandomForest hyperparameters.
        train (TrainConfigSupervised): Sample split for validation.
        imputer (ImputerConfigSupervised): IterativeImputer scaffolding (neighbors/iters).
        sim (SimConfigSupervised): Simulated missingness used during evaluation.
        plot (PlotConfigSupervised): Plot styling and export options.
        tune (TuningConfigSupervised): Optuna knobs (not required by RF itself).
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

        This method allows for easy construction of an RFConfig instance with sensible defaults based on the chosen preset. Presets adjust both model capacity and training/tuning behavior across speed/quality tradeoffs.

        Args:
            preset: One of {"fast", "balanced", "thorough"}.
                - fast:      Quick baseline; fewer trees; fewer imputer iters.
                - balanced:  Balances speed and model performance; moderate trees and imputer iters.
                - thorough:  Prioritizes model performance; more trees; more imputer iters.

        Returns:
            RFConfig: Config with preset values applied.
        """
        cfg = cls()
        if preset == "fast":
            cfg.model.n_estimators = 50
            cfg.model.max_depth = None
            cfg.imputer.max_iter = 5
            cfg.io.n_jobs = 1
            cfg.tune.enabled = False
        elif preset == "balanced":
            cfg.model.n_estimators = 100
            cfg.model.max_depth = None
            cfg.imputer.max_iter = 10
            cfg.io.n_jobs = 1
            cfg.tune.enabled = False
            cfg.tune.n_trials = 100
        elif preset == "thorough":
            cfg.model.n_estimators = 500
            cfg.model.max_depth = None
            cfg.imputer.max_iter = 15
            cfg.io.n_jobs = 1
            cfg.tune.enabled = False
            cfg.tune.n_trials = 250
        else:
            raise ValueError(f"Unknown preset: {preset}")

        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "RFConfig":
        """Load from YAML; honors optional top-level 'preset' then merges keys.

        This method allows for easy construction of an RFConfig instance from a YAML file, with support for presets. If the YAML file specifies a top-level 'preset', the corresponding preset values are applied first, and then any additional keys in the YAML file override those preset values.

        Args:
            path (str): Path to the YAML configuration file.

        Returns:
            RFConfig: Config instance populated from the YAML file.
        """
        return load_yaml_to_dataclass(path, cls)

    def apply_overrides(self, overrides: Dict[str, Any] | None) -> "RFConfig":
        """Apply flat dot-key overrides (e.g., {'model.n_estimators': 500}).

        This method allows for easy application of overrides to the config instance using a flat dictionary structure.

        Args:
            overrides (Dict[str, Any] | None): Mapping of dot-key paths to values to override.

        Returns:
            RFConfig: This instance after applying overrides.
        """
        if overrides:
            apply_dot_overrides(self, overrides)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Return as nested dictionary.

        This method converts the config instance into a nested dictionary format, which can be useful for serialization or inspection.

        Returns:
            Dict[str, Any]: The config as a nested dictionary.
        """
        return asdict(self)

    def to_imputer_kwargs(self) -> Dict[str, Any]:
        """Map config fields to current ImputeRandomForest ``__init__`` kwargs.

        This method extracts relevant configuration fields and maps them to keyword arguments suitable for initializing the ImputeRandomForest class.

        Returns:
            Dict[str, Dict[str, Any]]: kwargs compatible with ImputeRandomForest(..., kwargs).
        """
        return {
            # General
            "prefix": self.io.prefix,
            "seed": self.io.seed,
            "n_jobs": self.io.n_jobs,
            "verbose": self.io.verbose,
            "debug": self.io.debug,
            # Model hyperparameters
            "model_n_estimators": self.model.n_estimators,
            "model_max_depth": self.model.max_depth,
            "model_min_samples_split": self.model.min_samples_split,
            "model_min_samples_leaf": self.model.min_samples_leaf,
            "model_max_features": self.model.max_features,
            "model_criterion": self.model.criterion,
            "model_validation_split": self.train.validation_split,
            "model_n_nearest_features": self.imputer.n_nearest_features,
            "model_max_iter": self.imputer.max_iter,
            # Simulation
            "sim_prop_missing": self.sim.prop_missing,
            "sim_strategy": self.sim.strategy,
            "sim_het_boost": self.sim.het_boost,
            # Plotting
            "plot_format": self.plot.fmt,
            "plot_fontsize": self.plot.fontsize,
            "plot_despine": self.plot.despine,
            "plot_dpi": self.plot.dpi,
            "plot_show_plots": self.plot.show,
        }


@dataclass
class HGBConfig:
    """Configuration for ImputeHistGradientBoosting.

    Mirrors the legacy __init__ signature and provides presets/YAML/overrides.
    Use `to_imputer_kwargs()` now, or switch the imputer to accept `config: HGBConfig`.

    Attributes:
        io (IOConfigSupervised): Run identity, logging, and seeds.
        model (HGBModelConfig): HistGradientBoosting hyperparameters.
        train (TrainConfigSupervised): Sample split for validation.
        imputer (ImputerConfigSupervised): IterativeImputer scaffolding (neighbors/iters).
        sim (SimConfigSupervised): Simulated missingness used during evaluation.
        plot (PlotConfigSupervised): Plot styling and export options.
        tune (TuningConfigSupervised): Optuna knobs (not required by HGB itself).
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

        This class method allows for easy construction of a HGBConfig instance with sensible defaults based on the chosen preset. Presets adjust both model capacity and training/tuning behavior across speed/quality tradeoffs.

        Args:
            preset (Literal["fast", "balanced", "thorough"]): One of {"fast", "balanced", "thorough"}. fast: Quick baseline; fewer trees; fewer imputer iters. balanced: Balances speed and model performance; moderate trees and imputer iters. thorough:  Prioritizes model performance; more trees; more imputer iterations.

        Returns:
            HGBConfig: Config with preset values applied.
        """
        cfg = cls()
        if preset == "fast":
            cfg.model.n_estimators = 50
            cfg.model.learning_rate = 0.15
            cfg.model.max_depth = None
            cfg.imputer.max_iter = 5
            cfg.io.n_jobs = 1
            cfg.tune.enabled = False
            cfg.tune.n_trials = 50
        elif preset == "balanced":
            cfg.model.n_estimators = 100
            cfg.model.learning_rate = 0.1
            cfg.model.max_depth = None
            cfg.imputer.max_iter = 10
            cfg.io.n_jobs = 1
            cfg.tune.enabled = False
            cfg.tune.n_trials = 100
        elif preset == "thorough":
            cfg.model.n_estimators = 500
            cfg.model.learning_rate = 0.08
            cfg.model.max_depth = None
            cfg.imputer.max_iter = 15
            cfg.io.n_jobs = 1
            cfg.tune.enabled = False
            cfg.tune.n_trials = 250
        else:
            raise ValueError(f"Unknown preset: {preset}")
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "HGBConfig":
        """Load from YAML; honors optional top-level 'preset' then merges keys.

        This method allows for easy construction of a HGBConfig instance from a YAML file, with support for presets. If the YAML file specifies a top-level 'preset', the corresponding preset values are applied first, and then any additional keys in the YAML file override those preset values.

        Args:
            path (str): Path to the YAML configuration file.

        Returns:
            HGBConfig: Config instance populated from the YAML file.
        """
        return load_yaml_to_dataclass(path, cls)

    def apply_overrides(self, overrides: Dict[str, Any] | None) -> "HGBConfig":
        """Apply flat dot-key overrides (e.g., {'model.learning_rate': 0.05}).

        This method allows for easy application of overrides to the configuration fields using a flat dot-key notation.

        Args:
            overrides (Dict[str, Any] | None): Mapping of dot-key paths to values to override.

        Returns:
            HGBConfig: This instance after applying overrides.
        """
        if overrides:
            apply_dot_overrides(self, overrides)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Return as nested dict.

        This method converts the configuration instance into a nested dictionary format, which can be useful for serialization or inspection.

        Returns:
            Dict[str, Any]: The config as a nested dictionary.
        """
        return asdict(self)

    def to_imputer_kwargs(self) -> Dict[str, Any]:
        """Map config fields to current ImputeHistGradientBoosting ``__init__`` kwargs.

        This method maps the configuration fields to the keyword arguments expected by the ImputeHistGradientBoosting class.

        Returns:
            Dict[str, Dict[str, Any]]: kwargs compatible with ImputeHistGradientBoosting(..., kwargs).
        """
        return {
            # General
            "prefix": self.io.prefix,
            "seed": self.io.seed,
            "n_jobs": self.io.n_jobs,
            "verbose": self.io.verbose,
            "debug": self.io.debug,
            # Model hyperparameters (note the mapping to sklearn's HGB)
            "model_n_estimators": self.model.n_estimators,  # -> max_iter
            "model_learning_rate": self.model.learning_rate,
            "model_n_iter_no_change": self.model.n_iter_no_change,
            "model_tol": self.model.tol,
            "model_max_depth": self.model.max_depth,
            "model_min_samples_leaf": self.model.min_samples_leaf,
            "model_max_features": self.model.max_features,
            "model_validation_split": self.train.validation_split,
            "model_n_nearest_features": self.imputer.n_nearest_features,
            "model_max_iter": self.imputer.max_iter,
            # Simulation
            "sim_prop_missing": self.sim.prop_missing,
            "sim_strategy": self.sim.strategy,
            "sim_het_boost": self.sim.het_boost,
            # Plotting
            "plot_format": self.plot.fmt,
            "plot_fontsize": self.plot.fontsize,
            "plot_despine": self.plot.despine,
            "plot_dpi": self.plot.dpi,
            "plot_show_plots": self.plot.show,
        }
