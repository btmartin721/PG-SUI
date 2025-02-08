Tutorial: Implementing New Imputation Models
============================================

This document provides a step-by-step guide to implementing a new model using the base class `BaseNNImputer`. The guide is structured to help you define the model architecture, configure the model parameters, and integrate it into the existing imputation framework.

Prerequisites
-------------

- Familiarity with PyTorch for defining and training neural network models.
- Knowledge of genotype data encoding and missing value handling.
- Understanding of the `BaseNNImputer` class and its methods.

Step 1: Define the Model Architecture
-------------------------------------

You need to create a new model class that defines the architecture of your model. The model should inherit from `torch.nn.Module` and implement the following methods:

1. **`__init__` method:** Define the network architecture, including layers, activations, and dropout if applicable.
2. **`forward` method:** Define the forward pass that computes the output given an input.
3. **`compute_loss` method:** Define the loss function to compute the loss between predictions and targets.

Example:

.. code-block:: python

    import torch
    from torch import nn

    from pgsui.impute.unsupervised.loss_functions import MaskedFocalLoss

    class MyNewModel(nn.Module):
        def __init__(self, n_features, latent_dim, hidden_layer_sizes, dropout_rate, activation):
            super(MyNewModel, self).__init__()
            
            layers = nn.ModuleList()

            # Build hidden layers
            input_dim = n_features
            for hidden_size in hidden_layer_sizes:
                layers.append(nn.Linear(input_dim, hidden_size))
                layers.append(nn.ReLU()) # or any other torch activation.
                layers.append(nn.Dropout(dropout_rate))
                input_dim = hidden_size

            # Latent layer
            layers.append(nn.Linear(input_dim, latent_dim))

            self.encoder = nn.Sequential(*layers)

            decoder_layer_sizes = list(reversed(hidden_layer_sizes))


            decoder_layers = nn.ModuleList()
            input_dim = latent_dim
            for hidden_size in decoder_layer_sizes:
                decoder_layers.append(nn.Linear(input_dim, hidden_size))
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Dropout(dropout_rate))
                input_dim = hidden_size

            decoder_layers.append(nn.Linear(decoder_layer_sizes[-1], n_features))

            self.decoder = nn.Sequential(*decoder_layers)

        def forward(self, x):
            z = self.encoder(x)
            reconstruction = self.decoder(z)
            return reconstruction

        def compute_loss(
            self,
            y: torch.Tensor,
            outputs: torch.Tensor,
            mask: torch.Tensor | None = None,
            class_weights: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Compute the masked focal loss between predictions and targets.

            This method computes the masked focal loss between the model predictions and the ground truth labels. The mask tensor is used to ignore certain values (< 0), and class weights can be provided to balance the loss.

            Args:
                y (torch.Tensor): Ground truth labels of shape (batch, seq).
                outputs (torch.Tensor): Model outputs.
                mask (torch.Tensor, optional): Mask tensor to ignore certain values. Default is None.
                class_weights (torch.Tensor, optional): Class weights for the loss. Default is None.

            Returns:
                torch.Tensor: Computed focal loss value.
            """
            if class_weights is None:
                class_weights = torch.ones(self.num_classes, device=y.device)

            if mask is None:
                mask = torch.ones_like(y, dtype=torch.bool)

            criterion = MaskedFocalLoss(alpha=class_weights, gamma=self.gamma)
            reconstruction_loss = criterion(outputs, y, valid_mask=mask)
            return reconstruction_loss


The model should define the architecture of the encoder and decoder networks, including the hidden layers, latent layer, and activation functions. The `forward` method should compute the output given an input tensor `x`.

Additionally, you must define your loss function within the model class to handle specific requirements. For example, the above `compute_loss` method computes the masked focal loss between the model predictions and the ground truth labels.
            

Step 2: Implement the Model Wrapper
-----------------------------------

Create a new class that wraps the model architecture. The wrapper should inherit from `BaseNNImputer` and implement the following methods:

1. **`fit` method:** Fit the model using the provided data.
2. **`transform` method:** Transform and impute the data using the trained model.
3. **`objective` method:** Define the objective function for hyperparameter tuning (if applicable).
4. **`set_best_params` method:** Set the best hyperparameters after tuning.

The model wrapper class should define the model-specific parameters, such as latent dimension, hidden layer sizes, dropout rate, and activation function. The class should also set the model name and model-specific parameters in the `__init__` method.

Other than the required methods, the following class attributes should be defined in the model wrapper class:

- `Model`: The model class that defines the architecture.
- `model_name`: The string name of the model (e.g., "ImputeMyNewModel").
- `model_params`: A dictionary containing the model-specific parameters.
- `logger`: A logger object for logging messages during training and evaluation.
- `sim_missing_mask_`: An attribute with simulated values in the input data for evaluation purposes.
- `original_missing_mask_`: An attribute to store the original missing values from the input data.
- `loader_`: A data loader object for training and validation data.
- `best_params_`: A dictionary containing the best hyperparameters after tuning.
- `tt_`: A transformer object for encoding the input data.
- `sim_`: A transformer object for simulating missing values.
- `plotter_`: A plotter object for visualizing the training process.
- `scorers_`: A dictionary of evaluation metrics.
- `num_features_`: The number of features in the input data.
- `num_classes_`: The number of classes in the input data.
- `class_weights_`: Class weights for
- `activate_`: Activation function for the final layer. Currently, only "softmax" is supported.
- `metrics_`: Evaluation metrics of the model.
- `tune_metric`: The metric to optimize during hyperparameter tuning.
- `weights_log_scale`: Whether to use a log scale for class weights.
- `weights_alpha`: The alpha parameter for class weights.
- `weights_normalize`: Whether to normalize class weights.
- `weights_temperature`: The temperature parameter for class weights.
- `l1_penalty`: The L1 penalty for regularization.
- `lr_`: The learning rate for training.
- `gamma`: The gamma parameter for the focal loss.
- `tune`: Whether to enable hyperparameter tuning.
- `tune_n_trials`: The number of trials for hyperparameter tuning.
- `n_jobs`: The number of parallel jobs for hyperparameter tuning.

Example:

.. code-block:: python

    from pgsui.impute.unsupervised.base import BaseNNImputer
    from my_module.my_new_model import MyNewModel

    class ImputeMyNewModel(BaseNNImputer):
        def __init__(self, genotype_data, **kwargs):

            self.model_name = "ImputeUBP"
            self.is_backprop = self.model_name in {"ImputeUBP", "ImputeNLPCA"}

            kwargs = {"prefix": prefix, "debug": debug, "verbose": verbose >= 1}
            logman = LoggerManager(__name__, **kwargs)
            self.logger = logman.get_logger()

            super().__init__(
                prefix=prefix,
                output_dir=output_dir,
                device=model_device,
                verbose=verbose,
                debug=debug,
            )
            self.Model = MyNewModel
            self.model_name = "ImputeMyNewModel"

            # Model-specific parameters
            self.latent_dim = kwargs.get('model_latent_dim', 2)
            self.hidden_layer_sizes = kwargs.get('model_hidden_layer_sizes', [128, 64])
            self.dropout_rate = kwargs.get('model_dropout_rate', 0.2)
            self.activation = kwargs.get('model_hidden_activation', 'relu')

            # Set other necessary kwargs

            # Prepare model parameters dictionary
            self.model_params = {
                "n_features": genotype_data.shape[1],
                "latent_dim": self.latent_dim,
                "hidden_layer_sizes": self.hidden_layer_sizes,
                "dropout_rate": self.dropout_rate,
                "activation": self.activation,
            }

        def fit(self, X: np.ndarray | pd.DataFrame | list | Tensor, y: Any | None = None):
            """Fit the model using the input data.

            This method fits the model using the input data. The ``transform`` method then transforms the input data and imputes the missing values using the trained model.

            Args:
                X (numpy.ndarray): Input data to fit the model.
                y (None): Ignored. Only for compatibility with the scikit-learn API.

            Returns:
                self: Returns an instance of the class.
            """
            self.logger.info(f"Fitting the {self.model_name} model...")

            # Activation for final layer.
            # Currently, only 'softmax' is supported.
            self.activate_ = "softmax"

            # Validate input and unify missing indicators
            # Ensure NaNs are replaced by -1
            X = validate_input_type(X)
            mask = np.logical_or(X < 0, np.isnan(X))
            X = X.astype(float)

            # Count number of classes for activation.
            # If 4 classes, use sigmoid, else use softmax.
            # Ignore missing values (-9) in counting of classes.
            # 1. Compute the number of distinct classes
            self.num_classes_ = len(np.unique(X[X >= 0 & ~mask]))
            self.model_params["num_classes"] = self.num_classes_

            # 2. Compute class weights
            self.class_weights_ = self.compute_class_weights(
                X,
                use_log_scale=self.weights_log_scale,
                alpha=self.weights_alpha,
                normalize=self.weights_normalize,
                temperature=self.weights_temperature,
                max_weight=20.0,
                min_weight=0.01,
            )

            # For final dictionary of hyperparameters
            self.best_params_ = self.model_params

            self.tt_, self.sim_, self.plotter_, self.scorers_ = self.init_transformers()

            Xsim, missing_masks = self.sim_.fit_transform(X)
            self.original_missing_mask_ = missing_masks["original"]
            self.sim_missing_mask_ = missing_masks["simulated"]
            self.all_missing_mask = missing_masks["all"]

            # Encode the data.
            Xsim_enc = self.tt_.fit_transform(Xsim)

            self.num_features_ = Xsim_enc.shape[1]
            self.model_params["n_features"] = self.num_features_

            self.Xsim_enc_ = Xsim_enc
            self.X_ = X

            if self.tune:
                self.tune_hyperparameters()

            self.loader_ = self.get_data_loaders(Xsim_enc, X, self.latent_dim)

            self.best_loss_, self.model_, self.history_ = self.train_final_model(
                self.loader_
            )
            self.metrics_ = self.evaluate_model(
                objective_mode=False, trial=None, model=self.model_, loader=self.loader_
            )
            self.plotter_.plot_history(self.history_)

            return self

        def transform(self, X: np.ndarray | pd.DataFrame | list | Tensor) -> np.ndarray:
            """Transform and impute the data using the trained model.

            This method transforms the input data and imputes the missing values using the trained model. The input data is transformed using the transformers and then imputed using the trained model. The ``fit()`` method must be called before calling this method.

            Args:
                X (numpy.ndarray): Data to transform and impute.

            Returns:
                numpy.ndarray: Transformed and imputed data.
            """
            Xenc = self.tt_.transform(validate_input_type(X))
            X_imputed = self.impute(Xenc, self.model_)

            self.plotter_.plot_gt_distribution(X, is_imputed=False)
            self.plotter_.plot_gt_distribution(X_imputed, is_imputed=True)

            return X_imputed

        def objective(self, trial: optuna.Trial, Model: torch.nn.Module) -> float:
            """Optimized Objective function for Optuna.

            This method is used as the objective function for hyperparameter tuning using Optuna. It is used to optimize the hyperparameters of the model.

            Args:
                trial (optuna.Trial): Optuna trial object.
                Model (torch.nn.Module): Model class to instantiate.

            Returns:
                float: The metric value to optimize. Which metric to use is based on the `tune_metric` attribute. Defaults to 'pr_macro', which works well with imbalanced classes.
            """
            # Efficient hyperparameter sampling
            latent_dim = trial.suggest_int("latent_dim", 2, 4)
            dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.05)
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 10)
            hidden_layer_sizes = [
                int(x) for x in np.linspace(16, 256, num_hidden_layers)[::-1]
            ]
            gamma = trial.suggest_float("gamma", 0.025, 5.0, step=0.025)
            activation = trial.suggest_categorical(
                "activation", ["relu", "elu", "selu", "leaky_relu"]
            )

            # Model parameters
            model_params = {
                "n_features": self.num_features_,
                "num_classes": self.num_classes_,
                "latent_dim": latent_dim,
                "dropout_rate": dropout_rate,
                "hidden_layer_sizes": hidden_layer_sizes,
                "activation": activation,
                "gamma": gamma,
            }

            # Build and initialize model
            model = self.build_model(Model, model_params)
            model.apply(self.initialize_weights)

            train_loader = self.get_data_loaders(self.Xsim_enc_, self.X_, latent_dim)

            try:
                # Train and validate the model
                _, model = self.train_and_validate_model(
                    model=model,
                    loader=train_loader,
                    l1_penalty=self.l1_penalty,
                    lr=learning_rate,
                    trial=trial,
                )

                if model is None:
                    self.logger.warning(
                        f"Trial {trial.number} pruned due to failed model training. Model was NoneType."
                    )
                    raise optuna.exceptions.TrialPruned()

                # Efficient evaluation
                metrics = self.evaluate_model(
                    objective_mode=True, trial=trial, model=model, loader=train_loader
                )

                if self.tune_metric not in metrics:
                    msg = f"Invalid tuning metric: {self.tune_metric}"
                    self.logger.error(msg)
                    raise KeyError(msg)

                return metrics[self.tune_metric]

            except Exception as e:
                self.logger.warning(f"Trial {trial.number} pruned due to exception: {e}")
                raise optuna.exceptions.TrialPruned()

            finally:
                self.reset_weights(model.phase1_decoder)
                self.reset_weights(model.phase23_decoder)
                self.reset_weights(model)

        def set_best_params(self, best_params: dict) -> dict:
            """Set the best hyperparameters.

            This method sets the best hyperparameters for the model after tuning.

            Args:
                best_params (dict): Dictionary of best hyperparameters.

            Returns:
                dict: Dictionary of best hyperparameters. The best hyperparameters are set as attributes of the class.
            """
            # Load best hyperparameters
            self.latent_dim = best_params["latent_dim"]
            self.hidden_layer_sizes = np.linspace(
                16, 256, best_params["num_hidden_layers"]
            ).astype(int)[::-1]
            self.dropout_rate = best_params["dropout_rate"]
            self.lr_ = best_params["learning_rate"]
            self.gamma = best_params["gamma"]

            best_params_ = {
                "n_features": self.num_features_,
                "latent_dim": self.latent_dim,
                "hidden_layer_sizes": self.hidden_layer_sizes,
                "dropout_rate": self.dropout_rate,
                "activation": self.activation,
                "gamma": self.gamma,
            }

            return best_params_

The `ImputeMyNewModel` class inherits from `BaseNNImputer` and defines the model-specific parameters such as latent dimension, hidden layer sizes, dropout rate, and activation function. The `fit` method initializes and trains the model, while the `transform` method imputes the missing values in the input data. The `objective` method defines the objective function for hyperparameter tuning using Optuna. The `set_best_params` method sets the best hyperparameters after tuning.

Step 3: Configure and Train the Model
-------------------------------------

You can configure the new model by setting the appropriate parameters during instantiation and call the `fit` and `transform` methods for training and prediction.

Example:

.. code-block:: python

    from snpio import VCFReader, GenotypeEncoder

    # Load genotype data
    genotype_data = VCFReader(filename="example.vcf", popmapfile="example.popmap")

    # Instantiate and configure the model
    model = ImputeMyNewModel(
        genotype_data=genotype_data,
        model_latent_dim=3,
        model_hidden_layer_sizes=[256, 128],
        model_dropout_rate=0.3,
        model_hidden_activation="relu",
        model_learning_rate=0.001
    )

    # Encode the genotype data.
    ge = GenotypeEncoder(genotype_data)

    # Train the model and impute the missing values.
    imputed_data = model.fit_transform(ge.genotypes_012)

Step 4: Optional - Hyperparameter Tuning
----------------------------------------

If you want to enable hyperparameter tuning, define the `objective` method and use Optuna to optimize the hyperparameters by setting the `tune` parameter to `True`. You can also specify the number of trials and the number of parallel jobs for tuning. The `objective` method should return the metric value to optimize during tuning, and the `set_best_params` method should set the best hyperparameters as class attributes after tuning.

Example:

.. code-block:: python

    from pgsui import ImputeMyNewModel

    model = ImputeMyNewModel(
        genotype_data=genotype_data,
        model_latent_dim=3,
        model_hidden_layer_sizes=[256, 128],
        model_dropout_rate=0.3,
        model_hidden_activation="relu",
        model_learning_rate=0.001,
        tune=True
        tune_n_trials=100,
        n_jobs=8,
    )
    
Final Remarks
-------------

By following these steps, you can define, train, and evaluate a new model using the provided framework. Be sure to implement any custom behavior in the `objective`, `fit`, `transform`, and `set_best_params` methods to match the specific needs of your model. Additionally, you can extend the model wrapper class with additional methods for evaluation, visualization, or other tasks as needed. The provided framework is designed to be flexible and extensible, allowing you to implement and integrate new models efficiently.
