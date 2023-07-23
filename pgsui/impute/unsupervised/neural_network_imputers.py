# Standard Library Imports
import logging
import os
import pprint
import sys
import warnings

# Third-party Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Grid search imports
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Scikit-learn imports
from sklearn.base import BaseEstimator, TransformerMixin

# Genetic algorithm grid search imports
from sklearn_genetic import GASearchCV
from sklearn_genetic.callbacks import ConsecutiveStopping, ProgressBar
from sklearn_genetic.plots import plot_fitness_evolution

# Import tensorflow with reduced warnings.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").disabled = True
warnings.filterwarnings("ignore", category=UserWarning)

# noinspection PyPackageRequirements
import tensorflow as tf

# Disable can't find cuda .dll errors. Also turns of GPU support.
tf.config.set_visible_devices([], "GPU")

from tensorflow.python.util import deprecation

# Disable warnings and info logs.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)


# Monkey patching deprecation utils to supress warnings.
# noinspection PyUnusedLocal
def deprecated(
    date, instructions, warn_once=True
):  # pylint: disable=unused-argument
    def deprecated_wrapper(func):
        return func

    return deprecated_wrapper


deprecation.deprecated = deprecated

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
)

# For development purposes
# from memory_profiler import memory_usage

# Custom module imports
try:
    from ...utils.misc import timer
    from ...utils.misc import isnotebook
    from ...utils.misc import validate_input_type
    from .neural_network_methods import NeuralNetworkMethods, DisabledCV
    from ...utils.scorers import Scorers
    from ...utils.plotting import Plotting
    from .callbacks import (
        UBPCallbacks,
        VAECallbacks,
        CyclicalAnnealingCallback,
    )
    from .keras_classifiers import VAEClassifier, MLPClassifier, SAEClassifier
    from ...data_processing.transformers import (
        SimGenotypeDataTransformer,
        AutoEncoderFeatureTransformer,
    )
except (ModuleNotFoundError, ValueError, ImportError):
    from utils.misc import timer
    from utils.misc import isnotebook
    from utils.misc import validate_input_type
    from impute.unsupervised.neural_network_methods import (
        NeuralNetworkMethods,
        DisabledCV,
    )
    from utils.scorers import Scorers
    from utils.plotting import Plotting
    from impute.unsupervised.callbacks import (
        UBPCallbacks,
        VAECallbacks,
        CyclicalAnnealingCallback,
    )
    from impute.unsupervised.keras_classifiers import (
        VAEClassifier,
        MLPClassifier,
        SAEClassifier,
    )
    from data_processing.transformers import (
        SimGenotypeDataTransformer,
        AutoEncoderFeatureTransformer,
    )

is_notebook = isnotebook()

if is_notebook:
    from tqdm.notebook import tqdm as progressbar
else:
    from tqdm import tqdm as progressbar

from tqdm.keras import TqdmCallback


class BaseNNImputer(BaseEstimator, TransformerMixin):
    """Base transformer class for neural network imputers.

    Args:
        genotype_data (GenotypeData): Input GenotypeData instance.

        prefix (str, optional): Prefix for output files. Defaults to "output".

        gridparams (Dict[str, Any] or None, optional): Dictionary with keys=keyword arguments for the specified estimator and values=lists of parameter values or distributions. If using GridSearchCV, distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). ``gridparams`` will be used for a randomized grid search with cross-validation. If using the genetic algorithm grid search (GASearchCV) by setting ``ga=True``\, the parameters can be specified as ``sklearn_genetic.space`` objects. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. If ``gridparams=None``\, a grid search is not performed. Defaults to None.

        disable_progressbar (bool, optional): Whether to disable the tqdm progress bar. Useful if you are doing the imputation on e.g. a high-performance computing cluster, where sometimes tqdm does not work correctly. If False, uses tqdm progress bar. If True, does not use tqdm. Defaults to False.

        batch_size (int, optional): Batch size per epoch to train the model with.

        n_components (int, optional): Number of components to use as the input data. Defaults to 3.

        early_stop_gen (int, optional): Early stopping criterion for epochs. Training will stop if the loss (error) does not decrease past the tolerance level for ``early_stop_gen`` epochs. Will save the optimal model and reload it once ``early_stop_gen`` has been reached. Defaults to 25.

        num_hidden_layers (int, optional): Number of hidden layers to use in the model. Adjust if overfitting occurs. Defaults to 3.

        hidden_layer_sizes (str, List[int], List[str], or int, optional): Number of neurons to use in hidden layers. If string or a list of strings is supplied, the strings must be either "midpoint", "sqrt", or "log2". "midpoint" will calculate the midpoint as ``(n_features + n_components) / 2``. If "sqrt" is supplied, the square root of the number of features will be used to calculate the output units. If "log2" is supplied, the units will be calculated as ``log2(n_features)``. hidden_layer_sizes will calculate and set the number of output units for each hidden layer. If one string or integer is supplied, the model will use the same number of output units for each hidden layer. If a list of integers or strings is supplied, the model will use the values supplied in the list, which can differ. The list length must be equal to the ``num_hidden_layers``. Defaults to "midpoint".

        optimizer (str, optional): The optimizer to use with gradient descent. Possible value include: "adam", "sgd", "adagrad", "adadelta", "adamax", "ftrl", "nadam", and "rmsprop" are supported. See tf.keras.optimizers for more info. Defaults to "adam".

        hidden_activation (str, optional): The activation function to use for the hidden layers. See tf.keras.activations for more info. Commonly used activation functions include "elu", "relu", and "sigmoid". Defaults to "elu".

        learning_rate (float, optional): The learning rate for the optimizers. Adjust if the loss is learning too slowly. Defaults to 0.1.

        lr_patience (int, optional): Number of epochs with no loss improvement to wait before reducing the learning rate.

        epochs (int, optional): Maximum number of epochs to run if the ``early_stop_gen`` criterion is not met.

        weights_initializer (str, optional): Initializer to use for the model weights. See tf.keras.initializers for more info. Defaults to "glorot_normal".

        l1_penalty (float, optional): L1 regularization penalty to apply to reduce overfitting. Defaults to 0.01.

        l2_penalty (float, optional): L2 regularization penalty to apply to reduce overfitting. Defaults to 0.01.

        dropout_rate (float, optional): Dropout rate during training to reduce overfitting. Must be a float between 0 and 1. Defaults to 0.2.

        recurrent_weight (float, optional): Recurrent weight to calculate predictions. Defaults to 0.5.

        sample_weights (str or Dict[int, float], optional): Whether to weight each genotype by its class frequency. If ``sample_weights='auto'`` then it automatically calculates sample weights based on genotype class frequencies per locus; for example, if there are a lot more 0s and fewer 2s, then it will balance out the classes by weighting each genotype accordingly. ``sample_weights`` can also be a dictionary with the genotypes (0, 1, and 2) as the keys and the weights as the keys. If ``sample_weights`` is anything else, then they are not calculated. Defaults to False.

        grid_iter (int, optional): Number of iterations for grid search. Defaults to 50.

        gridsearch_method (str, optional): Grid search method to use. Possible options include: 'gridsearch', 'randomized_gridsearch', and 'genetic_algorithm'. 'gridsearch' runs all possible permutations of parameters, 'randomized_gridsearch' runs a random subset of parameters, and 'genetic_algorithm' uses a genetic algorithm gridsearch (via GASearchCV). Defaults to 'gridsearch'.

        ga_kwargs (Dict[str, Any] or None): Keyword arguments to be passed to a Genetic Algorithm grid search. Only used if ``ga==True``\.

        scoring_metric (str, optional): Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options. Defaults to "accuracy".

        sim_strategy (str, optional): Strategy to use for simulating missing data. Only used to validate the accuracy of the imputation. The final model will be trained with the non-simulated dataset. Supported options include: "random", "nonrandom", and "nonrandom_weighted". "random" randomly simulates missing data. When set to "nonrandom", branches from ``GenotypeData.guidetree`` will be randomly sampled to generate missing data on descendant nodes. For "nonrandom_weighted", missing data will be placed on nodes proportionally to their branch lengths (e.g., to generate data distributed as might be the case with mutation-disruption of RAD sites). Defaults to "random".

        sim_prop_missing (float, optional): Proportion of missing data to simulate with the SimGenotypeDataTransformer. Defaults to 0.1.

        n_jobs (int, optional): Number of parallel jobs to use in the grid search if ``gridparams`` is not None. -1 means use all available processors. Defaults to 1.

        verbose (int, optional): Verbosity setting. Can be 0, 1, or 2. 0 is the least and 2 is the most verbose. Defaults to 0.

        ToDo:
            Fix sample_weight for multi-label encodings.
    """

    def __init__(
        self,
        activate,
        nn_method,
        num_classes,
        act_func,
        *,
        genotype_data=None,
        prefix="imputer",
        gridparams=None,
        disable_progressbar=False,
        batch_size=32,
        n_components=3,
        early_stop_gen=25,
        num_hidden_layers=3,
        hidden_layer_sizes="midpoint",
        optimizer="adam",
        hidden_activation="elu",
        learning_rate=0.01,
        lr_patience=1,
        epochs=100,
        weights_initializer="glorot_normal",
        l1_penalty=0.0001,
        l2_penalty=0.0001,
        dropout_rate=0.2,
        sample_weights=False,
        grid_iter=80,
        gridsearch_method="gridsearch",
        ga_kwargs=None,
        scoring_metric="auc_macro",
        sim_strategy="random",
        sim_prop_missing=0.2,
        n_jobs=1,
        verbose=0,
        kl_beta=tf.Variable(1.0, trainable=False),
        validation_split=0.0,
        nlpca=False,
        testing=False,
    ):
        self.activate = activate
        self.act_func_ = act_func
        self.num_classes = num_classes
        self.testing = testing
        self.nn_method_ = nn_method

        self.genotype_data = genotype_data
        self.prefix = prefix
        self.gridparams = gridparams
        self.disable_progressbar = disable_progressbar
        self.batch_size = batch_size
        self.n_components = n_components

        self.early_stop_gen = early_stop_gen
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.optimizer = optimizer
        self.hidden_activation = hidden_activation
        self.learning_rate = learning_rate
        self.lr_patience = lr_patience
        self.epochs = epochs
        self.weights_initializer = weights_initializer
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.dropout_rate = dropout_rate
        self.sample_weights = sample_weights
        self.grid_iter = grid_iter
        self.gridsearch_method = gridsearch_method
        self.ga_kwargs = ga_kwargs
        self.scoring_metric = scoring_metric
        self.sim_strategy = sim_strategy
        self.sim_prop_missing = sim_prop_missing
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.kl_beta = kl_beta
        self.validation_split = validation_split
        self.nlpca = nlpca

        self.run_gridsearch_ = False if self.gridparams is None else True
        self.is_multiclass_ = True if self.num_classes != 4 else False

        # Simulate missing data and get missing masks.
        self.sim = SimGenotypeDataTransformer(
            self.genotype_data,
            prop_missing=self.sim_prop_missing,
            strategy=self.sim_strategy,
            mask_missing=True,
        )

        # Binary encode y to get y_train.
        self.tt_ = AutoEncoderFeatureTransformer(
            num_classes=self.num_classes, activate=self.activate
        )

    @timer
    def fit(self, X):
        """Train the VAE model on input data X.

        Args:
            X (pandas.DataFrame, numpy.ndarray, or List[List[int]]): Input 012-encoded genotypes.

        Returns:
            self: Current instance; allows method chaining.

        Raises:
            TypeError: Must be either pandas.DataFrame, numpy.ndarray, or List[List[int]].
        """
        # Treating y as X here for compatibility with UBP/NLPCA.
        # With VAE, y=X anyways.
        y = X
        y = validate_input_type(y, return_type="array")

        self.nn_ = NeuralNetworkMethods()
        plotting = Plotting()

        if self.gridsearch_method == "genetic_algorithm":
            self.ga_ = True
        else:
            self.ga_ = False

        self.y_original_ = y.copy()
        self.y_simulated_ = self.sim.fit_transform(self.y_original_)

        # Get values where original value was not missing but missing data
        # was simulated.
        self.sim_missing_mask_ = self.sim.sim_missing_mask_

        # Original missing data.
        self.original_missing_mask_ = self.sim.original_missing_mask_

        # Both simulated and original missing data.
        self.all_missing_ = self.sim.all_missing_mask_

        # Just y_original with missing values encoded as -1.
        y_train = self.tt_.fit_transform(self.y_original_)

        if self.gridparams is not None:
            self.scoring_metrics_ = [
                "precision_recall_macro",
                "precision_recall_micro",
                "f1_score",
                "auc_macro",
                "auc_micro",
                "accuracy",
                "hamming",
            ]

        (
            logfile,
            callbacks,
            compile_params,
            model_params,
            fit_params,
        ) = self._initialize_parameters(y_train)

        if self.nn_method_ == "VAE":
            func = self.run_vae
        elif self.nn_method_ == "SAE":
            func = self.run_sae
        elif self.nn_method_ == "NLPCA":
            func = self.run_nlpca
        elif self.nn_method_ == "UBP":
            func = self.run_ubp
        else:
            raise ValueError(f"Invalid nn_method specified: {self.nn_method_}")

        (
            self.models_,
            self.histories_,
            self.best_params_,
            self.best_score_,
            self.best_estimator_,
            self.search_,
            self.metrics_,
        ) = func(
            self.y_original_,
            y_train,
            model_params,
            compile_params,
            fit_params,
        )

        if (
            self.best_params_ is not None
            and "optimizer__learning_rate" in self.best_params_
        ):
            self.best_params_["learning_rate"] = self.best_params_.pop(
                "optimizer__learning_rate"
            )

        if self.gridparams is not None:
            if self.verbose > 0:
                print("\nBest found parameters:")
                pprint.pprint(self.best_params_)
                print(f"\nBest score: {self.best_score_}")
            plotting.plot_grid_search(
                self.search_.cv_results_, self.nn_method_, self.prefix
            )

        plotting.plot_history(
            self.histories_, self.nn_method_, prefix=self.prefix
        )
        plotting.plot_metrics(
            self.metrics_, self.num_classes, self.prefix, self.nn_method_
        )

        if self.ga_:
            plot_fitness_evolution(self.search_)
            plt.savefig(
                os.path.join(
                    f"{self.prefix}_output",
                    "plots",
                    "Unsupervised",
                    self.nn_method_,
                    "fitness_evolution.pdf",
                ),
                bbox_inches="tight",
                facecolor="white",
            )
            plt.cla()
            plt.clf()
            plt.close()

            g = plotting.plot_search_space(self.search_)
            plt.savefig(
                os.path.join(
                    f"{self.prefix}_output",
                    "plots",
                    "Unsupervised",
                    self.nn_method_,
                    "search_space.pdf",
                ),
                bbox_inches="tight",
                facecolor="white",
            )
            plt.cla()
            plt.clf()
            plt.close()

        return self

    def transform(self, X):
        """Predict and decode imputations and return transformed array.

        Args:
            X (pandas.DataFrame, numpy.ndarray, or List[List[int]]): Input data to transform.

        Returns:
            numpy.ndarray: Imputed data.
        """
        y = X
        y = validate_input_type(y, return_type="array")

        if self.nn_method_ not in ["UBP", "NLPCA"]:
            model = self.models_[0]
        else:
            if len(self.models_) == 1:
                model = self.models_[0]
            else:
                model = self.models_[-1]

        y_true = y.copy()
        y_train = self.tt_.transform(y_true)
        y_true_1d = y_true.ravel()
        y_size = y_true.size
        y_missing_idx = np.flatnonzero(self.original_missing_mask_)

        if self.nn_method_ == "VAE":
            y_pred = model(
                tf.convert_to_tensor(y_train),
                training=False,
            )
        elif self.nn_method_ == "SAE":
            y_pred = model(y_train, training=False)
        else:
            y_pred = model(model.V_latent, training=False)
        y_pred = self.tt_.inverse_transform(y_pred)

        y_pred_decoded = self.nn_.decode_masked(
            y_train,
            y_pred,
            is_multiclass=self.is_multiclass_,
        )
        # y_pred_decoded, y_pred_certainty = self.nn_.decode_masked(
        #     y_train, y_pred, return_proba=True
        # )

        y_pred_1d = y_pred_decoded.ravel()

        # Only replace originally missing values at missing indexes.
        for i in np.arange(y_size):
            if i in y_missing_idx:
                y_true_1d[i] = y_pred_1d[i]

        self.nn_.write_gt_state_probs(
            y_pred,
            y_pred_1d,
            y_true,
            y_true_1d,
            self.nn_method_,
            self.sim_missing_mask_,
            self.original_missing_mask_,
            prefix=self.prefix,
        )

        Plotting.plot_confusion_matrix(
            y_true_1d, y_pred_1d, self.nn_method_, prefix=self.prefix
        )

        # if self.nn_method_ == "VAE":
        # Plotting.plot_label_clusters(z_mean, y_true_1d)

        # Return to original shape.
        return np.reshape(y_true_1d, y_true.shape)

    def run_clf(
        self,
        y_train,
        y_true,
        model_params,
        compile_params,
        fit_params,
        ubp_weights=None,
        phase=None,
        scoring=None,
        testing=False,
        **kwargs,
    ):
        """Run KerasClassifier with neural network model and grid search.

        Args:
            y_train (numpy.ndarray): Onehot-encoded training input data of shape (n_samples, n_features, num_classes).

            y_true (numpy.ndarray): Original 012-encoded input data of shape (n_samples, n_features).

            model_params (Dict[str, Any]): Dictionary with model parameters to be passed to KerasClassifier model.

            compile_params (Dict[str, Any]): Dictionary with params to be passed to Keras model.compile() in KerasClassifier.

            fit_params (Dict[str, Any]): Dictionary with parameters to be passed to fit in KerasClassifier.

            scoring (Dict[str, Callable], optional): Multimetric scorer made using sklearn.metrics.make_scorer. To be used with grid search.

        Returns:
            List[tf.keras.Model]: List of keras model objects. One for each phase (len=1 if NLPCA, len=3 if UBP).

            List[Dict[str, float]]: List of dictionaries with best neural network model history.

            Dict[str, Any] or None: Best parameters found during a grid search, or None if a grid search was not run.

            float: Best score obtained during grid search.

            tf.keras.Model: Best model found during grid search.

            sklearn.model_selection object (GridSearchCV, RandomizedSearchCV) or GASearchCV object.

            Dict[str, Any]: Per-class, micro, and macro-averaged metrics including accuracy, ROC-AUC, and Precision-Recall with Average Precision scores.
        """
        # This reduces memory usage.
        # tensorflow builds graphs that
        # will stack if not cleared before
        # building a new model.
        tf.keras.backend.clear_session()
        self.nn_.reset_seeds()

        model = None
        if self.nn_method_ in ["UBP", "NLPCA"]:
            V = model_params.pop("V")
            if phase is not None:
                desc = f"Epoch (Phase {phase}): "
            else:
                desc = "Epoch: "

        else:
            desc = "Epoch: "

        if not self.disable_progressbar and not self.run_gridsearch_:
            fit_params["callbacks"][-1] = TqdmCallback(
                epochs=self.epochs, verbose=self.verbose, desc=desc
            )

        if self.nn_method_ == "VAE":
            clf = VAEClassifier(
                **model_params,
                optimizer=compile_params["optimizer"],
                optimizer__learning_rate=compile_params["learning_rate"],
                loss=compile_params["loss"],
                metrics=compile_params["metrics"],
                run_eagerly=compile_params["run_eagerly"],
                callbacks=fit_params["callbacks"],
                epochs=fit_params["epochs"],
                verbose=0,
                num_classes=self.num_classes,
                activate=self.act_func_,
                fit__validation_split=fit_params["validation_split"],
                score__missing_mask=self.sim_missing_mask_,
                score__scoring_metric=self.scoring_metric,
                score__num_classes=self.num_classes,
                score__n_classes=self.num_classes,
            )
        elif self.nn_method_ == "SAE":
            clf = SAEClassifier(
                **model_params,
                optimizer=compile_params["optimizer"],
                optimizer__learning_rate=compile_params["learning_rate"],
                loss=compile_params["loss"],
                metrics=compile_params["metrics"],
                callbacks=fit_params["callbacks"],
                epochs=fit_params["epochs"],
                verbose=0,
                activate=self.act_func_,
                fit__validation_split=fit_params["validation_split"],
                score__missing_mask=self.sim_missing_mask_,
                score__scoring_metric=self.scoring_metric,
                score__num_classes=self.num_classes,
                score__n_classes=self.num_classes,
            )
        else:
            clf = MLPClassifier(
                V,
                y_train,
                **model_params,
                ubp_weights=ubp_weights,
                optimizer=compile_params["optimizer"],
                optimizer__learning_rate=compile_params["learning_rate"],
                loss=compile_params["loss"],
                metrics=compile_params["metrics"],
                epochs=fit_params["epochs"],
                phase=phase,
                callbacks=fit_params["callbacks"],
                validation_split=fit_params["validation_split"],
                verbose=0,
                score__missing_mask=self.sim_missing_mask_,
                score__scoring_metric=self.scoring_metric,
            )

        if self.run_gridsearch_:
            # Cannot do CV because there is no way to use test splits
            # given that the input gets refined. If using a test split,
            # then it would just be the randomly initialized values and
            # would not accurately represent the model.
            # Thus, we disable cross-validation for the grid searches.
            cross_val = DisabledCV()
            verbose = False if self.verbose == 0 else True

            if self.ga_:
                # Stop searching if GA sees no improvement.
                callback = [
                    ConsecutiveStopping(
                        generations=self.early_stop_gen, metric="fitness"
                    )
                ]

                if not self.disable_progressbar:
                    callback.append(ProgressBar())

                # Do genetic algorithm
                # with HiddenPrints():
                search = GASearchCV(
                    estimator=clf,
                    cv=cross_val,
                    scoring=scoring,
                    generations=self.grid_iter,
                    param_grid=self.gridparams,
                    n_jobs=self.n_jobs,
                    refit=self.scoring_metric,
                    verbose=verbose,
                    **self.ga_kwargs,
                    error_score="raise",
                )

                if self.nn_method_ in ["UBP", "NLPCA"]:
                    search.fit(V[self.n_components], y=y_true)
                else:
                    search.fit(y_true, y_true, callbacks=callback)

            else:
                # Write GridSearchCV to log file instead of STDOUT.
                if self.verbose >= 10:
                    old_stdout = sys.stdout
                    log_file = open(
                        os.path.join(
                            f"{self.prefix}_output",
                            "logs",
                            "Unsupervised",
                            self.nn_method_,
                            "gridsearch_progress_log.txt",
                        ),
                        "w",
                    )
                    sys.stdout = log_file

                if self.gridsearch_method.lower() == "gridsearch":
                    # Do GridSearchCV
                    search = GridSearchCV(
                        clf,
                        param_grid=self.gridparams,
                        n_jobs=self.n_jobs,
                        cv=cross_val,
                        scoring=scoring,
                        refit=self.scoring_metric,
                        verbose=self.verbose * 4,
                        error_score="raise",
                    )

                elif self.gridsearch_method.lower() == "randomized_gridsearch":
                    search = RandomizedSearchCV(
                        clf,
                        param_distributions=self.gridparams,
                        n_iter=self.grid_iter,
                        n_jobs=self.n_jobs,
                        cv=cross_val,
                        scoring=scoring,
                        refit=self.scoring_metric,
                        verbose=verbose * 4,
                        error_score="raise",
                    )

                else:
                    raise ValueError(
                        f"Invalid gridsearch_method specified: "
                        f"{self.gridsearch_method}"
                    )

                if self.nn_method_ in ["UBP", "NLPCA"]:
                    search.fit(V[self.n_components], y=y_true)
                else:
                    search.fit(y_true, y=y_true)

                if self.verbose >= 10:
                    # Make sure to revert STDOUT back to original.
                    sys.stdout = old_stdout
                    log_file.close()

            best_params = search.best_params_
            best_score = search.best_score_
            best_clf = search.best_estimator_

            fp = os.path.join(
                f"{self.prefix}_output",
                "reports",
                "Unsupervised",
                self.nn_method_,
                f"cvresults_{self.nn_method_}.csv",
            )

            cv_results = pd.DataFrame(search.cv_results_)
            cv_results.to_csv(fp, index=False)

        else:
            if self.nn_method_ in ["UBP", "NLPCA"]:
                clf.fit(V[self.n_components], y=y_true)
            else:
                clf.fit(y_true, y=y_true)
            best_params = None
            best_score = None
            search = None
            best_clf = clf

        model = best_clf.model_
        best_history = best_clf.history_

        if self.nn_method_ == "VAE":
            y_pred = model(
                tf.convert_to_tensor(y_train),
                training=False,
            )
            y_pred = self.tt_.inverse_transform(y_pred)
        elif self.nn_method_ == "SAE":
            y_pred = model(y_train, training=False)
            y_pred = self.tt_.inverse_transform(y_pred)
        elif self.nn_method_ in ["UBP", "NLPCA"]:
            # Third run_clf function
            y_pred_proba = model(model.V_latent, training=False)
            y_pred = self.tt_.inverse_transform(y_pred_proba)

        # Get metric scores.
        metrics = Scorers.scorer(
            y_true,
            y_pred,
            missing_mask=self.sim_missing_mask_,
            num_classes=self.num_classes,
            testing=self.testing,
        )

        if self.nn_method_ in ["UBP", "NLPCA"]:
            return (
                V,
                model,
                best_history,
                best_params,
                best_score,
                best_clf,
                search,
                metrics,
            )
        else:
            return (
                model,
                best_history,
                best_params,
                best_score,
                best_clf,
                search,
                metrics,
            )

    def _initialize_parameters(self, y_train):
        """Initialize important parameters.

        Args:
            y_train (numpy.ndarray): Training subset of original input data.

        Returns:
            Dict[str, Any]: Parameters to use for model.compile().
            Dict[str, Any]: Other parameters to pass to KerasClassifier().
            Dict[str, Any]: Parameters to pass to fit_params() in grid search.
        """
        # For CSVLogger() callback.

        append = True if self.nn_method_ == "UBP" else False
        logfile = os.path.join(
            f"{self.prefix}_output",
            "logs",
            "Unsupervised",
            self.nn_method_,
            "training_log.csv",
        )

        callbacks = [
            CSVLogger(filename=logfile, append=append),
            ReduceLROnPlateau(
                patience=self.lr_patience, min_lr=1e-6, min_delta=1e-6
            ),
        ]

        if self.nn_method_ in ["VAE", "SAE"]:
            callbacks.append(VAECallbacks())

            if self.nn_method_ == "VAE":
                callbacks.append(
                    CyclicalAnnealingCallback(
                        self.epochs, schedule_type="sigmoid"
                    )
                )
        else:
            callbacks.append(UBPCallbacks())

        search_mode = True if self.run_gridsearch_ else False

        if not self.disable_progressbar and not search_mode:
            callbacks.append(
                TqdmCallback(epochs=self.epochs, verbose=0, desc="Epoch: ")
            )

        if self.nn_method_ in ["UBP", "NLPCA"]:
            vinput = self._initV(y_train, search_mode)
            compile_params = self.nn_.set_compile_params(self.optimizer)
        else:
            vae = True if self.nn_method_ in ["VAE", "SAE"] else False

        if self.sample_weights == "auto" or self.sample_weights == "logsmooth":
            # Get class weights for each column.
            sample_weights = self.nn_.get_class_weights(
                self.y_original_,
                self.original_missing_mask_,
                return_1d=False,
                method=self.sample_weights,
            )
            sample_weights = self.nn_.normalize_data(sample_weights)

        elif isinstance(self.sample_weights, dict):
            for i in range(self.num_classes):
                if self.sample_weights[i] == 0.0:
                    self.sim_missing_mask_[self.y_original_ == i] = False

            sample_weights = self.nn_.get_class_weights(
                self.y_original_, user_weights=self.sample_weights
            )

        else:
            sample_weights = None

        vae = True if self.nn_method_ == "VAE" else False

        compile_params = self.nn_.set_compile_params(
            self.optimizer,
            sample_weights,
            vae=vae,
            act_func=self.act_func_,
        )

        compile_params["learning_rate"] = self.learning_rate

        if self.nn_method_ in ["VAE", "SAE"]:
            model_params = {
                "y": y_train,
                "batch_size": self.batch_size,
                "sample_weight": sample_weights,
                "missing_mask": self.original_missing_mask_,
                "output_shape": y_train.shape[1],
                "weights_initializer": self.weights_initializer,
                "n_components": self.n_components,
                "hidden_layer_sizes": self.hidden_layer_sizes,
                "num_hidden_layers": self.num_hidden_layers,
                "hidden_activation": self.hidden_activation,
                "l1_penalty": self.l1_penalty,
                "l2_penalty": self.l2_penalty,
                "dropout_rate": self.dropout_rate,
            }

            if self.nn_method_ == "VAE":
                model_params["kl_beta"] = (1.0 / y_train.shape[0],)
        else:
            model_params = {
                "V": vinput,
                "y_train": y_train,
                "batch_size": self.batch_size,
                "missing_mask": self.original_missing_mask_,
                "output_shape": y_train.shape[1],
                "weights_initializer": self.weights_initializer,
                "n_components": self.n_components,
                "hidden_layer_sizes": self.hidden_layer_sizes,
                "num_hidden_layers": self.num_hidden_layers,
                "hidden_activation": self.hidden_activation,
                "l1_penalty": self.l1_penalty,
                "l2_penalty": self.l2_penalty,
                "dropout_rate": self.dropout_rate,
                "num_classes": self.num_classes,
            }

        model_params["sample_weight"] = sample_weights

        fit_verbose = 1 if self.verbose == 2 else 0

        fit_params = {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "callbacks": callbacks,
            "shuffle": True,
            "verbose": fit_verbose,
            "sample_weight": sample_weights,
        }

        if self.nn_method_ in ["VAE", "SAE"]:
            shuffle = True
            fit_params["validation_split"] = self.validation_split
        else:
            shuffle = False
            fit_params["validation_split"] = 0.0

        fit_params["shuffle"] = shuffle

        if self.run_gridsearch_ and "learning_rate" in self.gridparams:
            self.gridparams["optimizer__learning_rate"] = self.gridparams[
                "learning_rate"
            ]

            self.gridparams.pop("learning_rate")

        return (
            logfile,
            callbacks,
            compile_params,
            model_params,
            fit_params,
        )


class VAE(BaseNNImputer):
    """Class to impute missing data using a Variational Autoencoder neural network."""

    def __init__(
        self,
        kl_beta=tf.Variable(1.0, trainable=False),
        validation_split=0.2,
        **kwargs,
    ):
        self.kl_beta = kl_beta
        self.validation_split = validation_split

        self.nn_method_ = "VAE"
        self.num_classes = 4
        self.activate = None
        self.is_multiclass_ = True if self.num_classes != 4 else False
        self.testing = kwargs.get("testing", False)
        self.do_act_in_model_ = True if self.activate is None else False

        if self.do_act_in_model_ and self.is_multiclass_:
            self.act_func_ = "softmax"
        elif self.do_act_in_model_ and not self.is_multiclass_:
            self.act_func_ = "sigmoid"
        else:
            self.act_func_ = None

        super().__init__(
            self.activate,
            self.nn_method_,
            self.num_classes,
            self.act_func_,
            **kwargs,
            kl_beta=self.kl_beta,
            validation_split=self.validation_split,
        )

    def run_vae(
        self,
        y_true,
        y_train,
        model_params,
        compile_params,
        fit_params,
    ):
        """Run VAE using custom subclassed model.

        Args:
            y_true (numpy.ndarray): Original genotypes (training dataset) with known and missing values, of shape (n_samples, n_features).

            y_train (numpy.ndarray): Onehot encoded genotypes (training dataset) with known and missing values, of shape (n_samples, n_features, num_classes).

            model_params (Dict[str, Any]): Dictionary with parameters to pass to the classifier model.

            compile_params (Dict[str, Any]): Dictionary with parameters to pass to the tensorflow compile function.

            fit_params (Dict[str, Any]): Dictionary with parameters to pass to the fit() function.

        Returns:
            List[tf.keras.Model]: List of keras model objects. One for each phase (len=1 if NLPCA, len=3 if UBP).

            List[Dict[str, float]]: List of dictionaries with best neural network model history.

            Dict[str, Any] or None: Best parameters found during a grid search, or None if a grid search was not run.

            float: Best score obtained during grid search.

            tf.keras.Model: Best model found during grid search.

            sklearn.model_selection object (GridSearchCV, RandomizedSearchCV) or GASearchCV object.

            Dict[str, Any]: Per-class, micro, and macro-averaged metrics including accuracy, ROC-AUC, and Precision-Recall with Average Precision scores.
        """
        scorers = Scorers()
        scoring = None

        histories = list()
        models = list()

        if self.run_gridsearch_:
            scoring = scorers.make_multimetric_scorer(
                self.scoring_metrics_,
                self.sim_missing_mask_,
                num_classes=self.num_classes,
            )

        (
            model,
            best_history,
            best_params,
            best_score,
            best_clf,
            search,
            metrics,
        ) = self.run_clf(
            y_train,
            y_true,
            model_params,
            compile_params,
            fit_params,
            scoring=scoring,
        )

        histories.append(best_history)
        models.append(model)
        del model

        return (
            models,
            histories,
            best_params,
            best_score,
            best_clf,
            search,
            metrics,
        )


class SAE(BaseNNImputer):
    def __init__(
        self,
        **kwargs,
    ):
        self.num_classes = 3
        self.activate = "softmax"
        self.nn_method_ = "SAE"
        self.act_func_ = "softmax"
        self.testing = kwargs.get("testing", False)

        super().__init__(
            self.activate,
            self.nn_method_,
            self.num_classes,
            self.act_func_,
            **kwargs,
        )

    def run_sae(
        self,
        y_true,
        y_train,
        model_params,
        compile_params,
        fit_params,
    ):
        """Run standard autoencoder using custom subclassed model.

        Args:
            y_true (numpy.ndarray): Original genotypes (training dataset) with known and missing values of shape (n_samples, n_features).

            y_train (numpy.ndarray): Onehot-encoded genotypes (training dataset) with known and missing values of shape (n_samples, n_features, num_classes.)

            model_params (Dict[str, Any]): Dictionary with parameters to pass to the classifier model.

            compile_params (Dict[str, Any]): Dictionary with parameters to pass to the tensorflow compile function.

            fit_params (Dict[str, Any]): Dictionary with parameters to pass to the fit() function.

        Returns:
            List[tf.keras.Model]: List of keras model objects. One for each phase (len=1 if NLPCA, len=3 if UBP).

            List[Dict[str, float]]: List of dictionaries with best neural network model history.

            Dict[str, Any] or None: Best parameters found during a grid search, or None if a grid search was not run.

            float: Best score obtained during grid search.

            tf.keras.Model: Best model found during grid search.

            sklearn.model_selection object (GridSearchCV, RandomizedSearchCV) or GASearchCV object.

            Dict[str, Any]: Per-class, micro, and macro-averaged metrics including accuracy, ROC-AUC, and Precision-Recall with Average Precision scores.
        """
        scorers = Scorers()
        scoring = None

        histories = list()
        models = list()

        if self.run_gridsearch_:
            scoring = scorers.make_multimetric_scorer(
                self.scoring_metrics_, self.sim_missing_mask_
            )

        (
            model,
            best_history,
            best_params,
            best_score,
            best_clf,
            search,
            metrics,
        ) = self.run_clf(
            y_train,
            y_true,
            model_params,
            compile_params,
            fit_params,
            scoring=scoring,
            testing=False,
        )

        histories.append(best_history)
        models.append(model)
        del model

        return (
            models,
            histories,
            best_params,
            best_score,
            best_clf,
            search,
            metrics,
        )


class UBP(BaseNNImputer):
    def __init__(
        self,
        *,
        nlpca=False,
        **kwargs,
    ):
        # TODO: Make estimators compatible with variable number of classes.
        # E.g., with morphological data.
        self.nlpca = nlpca
        self.nn_method_ = "NLPCA" if self.nlpca else "UBP"
        self.num_classes = 3
        self.testing = kwargs.get("testing", False)
        self.activate = None
        self.act_func_ = "softmax"

        super().__init__(
            self.activate,
            self.nn_method_,
            self.num_classes,
            self.act_func_,
            **kwargs,
            nlpca=self.nlpca,
        )

    def run_nlpca(
        self,
        y_true,
        y_train,
        model_params,
        compile_params,
        fit_params,
    ):
        """Run NLPCA using custom subclassed model.

        Args:
            y_true (numpy.ndarray): Original genotypes with known and missing values.

            y_train (numpy.ndarray): For compatibility with VAE and SAE. Not used.

            model_params (Dict[str, Any]): Dictionary with parameters to pass to the classifier model.

            compile_params (Dict[str, Any]): Dictionary with parameters to pass to the tensorflow compile function.

            fit_params (Dict[str, Any]): Dictionary with parameters to pass to the fit() function.

        Returns:
            List[tf.keras.Model]: List of keras model objects. One for each phase (len=1 if NLPCA, len=3 if UBP).

            List[Dict[str, float]]: List of dictionaries with best neural network model history.

            Dict[str, Any] or None: Best parameters found during a grid search, or None if a grid search was not run.

            float: Best score obtained during grid search.

            tf.keras.Model: Best model found during grid search.

            sklearn.model_selection object (GridSearchCV, RandomizedSearchCV) or GASearchCV object.

            Dict[str, Any]: Per-class, micro, and macro-averaged metrics including accuracy, ROC-AUC, and Precision-Recall with Average Precision scores.
        """
        scorers = Scorers()

        histories = list()
        models = list()
        y_train = model_params.pop("y_train")
        ubp_weights = None
        phase = None
        scoring = None

        if self.run_gridsearch_:
            scoring = scorers.make_multimetric_scorer(
                self.scoring_metrics_, self.sim_missing_mask_
            )

        (
            V,
            model,
            best_history,
            best_params,
            best_score,
            best_clf,
            search,
            metrics,
        ) = self.run_clf(
            y_train,
            y_true,
            model_params,
            compile_params,
            fit_params,
            ubp_weights=ubp_weights,
            phase=phase,
            scoring=scoring,
            testing=False,
        )

        histories.append(best_history)
        models.append(model)
        del model

        return (
            models,
            histories,
            best_params,
            best_score,
            best_clf,
            search,
            metrics,
        )

    def run_ubp(
        self,
        y_true,
        y_train,
        model_params,
        compile_params,
        fit_params,
    ):
        """Run UBP using custom subclassed model.

        Args:
            y_true (numpy.ndarray): Original genotypes with known and missing values.

            y_train (numpy.ndarray): For compatibility with VAE and SAE. Not used.

            model_params (Dict[str, Any]): Dictionary with parameters to pass to the classifier model.

            compile_params (Dict[str, Any]): Dictionary with parameters to pass to the tensorflow compile function.

            fit_params (Dict[str, Any]): Dictionary with parameters to pass to the fit() function.

        Returns:
            List[tf.keras.Model]: List of keras model objects. One for each phase (len=1 if NLPCA, len=3 if UBP).

            List[Dict[str, float]]: List of dictionaries with best neural network model history.

            Dict[str, Any] or None: Best parameters found during a grid search, or None if a grid search was not run.

            float: Best score obtained during grid search.

            tf.keras.Model: Best model found during grid search.

            sklearn.model_selection object (GridSearchCV, RandomizedSearchCV) or GASearchCV object.

            Dict[str, Any]: Per-class, micro, and macro-averaged metrics including accuracy, ROC-AUC, and Precision-Recall with Average Precision scores.
        """
        scorers = Scorers()

        histories = list()
        models = list()
        search_n_components = False

        y_train = model_params.pop("y_train")

        if self.run_gridsearch_:
            # Cannot do CV because there is no way to use test splits
            # given that the input gets refined. If using a test split,
            # then it would just be the randomly initialized values and
            # would not accurately represent the model.
            # Thus, we disable cross-validation for the grid searches.
            scoring = scorers.make_multimetric_scorer(
                self.scoring_metrics_, self.sim_missing_mask_
            )

            if "n_components" in self.gridparams:
                search_n_components = True
                n_components_searched = self.n_components
        else:
            scoring = None

        for phase in range(1, 4):
            ubp_weights = models[1].get_weights() if phase == 3 else None

            (
                V,
                model,
                best_history,
                best_params,
                best_score,
                best_clf,
                search,
                metrics,
            ) = self.run_clf(
                y_train,
                y_true,
                model_params,
                compile_params,
                fit_params,
                ubp_weights=ubp_weights,
                phase=phase,
                scoring=scoring,
                testing=False,
            )

            if phase == 1:
                # Cannot have V input with different n_components
                # in other phases than are in phase 1.
                # So the n_components search has to happen in phase 1.
                if best_params is not None and search_n_components:
                    n_components_searched = best_params["n_components"]
                    model_params["V"] = {
                        n_components_searched: model.V_latent.copy()
                    }
                    model_params["n_components"] = n_components_searched
                    self.n_components = n_components_searched
                    self.gridparams.pop("n_components")

                else:
                    model_params["V"] = V
            elif phase == 2:
                model_params["V"] = V

            elif phase == 3:
                if best_params is not None and search_n_components:
                    best_params["n_components"] = n_components_searched

            histories.append(best_history)
            models.append(model)
            del model

        return (
            models,
            histories,
            best_params,
            best_score,
            best_clf,
            search,
            metrics,
        )

    def _initV(self, y_train, search_mode):
        """Initialize random input V as dictionary of numpy arrays.

        Args:
            y_train (numpy.ndarray): One-hot encoded training dataset (actual data).

            search_mode (bool): Whether doing grid search.

        Returns:
            Dict[int, numpy.ndarray]: Dictionary with n_components: V as key-value pairs.

        Raises:
            ValueError: Number of components must be >= 2.
        """
        vinput = dict()
        if search_mode:
            if "n_components" in self.gridparams:
                n_components = self.gridparams["n_components"]
            else:
                n_components = self.n_components

            if not isinstance(n_components, int):
                if min(n_components) < 2:
                    raise ValueError(
                        f"n_components must be >= 2, but a value of {n_components} was specified."
                    )

                elif len(n_components) == 1:
                    vinput[n_components[0]] = self.nn_.init_weights(
                        y_train.shape[0], n_components[0]
                    )

                else:
                    for c in n_components:
                        vinput[c] = self.nn_.init_weights(y_train.shape[0], c)
            else:
                vinput[self.n_components] = self.nn_.init_weights(
                    y_train.shape[0], self.n_components
                )

        else:
            vinput[self.n_components] = self.nn_.init_weights(
                y_train.shape[0], self.n_components
            )

        return vinput
