import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score

from scikeras.wrappers import KerasClassifier

try:
    from .neural_network_methods import NeuralNetworkMethods
    from .scorers import Scorers
    from .vae_model import VAEModel
    from impute.autoencoder_model import AutoEncoderModel
    from .nlpca_model import NLPCAModel
    from .ubp_model import UBPPhase1, UBPPhase2, UBPPhase3
    from ..data_processing.transformers import (
        MLPTargetTransformer,
        UBPInputTransformer,
        AutoEncoderFeatureTransformer,
    )
except (ModuleNotFoundError, ValueError):
    from impute.neural_network_methods import NeuralNetworkMethods
    from impute.scorers import Scorers
    from impute.vae_model import VAEModel
    from impute.autoencoder_model import AutoEncoderModel
    from impute.nlpca_model import NLPCAModel
    from impute.ubp_model import UBPPhase1, UBPPhase2, UBPPhase3
    from data_processing.transformers import (
        MLPTargetTransformer,
        UBPInputTransformer,
        AutoEncoderFeatureTransformer,
    )


class SAEClassifier(KerasClassifier):
    """Estimator to be used with the scikit-learn API.

    Args:
        output_shape (int): Number of units in model output layer. Defaults to None.

        weights_initializer (str): Kernel initializer to use for model weights. Defaults to "glorot_normal".

        hidden_layer_sizes (List[int]): Output unit size for each hidden layer. Should be list of length num_hidden_layers. Defaults to None.

        num_hidden_layers (int): Number of hidden layers to use. Defaults to 1.

        hidden_activation (str): Hidden activation function to use. Defaults to "elu".

        l1_penalty (float): L1 regularization penalty to use to reduce overfitting. Defautls to 0.01.

        l2_penalty (float): L2 regularization penalty to use to reduce overfitting. Defaults to 0.01.

        dropout_rate (float): Dropout rate for each hidden layer to reduce overfitting. Defaults to 0.2.

        n_components (int): Number of components to use for input V. Defaults to 3.

        num_classes (int, optional): Number of classes in y_train. 012-encoded data should have 3 classes. Defaults to 3.

        kwargs (Any): Other keyword arguments to route to fit, compile, callbacks, etc. Should have the routing prefix (e.g., optimizer__learning_rate=0.01).
    """

    def __init__(
        self,
        output_shape=None,
        weights_initializer="glorot_normal",
        hidden_layer_sizes=None,
        num_hidden_layers=1,
        hidden_activation="elu",
        l1_penalty=0.01,
        l2_penalty=0.01,
        dropout_rate=0.2,
        n_components=3,
        num_classes=3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.output_shape = output_shape
        self.weights_initializer = weights_initializer
        self.hidden_layer_sizes = hidden_layer_sizes
        self.num_hidden_layers = num_hidden_layers
        self.hidden_activation = hidden_activation
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.dropout_rate = dropout_rate
        self.n_components = n_components
        self.num_classes = num_classes

    def _keras_build_fn(self, compile_kwargs):
        """Build model with custom parameters.

        Args:
            compile_kwargs (Dict[str, Any]): Dictionary with parameters: values. The parameters should be passed to the class constructor, but should be captured as kwargs. They should also have the routing prefix (e.g., optimizer__learning_rate=0.01). compile_kwargs will automatically be parsed from **kwargs by KerasClassifier and sent here.

        Returns:
            tf.keras.Model: Model instance. The chosen model depends on which phase is passed to the class constructor.
        """
        model = AutoEncoderModel(
            output_shape=self.output_shape,
            n_components=self.n_components,
            weights_initializer=self.weights_initializer,
            hidden_layer_sizes=self.hidden_layer_sizes,
            num_hidden_layers=self.num_hidden_layers,
            hidden_activation=self.hidden_activation,
            l1_penalty=self.l1_penalty,
            l2_penalty=self.l2_penalty,
            dropout_rate=self.dropout_rate,
            num_classes=self.num_classes,
        )

        model.compile(
            optimizer=compile_kwargs["optimizer"],
            loss=compile_kwargs["loss"],
            metrics=compile_kwargs["metrics"],
            run_eagerly=False,
        )

        return model

    @staticmethod
    def scorer(y_true, y_pred, **kwargs):
        """Scorer for grid search that masks missing data.

        To use this, do not specify a scoring metric when initializing the grid search object. By default if the scoring_metric option is left as None, then it uses the estimator's scoring metric (this one).

        Args:
            y_true (numpy.ndarray): True target values input to fit().
            y_pred (numpy.ndarray): Predicted target values from estimator. The predictions are modified by self.target_encoder().inverse_transform() before being sent here.
            kwargs (Any): Other parameters sent to sklearn scoring metric. Supported options include missing_mask, scoring_metric, and testing.

        Returns:
            float: Calculated score.
        """
        # Get missing mask if provided.
        # Otherwise default is all missing values (array all True).
        missing_mask = kwargs.get(
            "missing_mask", np.ones(y_true.shape, dtype=bool)
        )
        is_vae = kwargs.get("is_vae", False)

        testing = kwargs.get("testing", False)
        scoring_metric = kwargs.get("scoring_metric", "accuracy")

        nn = NeuralNetworkMethods()

        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]

        y_true_masked = y_true[missing_mask]
        y_pred_masked = y_pred[missing_mask]

        if scoring_metric.startswith("auc"):
            roc_auc = Scorers.compute_roc_auc_micro_macro(
                y_true_masked, y_pred_masked, missing_mask, is_vae=is_vae
            )

            if scoring_metric == "auc_macro":
                return roc_auc["macro"]

            elif scoring_metric == "auc_micro":
                return roc_auc["micro"]

            else:
                raise ValueError(
                    f"Invalid scoring_metric provided: {scoring_metric}"
                )

        elif scoring_metric.startswith("precision"):
            pr_ap = Scorers.compute_pr(
                y_true_masked, y_pred_masked, is_vae=is_vae
            )

            if scoring_metric == "precision_recall_macro":
                return pr_ap["macro"]

            elif scoring_metric == "precision_recall_micro":
                return pr_ap["micro"]

            else:
                raise ValueError(
                    f"Invalid scoring_metric provided: {scoring_metric}"
                )

        elif scoring_metric == "accuracy":
            y_pred_masked_decoded = nn.decode_onehot(y_pred_masked)
            return accuracy_score(y_true_masked, y_pred_masked_decoded)

        else:
            raise ValueError(
                f"Invalid scoring_metric provided: {scoring_metric}"
            )

        if testing:
            np.set_printoptions(threshold=np.inf)
            print(y_true_masked)

            y_pred_masked_decoded = nn.decode_onehot(y_pred_masked)

            print(y_pred_masked_decoded)

    @property
    def feature_encoder(self):
        """Handles feature input, X, before training.

        Returns:
            MLPTargetTransformer: InputTransformer object that includes fit() and transform() methods to transform input before estimator fitting.
        """
        return AutoEncoderFeatureTransformer(num_classes=self.num_classes)

    @property
    def target_encoder(self):
        """Handles target input and output, y_true and y_pred, both before and after training.

        Returns:
            NNOutputTransformer: NNOutputTransformer object that includes fit(), transform(), and inverse_transform() methods.
        """
        return AutoEncoderFeatureTransformer(num_classes=self.num_classes)

    def predict(self, X, **kwargs):
        """Returns predictions for the given test data.

        Args:
            X (Union[array-like, sparse matrix, dataframe] of shape (n_samples, n_features)): Training samples where n_samples is the number of samples and n_features is the number of features.
        **kwargs (Dict[str, Any]): Extra arguments to route to ``Model.predict``\.

        Warnings:
            Passing estimator parameters as keyword arguments (aka as ``**kwargs``) to ``predict`` is not supported by the Scikit-Learn API, and will be removed in a future version of SciKeras. These parameters can also be specified by prefixing ``predict__`` to a parameter at initialization (``BaseWrapper(..., fit__batch_size=32, predict__batch_size=1000)``) or by using ``set_params`` (``est.set_params(fit__batch_size=32, predict__batch_size=1000)``\).

        Returns:
            array-like: Predictions, of shape shape (n_samples,) or (n_samples, n_outputs).

        Notes:
            Had to override predict() here in order to do the __call__ with the refined input, V_latent.
        """
        X_train = self.feature_encoder_.transform(X)
        y_pred, z_mean, z_log_var = self.model_(X_train, training=False)
        return y_pred.numpy(), z_mean, z_log_var


class VAEClassifier(KerasClassifier):
    """Estimator to be used with the scikit-learn API and a keras model.

    Args:
        output_shape (int): Number of units in model output layer. Defaults to None.

        weights_initializer (str, optional): Kernel initializer to use for model weights. Defaults to "glorot_normal".

        hidden_layer_sizes (List[int]): Output unit size for each hidden layer. Should be list of length num_hidden_layers. Defaults to None.

        num_hidden_layers (int, optional): Number of hidden layers to use. Defaults to 1.

        hidden_activation (str, optional): Hidden activation function to use. Defaults to "elu".

        l1_penalty (float, optional): L1 regularization penalty to use to reduce overfitting. Defautls to 0.01.

        l2_penalty (float, optional): L2 regularization penalty to use to reduce overfitting. Defaults to 0.01.

        dropout_rate (float, optional): Dropout rate for each hidden layer to reduce overfitting. Defaults to 0.2.

        kl_beta (float, optional): Kullback-Liebler divergence weight (beta) to apply to KL loss. 1.0 means unweighted, 0.0 means KL loss is not applied at all. Defaults to 1.0.

        n_components (int, optional): Number of components to use for input V. Defaults to 3.

        num_classes (int, optional): Number of classes in y_train. [A,G,C,T]-encoded data should have 4 classes. Defaults to 4.

        kwargs (Any): Other keyword arguments to route to fit, compile, callbacks, etc. Should have the routing prefix (e.g., optimizer__learning_rate=0.01).
    """

    def __init__(
        self,
        output_shape=None,
        weights_initializer="glorot_normal",
        hidden_layer_sizes=None,
        num_hidden_layers=1,
        hidden_activation="elu",
        l1_penalty=0.01,
        l2_penalty=0.01,
        dropout_rate=0.2,
        kl_beta=1.0,
        n_components=3,
        num_classes=4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.output_shape = output_shape
        self.weights_initializer = weights_initializer
        self.hidden_layer_sizes = hidden_layer_sizes
        self.num_hidden_layers = num_hidden_layers
        self.hidden_activation = hidden_activation
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.dropout_rate = dropout_rate
        self.kl_beta = kl_beta
        self.n_components = n_components
        self.num_classes = num_classes

    def _keras_build_fn(self, compile_kwargs):
        """Build model with custom parameters.

        Args:
            compile_kwargs (Dict[str, Any]): Dictionary with parameters: values. The parameters should be passed to the class constructor, but should be captured as kwargs. They should also have the routing prefix (e.g., optimizer__learning_rate=0.01). compile_kwargs will automatically be parsed from **kwargs by KerasClassifier and sent here.

        Returns:
            tf.keras.Model: Model instance. The chosen model depends on which phase is passed to the class constructor.
        """
        model = VAEModel(
            output_shape=self.output_shape,
            n_components=self.n_components,
            weights_initializer=self.weights_initializer,
            hidden_layer_sizes=self.hidden_layer_sizes,
            num_hidden_layers=self.num_hidden_layers,
            hidden_activation=self.hidden_activation,
            l1_penalty=self.l1_penalty,
            l2_penalty=self.l2_penalty,
            dropout_rate=self.dropout_rate,
            kl_beta=self.kl_beta,
            num_classes=self.num_classes,
        )

        model.compile(
            optimizer=compile_kwargs["optimizer"],
            loss=compile_kwargs["loss"],
            metrics=compile_kwargs["metrics"],
            run_eagerly=False,
        )

        return model

    @staticmethod
    def scorer(y_true, y_pred, **kwargs):
        """Scorer for grid search that masks missing data.

        To use this, do not specify a scoring metric when initializing the grid search object. By default if the scoring_metric option is left as None, then it uses the estimator's scoring metric (this one).

        Args:
            y_true (numpy.ndarray): True target values input to fit().
            y_pred (numpy.ndarray): Predicted target values from estimator. The predictions are modified by self.target_encoder().inverse_transform() before being sent here.
            kwargs (Any): Other parameters sent to sklearn scoring metric. Supported options include missing_mask, scoring_metric, and testing.

        Returns:
            float: Calculated score.
        """
        # Get missing mask if provided.
        # Otherwise default is all missing values (array all True).
        missing_mask = kwargs.get(
            "missing_mask", np.ones(y_true.shape, dtype=bool)
        )

        testing = kwargs.get("testing", False)
        scoring_metric = kwargs.get("scoring_metric", "accuracy")

        nn = NeuralNetworkMethods()

        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]

        y_true_masked = y_true[missing_mask]
        y_pred_masked = y_pred[missing_mask]

        if scoring_metric.startswith("auc"):
            roc_auc = Scorers.compute_roc_auc_micro_macro(
                y_true_masked, y_pred_masked, missing_mask
            )

            if scoring_metric == "auc_macro":
                return roc_auc["macro"]

            elif scoring_metric == "auc_micro":
                return roc_auc["micro"]

            else:
                raise ValueError(
                    f"Invalid scoring_metric provided: {scoring_metric}"
                )

        elif scoring_metric.startswith("precision"):
            pr_ap = Scorers.compute_pr(y_true_masked, y_pred_masked)

            if scoring_metric == "precision_recall_macro":
                return pr_ap["macro"]

            elif scoring_metric == "precision_recall_micro":
                return pr_ap["micro"]

            else:
                raise ValueError(
                    f"Invalid scoring_metric provided: {scoring_metric}"
                )

        elif scoring_metric == "accuracy":
            y_pred_masked_decoded = nn.decode_onehot(y_pred_masked)
            return accuracy_score(y_true_masked, y_pred_masked_decoded)

        else:
            raise ValueError(
                f"Invalid scoring_metric provided: {scoring_metric}"
            )

        if testing:
            np.set_printoptions(threshold=np.inf)
            print(y_true_masked)

            y_pred_masked_decoded = nn.decode_onehot(y_pred_masked)

            print(y_pred_masked_decoded)

    @property
    def feature_encoder(self):
        """Handles feature input, X, before training.

        Returns:
            MLPTargetTransformer: InputTransformer object that includes fit() and transform() methods to transform input before estimator fitting.
        """
        return AutoEncoderFeatureTransformer(num_classes=self.num_classes)

    @property
    def target_encoder(self):
        """Handles target input and output, y_true and y_pred, both before and after training.

        Returns:
            NNOutputTransformer: NNOutputTransformer object that includes fit(), transform(), and inverse_transform() methods.
        """
        return AutoEncoderFeatureTransformer(num_classes=self.num_classes)

    def predict(self, X, **kwargs):
        """Returns predictions for the given test data.

        Args:
            X (Union[array-like, sparse matrix, dataframe] of shape (n_samples, n_features)): Training samples where n_samples is the number of samples and n_features is the number of features.
        **kwargs (Dict[str, Any]): Extra arguments to route to ``Model.predict``\.

        Warnings:
            Passing estimator parameters as keyword arguments (aka as ``**kwargs``) to ``predict`` is not supported by the Scikit-Learn API, and will be removed in a future version of SciKeras. These parameters can also be specified by prefixing ``predict__`` to a parameter at initialization (``BaseWrapper(..., fit__batch_size=32, predict__batch_size=1000)``) or by using ``set_params`` (``est.set_params(fit__batch_size=32, predict__batch_size=1000)``\).

        Returns:
            array-like: Predictions, of shape shape (n_samples,) or (n_samples, n_outputs).

        Notes:
            Had to override predict() here in order to do the __call__ with the refined input, V_latent.
        """
        X_train = self.feature_encoder_.transform(X)
        y_pred, z_mean, z_log_var, z = self.model_(X_train, training=False)
        return y_pred.numpy(), z_mean, z_log_var, z


class MLPClassifier(KerasClassifier):
    """Estimator to be used with the scikit-learn API.

    Args:
        V (numpy.ndarray or Dict[str, Any]): Input X values of shape (n_samples, n_components). If a dictionary is passed, each key: value pair should have randomly initialized values for n_components: V. self.feature_encoder() will parse it and select the key: value pair with the current n_components. This allows n_components to be grid searched using GridSearchCV. Otherwise, it throws an error that the dimensions are off. Defaults to None.

        y_train (numpy.ndarray): One-hot encoded target data. Defaults to None.

        y_original (numpy.ndarray): Original target data, y, that is not one-hot encoded. Should have shape (n_samples, n_features). Should be 012-encoded. Defaults to None.

        batch_size (int): Batch size to train with. Defaults to 32.

        missing_mask (np.ndarray): Missing mask with missing values set to False (0) and observed values as True (1). Defaults to None. Defaults to None.

        output_shape (int): Number of units in model output layer. Defaults to None.

        weights_initializer (str): Kernel initializer to use for model weights. Defaults to "glorot_normal".

        hidden_layer_sizes (List[int]): Output unit size for each hidden layer. Should be list of length num_hidden_layers. Defaults to None.

        num_hidden_layers (int): Number of hidden layers to use. Defaults to 1.

        hidden_activation (str): Hidden activation function to use. Defaults to "elu".

        l1_penalty (float): L1 regularization penalty to use to reduce overfitting. Defautls to 0.01.

        l2_penalty (float): L2 regularization penalty to use to reduce overfitting. Defaults to 0.01.

        dropout_rate (float): Dropout rate for each hidden layer to reduce overfitting. Defaults to 0.2.

        num_classes (int): Number of classes in output predictions. Defaults to 3.

        phase (int or None): Current phase (if doing UBP), or None if doing NLPCA. Defults to None.

        n_components (int): Number of components to use for input V. Defaults to 3.

        ubp_weights (tensorflow.Tensor): Weights from UBP model. Fetched by doing model.get_weights() on phase 2 model. Only used if phase 3. Defaults to None.

        kwargs (Any): Other keyword arguments to route to fit, compile, callbacks, etc. Should have the routing prefix (e.g., optimizer__learning_rate=0.01).
    """

    def __init__(
        self,
        V,
        y_train,
        ubp_weights=None,
        batch_size=32,
        missing_mask=None,
        output_shape=None,
        weights_initializer="glorot_normal",
        hidden_layer_sizes=None,
        num_hidden_layers=1,
        hidden_activation="elu",
        l1_penalty=0.01,
        l2_penalty=0.01,
        dropout_rate=0.2,
        num_classes=3,
        phase=None,
        sample_weight=None,
        n_components=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.V = V
        self.y_train = y_train
        self.ubp_weights = ubp_weights
        self.batch_size = batch_size
        self.missing_mask = missing_mask
        self.output_shape = output_shape
        self.weights_initializer = weights_initializer
        self.hidden_layer_sizes = hidden_layer_sizes
        self.num_hidden_layers = num_hidden_layers
        self.hidden_activation = hidden_activation
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.phase = phase
        self.sample_weight = sample_weight
        self.n_components = n_components

    def _keras_build_fn(self, compile_kwargs):
        """Build model with custom parameters.

        Args:
            compile_kwargs (Dict[str, Any]): Dictionary with parameters: values. The parameters should be passed to the class constructor, but should be captured as kwargs. They should also have the routing prefix (e.g., optimizer__learning_rate=0.01). compile_kwargs will automatically be parsed from **kwargs by KerasClassifier and sent here.

        Returns:
            tf.keras.Model: Model instance. The chosen model depends on which phase is passed to the class constructor.
        """
        if self.phase is None:
            model = NLPCAModel(
                V=self.V,
                y=self.y_train,
                batch_size=self.batch_size,
                missing_mask=self.missing_mask,
                output_shape=self.output_shape,
                n_components=self.n_components,
                weights_initializer=self.weights_initializer,
                hidden_layer_sizes=self.hidden_layer_sizes,
                num_hidden_layers=self.num_hidden_layers,
                hidden_activation=self.hidden_activation,
                l1_penalty=self.l1_penalty,
                l2_penalty=self.l2_penalty,
                dropout_rate=self.dropout_rate,
                num_classes=self.num_classes,
                phase=self.phase,
                sample_weight=self.sample_weight,
            )

        elif self.phase == 1:
            model = UBPPhase1(
                V=self.V,
                y=self.y_train,
                batch_size=self.batch_size,
                missing_mask=self.missing_mask,
                output_shape=self.output_shape,
                n_components=self.n_components,
                weights_initializer=self.weights_initializer,
                hidden_layer_sizes=self.hidden_layer_sizes,
                num_hidden_layers=self.num_hidden_layers,
                hidden_activation=self.hidden_activation,
                l1_penalty=self.l1_penalty,
                l2_penalty=self.l2_penalty,
                dropout_rate=self.dropout_rate,
                num_classes=self.num_classes,
                phase=self.phase,
            )

        elif self.phase == 2:
            model = UBPPhase2(
                V=self.V,
                y=self.y_train,
                batch_size=self.batch_size,
                missing_mask=self.missing_mask,
                output_shape=self.output_shape,
                n_components=self.n_components,
                weights_initializer=self.weights_initializer,
                hidden_layer_sizes=self.hidden_layer_sizes,
                num_hidden_layers=self.num_hidden_layers,
                hidden_activation=self.hidden_activation,
                l1_penalty=self.l1_penalty,
                l2_penalty=self.l2_penalty,
                dropout_rate=self.dropout_rate,
                num_classes=self.num_classes,
                phase=self.phase,
            )

        elif self.phase == 3:
            model = UBPPhase3(
                V=self.V,
                y=self.y_train,
                batch_size=self.batch_size,
                missing_mask=self.missing_mask,
                output_shape=self.output_shape,
                n_components=self.n_components,
                weights_initializer=self.weights_initializer,
                hidden_layer_sizes=self.hidden_layer_sizes,
                num_hidden_layers=self.num_hidden_layers,
                hidden_activation=self.hidden_activation,
                l1_penalty=self.l1_penalty,
                l2_penalty=self.l2_penalty,
                dropout_rate=self.dropout_rate,
                num_classes=self.num_classes,
                phase=self.phase,
            )

            model.build((None, self.n_components))

        model.compile(
            optimizer=compile_kwargs["optimizer"],
            loss=compile_kwargs["loss"],
            metrics=compile_kwargs["metrics"],
            run_eagerly=True,
        )

        model.set_model_outputs()

        if self.phase == 3:
            model.set_weights(self.ubp_weights)

        return model

    @staticmethod
    def scorer(y_true, y_pred, **kwargs):
        """Scorer for grid search that masks missing data.

        To use this, do not specify a scoring metric when initializing the grid search object. By default if the scoring_metric option is left as None, then it uses the estimator's scoring metric (this one).

        Args:
            y_true (numpy.ndarray): True target values input to fit().
            y_pred (numpy.ndarray): Predicted target values from estimator. The predictions are modified by self.target_encoder().inverse_transform() before being sent here.
            kwargs (Any): Other parameters sent to sklearn scoring metric. Supported options include missing_mask, scoring_metric, and testing.

        Returns:
            float: Calculated score.
        """
        # Get missing mask if provided.
        # Otherwise default is all missing values (array all True).
        missing_mask = kwargs.get(
            "missing_mask", np.ones(y_true.shape, dtype=bool)
        )

        testing = kwargs.get("testing", False)
        scoring_metric = kwargs.get("scoring_metric", "accuracy")

        y_true_masked = y_true[missing_mask]
        y_pred_masked = y_pred[missing_mask]

        if scoring_metric.startswith("auc"):
            roc_auc = Scorers.compute_roc_auc_micro_macro(
                y_true_masked, y_pred_masked, missing_mask
            )

            if scoring_metric == "auc_macro":
                return roc_auc["macro"]

            elif scoring_metric == "auc_micro":
                return roc_auc["micro"]

            else:
                raise ValueError(
                    f"Invalid scoring_metric provided: {scoring_metric}"
                )

        elif scoring_metric.startswith("precision"):
            pr_ap = Scorers.compute_pr(y_true_masked, y_pred_masked)

            if scoring_metric == "precision_recall_macro":
                return pr_ap["macro"]

            elif scoring_metric == "precision_recall_micro":
                return pr_ap["micro"]

            else:
                raise ValueError(
                    f"Invalid scoring_metric provided: {scoring_metric}"
                )

        elif scoring_metric == "accuracy":
            y_pred_masked_decoded = NeuralNetworkMethods.decode_masked(
                y_pred_masked
            )
            return accuracy_score(y_true_masked, y_pred_masked_decoded)

        else:
            raise ValueError(
                f"Invalid scoring_metric provided: {scoring_metric}"
            )

        if testing:
            np.set_printoptions(threshold=np.inf)
            print(y_true_masked)

            y_pred_masked_decoded = NeuralNetworkMethods.decode_masked(
                y_pred_masked
            )

            print(y_pred_masked_decoded)

    @property
    def feature_encoder(self):
        """Handles feature input, X, before training.

        Returns:
            UBPInputTransformer: InputTransformer object that includes fit() and transform() methods to transform input before estimator fitting.
        """
        return UBPInputTransformer(self.n_components, self.V)

    @property
    def target_encoder(self):
        """Handles target input and output, y_true and y_pred, both before and after training.

        Returns:
            NNOutputTransformer: NNOutputTransformer object that includes fit(), transform(), and inverse_transform() methods.
        """
        return MLPTargetTransformer()

    def predict(self, X, **kwargs):
        """Returns predictions for the given test data.

        Args:
            X (Union[array-like, sparse matrix, dataframe] of shape (n_samples, n_features)): Training samples where n_samples is the number of samples and n_features is the number of features.
        **kwargs (Dict[str, Any]): Extra arguments to route to ``Model.predict``\.

        Warnings:
            Passing estimator parameters as keyword arguments (aka as ``**kwargs``) to ``predict`` is not supported by the Scikit-Learn API, and will be removed in a future version of SciKeras. These parameters can also be specified by prefixing ``predict__`` to a parameter at initialization (``BaseWrapper(..., fit__batch_size=32, predict__batch_size=1000)``) or by using ``set_params`` (``est.set_params(fit__batch_size=32, predict__batch_size=1000)``\).

        Returns:
            array-like: Predictions, of shape shape (n_samples,) or (n_samples, n_outputs).

        Notes:
            Had to override predict() here in order to do the __call__ with the refined input, V_latent.
        """
        y_pred_proba = self.model_(self.model_.V_latent, training=False)
        return self.target_encoder_.inverse_transform(y_pred_proba)
