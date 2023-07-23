import numpy as np
import tensorflow as tf

from scikeras.wrappers import KerasClassifier

try:
    from ...utils.scorers import Scorers
    from .models.autoencoder_model import AutoEncoderModel
    from .models.nlpca_model import NLPCAModel
    from .models.ubp_model import UBPPhase1, UBPPhase2, UBPPhase3
    from .models.vae_model import VAEModel
    from ...data_processing.transformers import (
        MLPTargetTransformer,
        UBPInputTransformer,
        AutoEncoderFeatureTransformer,
    )
except (ModuleNotFoundError, ValueError, ImportError):
    from utils.scorers import Scorers
    from impute.unsupervised.neural_network_methods import NeuralNetworkMethods
    from impute.unsupervised.models.vae_model import (
        VAEModel,
    )
    from impute.unsupervised.models.autoencoder_model import AutoEncoderModel
    from impute.unsupervised.models.nlpca_model import NLPCAModel
    from impute.unsupervised.models.ubp_model import (
        UBPPhase1,
        UBPPhase2,
        UBPPhase3,
    )
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
        y=None,
        output_shape=None,
        weights_initializer="glorot_normal",
        hidden_layer_sizes=None,
        num_hidden_layers=1,
        hidden_activation="elu",
        l1_penalty=0.01,
        l2_penalty=0.01,
        dropout_rate=0.2,
        n_components=3,
        sample_weight=None,
        missing_mask=None,
        num_classes=3,
        activate="softmax",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.y = y
        self.output_shape = output_shape
        self.weights_initializer = weights_initializer
        self.hidden_layer_sizes = hidden_layer_sizes
        self.num_hidden_layers = num_hidden_layers
        self.hidden_activation = hidden_activation
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.dropout_rate = dropout_rate
        self.n_components = n_components
        self.sample_weight = sample_weight
        self.missing_mask = missing_mask
        self.num_classes = num_classes
        self.activate = activate

        self.classes_ = np.arange(self.num_classes)
        self.n_classes_ = self.num_classes

    def _keras_build_fn(self, compile_kwargs):
        """Build model with custom parameters.

        Args:
            compile_kwargs (Dict[str, Any]): Dictionary with parameters: values. The parameters should be passed to the class constructor, but should be captured as kwargs. They should also have the routing prefix (e.g., optimizer__learning_rate=0.01). compile_kwargs will automatically be parsed from **kwargs by KerasClassifier and sent here.

        Returns:
            tf.keras.Model: Model instance. The chosen model depends on which phase is passed to the class constructor.
        """

        ######### REMOVING THIS LINE WILL BREAK THE MODEL!!!!! ########
        self.classes_ = np.arange(self.num_classes)

        model = AutoEncoderModel(
            self.y,
            output_shape=self.output_shape,
            n_components=self.n_components,
            weights_initializer=self.weights_initializer,
            hidden_layer_sizes=self.hidden_layer_sizes,
            num_hidden_layers=self.num_hidden_layers,
            hidden_activation=self.hidden_activation,
            l1_penalty=self.l1_penalty,
            l2_penalty=self.l2_penalty,
            dropout_rate=self.dropout_rate,
            sample_weight=self.sample_weight,
            missing_mask=self.missing_mask,
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
        n_classes_ = kwargs.get("num_classes", 3)
        classes_ = np.arange(n_classes_)
        missing_mask = kwargs.get("missing_mask")

        num_classes = kwargs.get("num_classes", 3)
        testing = kwargs.get("testing", False)

        scorers = Scorers()

        return scorers.scorer(
            y_true,
            y_pred,
            missing_mask=missing_mask,
            num_classes=num_classes,
            testing=testing,
        )

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
        return AutoEncoderFeatureTransformer(
            num_classes=self.num_classes, activate=self.activate
        )

    def predict(self, X, **kwargs):
        """Returns predictions for the given test data.

        Args:
            X (Union[array-like, sparse matrix, dataframe] of shape (n_samples, n_features)): Training samples where n_samples is the number of samples and n_features is the number of features.
        kwargs (Dict[str, Any]): Extra arguments to route to ``Model.predict``\.

        Warnings:
            Passing estimator parameters as keyword arguments (aka as ``**kwargs``) to ``predict`` is not supported by the Scikit-Learn API, and will be removed in a future version of SciKeras. These parameters can also be specified by prefixing ``predict__`` to a parameter at initialization (``BaseWrapper(..., fit__batch_size=32, predict__batch_size=1000)``) or by using ``set_params`` (``est.set_params(fit__batch_size=32, predict__batch_size=1000)``\).

        Returns:
            array-like: Predictions, of shape shape (n_samples,) or (n_samples, n_outputs).

        Notes:
            Had to override predict() here in order to do the __call__ with the refined input, V_latent.
        """
        X_train = self.target_encoder_.transform(X)
        y_pred = self.model_(X_train, training=False)
        return self.target_encoder_.inverse_transform(y_pred)

    def get_metadata(self):
        """Returns a dictionary of meta-parameters generated when this transformer was fitted.

        Used by SciKeras to bind these parameters to the SciKeras estimator itself and make them available as inputs to the Keras model.

        Returns:
            Dict[str, Any]: Dictionary of meta-parameters generated when this transfromer was fitted.
        """
        return {
            "classes_": self.classes_,
            "n_classes_": self.n_classes_,
            "n_outputs_": self.n_outputs_,
            "n_outputs_expected_": self.n_outputs_expected_,
        }


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

        num_classes (int, optional): Number of classes in y_train. [A,G,C,T...IUPAC codes]-encoded data should have 10 classes. Defaults to 4.

        activate (str or None, optional): If not None, then does the appropriate activation. Multilabel learning uses sigmoid activation, and multiclass uses softmax. If set to None, then the function assumes that the input has already been activated. Possible values include: {None, 'sigmoid', 'softmax'}. Defaults to None.

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
        sample_weight=None,
        activate=None,
        y=None,
        missing_mask=None,
        batch_size=None,
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
        self.sample_weight = sample_weight
        self.activate = activate
        self.y = y
        self.missing_mask = missing_mask
        self.batch_size = batch_size

    def _keras_build_fn(self, compile_kwargs):
        """Build model with custom parameters.

        Args:
            compile_kwargs (Dict[str, Any]): Dictionary with parameters: values. The parameters should be passed to the class constructor, but should be captured as kwargs. They should also have the routing prefix (e.g., optimizer__learning_rate=0.01). compile_kwargs will automatically be parsed from **kwargs by KerasClassifier and sent here.

        Returns:
            tf.keras.Model: Model instance. The chosen model depends on which phase is passed to the class constructor.
        """

        ######### REMOVING THIS LINE WILL BREAK THE MODEL!!!!! ########
        self.classes_ = np.arange(self.num_classes)

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
            sample_weight=self.sample_weight,
            missing_mask=self.missing_mask,
            batch_size=self.batch_size,
            y=self.y,
            final_activation=self.activate,
        )

        model.compile(
            optimizer=compile_kwargs["optimizer"],
            loss=compile_kwargs["loss"],
            metrics=compile_kwargs["metrics"],
            run_eagerly=compile_kwargs["run_eagerly"],
            # sample_weight_mode="temporal",
        )

        return model

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
        return AutoEncoderFeatureTransformer(
            num_classes=self.num_classes,
            activate=self.activate,
        )

    def predict(self, X, **kwargs):
        """Returns predictions for the given test data.

        Args:
            X (Union[array-like, sparse matrix, dataframe] of shape (n_samples, n_features)): Training samples where n_samples is the number of samples and n_features is the number of features.
        kwargs (Dict[str, Any]): Extra arguments to route to ``Model.predict``\.

        Warnings:
            Passing estimator parameters as keyword arguments (aka as ``**kwargs``) to ``predict`` is not supported by the Scikit-Learn API, and will be removed in a future version of SciKeras. These parameters can also be specified by prefixing ``predict__`` to a parameter at initialization (``BaseWrapper(..., fit__batch_size=32, predict__batch_size=1000)``) or by using ``set_params`` (``est.set_params(fit__batch_size=32, predict__batch_size=1000)``\).

        Returns:
            array-like: Predictions, of shape shape (n_samples,) or (n_samples, n_outputs).

        Notes:
            Had to override predict() here in order to do the __call__ with the refined input, V_latent.
        """
        X_train = self.target_encoder_.transform(X)
        y_pred = self.model_(X_train, training=False)
        return self.target_encoder_.inverse_transform(y_pred)

    def get_metadata(self):
        """Returns a dictionary of meta-parameters generated when this transformer was fitted.

        Used by SciKeras to bind these parameters to the SciKeras estimator itself and make them available as inputs to the Keras model.

        Returns:
            Dict[str, Any]: Dictionary of meta-parameters generated when this transfromer was fitted.
        """
        return {
            "classes_": self.classes_,
            "n_classes_": self.n_classes_,
            "n_outputs_": self.n_outputs_,
            "n_outputs_expected_": self.n_outputs_expected_,
        }

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

        n_classes_ = kwargs.get("num_classes", 3)
        classes_ = np.arange(n_classes_)
        missing_mask = kwargs.get("missing_mask")

        num_classes = kwargs.get("num_classes", 3)
        testing = kwargs.get("testing", False)

        y_pred = y_pred.reshape(y_pred.shape[0], -1, num_classes)

        scorers = Scorers()

        return scorers.scorer(
            y_true,
            y_pred,
            missing_mask=missing_mask,
            num_classes=num_classes,
            testing=testing,
        )


class MLPClassifier(KerasClassifier):
    """Estimator to be used with the scikit-learn API.

    Args:
        V (numpy.ndarray or Dict[str, Any]): Input X values of shape (n_samples, n_components). If a dictionary is passed, each key: value pair should have randomly initialized values for n_components: V. self.feature_encoder() will parse it and select the key: value pair with the current n_components. This allows n_components to be grid searched using GridSearchCV. Otherwise, it throws an error that the dimensions are off. Defaults to None.

        y_train (numpy.ndarray): One-hot encoded target data. Defaults to None.

        ubp_weights (tensorflow.Tensor): Weights from UBP model. Fetched by doing model.get_weights() on phase 2 model. Only used if phase 3. Defaults to None.

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

        sample_weight (numpy.ndarray): Sample weight matrix for reducing the impact of class imbalance. Should be of shape (n_samples, n_features).

        n_components (int): Number of components to use for input V. Defaults to 3.

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
        ######### REMOVING THIS LINE WILL BREAK THE MODEL!!!!! ########
        self.classes_ = np.arange(self.num_classes)

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
        missing_mask = kwargs.get(
            "missing_mask", np.ones(y_true.shape, dtype=bool)
        )
        num_classes = kwargs.get("num_classes", 3)
        testing = kwargs.get("testing", False)

        scorers = Scorers()

        return scorers.scorer(
            y_true,
            y_pred,
            missing_mask=missing_mask,
            num_classes=num_classes,
            testing=testing,
        )

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
        kwargs (Dict[str, Any]): Extra arguments to route to ``Model.predict``\.

        Warnings:
            Passing estimator parameters as keyword arguments (aka as ``**kwargs``) to ``predict`` is not supported by the Scikit-Learn API, and will be removed in a future version of SciKeras. These parameters can also be specified by prefixing ``predict__`` to a parameter at initialization (``BaseWrapper(..., fit__batch_size=32, predict__batch_size=1000)``) or by using ``set_params`` (``est.set_params(fit__batch_size=32, predict__batch_size=1000)``\).

        Returns:
            array-like: Predictions, of shape shape (n_samples,) or (n_samples, n_outputs).

        Notes:
            Had to override predict() here in order to do the __call__ with the refined input, V_latent.
        """
        y_pred_proba = self.model_(self.model_.V_latent, training=False)
        return self.target_encoder_.inverse_transform(y_pred_proba)

    def get_metadata(self):
        """Returns a dictionary of meta-parameters generated when this transformer was fitted.

        Used by SciKeras to bind these parameters to the SciKeras estimator itself and make them available as inputs to the Keras model.

        Returns:
            Dict[str, Any]: Dictionary of meta-parameters generated when this transfromer was fitted.
        """
        return {
            "classes_": self.classes_,
            "n_classes_": self.n_classes_,
            "n_outputs_": self.n_outputs_,
            "n_outputs_expected_": self.n_outputs_expected_,
        }
