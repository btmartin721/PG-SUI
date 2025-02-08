# Standard library imports
from typing import Any, List, Optional, Tuple

# Third-party imports
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from snpio.utils.logging import LoggerManager
from torch import Tensor

# Custom imports
from pgsui.utils.misc import validate_input_type


def encode_onehot(X):
    """Convert 012-encoded data to one-hot encodings.
    Args:
        X (numpy.ndarray): Input array with 012-encoded data and -9 as the missing data value.
    Returns:
        pandas.DataFrame: One-hot encoded data, ignoring missing values (np.nan).
    """
    Xt = np.zeros(shape=(X.shape[0], X.shape[1], 3))
    mappings = {
        0: np.array([1, 0, 0]),
        1: np.array([0, 1, 0]),
        2: np.array([0, 0, 1]),
        -9: np.array([np.nan, np.nan, np.nan]),
    }
    for row in np.arange(X.shape[0]):
        Xt[row] = [mappings[enc] for enc in X[row]]
    return Xt


def mle(row):
    """Get the Maximum Likelihood Estimation for the best prediction. Basically, it sets the index of the maxiumum value in a vector (row) to 1.0, since it is one-hot encoded.

    Args:
        row (numpy.ndarray(float)): Row vector with predicted values as floating points.

    Returns:
        numpy.ndarray(float): Row vector with the highest prediction set to 1.0 and the others set to 0.0.
    """
    res = np.zeros(row.shape[0])
    res[np.argmax(row)] = 1
    return res


class UBPInputTransformer(BaseEstimator, TransformerMixin):
    """Transform input X prior to estimator fitting.

    Args:
        n_components (int): Number of principal components currently being used in V.

        V (numpy.ndarray or Dict[str, Any]): If doing grid search, should be a dictionary with current_component: numpy.ndarray. If not doing grid search, then it should be a numpy.ndarray.
    """

    def __init__(self, n_components, V):
        self.n_components = n_components
        self.V = V

    def fit(self, X):
        """Fit transformer to input data X.

        Args:
            X (numpy.ndarray): Input data to fit. If numpy.ndarray, then should be of shape (n_samples, n_components). If dictionary, then should be component: numpy.ndarray.

        Returns:
            self: Class instance.
        """
        self.n_features_in_ = self.n_components
        return self

    def transform(self, X):
        """Transform input data X to the needed format.

        Args:
            X (numpy.ndarray): Input data to fit. If numpy.ndarray, then should be of shape (n_samples, n_components). If dictionary, then should be component: numpy.ndarray.

        Returns:
            numpy.ndarray: Formatted input data with correct component.

        Raises:
            TypeError: V must be a dictionary if phase is None or phase == 1.
            TypeError: V must be a numpy array if phase is 2 or 3.
        """
        if not isinstance(self.V, dict):
            raise TypeError(f"V must be a dictionary, but got {type(self.V)}")
        return self.V[self.n_components]


class AutoEncoderFeatureTransformer(BaseEstimator, TransformerMixin):
    """Formats autoencoder features and targets before model fitting.

    The input data is converted to one-hot format, and missing values are replaced with [-1] * num_classes. Missing and observed boolean masks are also generated. The transformer can also return integer-encoded SNP data. The inverse_transform method decodes the predicted outputs back to the original format. The optimal threshold for sigmoid activation can be found using the inverse_transform method. The encoding function is selected based on the number of SNP categories. The encoding functions are encode_012, encode_multilabel, and encode_multiclass.

    Attributes:
        num_classes (int): Number of SNP categories (A, T, C, G).
        return_int (bool): Whether to return integer-encoded SNP data.
        activate (str): Activation function used in the autoencoder.
        threshold_increment (float): Step size for optimal threshold search.
        logger (LoggerManager): Logger for debugging and tracking.
        debug (bool): Enables debug mode.
        encoding_function (function): Selected encoding function.
        encoded_data (numpy.ndarray): One-hot encoded SNP data.
        original_data (numpy.ndarray): Raw input SNP data.
        missing_data_mask (numpy.ndarray): Boolean mask for missing values.
        observed_data_mask (numpy.ndarray): Boolean mask for observed values.
    """

    def __init__(
        self,
        num_classes: int = 4,  # Adjusted for SNP categories (A, T, C, G)
        return_int: bool = False,
        activate: str = "softmax",
        threshold_increment: float = 0.05,
        logger: Optional[LoggerManager] = None,
        verbose: int = 0,
        debug: bool = False,
    ) -> None:
        """Initialize the AutoEncoderFeatureTransformer.

        This class formats autoencoder features and targets before model fitting. The input data is converted to one-hot format, and missing values are replaced with [-1] * num_classes. Missing and observed boolean masks are also generated. The transformer can also return integer-encoded SNP data. The inverse_transform method decodes the predicted outputs back to the original format. The optimal threshold for sigmoid activation can be found using the inverse_transform method. The encoding function is selected based on the number of SNP categories. The encoding functions are encode_012, encode_multilabel, and encode_multiclass.

        Args:
            num_classes (int): Number of SNP categories (A, T, C, G).
            return_int (bool): Whether to return integer-encoded SNP data.
            activate (str): Activation function used in the autoencoder.
            threshold_increment (float): Step size for optimal threshold search.
            logger (LoggerManager): Logger for debugging and tracking.
            verbose (int): Verbosity level.
            debug (bool): Enable debug mode.

        Raises:
            ValueError: If the number of classes is invalid (not 3, 4, or 10).

        Examples:
            >>>from skbio.transformers import AutoEncoderFeatureTransformer
            >>>transformer = AutoEncoderFeatureTransformer(num_classes=4, return_int=True, activate="softmax", threshold_increment=0.05, verbose=1, debug=False)
            >>>transformer.fit(X)
            >>>Xenc = transformer.transform(X)
            >>>Xdec = transformer.inverse_transform(Xenc)
        """
        self.num_classes = num_classes
        self.return_int = return_int
        self.activate = activate
        self.threshold_increment = threshold_increment

        if logger is None:
            logman = LoggerManager(
                name=__name__,
                prefix="autoencoder_feature_transformer",
                debug=debug,
                verbose=verbose >= 1,
            )
            self.logger = logman.get_logger()
        else:
            self.logger = logger

    def fit(self, X) -> Any:
        """Fit the transformer by encoding X.

        Args:
            X (numpy.ndarray): Input data to fit.

        Returns:
            self: Class instance.

        Raises:
            ValueError: If the number of classes is invalid.
        """
        X = validate_input_type(X, "array")

        # Fill missing values in X with -1
        X = np.nan_to_num(X, nan=-1)

        self.classes_ = np.arange(self.num_classes)

        # Select encoding function
        if self.num_classes == 3:
            encoding_function = self.encode_012
        elif self.num_classes == 4:
            encoding_function = self.encode_multilabel
        elif self.num_classes == 10:
            encoding_function = self.encode_multiclass
        else:
            msg = f"Invalid num_classes {self.num_classes}. Only 3, 4, or 10 are supported."
            self.logger.error(msg)
            raise ValueError(msg)

        self.encoding_function_ = encoding_function

        # Encode the input data
        Xenc = self.encoding_function_(X)
        self.data_shape_ = Xenc.shape

        # Generate missing and observed data masks
        self.missing_mask_, self.observed_mask_ = self._get_masks(Xenc)

        return self

    def transform(self, X):
        """Transform input data X to the needed format.

        Args:
            X (numpy.ndarray): Input data to fit.

        Returns:
            numpy.ndarray: Formatted input data with correct component.

        Raises:
            ValueError: If the encoding produced an unexpected shape.
        """
        X = validate_input_type(X, return_type="array")
        X = np.nan_to_num(X, nan=-1)
        X_encoded = self.encoding_function_(X)

        # Ensure correct shape
        if (
            X_encoded.shape[1] != self.data_shape_
            and X_encoded.shape[2] != self.data_shape_[2]
        ):
            msg = f"Encoding produced unexpected shape {tuple(X_encoded.shape[1:])}, expected {tuple(self.data_shape_[1:])}."
            self.logger.error(msg)
            raise ValueError(msg)

        return X_encoded

    def inverse_transform(self, X) -> np.ndarray:
        """Inverse transform predicted outputs back to original format.

        This method decodes the predicted outputs from the autoencoder back to the original SNP format. If return_proba is True, then the predicted probabilities are also returned.

        Args:
            X (numpy.ndarray): Input data. Should be encoded of shape (n_samples, n_features, num_classes).

        Returns:
            numpy.ndarray: Decoded SNP data.

        Raises:
            ValueError: If the activation function is invalid.
        """
        if self.activate not in {"sigmoid", "softmax"}:
            msg = f"Invalid activation function: {self.activate}; must be 'sigmoid' or 'softmax'."
            self.logger.error(msg)
            raise ValueError(msg)

        if self.activate == "softmax":
            Xarr = validate_input_type(X, return_type="array")
            if not Xarr.shape[-1] == self.num_classes:
                msg = f"Unexpected shape: {Xarr.shape}; expected (n_samples, n_features, num_classes)."
                self.logger.error(msg)
                raise ValueError(msg)

            return np.argmax(Xarr, axis=-1)
        else:
            Xarr = validate_input_type(X, return_type="array")
            if Xarr.shape[-1] != self.num_classes:
                msg = f"Unexpected shape: {Xarr.shape}; expected {self.data_shape_}, {self.num_classes})."
                self.logger.error(msg)
                raise ValueError(msg)

            is_binarized = np.all(np.isin(Xarr, [0, 1, -1]))

            if not is_binarized:
                msg = "Input X is not binarized. Please decode the predicted probabilities first."
                self.logger.error(msg)
                raise ValueError(msg)

            return self.decode_multilabel(Xarr)

    def decode_multilabel(self, y_binarized):
        """Decode binarized multilabel outputs into integer encodings.

        Args:
            y_binarized (np.ndarray): Binarized predictions (0 or 1) after applying the optimal threshold.

        Returns:
            np.ndarray: Decoded integer-encoded multilabel predictions.
        """
        y_binarized = y_binarized.reshape(-1, self.num_classes)

        # Replace all-zero predictions with the most likely class.
        # Get row indices where all values are zero
        zero_rows = np.where(np.sum(y_binarized, axis=1) == 0)[0]

        if len(zero_rows) > 0:
            # Get the index of the maximum value in each row.
            argmax_indices = np.argmax(y_binarized, axis=1)

            # Assign argmax in only those rows where all values are zero.
            y_binarized[zero_rows, argmax_indices[zero_rows]] = 1

        # Define mapping from multi-hot encodings to integer values
        mappings = {
            (1, 0, 0, 0): 0,
            (0, 1, 0, 0): 1,
            (0, 0, 1, 0): 2,
            (0, 0, 0, 1): 3,
            (1, 1, 0, 0): 4,
            (1, 0, 1, 0): 5,
            (1, 0, 0, 1): 6,
            (0, 1, 1, 0): 7,
            (0, 1, 0, 1): 8,
            (0, 0, 1, 1): 9,
            (0, 0, 0, 0): -1,
        }

        # Convert each row of one-hot to tuple and map to integer
        decoded = np.array(
            [mappings.get(tuple(row), -1) for row in y_binarized],
            dtype=int,
        )

        return decoded.reshape(self.data_shape_[0], self.data_shape_[1])

    def _get_masks(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """Generate missing and observed data masks.

        Args:
            X (np.ndarray): Input data to generate masks from.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Missing data mask, observed data mask.
        """
        missing_mask = np.isnan(X).all(axis=2)
        observed_mask = ~missing_mask
        return missing_mask, observed_mask

    def encode_012(self, X):
        """Convert 012-encoded data to one-hot encodings.
        Args:
            X (numpy.ndarray): Input array with 012-encoded data and -9 as the missing data value.
        Returns:
            pandas.DataFrame: One-hot encoded data, ignoring missing values (np.nan).
        """
        Xt = np.zeros(shape=(X.shape[0], X.shape[1], 3))
        mappings = {
            0: np.array([1, 0, 0]),
            1: np.array([0, 1, 0]),
            2: np.array([0, 0, 1]),
            -9: np.array([np.nan, np.nan, np.nan]),
            -1: np.array([np.nan, np.nan, np.nan]),
        }
        for row in np.arange(X.shape[0]):
            Xt[row] = [mappings[enc] for enc in X[row]]
        return Xt

    def encode_multilabel(self, X, multilabel_value=1.0):
        """Ensure SNP encoding remains (samples, loci, 4) without unintended expansion."""

        # **Force input X to be 2D**
        if X.ndim != 2:
            raise ValueError(f"Unexpected X shape: {X.shape}, expected (samples, loci)")

        # Initialize output matrix
        Xt = np.full((X.shape[0], X.shape[1], 4), np.nan)  # (samples, loci, 4)

        # Define encoding map
        encoding_map = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # A (0)
                [0.0, 1.0, 0.0, 0.0],  # T (1)
                [0.0, 0.0, 1.0, 0.0],  # C (2)
                [0.0, 0.0, 0.0, 1.0],  # G (3)
                [multilabel_value, multilabel_value, 0.0, 0.0],  # AT (4)
                [multilabel_value, 0.0, multilabel_value, 0.0],  # AC (5)
                [multilabel_value, 0.0, 0.0, multilabel_value],  # AG (6)
                [0.0, multilabel_value, multilabel_value, 0.0],  # TC (7)
                [0.0, multilabel_value, 0.0, multilabel_value],  # TG (8)
                [0.0, 0.0, multilabel_value, multilabel_value],  # CG (9)
            ]
        )

        # **Step 1: Ensure `valid_mask` is 2D**
        valid_mask = (X >= 0) & (X <= 9)  # Boolean mask of valid SNPs

        if valid_mask.ndim == 3:
            valid_mask = valid_mask[:, :, 0]  # Force 2D selection

        if valid_mask.shape != (X.shape[0], X.shape[1]):
            raise ValueError(
                f"Valid mask has incorrect shape: {valid_mask.shape}, expected (samples, loci)"
            )

        # **Step 2: Extract valid SNPs using the corrected mask**
        row_indices, col_indices = np.where(valid_mask)

        if len(row_indices) == 0:
            raise ValueError("No valid SNPs found. Ensure correct input.")

        # **Step 3: Get valid SNP values**
        valid_indices = X[row_indices, col_indices].astype(int)  # (N,)

        if valid_indices.ndim != 1:
            raise ValueError(
                f"Unexpected valid_indices shape: {valid_indices.shape}, expected (N,)"
            )

        # **Step 4: Lookup encoding (ensuring correct shape)**
        encoding_selected = encoding_map[valid_indices]  # (N, 4)

        if encoding_selected.shape != (len(valid_indices), 4):
            raise ValueError(
                f"Unexpected encoding_selected shape: {encoding_selected.shape}, expected (N, 4)"
            )

        # **Step 5: Assign to Xt**
        Xt[row_indices, col_indices, :] = (
            encoding_selected  # Assign to correct positions
        )

        return Xt  # Expected: (samples, loci, 4)


class MLPTargetTransformer(BaseEstimator, TransformerMixin):
    """Transformer to format UBP / NLPCA target data both before and after model fitting."""

    def fit(self, y):
        """Fit 012-encoded target data.

        Args:
            y (numpy.ndarray): Target data that is 012-encoded.

        Returns:
            self: Class instance.
        """
        y = validate_input_type(y, return_type="array")

        # Original 012-encoded y
        self.y_decoded_ = y

        y_train = encode_onehot(y)

        # Get missing and observed data boolean masks.
        self.missing_mask_, self.observed_mask_ = self._get_masks(y_train)

        # To accomodate multiclass-multioutput.
        self.n_outputs_expected_ = 1

        return self

    def transform(self, y):
        """Transform y_true to one-hot encoded.

        Accomodates multiclass-multioutput targets.

        Args:
            y (numpy.ndarray): One-hot encoded target data.

        Returns:
            numpy.ndarray: y_true target data.
        """
        y = validate_input_type(y, return_type="array")
        y_train = encode_onehot(y)
        return self._fill(y_train, self.missing_mask_)

    def inverse_transform(self, y):
        """Decode y_pred from one-hot to 012-based encoding.

        This allows sklearn.metrics to be used.

        Args:
            y (numpy.ndarray): One-hot encoded predicted probabilities after model fitting.

        Returns:
            numpy.ndarray: y predictions in same format as y_true.
        """
        # VAE has tuple output
        if isinstance(y, tuple):
            y = y[0]

        # Return predictions.
        return tf.nn.softmax(y).numpy()

    def _fill(self, data, missing_mask, missing_value=-1, num_classes=3):
        """Mask missing data as ``missing_value.

        Args:
            data (numpy.ndarray): Input with missing values of shape (n_samples, n_features, num_classes).

            missing_mask (np.ndarray(bool)): Missing data mask with True corresponding to a missing value.

            missing_value (int): Value to set missing data to. If a list is provided, then its length should equal the number of one-hot classes. Defaults to -1.

            num_classes (int): Number of classes in dataset. Defaults to 3.
        """
        if num_classes > 1:
            missing_value = [missing_value] * num_classes
        data[missing_mask] = missing_value
        return data

    def _get_masks(self, X):
        """Format the provided target data for use with UBP/NLPCA.

        Args:
            X (numpy.ndarray(float)): Input data that will be used as the target.

        Returns:
            numpy.ndarray(float): Missing data mask, with missing values encoded as 1's and non-missing as 0's.

            numpy.ndarray(float): Observed data mask, with non-missing values encoded as 1's and missing values as 0's.
        """
        missing_mask = self._create_missing_mask(X)
        observed_mask = ~missing_mask
        return missing_mask, observed_mask

    def _create_missing_mask(self, data):
        """Creates a missing data mask with boolean values.
        Args:
            data (numpy.ndarray): Data to generate missing mask from, of shape (n_samples, n_features, n_classes).
        Returns:
            numpy.ndarray(bool): Boolean mask of missing values of shape (n_samples, n_features), with True corresponding to a missing data point.
        """
        return np.isnan(data).all(axis=2)

    def _decode(self, y):
        """Evaluate UBP / NLPCA predictions by calculating the highest predicted value.

        Calucalates highest predicted value for each row vector and each class, setting the most likely class to 1.0.

        Args:
            y (numpy.ndarray): Input one-hot encoded data.

        Returns:
            numpy.ndarray: Imputed one-hot encoded values.
        """
        Xprob = y
        Xt = np.apply_along_axis(mle, axis=2, arr=Xprob)
        Xpred = np.argmax(Xt, axis=2)
        Xtrue = np.argmax(y, axis=2)
        Xdecoded = np.zeros((Xpred.shape[0], Xpred.shape[1]))
        for idx in np.arange(Xdecoded):
            imputed_idx = np.where(self.observed_mask_[idx] == 0)
            known_idx = np.nonzero(self.observed_mask_[idx])
            Xdecoded[idx, imputed_idx] = Xpred[idx, imputed_idx]
            Xdecoded[idx, known_idx] = Xtrue[idx, known_idx]
        return Xdecoded.astype("int8")


class UBPTargetTransformer(BaseEstimator, TransformerMixin):
    """Transformer to format UBP / NLPCA target data both before model fitting.

    Examples:
        >>>ubp_tt = UBPTargetTransformer()
        >>>y_train = ubp_tt.fit_transform(y)
    """

    def fit(self, y):
        """Fit 012-encoded target data.

        Args:
            y (numpy.ndarray): Target data that is 012-encoded, of shape (n_samples, n_features).

        Returns:
            self: Class instance.
        """
        y = validate_input_type(y, return_type="array")

        # Original 012-encoded y
        self.y_decoded_ = y

        # One-hot encode y.
        y_train = encode_onehot(y)

        # Get missing and observed data boolean masks.
        self.missing_mask_, self.observed_mask_ = self._get_masks(y_train)

        # To accomodate multiclass-multioutput.
        self.n_outputs_expected_ = 1

        return self

    def transform(self, y):
        """Transform 012-encoded target to one-hot encoded format.

        Accomodates multiclass-multioutput targets.

        Args:
            y (numpy.ndarray): One-hot encoded target data of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: y_true target data.
        """
        y = validate_input_type(y, return_type="array")
        y_train = encode_onehot(y)
        return self._fill(y_train, self.missing_mask_)

    def inverse_transform(self, y):
        """Decode y_predicted from one-hot to 012-integer encoding.

        Performs a softmax activation for multiclass classification.

        This allows sklearn.metrics to be used.

        Args:
            y (numpy.ndarray): One-hot encoded predicted probabilities after model fitting, of shape (n_samples, n_features, num_classes).

        Returns:
            numpy.ndarray: y predictions in same format as y_true (n_samples, n_features).
        """
        return tf.nn.softmax(y).numpy()

    def _fill(self, data, missing_mask, missing_value=-1, num_classes=3):
        """Mask missing data as ``missing_value.

        Args:
            data (numpy.ndarray): Input with missing values of shape (n_samples, n_features, num_classes).

            missing_mask (np.ndarray(bool)): Missing data mask with True corresponding to a missing value, of shape (n_samples, n_features).

            missing_value (int, optional): Value to set missing data to. If a list is provided, then its length should equal the number of one-hot classes. Defaults to -1.

            num_classes (int, optional): Number of classes to use. Defaults to 3.
        """
        if num_classes > 1:
            missing_value = [missing_value] * num_classes
        data[missing_mask] = missing_value
        return data

    def _get_masks(self, y):
        """Format the provided target data for use with UBP/NLPCA models.

        Args:
            y (numpy.ndarray(float)): Input data that will be used as the target of shape (n_samples, n_features, num_classes).

        Returns:
            numpy.ndarray(float): Missing data mask, with missing values encoded as 1's and non-missing as 0's.

            numpy.ndarray(float): Observed data mask, with non-missing values encoded as 1's and missing values as 0's.
        """
        missing_mask = self._create_missing_mask(y)
        observed_mask = ~missing_mask
        return missing_mask, observed_mask

    def _create_missing_mask(self, data):
        """Creates a missing data mask with boolean values.

        Args:
            data (numpy.ndarray): Data to generate missing mask from, of shape (n_samples, n_features, n_classes).

        Returns:
            numpy.ndarray(bool): Boolean mask of missing values of shape (n_samples, n_features), with True corresponding to a missing data point.
        """
        return np.isnan(data).all(axis=2)

    def _decode(self, y):
        """Evaluate UBP/NLPCA predictions by calculating the argmax.

        Calucalates highest predicted value for each row vector and each class, setting the most likely class to 1.0.

        Args:
            y (numpy.ndarray): Input one-hot encoded data of shape (n_samples, n_features, num_classes).

        Returns:
            numpy.ndarray: Imputed one-hot encoded values.
        """
        Xprob = y
        Xt = np.apply_along_axis(mle, axis=2, arr=Xprob)
        Xpred = np.argmax(Xt, axis=2)
        Xtrue = np.argmax(y, axis=2)
        Xdecoded = np.zeros((Xpred.shape[0], Xpred.shape[1]))
        for idx in np.arange(Xdecoded):
            imputed_idx = np.where(self.observed_mask_[idx] == 0)
            known_idx = np.nonzero(self.observed_mask_[idx])
            Xdecoded[idx, imputed_idx] = Xpred[idx, imputed_idx]
            Xdecoded[idx, known_idx] = Xtrue[idx, known_idx]
        return Xdecoded.astype("int8")


class SimGenotypeDataTransformer(BaseEstimator, TransformerMixin):
    """Simulate missing data on genotypes, globally (rather than per-column)."""

    def __init__(
        self,
        genotype_data: Any,
        *,
        prop_missing: float = 0.1,
        strategy: str = "random",
        missing_val: float = -1,
        mask_missing: bool = True,
        seed: int = None,
        tree: any = None,
        logger=None,
        verbose: int = 0,
        debug: bool = False,
        n_focal: int = 2,
        class_weights: List[float] | np.ndarray | Tensor | None = None,
    ):
        """
        Args:
            genotype_data (np.ndarray): Genotype data to transform.
            prop_missing (float): Proportion of data to mask as missing.
            strategy (str): Strategy to use for missing data simulation.
            missing_val (float): Value to use for missing data.
            mask_missing (bool): Whether to mask existing missing data.
            seed (int): Random seed for reproducibility.
            tree (any): Tree object for nonrandom strategies.
            logger (LoggerManager): Logger for debugging and tracking.
            verbose (int): Verbosity level.
            debug (bool): Enable debug mode.
            n_focal (int): Number of focal individuals.
            class_weights (List[float] | np.ndarray | Tensor, optional): Class weights for nonrandom strategies.
        """
        self.genotype_data = genotype_data
        self.prop_missing = prop_missing
        self.strategy = strategy
        self.missing_val = missing_val
        self.mask_missing = mask_missing
        self.seed = seed
        self.tree = tree
        self.n_focal = n_focal

        self.class_weights = (
            None if class_weights is None else validate_input_type(class_weights)
        )

        self.verbose = verbose
        self.debug = debug
        self.rng = np.random.default_rng(seed)

        if logger is None:
            logman = LoggerManager(
                name=__name__,
                prefix="sim_genotype_data_transformer",
                debug=debug,
                verbose=verbose >= 1,
            )
            self.logger = logman.get_logger()
        else:
            self.logger = logger

        self.logger.debug(
            f"Initialized {self.__class__.__name__} with strategy={strategy}"
        )

    def fit(self, X, y=None):
        """No-op fit. Optionally log info."""
        X = validate_input_type(X, "array")
        self.logger.info(
            f"Global missing simulation: strategy={self.strategy}, prop={self.prop_missing:.2f}"
        )
        return self

    def transform(self, X):
        """Apply global missing-data simulation to new data."""
        X = validate_input_type(X)
        X = X.astype(float)

        # Convert existing missing_val or negative placeholders to NaN
        if not np.isnan(self.missing_val):
            X = np.where(X == self.missing_val, np.nan, X)
        X = np.where(X < 0, np.nan, X)

        original_missing_mask = np.isnan(X)

        # Build the simulated mask
        sim_missing_mask = self._simulate_missing_mask(X, original_missing_mask)

        # Do not overwrite original missing
        if self.mask_missing:
            sim_missing_mask[original_missing_mask] = False

        # Validate columns or rows, if desired
        sim_missing_mask = self._validate_mask_columns(sim_missing_mask)

        all_missing_mask = np.logical_or(original_missing_mask, sim_missing_mask)

        # Apply
        Xt = X.copy()
        Xt[sim_missing_mask] = self.missing_val

        masks = {
            "original": original_missing_mask,
            "simulated": sim_missing_mask,
            "all": all_missing_mask,
        }
        return Xt, masks

    def _simulate_missing_mask(self, X, original_missing_mask):
        """Select which sites to mask, globally, depending on self.strategy.

        Args:
            X (np.ndarray): Genotype data.
            original_missing_mask (np.ndarray): Mask of original missing data.

        Returns:
            np.ndarray: Mask of simulated missing data.

        Raises:
            ValueError: If the strategy is invalid.
        """
        if self.strategy == "random":
            return self._simulate_random(X, original_missing_mask)

        elif self.strategy == "random_balanced":
            return self._simulate_random_weight(X, original_missing_mask, inverse=False)

        elif self.strategy == "random_inv":
            return self._simulate_random_weight(X, original_missing_mask, inverse=True)

        elif self.strategy == "random_balanced_multinom":
            return self._simulate_multinom(X, original_missing_mask, inverse=False)

        elif self.strategy == "random_inv_multinom":
            return self._simulate_multinom(X, original_missing_mask, inverse=True)

        elif self.strategy == "nonrandom":
            return self._simulate_tree_based(X, original_missing_mask, weighted=False)

        elif self.strategy == "nonrandom_weighted":
            return self._simulate_tree_based(X, original_missing_mask, weighted=True)

        elif self.strategy == "nonrandom_distance":
            return self._simulate_nonrandom_distance(X, original_missing_mask)

        else:
            raise ValueError(
                f"Invalid strategy '{self.strategy}'. "
                "Choose from: ['random', 'random_balanced', 'random_balanced_multinom', 'random_inv', 'random_inv_multinom', 'nonrandom', 'nonrandom_weighted', 'nonrandom_distance']."
            )

    def _simulate_random(self, X, original_missing_mask):
        """Randomly mask ~prop_missing fraction of all known genotype calls.

        Method to simulate missing data by randomly selecting calls to mask. This method is the simplest and most straightforward, but it does not take into account the distribution of missing data.

        Args:
            X (np.ndarray): Genotype data.
            original_missing_mask (np.ndarray): Mask of original missing data.

        Returns:
            np.ndarray: Mask of simulated missing data.
        """
        known_locs = np.where(~original_missing_mask)
        n_known = len(known_locs[0])
        mask = np.zeros_like(original_missing_mask, dtype=bool)

        if n_known == 0:
            return mask

        n_to_mask = int(np.floor(self.prop_missing * n_known))
        if n_to_mask == 0:
            return mask

        chosen = self.rng.choice(n_known, size=n_to_mask, replace=False)
        chosen_rows = known_locs[0][chosen]
        chosen_cols = known_locs[1][chosen]
        mask[chosen_rows, chosen_cols] = True

        return mask

    def _simulate_random_weight(self, X, original_missing_mask, inverse=False):
        """
        Globally subdivide missingness by genotype (0,1,2) across the entire matrix. If inverse=True => classes with fewer counts get proportionally more masked. If inverse=False => classes are masked equally. This method is a simple extension of the random method, but it takes into account the distribution of missing data.

        Args:
            X (np.ndarray): Genotype data.
            original_missing_mask (np.ndarray): Mask of original missing data.
            inverse (bool): Whether to use inverse weighting.

        Returns:
            np.ndarray: Mask of simulated missing data.
        """
        nrows, ncols = X.shape
        mask = np.zeros((nrows, ncols), dtype=bool)

        # Gather all known calls
        known_locs = np.where(~original_missing_mask)
        n_known = len(known_locs[0])
        if n_known == 0:
            return mask

        # Separate known calls by genotype 0,1,2
        row_coords = known_locs[0]
        col_coords = known_locs[1]
        genotypes = X[row_coords, col_coords]

        idx0 = np.where(genotypes == 0)[0]
        idx1 = np.where(genotypes == 1)[0]
        idx2 = np.where(genotypes == 2)[0]

        n_to_mask = int(np.floor(self.prop_missing * n_known))
        if n_to_mask == 0:
            return mask

        # Decide how many to remove from each genotype
        if inverse:
            # Class weights approach
            inv0 = self.class_weights[0]
            inv1 = self.class_weights[1]
            inv2 = self.class_weights[2]
            denom = inv0 + inv1 + inv2
            frac0, frac1, frac2 = inv0 / denom, inv1 / denom, inv2 / denom
        else:
            # Balanced => ~1/3 each
            frac0 = frac1 = frac2 = 1 / 3

        picks0 = int(np.floor(n_to_mask * frac0))
        picks1 = int(np.floor(n_to_mask * frac1))
        picks2 = int(np.floor(n_to_mask * frac2))

        chosen_indices = []

        # Pick for genotype 0
        if len(idx0) > 0 and picks0 > 0:
            c0 = self.rng.choice(idx0, size=min(picks0, len(idx0)), replace=False)
            chosen_indices.append(c0)

        # Pick for genotype 1
        if len(idx1) > 0 and picks1 > 0:
            c1 = self.rng.choice(idx1, size=min(picks1, len(idx1)), replace=False)
            chosen_indices.append(c1)

        # Pick for genotype 2
        if len(idx2) > 0 and picks2 > 0:
            c2 = self.rng.choice(idx2, size=min(picks2, len(idx2)), replace=False)
            chosen_indices.append(c2)

        remainder = n_to_mask - sum(len(arr) for arr in chosen_indices)
        if remainder > 0:
            # Fill remainder from the sets that still have capacity
            # We'll just cycle through 0,1,2 sets
            genotype_sets = [idx0, idx1, idx2]
            self.rng.shuffle(genotype_sets)
            pass_ctr = 0
            while remainder > 0 and pass_ctr < 10:
                for g_set in genotype_sets:
                    if remainder <= 0:
                        break
                    # Already chosen so far
                    already = (
                        set(np.concatenate(chosen_indices)) if chosen_indices else set()
                    )
                    available = np.setdiff1d(g_set, list(already))
                    if len(available) > 0:
                        pick = self.rng.choice(available, size=1, replace=False)
                        chosen_indices.append(pick)
                        remainder -= 1
                pass_ctr += 1

        if chosen_indices:
            chosen_indices = np.concatenate(chosen_indices)
            chosen_rows = row_coords[chosen_indices]
            chosen_cols = col_coords[chosen_indices]
            mask[chosen_rows, chosen_cols] = True

        return mask

    def _simulate_multinom(self, X, original_missing_mask, inverse=False):
        """Simulate missing data by randomly selecting calls to mask, with balanced genotype proportions.

        Args:
            X (np.ndarray): Genotype data.
            original_missing_mask (np.ndarray): Mask of original missing data.
            inverse (bool): Whether to use inverse weighting.

        Returns:
            np.ndarray: Mask of simulated missing data.
        """
        nrows, ncols = X.shape
        mask = np.zeros((nrows, ncols), dtype=bool)

        # Identify known (non-missing) calls
        known_locs = np.where(~original_missing_mask)
        n_known = len(known_locs[0])
        if n_known == 0:
            return mask

        row_coords = known_locs[0]
        col_coords = known_locs[1]
        genotypes = X[row_coords, col_coords]

        # Identify indices for each genotype
        idx0 = np.where(genotypes == 0)[0]
        idx1 = np.where(genotypes == 1)[0]
        idx2 = np.where(genotypes == 2)[0]

        n_to_mask = int(np.floor(self.prop_missing * n_known))
        if n_to_mask == 0:
            return mask

        # Determine probabilities for each genotype class
        if inverse:
            inv0, inv1, inv2 = (
                self.class_weights[0],
                self.class_weights[1],
                self.class_weights[2],
            )
            denom = inv0 + inv1 + inv2
            p = np.array([inv0 / denom, inv1 / denom, inv2 / denom])
        else:
            p = np.array([1 / 3, 1 / 3, 1 / 3])

        # Allocate number of picks per genotype using a multinomial draw
        picks = np.random.multinomial(n_to_mask, p)

        chosen_indices = []
        for idx, n_picks in zip([idx0, idx1, idx2], picks):
            if len(idx) > 0 and n_picks > 0:
                n_picks = min(n_picks, len(idx))
                chosen_indices.append(self.rng.choice(idx, size=n_picks, replace=False))

        if chosen_indices:
            chosen_indices = np.concatenate(chosen_indices)
            chosen_rows = row_coords[chosen_indices]
            chosen_cols = col_coords[chosen_indices]
            mask[chosen_rows, chosen_cols] = True

        return mask

    def _simulate_tree_based(self, X, original_missing_mask, weighted=False):
        """
        Global approach that uses subclades from a phylogenetic tree, but
        picks missing calls across the entire matrix.

        Args:
            X (np.ndarray): Genotype data.
            original_missing_mask (np.ndarray): Mask of original missing data.
            weighted (bool): Whether to use genotype-based weighting.

        Notes:
            For each row, identify which subclade it belongs to. Then we:
                - group all known calls by subclade and pick ~prop_missing fraction of calls to mask.
                - for subclade i with K_i known calls, we pick ~prop_missing*K_i calls to mask (randomly) from that subclade.
                - if weighted=True, we do a genotype-based weighting inside each subclade based on genotype frequencies there (0,1,2) and pick accordingly from each genotype group inside the subclade (proportional to genotype frequency).
        """
        nrows, ncols = X.shape
        mask = np.zeros((nrows, ncols), dtype=bool)

        if self.tree is None:
            # If no tree is provided, can't do subclade-based
            return mask

        tip_labels = self.tree.get_tip_labels()
        if len(tip_labels) != nrows:
            raise ValueError("Tree tip count does not match the number of rows in X.")

        # Identify subclades => map each tip row to a subclade index
        internodes = self.tree.idx_dict["internodes"]
        # subclade_rows_list[i] = list of row indices in subclade i
        subclade_rows_list = []
        for node_idx in internodes:
            node_obj = self.tree.idx_dict["node_obj"][node_idx]
            desc_tips = node_obj.get_tip_labels()
            subclade_rows = []
            for label in desc_tips:
                row_i = tip_labels.index(label)
                subclade_rows.append(row_i)
            subclade_rows_list.append(subclade_rows)

        # NOTE: Assign each row to exactly one subclade for simplicity
        # (Alternatively, rows can appear in multiple subclades if the tree is big.)
        # For now, we pick the subclade with the smallest node_idx that contains that row
        # or let each row only belong to the 'lowest' subclade in subclade_rows_list.
        # This is simplistic, but workable for demonstration.
        row_to_subclade = [-1] * nrows
        for i, rows_here in enumerate(subclade_rows_list):
            for r in rows_here:
                # If not already assigned
                if row_to_subclade[r] == -1:
                    row_to_subclade[r] = i

        # NOTE: For any rows not in any subclade, place them in subclade = -1

        # Gather all known calls
        known_locs = np.where(~original_missing_mask)
        row_coords = known_locs[0]
        col_coords = known_locs[1]
        n_known = len(row_coords)
        if n_known == 0:
            return mask

        # Group calls by subclade
        subclade_dict = {}
        for i in range(n_known):
            r, c = row_coords[i], col_coords[i]
            sub_idx = row_to_subclade[r]
            if sub_idx not in subclade_dict:
                subclade_dict[sub_idx] = []
            subclade_dict[sub_idx].append((r, c))

        # For each subclade, pick ~prop_missing fraction
        for sub_idx, rc_list in subclade_dict.items():
            rc_array = np.array(rc_list)  # shape (Nsub, 2)
            Nsub = rc_array.shape[0]
            if Nsub == 0 or sub_idx < 0:
                continue

            n_to_mask_sub = int(np.floor(self.prop_missing * Nsub))
            if n_to_mask_sub <= 0:
                continue

            if weighted:
                # Weighted by genotype frequency inside this subclade
                # We'll separate calls by genotype 0,1,2
                rvals = rc_array[:, 0]
                cvals = rc_array[:, 1]
                g_sub = X[rvals, cvals]

                idx0 = np.where(g_sub == 0)[0]
                idx1 = np.where(g_sub == 1)[0]
                idx2 = np.where(g_sub == 2)[0]

                freq0 = max(len(idx0), 1e-9)
                freq1 = max(len(idx1), 1e-9)
                freq2 = max(len(idx2), 1e-9)
                inv0, inv1, inv2 = 1 / freq0, 1 / freq1, 1 / freq2
                denom = inv0 + inv1 + inv2
                frac0, frac1, frac2 = inv0 / denom, inv1 / denom, inv2 / denom

                picks0 = int(np.floor(n_to_mask_sub * frac0))
                picks1 = int(np.floor(n_to_mask_sub * frac1))
                picks2 = int(np.floor(n_to_mask_sub * frac2))

                chosen_indices = []
                # pick from genotype 0
                if idx0.size > 0 and picks0 > 0:
                    c0 = self.rng.choice(
                        idx0, size=min(picks0, len(idx0)), replace=False
                    )
                    chosen_indices.append(c0)
                # pick from genotype 1
                if idx1.size > 0 and picks1 > 0:
                    c1 = self.rng.choice(
                        idx1, size=min(picks1, len(idx1)), replace=False
                    )
                    chosen_indices.append(c1)
                # pick from genotype 2
                if idx2.size > 0 and picks2 > 0:
                    c2 = self.rng.choice(
                        idx2, size=min(picks2, len(idx2)), replace=False
                    )
                    chosen_indices.append(c2)

                remainder = n_to_mask_sub - sum(len(arr) for arr in chosen_indices)
                if remainder > 0:
                    genotype_sets = [idx0, idx1, idx2]
                    self.rng.shuffle(genotype_sets)
                    pass_ctr = 0
                    while remainder > 0 and pass_ctr < 10:
                        for g_set in genotype_sets:
                            if remainder <= 0:
                                break
                            already = (
                                set(np.concatenate(chosen_indices))
                                if chosen_indices
                                else set()
                            )
                            available = np.setdiff1d(g_set, list(already))
                            if len(available) > 0:
                                pick = self.rng.choice(available, size=1, replace=False)
                                chosen_indices.append(pick)
                                remainder -= 1
                        pass_ctr += 1

                if chosen_indices:
                    chosen_indices = np.concatenate(chosen_indices)
                    # map to actual rows, cols
                    chosen_rc = rc_array[chosen_indices]
                    mask[chosen_rc[:, 0], chosen_rc[:, 1]] = True

            else:
                # Unweighted subclade => just pick a random subset
                chosen_idx = self.rng.choice(Nsub, size=n_to_mask_sub, replace=False)
                chosen_rc = rc_array[chosen_idx]  # shape (n_to_mask_sub, 2)
                mask[chosen_rc[:, 0], chosen_rc[:, 1]] = True

        return mask

    def _simulate_nonrandom_distance(self, X, original_missing_mask):
        """Simulate non-random missingness using a distance-based approach.

        Uses a class-aware weighting for genotype classes 0, 1, and 2. The weighting factors are specified in self.class_weights. If not provided, the default is 1, 2, 3 for 0, 1, 2, respectively. The weighting factors are used to determine how many calls to mask in each genotype class.

        Args:
            X (np.ndarray): Genotype matrix of shape (n_rows, n_cols) with values in {0,1,2}.
            original_missing_mask (np.ndarray): Boolean mask (same shape as X),
                where True indicates an already-missing value in X.

        Returns:
            np.ndarray: Boolean mask of the same shape as X, where True indicates
                a newly-masked value.

        Notes:
            The distance-based approach is as follows:
                1. Build a genetic distance matrix.
                2. Randomly pick focal samples.
                3. Identify neighbors for each focal sample.
                4. Collect candidate calls (row, col) into a set or list.
                5. Determine how many calls to mask in total (prop_missing * candidate calls).
                6. Classify calls by genotype and assign weighting factors for 0, 1, 2.
                7. Mask calls in each genotype class according to the weighting scheme.
                8. Return the combined mask. If a column is fully masked, unmask one row.
        """
        nrows, ncols = X.shape
        mask = np.zeros((nrows, ncols), dtype=bool)

        # 1. Build a distance matrix
        distmat = self.compute_genetic_distance_matrix(X)

        # 2. Randomly pick focal samples
        row_indices = np.arange(nrows)
        if len(row_indices) == 0:
            return mask

        n_focal_used = min(self.n_focal, len(row_indices))
        focal_samples = self.rng.choice(row_indices, size=n_focal_used, replace=False)

        # 3. For each focal sample, find neighbors
        n_neighbors = max(5, int(0.2 * nrows))  # e.g., 20% or at least 5
        candidate_set = set()
        for fs in focal_samples:
            drow = distmat[fs, :]
            sorted_idx = np.argsort(drow)
            neighbor_rows = sorted_idx[: n_neighbors + 1]  # includes focal itself
            for nr in neighbor_rows:
                # Only consider columns that are currently not missing
                known_cols = np.where(~original_missing_mask[nr, :])[0]
                for c in known_cols:
                    candidate_set.add((nr, c))

        # Convert set to list
        candidate_list = list(candidate_set)
        n_candidate = len(candidate_list)
        if n_candidate == 0:
            return mask

        # 4. Determine how many calls we want to mask in total
        n_to_mask = int(np.floor(self.prop_missing * n_candidate))
        if n_to_mask == 0:
            return mask

        # 5. Classify candidate calls by genotype (0,1,2)
        #    We'll store them in separate lists for easy sampling
        gen0_indices = []
        gen1_indices = []
        gen2_indices = []

        for r, c in candidate_list:
            genotype_value = X[r, c]
            if genotype_value == 0:
                gen0_indices.append((r, c))
            elif genotype_value == 1:
                gen1_indices.append((r, c))
            elif genotype_value == 2:
                gen2_indices.append((r, c))
            # If there are other values or missing, skip them

        # 6. Define weighting for each genotype class. Example: 2 is weighted 3x more likely to be masked than 0, and 1 is weighted 2x more likely to be masked than 0. Adjust these as needed or derive from inverse frequency, e.g. 1 / global_freq.
        if self.class_weights is None:
            w0, w1, w2 = 1.0, 2.0, 3.0
        else:
            w0, w1, w2 = self.class_weights

        # 7. Compute how many from each class to mask, given total n_to_mask
        #    Weighted approach: each group will be allocated a fraction of n_to_mask based on w0, w1, w2, but not exceeding the size of that group.
        gen0_size = len(gen0_indices)
        gen1_size = len(gen1_indices)
        gen2_size = len(gen2_indices)

        sum_weights = (
            (w0 if gen0_size > 0 else 0)
            + (w1 if gen1_size > 0 else 0)
            + (w2 if gen2_size > 0 else 0)
        )

        # If the sum of active weights is 0, nothing to mask
        if sum_weights == 0:
            return mask

        # fraction of total n_to_mask assigned to each group
        frac0 = w0 / sum_weights if gen0_size > 0 else 0
        frac1 = w1 / sum_weights if gen1_size > 0 else 0
        frac2 = w2 / sum_weights if gen2_size > 0 else 0

        # actual number to mask in each genotype
        n0_to_mask = min(gen0_size, int(np.floor(n_to_mask * frac0)))
        n1_to_mask = min(gen1_size, int(np.floor(n_to_mask * frac1)))
        n2_to_mask = min(gen2_size, int(np.floor(n_to_mask * frac2)))

        # 8. Randomly choose which calls to mask in each genotype group
        if n0_to_mask > 0:
            chosen_0 = self.rng.choice(gen0_indices, size=n0_to_mask, replace=False)
        else:
            chosen_0 = []

        if n1_to_mask > 0:
            chosen_1 = self.rng.choice(gen1_indices, size=n1_to_mask, replace=False)
        else:
            chosen_1 = []

        if n2_to_mask > 0:
            chosen_2 = self.rng.choice(gen2_indices, size=n2_to_mask, replace=False)
        else:
            chosen_2 = []

        # 9. Combine chosen calls for final mask
        for r, c in list(chosen_0) + list(chosen_1) + list(chosen_2):
            mask[r, c] = True

        return mask

    def _validate_mask_columns(self, mask):
        """Ensure no column is fully masked.

        If a column is fully masked, unmask exactly one row to avoid fully masked columns.

        Args:
            mask (np.ndarray): Mask of missing data.

        Returns:
            np.ndarray: Updated mask with exactly one row unmasked for each fully masked column.
        """
        nrows, ncols = mask.shape
        for col_idx in range(ncols):
            if np.all(mask[:, col_idx]):
                # unmask exactly one row to avoid fully masked column
                row_to_fix = self.rng.integers(low=0, high=nrows)
                mask[row_to_fix, col_idx] = False
        return mask

    def _validate_mask_rows(self, mask):
        """Ensure no row is fully masked."""
        nrows, ncols = mask.shape
        for row_idx in range(nrows):
            if np.all(mask[row_idx, :]):
                # unmask exactly one column to avoid fully masked row
                col_to_fix = self.rng.integers(low=0, high=ncols)
                mask[row_idx, col_to_fix] = False

    def compute_genetic_distance_matrix(self, X, measure="hamming"):
        """
        Compute pairwise distances ignoring any np.nan entries in X.
        measure='hamming' => fraction of differing calls among non-missing sites.
        """
        X = validate_input_type(X, "array")
        nrows, ncols = X.shape
        distmat = np.zeros((nrows, nrows), dtype=float)

        for i in range(nrows):
            for j in range(i + 1, nrows):
                g1 = X[i, :]
                g2 = X[j, :]
                valid_mask = (~np.isnan(g1)) & (~np.isnan(g2))
                if not np.any(valid_mask):
                    dist = np.nan
                else:
                    if measure == "hamming":
                        dist = np.mean(g1[valid_mask] != g2[valid_mask])
                    else:
                        dist = np.mean(np.abs(g1[valid_mask] - g2[valid_mask]))
                distmat[i, j] = dist
                distmat[j, i] = dist
        return distmat
