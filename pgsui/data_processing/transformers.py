# Standard library imports
import copy
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score
from snpio.utils.logging import LoggerManager

# Custom imports
from pgsui.utils import misc


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
    """Transformer to format autoencoder features and targets before model fitting.

    The input data, X, is encoded to one-hot format, and then missing values are filled to [-1] * num_classes. Missing and observed boolean masks are also generated.

    Example:
        >>> autoenc = AutoEncoderFeatureTransformer(num_classes=3)
        >>> autoenc.fit(X)
        >>> X_train = autoenc.transform(X)

    Attributes:
        num_classes (int): Number of classes in the last axis dimention of the input array. Defaults to 3.
        return_int (bool): Whether to return an integer-encoded array (If True) or a one-hot or multi-label encoded array (If False.). Defaults to False.
        activate (str or None): If not None, then does the appropriate activation. Multilabel learning uses sigmoid activation, and multiclass uses softmax. If set to None, then the function assumes that the input has already been activated. Possible values include: {None, 'sigmoid', 'softmax'}. Defaults to None.
        threshold_increment (float): The increment used to search for the best threshold. Defaults to 0.05.
        logger (LoggerManager): LoggerManager instance. If None, then a new instance will be created. Defaults to None.
        debug (bool): Debug mode. Defaults to False.
        n_classes_ (int): Number of classes in the last axis dimention of the input array.
        classes_ (numpy.ndarray): Array of classes.
        n_outputs_ (int): Number of outputs.
        n_outputs_expected_ (int): Number of expected outputs.
        X_train (numpy.ndarray): One-hot encoded input data.
        X_decoded (numpy.ndarray): Decoded input data.
        missing_mask_ (numpy.ndarray): Missing data mask.
        observed_mask_ (numpy.ndarray): Observed data mask.

    """

    def __init__(
        self,
        num_classes: int = 3,
        return_int: bool = False,
        activate: str = "softmax",
        threshold_increment: float = 0.05,
        logger: Optional[LoggerManager] = None,
        verbose: int = 0,
        debug: bool = False,
    ) -> None:
        """Initialize the AutoEncoderFeatureTransformer.

        This class is used to transform input data to one-hot encoded format, whether it be 3 or 4 classes. It also generates missing and observed data masks. The class can also be used to transform target data back to its original format with the ``inverse_transform`` method. The class can be used with both multiclass and multilabel learning. If multiclass, then the input data should be 012-encoded. If multilabel, then the input data should be 0123456789-encoded.

        Args:
            num_classes (int): The number of classes in the last axis dimention of the input array. Defaults to 3.
            return_int (bool): Whether to return an integer-encoded array (If True) or a one-hot or multi-label encoded array (If False.). Defaults to False.
            activate (str): If not None, then does the appropriate activation. Multilabel learning uses sigmoid activation, and multiclass uses softmax. If set to None, then the function assumes that the input has already been activated. Possible values include: {'sigmoid', 'softmax'}. Defaults to 'softmax'.
            threshold_increment (float): The increment used to search for the best threshold. Defaults to 0.05.
            logger (LoggerManager, optional): LoggerManager instance. If None, then a new instance will be created. Defaults to None.
            debug (bool): Debug mode. Defaults to False.

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
            self.logger - logman.get_logger()
        else:
            self.logger = logger

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame, List[List[int]], torch.Tensor],
        y: Optional[np.ndarray] = None,
    ) -> Any:
        """set attributes used to transform X (input features).

        Args:
            X (Union[numpy.ndarray, pd.DataFrame, List[List[int], torch.Tensor]]): Input integer-encoded numpy array.
            y (None): Ignored. Just for compatibility with sklearn API.

        Returns:
            self: Class instance for method chaining.

        Raises:
            ValueError: Invalid value passed to num_classes in AutoEncoderFeatureTransformer. Only 3 or 4 are supported.
        """
        X = misc.validate_input_type(X, return_type="array")

        self.X_decoded = X

        if self.num_classes == 3:
            enc_func = self.encode_012
        elif self.num_classes == 4:
            enc_func = self.encode_multilab
        elif self.num_classes == 10:
            enc_func = self.encode_multiclass
        else:
            msg = f"Invalid value passed to num_classes in AutoEncoderFeatureTransformer. Only 3 or 4 are supported, but got {self.num_classes}."
            self.logger.error(msg)
            raise ValueError(msg)

        self.enc_func_ = enc_func

        # Encode the data.
        self.X_train = enc_func(X)
        self.classes_ = np.arange(self.num_classes)
        self.n_classes_ = self.num_classes

        # Get missing and observed data boolean masks.
        self.missing_mask_, self.observed_mask_ = self._get_masks(self.X_train)

        # To accomodate multiclass-multioutput.
        self.n_outputs_expected_ = 1
        self.n_outputs_ = self.X_train.shape[1]

        return self

    def transform(
        self, X: Union[np.ndarray, pd.DataFrame, List[List[int]], torch.Tensor]
    ) -> np.ndarray:
        """Transform X to one-hot encoded format.

        Accomodates multiclass targets with a 3D shape. This method fills in missing values with -1.

        Args:
            X (Union[numpy.ndarray, pd.DataFrame, List[List[int]], torch.Tensor]): One-hot encoded target data of shape (n_samples, n_features, num_classes), unless return_int is True, in which case it should be integer-encoded data of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Transformed target data in one-hot format of shape (n_samples, n_features, num_classes).
        """
        X = misc.validate_input_type(X, return_type="array")

        if self.return_int:
            self.logger.debug("Returning integer-encoded data.")
            return X.copy()

        # One-hot encode the data.
        X_enc = self.enc_func_(X)

        # Fill missing values with -1.
        return self._fill(X_enc)

    def inverse_transform(self, y, return_proba=False):
        """Inverse transform the target data to the output format.

        This method is used to transform the target data to the output format by applying the appropriate activation function (softmax for multiclass or sigmoid for multilabel) and inverse transforming to integer-encoded data. The method can also return the probability values instead of the integer-encoded values.

        Args:
            y (torch.Tensor or np.ndarray): Array to inverse transform.
            return_proba (bool): If True, return the probability values instead of the integer-encoded values.

        Returns:
            np.ndarray: Inverse transformed data with appropriate activation function applied (softmax or sigmoid).

        Raises:
            ValueError: Invalid value passed to activate. Valid options: 'softmax', 'sigmoid'.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32).to(device)

        # Apply activation based on the type of problem
        if self.activate == "softmax":
            # Multiclass problem: Apply softmax and decode
            y = torch.softmax(y, dim=-1).cpu().numpy()

            if len(y.shape) != 3:
                y = y.reshape(-1, self.n_outputs_, self.n_classes_)

            if return_proba:
                return y
            else:
                # Number of classes determines output type
                if y.shape[-1] == 3:  # 0-1-2 integer encoding
                    return np.argmax(y, axis=-1)
                elif y.shape[-1] == 4:  # One-hot encoding
                    return np.eye(4)[np.argmax(y, axis=-1)]
                elif y.shape[-1] > 4:  # General integer encoding for >4 classes
                    return np.argmax(y, axis=-1)

        elif self.activate == "sigmoid":
            # Multilabel problem: Apply sigmoid activation
            y = torch.sigmoid(y).cpu().numpy()

            if len(y.shape) != 3:
                y = y.reshape(-1, self.n_outputs_, self.n_classes_)

            if return_proba:
                return y

            # Search for optimal threshold to maximize F1 score
            optimal_threshold = self._find_optimal_threshold(
                self.y_true_bin, y, increment=self.threshold_increment
            )

            # Binarize predictions using the optimal threshold
            y_binarized = np.where(y >= optimal_threshold, 1, 0)

            # Return binary or integer-decoded values (multilabel)
            return self._decode_multilabel(y_binarized)

        else:
            msg = f"Invalid value passed to activate. Valid options: 'softmax', 'sigmoid', but got {self.activate}"
            self.logger.error(msg)
            raise ValueError(msg)

        return y.cpu().numpy()  # Fallback if no activation applied

    # Helper Method to Find Optimal Threshold for Sigmoid Activation
    def _find_optimal_threshold(self, y_true_bin, y_pred_proba, increment=0.05):
        """Find the optimal threshold for sigmoid-activated outputs to maximize F1 score.

        This method is used to find the optimal threshold for sigmoid-activated outputs to maximize the F1 score. It searches for the best threshold value by incrementing from 0 to 1 in steps of the provided ``increment`` value.

        Args:
            y_true_bin (np.ndarray): Ground truth binary labels (0 or 1).
            y_pred_proba (np.ndarray): Predicted probabilities from sigmoid activation.
            increment (float): The increment used to search for the best threshold.

        Returns:
            float: The optimal threshold value.
        """
        thresholds = np.arange(0.0, 1.0, increment)
        best_threshold = 0.5
        best_f1 = 0

        for threshold in thresholds:
            # Binarize predictions with current threshold
            y_pred_bin = np.where(y_pred_proba >= threshold, 1, 0)

            # Compute F1 score
            current_f1 = f1_score(y_true_bin, y_pred_bin, average="macro")

            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold

        return best_threshold

    # Helper Method to Decode Multilabel Encodings
    def _decode_multilabel(self, y_binarized):
        """
        Decode binarized multilabel outputs into integer encodings.

        Args:
            y_binarized (np.ndarray): Binarized predictions (0 or 1) after applying the optimal threshold.

        Returns:
            np.ndarray: Decoded integer-encoded multilabel predictions.
        """
        # Generate integer encodings for multilabel combinations
        mappings = {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "01": 4,
            "02": 5,
            "03": 6,
            "12": 7,
            "13": 8,
            "23": 9,
        }

        # Create binary strings for each sample (e.g., "01" for classes 0 and 1)
        y_pred_idx = ["".join(map(str, row.nonzero()[0])) for row in y_binarized]

        # Map binary strings to integer encodings
        decoded = np.array([mappings.get(enc, -9) for enc in y_pred_idx])

        return decoded

    def encode_012(self, X):
        """Convert 012-encoded data to one-hot encodings using vectorized NumPy operations.

        This method is used to encode 012-encoded data to one-hot encodings. It uses vectorized NumPy operations to achieve this. Missing values are encoded as np.nan. Encodings are as follows: 0 -> [1, 0, 0], 1 -> [0, 1, 0], 2 -> [0, 0, 1], -9 -> [np.nan, np.nan, np.nan].

        Args:
            X (numpy.ndarray): Input array with 012-encoded data and -9 as the missing data value.

        Returns:
            numpy.ndarray: One-hot encoded data, ignoring missing values (np.nan).
        """
        # Initialize the output array with NaNs for missing data (-9)
        Xt = np.full((X.shape[0], X.shape[1], 3), np.nan)

        # Create a mask for valid data (0, 1, 2)
        valid_mask = (X >= 0) & (X <= 2)

        # Create identity matrix for one-hot encoding: 0 -> [1, 0, 0],
        # 1 -> [0, 1, 0], 2 -> [0, 0, 1]
        # Perform one-hot encoding for valid values only
        one_hot_values = np.eye(3)

        # Assign one-hot encodings based on values in X
        Xt[valid_mask] = one_hot_values[X[valid_mask].astype(int)]

        return Xt

    def encode_multilab(self, X, multilab_value=0.5):
        """Encode 0-9 integer data in multi-label one-hot format using vectorized NumPy operations.

        This method is used to encode 0-9 integer data in multi-label one-hot format. It uses vectorized NumPy operations to achieve this. Missing values are encoded as np.nan. Encodings are as follows: 0 -> [1, 0, 0, 0], 1 -> [0, 1, 0, 0], 2 -> [0, 0, 1, 0], 3 -> [0, 0, 0, 1], 4 -> [0.5, 0.5, 0, 0], 5 -> [0.5, 0, 0.5, 0], 6 -> [0.5, 0, 0, 0.5], 7 -> [0, 0.5, 0.5, 0], 8 -> [0, 0.5, 0, 0.5], 9 -> [0, 0, 0.5, 0.5].

        Args:
            X (numpy.ndarray): Input array with 012-encoded data and -9 as the missing data value.
            multilab_value (float): Value to use for multi-label target encodings. Defaults to 0.5.

        Returns:
            numpy.ndarray: One-hot encoded data, ignoring missing values (np.nan). Multi-label categories will be encoded as multilab_value.
        """
        # Initialize the output array with NaNs for missing data (-9)
        Xt = np.full((X.shape[0], X.shape[1], 4), np.nan)

        # Define the mappings for each category
        basic_mappings = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # 0
                [0.0, 1.0, 0.0, 0.0],  # 1
                [0.0, 0.0, 1.0, 0.0],  # 2
                [0.0, 0.0, 0.0, 1.0],  # 3
                [multilab_value, multilab_value, 0.0, 0.0],  # 4
                [multilab_value, 0.0, multilab_value, 0.0],  # 5
                [multilab_value, 0.0, 0.0, multilab_value],  # 6
                [0.0, multilab_value, multilab_value, 0.0],  # 7
                [0.0, multilab_value, 0.0, multilab_value],  # 8
                [0.0, 0.0, multilab_value, multilab_value],  # 9
            ]
        )

        # Create a mask for valid data (0-9)
        valid_mask = X >= 0

        # Assign values for valid data using advanced indexing
        Xt[valid_mask] = basic_mappings[X[valid_mask]]

        return Xt

    def encode_multiclass(self, X, num_classes=10, missing_value=-9):
        """Encode 0-9 integer data in multi-class one-hot format using vectorized NumPy operations.

        Missing values get encoded as ``[np.nan] * num_classes``. This method is used to encode 0-9 integer data in multi-class one-hot format. It uses vectorized NumPy operations to achieve this. Missing values are encoded as np.nan. Encodings are as follows: 0 -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1 -> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 4 -> [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 6 -> [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 7 -> [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 8 -> [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 9 -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 1].

        Args:
            X (numpy.ndarray): Input array with integer-encoded data and ``missing_value`` as the missing data value.
            num_classes (int, optional): Number of classes to use. Defaults to 10.
            missing_value (int, optional): Missing data value to replace with ``[np.nan] * num_classes``. Defaults to -9.

        Returns:
            numpy.ndarray: Multi-class one-hot encoded data, ignoring missing values (np.nan).
        """
        # Initialize output array with NaNs for missing data
        Xt = np.full((X.shape[0], X.shape[1], num_classes), np.nan)

        # Create a mask for valid data (i.e., not equal to the missing value)
        valid_mask = X != missing_value

        # One-hot encoding for valid data
        Xt[valid_mask, X[valid_mask]] = 1

        return Xt

    def _fill(self, data: np.ndarray, missing_value: int = -1) -> np.ndarray:
        """Mask missing data as ``missing_value``.

        This method is used to mask missing data as ``missing_value``. It sets the missing data to the provided value. If the number of classes is greater than 1, then it sets the missing data to a list of the same length as the number of classes.

        Args:
            data (numpy.ndarray): Input with missing values of shape (n_samples, n_features, num_classes).
            missing_value (int): Value to set missing data to. If a list is provided, then its length should equal the number of one-hot classes. Defaults to -1.

        Returns:
            numpy.ndarray: Data with missing values set to ``missing_value``.
        """
        missing_mask = self._create_missing_mask(data)
        if self.num_classes > 1:
            missing_value = [missing_value] * self.num_classes
        data[missing_mask] = missing_value
        return data

    def _get_masks(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Format the provided target data for use with UBP/NLPCA.

        This method generates missing and observed data masks for the input data.

        Args:
            y (numpy.ndarray(float)): Input data that will be used as the target of shape (n_samples, n_features, num_classes).

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: Missing and observed data masks.
        """
        missing_mask = self._create_missing_mask(X)
        observed_mask = ~missing_mask
        return missing_mask, observed_mask

    def _create_missing_mask(self, data: np.ndarray) -> np.ndarray:
        """Creates a missing data mask with boolean values.
        Args:
            data (numpy.ndarray): Data to generate missing mask from, of shape (n_samples, n_features, n_classes).
        Returns:
            numpy.ndarray: Boolean mask of missing values of shape (n_samples, n_features), with True corresponding to a missing data point.
        """
        return np.isnan(data).all(axis=2)


class MLPTargetTransformer(BaseEstimator, TransformerMixin):
    """Transformer to format UBP / NLPCA target data both before and after model fitting."""

    def fit(self, y):
        """Fit 012-encoded target data.

        Args:
            y (numpy.ndarray): Target data that is 012-encoded.

        Returns:
            self: Class instance.
        """
        y = misc.validate_input_type(y, return_type="array")

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
        y = misc.validate_input_type(y, return_type="array")
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
        y = misc.validate_input_type(y, return_type="array")

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
        y = misc.validate_input_type(y, return_type="array")
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
    """Simulate missing data on genotypes encoded with a GenotypeData object.

    Copies metadata from a GenotypeData object and simulates user-specified proportion of missing data using a specified strategy. The simulated missing data can be used to train machine learning models to impute missing data.

    There are five strategies available for simulating missing data:
        - "nonrandom": Branches from GenotypeData.guidetree will be randomly sampled to generate missing data on descendant nodes.
        - "nonrandom_weighted": Missing data will be placed on nodes proportionally to their branch lengths (e.g., to generate data distributed as might be the case with mutation-disruption of RAD sites).
        - "random_weighted": Missing data will be placed on nodes proportionally to their branch lengths, but the missing data will be randomly distributed.
        - "random_weighted_inv": Missing data will be placed on nodes proportionally to their branch lengths, but the missing data will be randomly distributed and inversely proportional to the branch lengths.
        - "random": Missing data will be randomly distributed across the data.

    Attributes:

        original_missing_mask_ (numpy.ndarray): Array with boolean mask for original missing locations.

        simulated_missing_mask_ (numpy.ndarray): Array with boolean mask for simulated missing locations, excluding the original ones.

        all_missing_mask_ (numpy.ndarray): Array with boolean mask for all missing locations, including both simulated and original.
    """

    def __init__(
        self,
        genotype_data: Any,
        *,
        prop_missing: float = 0.1,
        strategy: str = "random",
        missing_val: Union[int, float] = -9,
        mask_missing: bool = True,
        indices: Optional[Dict[str, List[int]]] = None,
        dataset_name: str = "train",
        seed: int = None,
        verbose: int = 0,
        tol: Optional[float] = None,
        max_tries: Optional[int] = None,
        logger: Optional[LoggerManager] = None,
        debug: bool = False,
    ) -> None:
        """Initialize the SimGenotypeDataTransformer.

        This transformer simulates missing data on genotypes encoded with a GenotypeData object. It copies metadata from a GenotypeData object and simulates a user-specified proportion of missing data using a specified strategy. The simulated missing data can be used to train machine learning models to impute missing data.

        Args:
            genotype_data (GenotypeData): GenotypeData instance.

            prop_missing (float): Proportion of missing data desired in output. Defaults to 0.1

            strategy (str): Strategy for simulating missing data. May be one of: "nonrandom", "nonrandom_weighted", "random_weighted", "random_weighted_inv", or "random". When set to "nonrandom", branches from GenotypeData.guidetree will be randomly sampled to generate missing data on descendant nodes. For "nonrandom_weighted", missing data will be placed on nodes proportionally to their branch lengths (e.g., to generate data distributed as might be the case with mutation-disruption of RAD sites). For "random_weighted", missing data will be placed on nodes proportionally to their branch lengths, but the missing data will be randomly distributed. For "random_weighted_inv", missing data will be placed on nodes proportionally to their branch lengths, but the missing data will be randomly distributed and inversely proportional to the branch lengths. For "random", missing data will be randomly distributed across the data. Defaults to "random".

            missing_val (int): Value that represents missing data. Defaults to -9.

            mask_missing (bool): True if you want to avoid array elements that were originally missing values when simulating new missing data, False to allow new missing data to be placed on top of original missing data. Defaults to True.

            indices (Dict[str, List[int]]): Dictionary of indices for samples and features. Defaults to None.

            dataset_name (str): Name of the dataset. Valid options are "train", "valid", or "test". Defaults to "train".

            seed (int): Seed for random number generation. If None, the random choice selector will be initialized with a random seed and will be non-deterministic. Defaults to None.

            verbose (bool, optional): Verbosity level. Defaults to 0.

            tol (float): Tolerance to reach proportion specified in self.prop_missing. Defaults to 1/num_snps*num_inds

            max_tries (int): Maximum number of tries to reach targeted missing data proportion within specified tol. If None, num_inds will be used. Defaults to None.

            logger (LoggerManager): Logger instance. If None, a new logger will be created. Defaults to None.

            debug (bool): If True, debug logging mode will be enabled. Defaults to False.
        """

        if logger is not None:
            self.logger = logger
        else:
            logman = LoggerManager(
                name=__name__,
                prefix="missing_data_simulator",
                level=verbose >= 1,
                debug=debug,
            )

            self.logger = logman.get_logger()

        self.genotype_data = genotype_data
        self.prop_missing = prop_missing
        self.strategy = strategy
        self.missing_val = missing_val
        self.mask_missing = mask_missing
        self.dataset_name = dataset_name
        self.seed = seed
        self.verbose = verbose
        self.tol = tol
        self.max_tries = max_tries
        self.indices = indices

        self.rng = np.random.default_rng(seed)

    def fit(self, X):
        """Fit to input data X by simulating missing data.

        Missing data will be simulated in varying ways depending on the ``strategy`` setting.

        Args:
            X (Union[pandas.DataFrame, numpy.ndarray, List[List[int]], torch.Tensor]): Data with which to simulate missing data. It should have already been imputed with one of the non-machine learning simple imputers, and there should be no missing data present in X.

        Raises:
            TypeError: SimGenotypeData.tree must not be NoneType when using strategy="nonrandom" or "nonrandom_weighted".
            ValueError: Invalid ``strategy`` parameter provided.
        """
        X = misc.validate_input_type(X, return_type="array").astype("float32")

        self.logger.info(
            f"Adding {self.prop_missing} missing data per column using strategy: {self.strategy}"
        )

        if ~np.isnan(self.missing_val):
            X[X == self.missing_val] = np.nan

        # Store the original missing mask (before simulation)
        self.original_missing_mask_ = np.isnan(X)

        # Simulate missing data for the training set
        self.sim_missing_mask_ = self._simulate_missing_mask(X)

        # Compute all missing values (original + simulated)
        self.all_missing_mask_ = np.logical_or(
            self.sim_missing_mask_, self.original_missing_mask_
        )

        return self

    def transform(self, X):
        """Apply the missing data simulation to X.

        This method applies the same missing data strategy to X but does not use the exact mask from the training set, so as to mitigate data leakage.

        Args:
            X (Union[numpy.ndarray, pd.DataFrame, List[List[int]], torch.Tensor]): Data to which missing data simulation should be applied.

        Returns:
            numpy.ndarray: Transformed data with simulated missing values.
        """
        X = misc.validate_input_type(X, return_type="array")

        # Generate a new mask for the validation or test set based on the same strategy
        sim_missing_mask = self._simulate_missing_mask(X)

        self.logger.debug(f"Simulated missing mask: {sim_missing_mask}")

        Xt = X.copy()

        # Apply the missing mask to generate missing values in the dataset
        Xt[sim_missing_mask] = self.missing_val

        return Xt

    def _simulate_missing_mask(self, X):
        """Simulate missing data on the input data X.

        This method simulates missing data on the input data X using the specified strategy. The missing data will be simulated in varying ways depending on the ``strategy`` setting. The simulated missing data will be returned as a boolean mask.

        Args:
            X (numpy.ndarray): Input data to simulate missing data on.

        Returns:
            numpy.ndarray: Boolean mask of simulated missing data.

        Raises:
            ValueError: Invalid ``strategy`` parameter provided
        """
        # Convert missing values to np.nan.
        if ~np.isnan(self.missing_val):
            X[X == self.missing_val] = np.nan

        mask = None

        if self.strategy not in {
            "random",
            "random_weighted",
            "random_weighted_inv",
            "nonrandom",
            "nonrandom_weighted",
        }:
            msg = f"Invalid 'strategy' provided: {self.strategy}; expected one of: 'random', 'random_weighted', 'random_weighted_inv', 'nonrandom', or 'nonrandom_weighted'."
            self.logger.error(msg)
            raise ValueError(msg)

        if self.strategy == "random":
            if self.mask_missing:
                # Get indexes where non-missing (Xobs) and missing (Xmiss).
                Xobs = np.where(~np.isnan(X).ravel())[0]
                Xmiss = np.where(np.isnan(X).ravel())[0]

                # Generate mask of 0's (non-missing) and 1's (missing).
                obs_mask = self.rng.choice(
                    [0, 1],
                    size=Xobs.size,
                    p=((1 - self.prop_missing), self.prop_missing),
                ).astype(bool)

                # Make missing data mask.
                mask = np.zeros(X.size, dtype=bool)
                mask[Xobs] = obs_mask
                mask[Xmiss] = True

                # Reshape from raveled to 2D.
                # With strategy=="random", mask_ is equal to all_missing_.
                mask = np.reshape(mask, X.shape)

            else:
                # Generate mask of 0's (non-missing) and 1's (missing).
                mask = self.rng.choice(
                    [0, 1], size=X.shape, p=((1 - self.prop_missing), self.prop_missing)
                ).astype(bool)

        elif self.strategy == "random_weighted":
            mask = self.random_weighted_missing_data(X, inv=False)

        elif self.strategy == "random_weighted_inv":
            mask = self.random_weighted_missing_data(X, inv=True)

        elif self.strategy in {"nonrandom", "nonrandom_weighted"}:
            if self.genotype_data.tree is None:
                msg = "SimGenotypeData.tree cannot be None when using strategy='nonrandom' or strategy='nonrandom_weighted'"
                self.logger.error(msg)
                raise TypeError(msg)

            mask = np.full_like(X, 0.0, dtype=bool)

            weighted = True if self.strategy == "nonrandom_weighted" else False

            sample_map = {}
            for i, sample in enumerate(self.genotype_data.samples):
                if self.indices is not None:
                    if i in self.indices[self.dataset_name]:
                        sample_map[sample] = i
                    else:
                        continue
                else:
                    sample_map[sample] = i

            # if no tolerance provided, set to 1 snp position
            if self.tol is None:
                self.tol = 1.0 / mask.size

            # if no max_tries provided, set to # inds
            if self.max_tries is None:
                self.max_tries = mask.shape[0]

            filled = False
            while not filled:
                # ToDO: Ensure this works with SNPio refactor.
                # Get list of samples from tree
                samples = self._sample_tree(
                    internal_only=False, skip_root=True, weighted=weighted
                )

                # Convert to row indices
                rows = [sample_map[i] for i in samples]

                # Randomly sample a column
                col_idx = self.rng.integers(low=0, high=mask.shape[1])
                sampled_col = copy.copy(mask[:, col_idx])
                miss_mask = copy.copy(np.isnan(X)[:, col_idx])

                # Mask column
                sampled_col[rows] = True

                # If original was missing, set back to False.
                if self.mask_missing:
                    sampled_col[miss_mask] = False
                if np.count_nonzero(sampled_col) == sampled_col.size:
                    # check that column is not 100% missing now
                    # if yes, sample again
                    continue

                # if not, set values in mask matrix
                mask[:, col_idx] = sampled_col

                # if this addition pushes missing % > self.prop_missing,
                # check previous prop_missing, remove masked samples from
                # this column until closest to target prop_missing
                current_prop = np.sum(mask) / mask.size
                if abs(current_prop - self.prop_missing) <= self.tol:
                    filled = True
                    break
                elif current_prop > self.prop_missing:
                    tries = 0
                    while (
                        abs(current_prop - self.prop_missing) > self.tol
                        and tries < self.max_tries
                    ):
                        r = self.rng.integers(low=0, high=mask.shape[0])
                        c = self.rng.integers(low=0, high=mask.shape[1])
                        mask[r, c] = False
                        tries += 1
                        current_prop = np.sum(mask) / mask.size

                    filled = True
                else:
                    continue

        self._validate_mask(mask)

        return mask.astype(bool)

    def _validate_mask(self, mask):
        """Ensure no entirely missing columns are simulated."""

        for i, column in enumerate(mask.T):
            if np.sum(column) == column.size:
                # If all missing, set one to False.
                mask[self.rng.integers(low=0, high=len(column)), i] = False

    def random_weighted_missing_data(self, X, inv=False):
        """Choose values for which to simulate missing data by biasing towards the minority or majority alleles, depending on whether inv is True or False.

        Args:
            X (np.ndarray): True values.

            inv (bool): If True, then biases towards choosing majority alleles. If False, then generates a stratified random sample (class proportions ~= full dataset) Defaults to False.

        Returns:
            np.ndarray: X with simulated missing values.

        """
        # Get unique classes and their counts
        classes, counts = np.unique(X, return_counts=True)

        # Compute class weights
        class_weights = 1 / counts if inv else counts

        # Normalize class weights
        class_weights = class_weights / sum(class_weights)

        # Compute mask
        if self.mask_missing:
            # Get indexes where non-missing (Xobs) and missing (Xmiss)
            Xobs = np.where(~self.original_missing_mask_.ravel())[0]
            Xmiss = np.where(self.original_missing_mask_.ravel())[0]

            # Generate mask of 0's (non-missing) and 1's (missing)
            obs_mask = self.rng.choice(classes, size=Xobs.size, p=class_weights)
            obs_mask = (obs_mask == classes[:, None]).argmax(axis=0)

            # Make missing data mask
            mask = np.zeros(X.size, dtype=bool)
            mask[Xobs] = obs_mask
            mask[Xmiss] = 1

            # Reshape from raveled to 2D
            mask = mask.reshape(X.shape)
        else:
            # Generate mask of 0's (non-missing) and 1's (missing)
            mask = self.rng.choice(classes, size=X.size, p=class_weights)
            mask = (mask == classes[:, None]).argmax(axis=0).reshape(X.shape)

        self._validate_mask(mask)

        return mask

    def _sample_tree(
        self, internal_only=False, tips_only=False, skip_root=True, weighted=False
    ):
        """Randomly sampling clades from GenotypeData.tree objects.

        Args:
            internal_only (bool): Only sample from NON-TIPS. Defaults to False.
            tips_only (bool): Only sample from tips. Defaults to False.
            skip_root (bool): Exclude sampling of root node. Defaults to True.
            weighted (bool): Weight sampling by branch length. Defaults to False.

        Returns:
            List[str]: List of descendant tips from the sampled node.

        Raises:
            ValueError: ``tips_only`` and ``internal_only`` cannot both be True.

        ToDo:
            Ensure this method works with SNPio refactor.
        """

        if tips_only and internal_only:
            msg = "'tips_only' and 'internal_only' cannot both be True"
            self.logger.error(msg)
            raise ValueError(msg)

        # to only sample internal nodes add  if not i.is_leaf()
        node_dict = {}

        for node in self.genotype_data.tree.treenode.traverse("preorder"):
            ## node.idx is node indexes.
            ## node.dist is branch lengths.
            if skip_root:
                if node.idx == self.genotype_data.tree.nnodes - 1:
                    # If root node.
                    continue

            if tips_only:
                if not node.is_leaf():
                    continue

            elif internal_only:
                if node.is_leaf():
                    continue
            node_dict[node.idx] = node.dist

        if weighted:
            s = sum(list(node_dict.values()))

            # Node index / sum of node distances.
            p = [i / s for i in list(node_dict.values())]
            node_idx = self.rng.choice(list(node_dict.keys()), size=1, p=p)[0]
        else:
            # Get missing choice from random clade.
            node_idx = self.rng.choice(list(node_dict.keys()), size=1)[0]
        return self.genotype_data.tree.get_tip_labels(idx=node_idx)

    def _mask_snps(self, X):
        """Mask positions in SimGenotypeData.snps and SimGenotypeData.onehot

        This method will mask the positions in the input data X based on the mask generated during the fit method.

        Args:
            X (numpy.ndarray): Data to mask.

        Returns:
            numpy.ndarray: Masked data.

        Raises:
            ValueError: Invalid shape of input X. Must be 2D or 3D.
        """
        if len(X.shape) == 3:
            # One-hot encoded.
            mask_val = [0.0, 0.0, 0.0, 0.0]
        elif len(X.shape) == 2:
            # 012-encoded.
            mask_val = -9
        else:
            msg = f"Invalid shape of input X: {X.shape}; must be 2D or 3D."
            self.logger.error(msg)
            raise ValueError(msg)

        Xt = X.copy()
        mask_boolean = self.sim_missing_mask_ != 0
        Xt[mask_boolean] = mask_val
        return Xt
