import copy
import os
import logging
import sys
import warnings

import numpy as np
import pandas as pd

# Third-party imports
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

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

# Custom Modules
try:
    from ..utils import misc

except (ModuleNotFoundError, ValueError, ImportError):
    from pgsui.utils import misc


# Pandas on pip gives a performance warning when doing the below code.
# Apparently it's a bug that exists in the pandas version I used here.
# It can be safely ignored.
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


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

    The input data, X, is encoded to one-hot format, and then missing values are filled to [-1] * num_classes.

    Missing and observed boolean masks are also generated.

    Args:
        num_classes (int, optional): The number of classes in the last axis dimention of the input array. Defaults to 3.

        return_int (bool, optional): Whether to return an integer-encoded array (If True) or a one-hot or multi-label encoded array (If False.). Defaults to False.

        activate (str or None, optional): If not None, then does the appropriate activation. Multilabel learning uses sigmoid activation, and multiclass uses softmax. If set to None, then the function assumes that the input has already been activated. Possible values include: {None, 'sigmoid', 'softmax'}. Defaults to None.
    """

    def __init__(self, num_classes=3, return_int=False, activate=None):
        self.num_classes = num_classes
        self.return_int = return_int
        self.activate = activate

    def fit(self, X, y=None):
        """set attributes used to transform X (input features).

        Args:
            X (numpy.ndarray): Input integer-encoded numpy array.

            y (None): Just for compatibility with sklearn API.
        """
        X = misc.validate_input_type(X, return_type="array")

        self.X_decoded = X

        # VAE uses 4 classes ([A,T,G,C]), SAE uses 3 ([0,1,2]).
        if self.num_classes == 3:
            enc_func = self.encode_012
        elif self.num_classes == 4:
            enc_func = self.encode_multilab
        elif self.num_classes == 10:
            enc_func = self.encode_multiclass
        else:
            raise ValueError(
                f"Invalid value passed to num_classes in "
                f"AutoEncoderFeatureTransformer. Only 3 or 4 are supported, "
                f"but got {self.num_classes}."
            )

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

    def transform(self, X):
        """Transform X to one-hot encoded format.

        Accomodates multiclass targets with a 3D shape.

        Args:
            X (numpy.ndarray): One-hot encoded target data of shape (n_samples, n_features, num_classes).

        Returns:
            numpy.ndarray: Transformed target data in one-hot format of shape (n_samples, n_features, num_classes).
        """
        if self.return_int:
            return X
        else:
            # X = misc.validate_input_type(X, return_type="array")
            return self._fill(self.X_train, self.missing_mask_)

    def inverse_transform(self, y, return_proba=False):
        """Transform target to output format.

        Args:
            y (numpy.ndarray): Array to inverse transform.

            return_proba (bool): Just for compatibility with scikeras API.
        """
        try:
            if self.activate is None:
                y = y.numpy()
            elif self.activate == "softmax":
                y = tf.nn.softmax(y).numpy()
            elif self.activate == "sigmoid":
                y = tf.nn.sigmoid(y).numpy()
            else:
                raise ValueError(
                    f"Invalid value passed to keyword argument activate. Valid "
                    f"options include: None, 'softmax', or 'sigmoid', but got "
                    f"{self.activate}"
                )
        except AttributeError:
            # If numpy array already.
            if self.activate is None:
                y = y.copy()
            elif self.activate == "softmax":
                y = tf.nn.softmax(tf.convert_to_tensor(y)).numpy()
            elif self.activate == "sigmoid":
                y = tf.nn.sigmoid(tf.convert_to_tensor(y)).numpy()
            else:
                raise ValueError(
                    f"Invalid value passed to keyword argument activate. Valid "
                    f"options include: None, 'softmax', or 'sigmoid', but got "
                    f"{self.activate}"
                )
        return y

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
        }
        for row in np.arange(X.shape[0]):
            Xt[row] = [mappings[enc] for enc in X[row]]
        return Xt

    def encode_multilab(self, X, multilab_value=1.0):
        """Encode 0-9 integer data in multi-label one-hot format.
        Args:
            X (numpy.ndarray): Input array with 012-encoded data and -9 as the missing data value.

            multilab_value (float): Value to use for multilabel target encodings. Defaults to 0.5.
        Returns:
            pandas.DataFrame: One-hot encoded data, ignoring missing values (np.nan). multi-label categories will be encoded as 0.5. Otherwise, it will be 1.0.
        """
        Xt = np.zeros(shape=(X.shape[0], X.shape[1], 4))
        mappings = {
            0: [1.0, 0.0, 0.0, 0.0],
            1: [0.0, 1.0, 0.0, 0.0],
            2: [0.0, 0.0, 1.0, 0.0],
            3: [0.0, 0.0, 0.0, 1.0],
            4: [multilab_value, multilab_value, 0.0, 0.0],
            5: [multilab_value, 0.0, multilab_value, 0.0],
            6: [multilab_value, 0.0, 0.0, multilab_value],
            7: [0.0, multilab_value, multilab_value, 0.0],
            8: [0.0, multilab_value, 0.0, multilab_value],
            9: [0.0, 0.0, multilab_value, multilab_value],
            -9: [np.nan, np.nan, np.nan, np.nan],
        }
        for row in np.arange(X.shape[0]):
            Xt[row] = [mappings[enc] for enc in X[row]]
        return Xt

    def decode_multilab(self, X, multilab_value=1.0):
        """Decode one-hot format data back to 0-9 integer data.

        Args:
            X (numpy.ndarray): Input array with one-hot-encoded data.

            multilab_value (float): Value to use for multilabel target encodings. Defaults to 0.5.

        Returns:
            pandas.DataFrame: Decoded data, with multi-label categories decoded to their original integer representation.
        """
        Xt = np.zeros(shape=(X.shape[0], X.shape[1]))
        mappings = {
            tuple([1.0, 0.0, 0.0, 0.0]): 0,
            tuple([0.0, 1.0, 0.0, 0.0]): 1,
            tuple([0.0, 0.0, 1.0, 0.0]): 2,
            tuple([0.0, 0.0, 0.0, 1.0]): 3,
            tuple([multilab_value, multilab_value, 0.0, 0.0]): 4,
            tuple([multilab_value, 0.0, multilab_value, 0.0]): 5,
            tuple([multilab_value, 0.0, 0.0, multilab_value]): 6,
            tuple([0.0, multilab_value, multilab_value, 0.0]): 7,
            tuple([0.0, multilab_value, 0.0, multilab_value]): 8,
            tuple([0.0, 0.0, multilab_value, multilab_value]): 9,
            tuple([np.nan, np.nan, np.nan, np.nan]): -9,
        }
        for row in np.arange(X.shape[0]):
            Xt[row] = [mappings[tuple(enc)] for enc in X[row]]
        return Xt

    def encode_multiclass(self, X, num_classes=10, missing_value=-9):
        """Encode 0-9 integer data in multi-class one-hot format.

        Missing values get encoded as ``[np.nan] * num_classes``
        Args:
            X (numpy.ndarray): Input array with 012-encoded data and ``missing_value`` as the missing data value.

            num_classes (int, optional): Number of classes to use. Defaults to 10.

            missing_value (int, optional): Missing data value to replace with ``[np.nan] * num_classes``\. Defaults to -9.
        Returns:
            pandas.DataFrame: Multi-class one-hot encoded data, ignoring missing values (np.nan).
        """
        int_cats, ohe_arr = np.arange(num_classes), np.eye(num_classes)
        mappings = dict(zip(int_cats, ohe_arr))
        mappings[missing_value] = np.array([np.nan] * num_classes)

        Xt = np.zeros(shape=(X.shape[0], X.shape[1], num_classes))
        for row in np.arange(X.shape[0]):
            Xt[row] = [mappings[enc] for enc in X[row]]
        return Xt

    def _fill(self, data, missing_mask, missing_value=-1):
        """Mask missing data as ``missing_value``\.

        Args:
            data (numpy.ndarray): Input with missing values of shape (n_samples, n_features, num_classes).

            missing_mask (np.ndarray(bool)): Missing data mask with True corresponding to a missing value.

            missing_value (int): Value to set missing data to. If a list is provided, then its length should equal the number of one-hot classes.
        """
        if self.num_classes > 1:
            missing_value = [missing_value] * self.num_classes
        data[missing_mask] = missing_value
        return data

    def _get_masks(self, X):
        """Format the provided target data for use with UBP/NLPCA.

        Args:
            y (numpy.ndarray(float)): Input data that will be used as the target of shape (n_samples, n_features, num_classes).

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
        """Mask missing data as ``missing_value``\.

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
        """Mask missing data as ``missing_value``\.

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
    """Simulate missing data on genotypes read/ encoded in a GenotypeData object.

    Copies metadata from a GenotypeData object and simulates user-specified proportion of missing data

    Args:
        genotype_data (GenotypeData object): GenotypeData instance.

        prop_missing (float, optional): Proportion of missing data desired in output. Defaults to 0.1

        strategy (str, optional): Strategy for simulating missing data. May be one of: "nonrandom", "nonrandom_weighted", "random_weighted", "random_weighted_inv", or "random". When set to "nonrandom", branches from GenotypeData.guidetree will be randomly sampled to generate missing data on descendant nodes. For "nonrandom_weighted", missing data will be placed on nodes proportionally to their branch lengths (e.g., to generate data distributed as might be the case with mutation-disruption of RAD sites). Defaults to "random"

        missing_val (int, optional): Value that represents missing data. Defaults to -9.

        mask_missing (bool, optional): True if you want to skip original missing values when simulating new missing data, False otherwise. Defaults to True.

        verbose (bool, optional): Verbosity level. Defaults to 0.

        tol (float): Tolerance to reach proportion specified in self.prop_missing. Defaults to 1/num_snps*num_inds

        max_tries (int): Maximum number of tries to reach targeted missing data proportion within specified tol. If None, num_inds will be used. Defaults to None.

    Attributes:

        original_missing_mask_ (numpy.ndarray): Array with boolean mask for original missing locations.

        simulated_missing_mask_ (numpy.ndarray): Array with boolean mask for simulated missing locations, excluding the original ones.

        all_missing_mask_ (numpy.ndarray): Array with boolean mask for all missing locations, including both simulated and original.

    Properties:
        missing_count (int): Number of genotypes masked by chosen missing data strategy

        prop_missing_real (float): True proportion of missing data generated using chosen strategy

        mask (numpy.ndarray): 2-dimensional array tracking the indices of sampled missing data sites (n_samples, n_sites)
    """

    def __init__(
        self,
        genotype_data,
        *,
        prop_missing=0.1,
        strategy="random",
        missing_val=-9,
        mask_missing=True,
        verbose=0,
        tol=None,
        max_tries=None,
    ) -> None:
        self.genotype_data = genotype_data
        self.prop_missing = prop_missing
        self.strategy = strategy
        self.missing_val = missing_val
        self.mask_missing = mask_missing
        self.verbose = verbose
        self.tol = tol
        self.max_tries = max_tries

    def fit(self, X):
        """Fit to input data X by simulating missing data.

        Missing data will be simulated in varying ways depending on the ``strategy`` setting.

        Args:
            X (pandas.DataFrame, numpy.ndarray, or List[List[int]]): Data with which to simulate missing data. It should have already been imputed with one of the non-machine learning simple imputers, and there should be no missing data present in X.

        Raises:
            TypeError: SimGenotypeData.tree must not be NoneType when using strategy="nonrandom" or "nonrandom_weighted".

            ValueError: Invalid ``strategy`` parameter provided.
        """
        X = misc.validate_input_type(X, return_type="array").astype("float32")

        if self.verbose > 0:
            print(
                f"\nAdding {self.prop_missing} missing data per column "
                f"using strategy: {self.strategy}"
            )

        if np.all(np.isnan(np.array([self.missing_val])) == False):
            X[X == self.missing_val] = np.nan

        self.original_missing_mask_ = np.isnan(X)

        if self.strategy == "random":
            if self.mask_missing:
                # Get indexes where non-missing (Xobs) and missing (Xmiss).
                Xobs = np.where(~self.original_missing_mask_.ravel())[0]
                Xmiss = np.where(self.original_missing_mask_.ravel())[0]

                # Generate mask of 0's (non-missing) and 1's (missing).
                obs_mask = np.random.choice(
                    [0, 1],
                    size=Xobs.size,
                    p=((1 - self.prop_missing), self.prop_missing),
                ).astype(bool)

                # Make missing data mask.
                mask = np.zeros(X.size)
                mask[Xobs] = obs_mask
                mask[Xmiss] = 1

                # Reshape from raveled to 2D.
                # With strategy=="random", mask_ is equal to all_missing_.
                self.mask_ = np.reshape(mask, X.shape)

            else:
                # Generate mask of 0's (non-missing) and 1's (missing).
                self.mask_ = np.random.choice(
                    [0, 1],
                    size=X.shape,
                    p=((1 - self.prop_missing), self.prop_missing),
                ).astype(bool)

            # Make sure no entirely missing columns were simulated.
            self._validate_mask()

        elif self.strategy == "random_weighted":
            self.mask_ = self.random_weighted_missing_data(X, inv=False)

        elif self.strategy == "random_weighted_inv":
            self.mask_ = self.random_weighted_missing_data(X, inv=True)

        elif (
            self.strategy == "nonrandom"
            or self.strategy == "nonrandom_weighted"
        ):
            if self.genotype_data.tree is None:
                raise TypeError(
                    "SimGenotypeData.tree cannot be NoneType when "
                    "strategy='nonrandom' or 'nonrandom_weighted'"
                )

            mask = np.full_like(X, 0.0, dtype=bool)

            if self.strategy == "nonrandom_weighted":
                weighted = True
            else:
                weighted = False

            sample_map = dict()
            for i, sample in enumerate(self.genotype_data.samples):
                sample_map[sample] = i

            # if no tolerance provided, set to 1 snp position
            if self.tol is None:
                self.tol = 1.0 / mask.size

            # if no max_tries provided, set to # inds
            if self.max_tries is None:
                self.max_tries = mask.shape[0]

            filled = False
            while not filled:
                # Get list of samples from tree
                samples = self._sample_tree(
                    internal_only=False, skip_root=True, weighted=weighted
                )

                # Convert to row indices
                rows = [sample_map[i] for i in samples]

                # Randomly sample a column
                col_idx = np.random.randint(0, mask.shape[1])
                sampled_col = copy.copy(mask[:, col_idx])
                miss_mask = copy.copy(self.original_missing_mask_[:, col_idx])

                # Mask column
                sampled_col[rows] = True

                # If original was missing, set back to False.
                if self.mask_missing:
                    sampled_col[miss_mask] = False

                # check that column is not 100% missing now
                # if yes, sample again
                if np.sum(sampled_col) == sampled_col.size:
                    continue

                # if not, set values in mask matrix
                else:
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
                            r = np.random.randint(0, mask.shape[0])
                            c = np.random.randint(0, mask.shape[1])
                            mask[r, c] = False
                            tries += 1
                            current_prop = np.sum(mask) / mask.size

                        filled = True
                    else:
                        continue

            # With strategy=="nonrandom" or "nonrandom_weighted",
            # mask_ is equal to sim_missing_mask_ if mask_missing is True.
            # Otherwise it is equal to all_missing_.
            self.mask_ = mask

            self._validate_mask()

        else:
            raise ValueError(
                "Invalid SimGenotypeData.strategy value:", self.strategy
            )

        # Get all missing values.
        self.all_missing_mask_ = np.logical_or(
            self.mask_, self.original_missing_mask_
        )
        # Get values where original value was not missing and simulated.
        # data is missing.
        self.sim_missing_mask_ = np.logical_and(
            self.all_missing_mask_, self.original_missing_mask_ == False
        )

        self._validate_mask(mask=self.mask_missing)

        return self

    def transform(self, X):
        """Function to generate masked sites in a SimGenotypeData object

        Args:
            X (pandas.DataFrame, numpy.ndarray, or List[List[int]]): Data to transform. No missing data should be present in X. It should have already been imputed with one of the non-machine learning simple imputers.

        Returns:
            numpy.ndarray: Transformed data with missing data added.
        """
        X = misc.validate_input_type(X, return_type="array")

        # mask 012-encoded and one-hot encoded genotypes.
        return self._mask_snps(X)

    def accuracy(self, X_true, X_pred):
        """Calculate imputation accuracy of the simulated genotypes.

        Args:
            X_true (np.ndarray): True values.

            X_pred (np.ndarray): Imputed values.

        Returns:
            float: Accuracy score between X_true and X_pred.
        '"""
        masked_sites = np.sum(self.sim_missing_mask_)
        num_correct = np.sum(
            X_true[self.sim_missing_mask_] == X_pred[self.sim_missing_mask_]
        )
        return num_correct / masked_sites

    def auc_roc_pr_ap(self, X_true, X_pred):
        """Calcuate AUC-ROC, Precision-Recall, and Average Precision (AP).

        Args:
            X_true (np.ndarray): True values.

            X_pred (np.ndarray): Imputed values.

        Returns:
            List[float]: List of AUC-ROC scores in order of: 0,1,2.
            List[float]: List of precision scores in order of: 0,1,2.
            List[float]: List of recall scores in order of: 0,1,2.
            List[float]: List of average precision scores in order of 0,1,2.

        """
        y_true = X_true[self.sim_missing_mask_]
        y_pred = X_pred[self.sim_missing_mask_]

        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        y_pred_bin = label_binarize(y_pred, classes=[0, 1, 2])

        # Initialize lists to hold the scores for each class
        auc_roc_scores = []
        precision_scores = []
        recall_scores = []
        avg_precision_scores = []

        for i in range(y_true_bin.shape[1]):
            # AUC-ROC score
            auc_roc = roc_auc_score(
                y_true_bin[:, i], y_pred_bin[:, i], average="weighted"
            )
            auc_roc_scores.append(auc_roc)

            # Precision-recall score
            precision, recall, _, _ = precision_recall_fscore_support(
                y_true_bin[:, i], y_pred_bin[:, i], average="weighted"
            )
            precision_scores.append(precision)
            recall_scores.append(recall)

            # Average precision score
            avg_precision = average_precision_score(
                y_true_bin[:, i], y_pred_bin[:, i], average="weighted"
            )
            avg_precision_scores.append(avg_precision)

        return (
            auc_roc_scores,
            precision_scores,
            recall_scores,
            avg_precision_scores,
        )

    def random_weighted_missing_data(self, X, inv=False):
        """Choose values for which to simulate missing data by biasing towards the minority or majority alleles, depending on whether inv is True or False.

        Args:
            X (np.ndarray): True values.

            inv (bool, optional): If True, then biases towards choosing majority alleles. If False, then biases towards choosing minority alleles. Defaults to False.

        Returns:
            np.ndarray: X with simulated missing values.

        """
        # Get unique classes and their counts
        classes, counts = np.unique(X, return_counts=True)
        # Compute class weights
        if inv:
            class_weights = 1 / counts
        else:
            class_weights = counts
        # Normalize class weights
        class_weights = class_weights / sum(class_weights)

        # Compute mask
        if self.mask_missing:
            # Get indexes where non-missing (Xobs) and missing (Xmiss)
            Xobs = np.where(~self.original_missing_mask_.ravel())[0]
            Xmiss = np.where(self.original_missing_mask_.ravel())[0]

            # Generate mask of 0's (non-missing) and 1's (missing)
            obs_mask = np.random.choice(
                classes, size=Xobs.size, p=class_weights
            )
            obs_mask = (obs_mask == classes[:, None]).argmax(axis=0)

            # Make missing data mask
            mask = np.zeros(X.size, dtype=bool)
            mask[Xobs] = obs_mask
            mask[Xmiss] = 1

            # Reshape from raveled to 2D
            mask = mask.reshape(X.shape)
        else:
            # Generate mask of 0's (non-missing) and 1's (missing)
            mask = np.random.choice(classes, size=X.size, p=class_weights)
            mask = (mask == classes[:, None]).argmax(axis=0).reshape(X.shape)

        # Assign mask to self before validation
        self.mask_ = mask

        self._validate_mask()

        return mask

    def _sample_tree(
        self,
        internal_only=False,
        tips_only=False,
        skip_root=True,
        weighted=False,
    ):
        """Function for randomly sampling clades from SimGenotypeData.tree.

        Args:
            internal_only (bool): Only sample from NON-TIPS. Defaults to False.

            tips_only (bool): Only sample from tips. Defaults to False.

            skip_root (bool): Exclude sampling of root node. Defaults to True.

            weighted (bool): Weight sampling by branch length. Defaults to False.

        Returns:
            List[str]: List of descendant tips from the sampled node.

        Raises:
            ValueError: ``tips_only`` and ``internal_only`` cannot both be True.
        """

        if tips_only and internal_only:
            raise ValueError("internal_only and tips_only cannot both be true")

        # to only sample internal nodes add  if not i.is_leaf()
        node_dict = dict()

        for node in self.genotype_data.tree.treenode.traverse("preorder"):
            ## node.idx is node indexes.
            ## node.dist is branch lengths.
            if skip_root:
                # If root node.
                if node.idx == self.genotype_data.tree.nnodes - 1:
                    continue

            if tips_only and internal_only:
                raise ValueError(
                    "tips_only and internal_only cannot both be True"
                )

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
            node_idx = np.random.choice(list(node_dict.keys()), size=1, p=p)[0]
        else:
            # Get missing choice from random clade.
            node_idx = np.random.choice(list(node_dict.keys()), size=1)[0]
        return self.genotype_data.tree.get_tip_labels(idx=node_idx)

    def _validate_mask(self, mask=False):
        """Make sure no entirely missing columns are simulated."""
        if mask is None:
            mask = self.mask_
        for i, column in enumerate(self.mask_.T):
            if mask:
                miss_mask = self.original_missing_mask_[:, i]
                col = column[~miss_mask]
                obs_idx = np.where(~miss_mask)
                idx = obs_idx[np.random.choice(np.arange(len(obs_idx)))]
            else:
                col = column
                idx = np.random.choice(np.arange(col.shape[0]))
            if np.sum(col) == col.size:
                self.mask_[idx, i] = False

    def _mask_snps(self, X):
        """Mask positions in SimGenotypeData.snps and SimGenotypeData.onehot"""
        if len(X.shape) == 3:
            # One-hot encoded.
            mask_val = [0.0, 0.0, 0.0, 0.0]
        elif len(X.shape) == 2:
            # 012-encoded.
            mask_val = -9
        else:
            raise ValueError(f"Invalid shape of input X: {X.shape}")

        Xt = X.copy()
        mask_boolean = self.mask_ != 0
        Xt[mask_boolean] = mask_val
        return Xt

    @property
    def missing_count(self) -> int:
        """Count of masked genotypes in SimGenotypeData.mask

        Returns:
            int: Integer count of masked alleles.
        """
        return np.sum(self.mask_)

    @property
    def prop_missing_real(self) -> float:
        """Proportion of genotypes masked in SimGenotypeData.mask

        Returns:
            float: Total number of masked alleles divided by SNP matrix size.
        """
        return np.sum(self.mask_) / self.mask_.size
