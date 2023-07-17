import copy
import os
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Any

import numpy as np
import pandas as pd

# Third-party imports
import numpy as np
import pandas as pd
import scipy.linalg
import toyplot.pdf
import toytree as tt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

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
    from snpio import GenotypeData
    from ..utils import misc
    from ..utils.misc import get_processor_name
    from ..utils.misc import isnotebook
except (ModuleNotFoundError, ValueError, ImportError):
    from snpio import GenotypeData
    from pgsui.utils import misc
    from pgsui.utils.misc import get_processor_name
    from pgsui.utils.misc import isnotebook

is_notebook = isnotebook()

if is_notebook:
    from tqdm.notebook import tqdm as progressbar
else:
    from tqdm import tqdm as progressbar

# Requires scikit-learn-intellex package
if get_processor_name().strip().startswith("Intel"):
    try:
        from sklearnex import patch_sklearn

        patch_sklearn()
        intelex = True
    except (ImportError, TypeError):
        print(
            "Warning: Intel CPU detected but scikit-learn-intelex is not installed. We recommend installing it to speed up computation."
        )
        intelex = False
else:
    intelex = False

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
    """Transformer to format autoencoder features before model fitting.

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

    def fit(self, X):
        """set attributes used to transform X (input features).

        Args:
            X (numpy.ndarray): Input integer-encoded numpy array.
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
                f"Invalid value passed to num_classes in AutoEncoderFeatureTransformer. Only 3 or 4 are supported, but got {self.num_classes}."
            )

        # Encode the data.
        self.X_train = enc_func(X)

        # Get missing and observed data boolean masks.
        self.missing_mask_, self.observed_mask_ = self._get_masks(self.X_train)

        # To accomodate multiclass-multioutput.
        self.n_outputs_expected_ = 1

        return self

    def transform(self, X):
        """Transform X to one-hot encoded format.

        Accomodates multiclass targets with a 3D shape.

        Args:
            y (numpy.ndarray): One-hot encoded target data of shape (n_samples, n_features, num_classes).

        Returns:
            numpy.ndarray: Transformed target data in one-hot format of shape (n_samples, n_features, num_classes).
        """
        if self.return_int:
            return X
        else:
            X = misc.validate_input_type(X, return_type="array")
            return self._fill(self.X_train, self.missing_mask_)

    def inverse_transform(self, y):
        """Transform target to output format.

        Args:
            y (numpy.ndarray): Array to inverse transform.
        """
        try:
            if self.activate is None:
                return y.numpy()
            elif self.activate == "softmax":
                return tf.nn.softmax(y).numpy()
            elif self.activate == "sigmoid":
                return tf.nn.sigmoid(y).numpy()
            else:
                raise ValueError(
                    f"Invalid value passed to keyword argument activate. Valid options include: None, 'softmax', or 'sigmoid', but got {self.activate}"
                )
        except AttributeError:
            # If numpy array already.
            if self.activate is None:
                return y
            elif self.activate == "softmax":
                return tf.nn.softmax(tf.convert_to_tensor(y)).numpy()
            elif self.activate == "sigmoid":
                return tf.nn.sigmoid(tf.convert_to_tensor(y)).numpy()
            else:
                raise ValueError(
                    f"Invalid value passed to keyword argument activate. Valid options include: None, 'softmax', or 'sigmoid', but got {self.activate}"
                )

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
    """Transformer to format target data both before and after model fitting.

    Args:
        y_decoded (numpy.ndarray): Original input data that is 012-encoded.
        model (tf.keras.model.Model): model to use for predicting y.
    """

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
        """Mask missing data as ``missing_value``.

        Args:
            data (numpy.ndarray): Input with missing values of shape (n_samples, n_features, num_classes).

            missing_mask (np.ndarray(bool)): Missing data mask with True corresponding to a missing value.

            missing_value (int): Value to set missing data to. If a list is provided, then its length should equal the number of one-hot classes.
        """
        if num_classes > 1:
            missing_value = [missing_value] * num_classes
        data[missing_mask] = missing_value
        return data

    def _get_masks(self, X):
        """Format the provided target data for use with UBP/NLPCA.

        Args:
            y (numpy.ndarray(float)): Input data that will be used as the target.

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
        """Evaluate VAE predictions by calculating the highest predicted value.

        Calucalates highest predicted value for each row vector and each class, setting the most likely class to 1.0.

        Args:
            y (numpy.ndarray): Input one-hot encoded data.

        Returns:
            numpy.ndarray: Imputed one-hot encoded values.

            pandas.DataFrame: One-hot encoded pandas DataFrame with no missing values.
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
    """Transformer to format UBP target data both before model fitting.

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

            pandas.DataFrame: One-hot encoded pandas DataFrame with no missing values.
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


class ImputePhyloTransformer(GenotypeData, BaseEstimator, TransformerMixin):
    """Impute missing data using a phylogenetic tree to inform the imputation.

    Args:
        alnfile (str or None, optional): Path to PHYLIP or STRUCTURE-formatted file to impute. Defaults to None.

        filetype (str or None, optional): Filetype for the input alignment. Valid options include: "phylip", "structure1row", "structure1rowPopID", "structure2row", "structure2rowPopId". Not required if ``genotype_data`` is defined. Defaults to "phylip".

        popmapfile (str or None, optional): Path to population map file. Required if filetype is "phylip", "structure1row", or "structure2row". If filetype is "structure1rowPopID" or "structure2rowPopID", then the population IDs must be the second column of the STRUCTURE file. Not required if ``genotype_data`` is defined. Defaults to None.

        treefile (str or None, optional): Path to Newick-formatted phylogenetic tree file. Not required if ``genotype_data`` is defined with the ``guidetree`` option. Defaults to None.

        siterates (str or None, optional): Path to file containing per-site rates, with 1 rate per line corresponding to 1 site. Not required if ``genotype_data`` is defined with the siterates or siterates_iqtree option. Defaults to None.

        siterates_iqtree (str or None, optional): Path to *.rates file output from IQ-TREE, containing a per-site rate table. If specified, ``ImputePhylo`` will read the site-rates from the IQ-TREE output file. Cannot be used in conjunction with ``siterates`` argument. Not required if the ``siterates`` or ``siterates_iqtree`` options were used with the ``GenotypeData`` object. Defaults to None.

        qmatrix (str or None, optional): Path to file containing only a Rate Matrix Q table. Not required if ``genotype_data`` is defined with the qmatrix or qmatrix_iqtree option. Defaults to None.

        str_encodings (Dict[str, int], optional): Integer encodings used in STRUCTURE-formatted file. Should be a dictionary with keys=nucleotides and values=integer encodings. The missing data encoding should also be included. Argument is ignored if using a PHYLIP-formatted file. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}

        prefix (str, optional): Prefix to use with output files.

        output_format (str, optional): Format of transformed imputed matrix. Possible values include: "df" for a pandas.DataFrame object, "array" for a numpy.ndarray object, and "list" for a 2D list. Defaults to "df".

        save_plots (bool, optional): Whether to save PDF files with genotype imputations for each site to disk. It makes one PDF file per locus, so if you have a lot of loci it will make a lot of PDF files. Defaults to False.

        write_output (bool, optional): Whether to save the imputed data to disk. Defaults to True.

        disable_progressbar (bool, optional): Whether to disable the progress bar during the imputation. Defaults to False.

        kwargs (Dict[str, Any] or None, optional): Additional keyword arguments intended for internal purposes only. Possible arguments: {"column_subset": List[int] or numpy.ndarray[int]}; Subset SNPs by a list of indices. Defauls to None.

    Attributes:
        imputed_ (pandas.DataFrame): Imputed data.

    Example:
        >>>data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID",
        >>>    guidetree="test.tre",
        >>>    qmatrix_iqtree="test.iqtree"
        >>>)
        >>>
        >>>phylo = ImputePhyloTransformer()
        >>>phylo_gtdata = phylo.fit_transform(data)
    """

    def __init__(
        self,
        *,
        alnfile: Optional[str] = None,
        filetype: Optional[str] = None,
        popmapfile: Optional[str] = None,
        treefile: Optional[str] = None,
        qmatrix_iqtree: Optional[str] = None,
        qmatrix: Optional[str] = None,
        siterates: Optional[str] = None,
        siterates_iqtree: Optional[str] = None,
        str_encodings: Dict[str, int] = {
            "A": 1,
            "C": 2,
            "G": 3,
            "T": 4,
            "N": -9,
        },
        prefix: str = "output",
        output_format: str = "df",
        save_plots: bool = False,
        disable_progressbar: bool = False,
        **kwargs: Optional[Any],
    ) -> None:
        GenotypeData.__init__(self)

        self.alnfile = alnfile
        self.filetype = filetype
        self.popmapfile = popmapfile
        self.treefile = treefile
        self.qmatrix_iqtree = qmatrix_iqtree
        self.qmatrix = qmatrix
        self.siterates = siterates
        self.siterates_iqtree = siterates_iqtree
        self.str_encodings = str_encodings
        self.prefix = prefix
        self.output_format = output_format
        self.save_plots = save_plots
        self.disable_progressbar = disable_progressbar
        self.kwargs = kwargs

        self.valid_sites = None
        self.valid_sites_count = None

    def fit(self, X):
        """Fit to input data.

        Args:
            X (GenotypeData): Instantiated GenotypeData object with data to impute of shape ``(n_samples, n_features)``\.
        """
        self._validate_arguments(X)

        self.column_subset_ = self.kwargs.get("column_subset", None)

        (
            data,
            tree,
            q,
            site_rates,
        ) = self._parse_arguments(X)

        self.imputed_ = self.impute_phylo(tree, data, q, site_rates)

        return self

    def transform(self, X):
        """Transform imputed data and return desired output format.

        X (GenotypeData): Initialized GenotypeData object with tree, q, and (optionally) site_rates data.

        Returns:
            pandas.DataFrame, numpy.ndarray, or List[List[int]]: Imputed 012-encoded data.
        """
        if self.output_format == "df":
            return self.imputed_
        elif self.output_format == "array":
            return self.imputed_.to_numpy()
        elif self.output_format == "list":
            return self.imputed_.values.tolist()
        else:
            raise ValueError(
                f"Unsupported output_format provided. Valid options include "
                f"'df', 'array', or 'list', but got {self.output_format}"
            )

    def impute_phylo(
        self,
        tree: tt.tree,
        genotypes: Dict[str, List[Union[str, int]]],
        Q: pd.DataFrame,
        site_rates=None,
    ) -> pd.DataFrame:
        """Imputes genotype values with a guide tree.

        Imputes genotype values by using a provided guide
        tree to inform the imputation, assuming maximum parsimony.

        Process Outline:
            For each SNP:
            1) if site_rates, get site-transformated Q matrix.

            2) Postorder traversal of tree to compute ancestral
            state likelihoods for internal nodes (tips -> root).
            If exclude_N==True, then ignore N tips for this step.

            3) Preorder traversal of tree to populate missing genotypes
            with the maximum likelihood state (root -> tips).

        Args:
            tree (toytree.tree object): Input tree.

            genotypes (Dict[str, List[Union[str, int]]]): Dictionary with key=sampleids, value=sequences.

            Q (pandas.DataFrame): Rate Matrix Q from .iqtree or separate file.

            site_rates (List): Site-specific substitution rates (used to weight per-site Q)

        Returns:
            pandas.DataFrame: Imputed genotypes.

        Raises:
            IndexError: If index does not exist when trying to read genotypes.
            AssertionError: Sites must have same lengths.
            AssertionError: Missing data still found after imputation.
        """
        try:
            if list(genotypes.values())[0][0][1] == "/":
                genotypes = self._str2iupac(genotypes, self.str_encodings)
        except IndexError:
            if self._is_int(list(genotypes.values())[0][0][0]):
                raise

        if self.column_subset_ is not None:
            if isinstance(self.column_subset_, np.ndarray):
                self.column_subset_ = self.column_subset_.tolist()

            genotypes = {
                k: [v[i] for i in self.column_subset_]
                for k, v in genotypes.items()
            }

        # For each SNP:
        nsites = list(set([len(v) for v in genotypes.values()]))
        assert len(nsites) == 1, "Some sites have different lengths!"

        outdir = f"{self.prefix}_imputation_plots"

        if self.save_plots:
            Path(outdir).mkdir(parents=True, exist_ok=True)

        for snp_index in progressbar(
            range(nsites[0]),
            desc="Feature Progress: ",
            leave=True,
            disable=self.disable_progressbar,
        ):
            node_lik = dict()

            # LATER: Need to get site rates
            rate = 1.0
            if site_rates is not None:
                rate = site_rates[snp_index]

            site_Q = Q.copy(deep=True) * rate
            # print(site_Q)

            # calculate state likelihoods for internal nodes
            for node in tree.treenode.traverse("postorder"):
                if node.is_leaf():
                    continue

                if node.idx not in node_lik:
                    node_lik[node.idx] = None

                for child in node.get_leaves():
                    # get branch length to child
                    # bl = child.edge.length
                    # get transition probs
                    pt = self._transition_probs(site_Q, child.dist)
                    if child.is_leaf():
                        if child.name in genotypes:
                            # get genotype
                            sum = None

                            for allele in self._get_iupac_full(
                                genotypes[child.name][snp_index]
                            ):
                                if sum is None:
                                    sum = list(pt[allele])
                                else:
                                    sum = [
                                        sum[i] + val
                                        for i, val in enumerate(
                                            list(pt[allele])
                                        )
                                    ]

                            if node_lik[node.idx] is None:
                                node_lik[node.idx] = sum

                            else:
                                node_lik[node.idx] = [
                                    sum[i] * val
                                    for i, val in enumerate(node_lik[node.idx])
                                ]
                        else:
                            # raise error
                            sys.exit(
                                f"Error: Taxon {child.name} not found in "
                                f"genotypes"
                            )

                    else:
                        l = self._get_internal_lik(pt, node_lik[child.idx])
                        if node_lik[node.idx] is None:
                            node_lik[node.idx] = l

                        else:
                            node_lik[node.idx] = [
                                l[i] * val
                                for i, val in enumerate(node_lik[node.idx])
                            ]

            # infer most likely states for tips with missing data
            # for each child node:
            bads = list()
            for samp in genotypes.keys():
                if genotypes[samp][snp_index].upper() == "N":
                    bads.append(samp)
                    # go backwards into tree until a node informed by
                    # actual data
                    # is found
                    # node = tree.search_nodes(name=samp)[0]
                    node = tree.idx_dict[
                        tree.get_mrca_idx_from_tip_labels(names=samp)
                    ]
                    dist = node.dist
                    node = node.up
                    imputed = None

                    while node and imputed is None:
                        if self._all_missing(
                            tree, node.idx, snp_index, genotypes
                        ):
                            dist += node.dist
                            node = node.up

                        else:
                            pt = self._transition_probs(site_Q, dist)
                            lik = self._get_internal_lik(
                                pt, node_lik[node.idx]
                            )
                            maxpos = lik.index(max(lik))
                            if maxpos == 0:
                                imputed = "A"

                            elif maxpos == 1:
                                imputed = "C"

                            elif maxpos == 2:
                                imputed = "G"

                            else:
                                imputed = "T"

                    genotypes[samp][snp_index] = imputed

            if self.save_plots:
                self._draw_imputed_position(
                    tree,
                    bads,
                    genotypes,
                    snp_index,
                    f"{outdir}/{self.prefix}_pos{snp_index}.pdf",
                )

        df = pd.DataFrame.from_dict(genotypes, orient="index")

        # Make sure no missing data remains in the dataset
        assert (
            not df.isin([-9]).any().any()
        ), "Imputation failed...Missing values found in the imputed dataset"

        imp_snps, self.valid_sites, self.valid_sites_count = self.convert_012(
            df.to_numpy().tolist(), impute_mode=True
        )

        df_imp = pd.DataFrame.from_records(imp_snps)

        return df_imp

    def _parse_arguments(
        self, genotype_data: Any
    ) -> Tuple[Dict[str, List[Union[int, str]]], tt.tree, pd.DataFrame]:
        """Determine which arguments were specified and set appropriate values.

        Args:
            genotype_data (GenotypeData): Initialized GenotypeData object.

        Returns:
            Dict[str, List[Union[int, str]]]: GenotypeData.snpsdict object. If genotype_data is not None, then this value gets set from the GenotypeData.snpsdict object. If alnfile is not None, then the alignment file gets read and the snpsdict object gets set from the alnfile.

            toytree.tree: Input phylogeny, either read from GenotypeData object or supplied with treefile.

            pandas.DataFrame: Q Rate Matrix, either from IQ-TREE file or from its own supplied file.
        """
        if genotype_data is not None:
            data = genotype_data.snpsdict
            filetype = genotype_data.filetype

        elif self.alnfile is not None:
            self.parse_filetype(filetype, genotype_data.popmapfile)

        if genotype_data.tree is not None and self.treefile is None:
            tree = genotype_data.tree

        elif genotype_data.tree is not None and self.treefile is not None:
            print(
                "WARNING: Both genotype_data.tree and treefile are defined; using local definition"
            )
            tree = self.read_tree(self.treefile)

        elif genotype_data.tree is None and self.treefile is not None:
            tree = self.read_tree(self.treefile)

        # read (optional) Q-matrix
        if (
            genotype_data.q is not None
            and self.qmatrix is None
            and self.qmatrix_iqtree is None
        ):
            q = genotype_data.q

        elif genotype_data.q is None:
            if self.qmatrix is not None:
                q = self.q_from_file(self.qmatrix)
            elif self.qmatrix_iqtree is not None:
                q = self.q_from_iqtree(self.qmatrix_iqtree)

        elif genotype_data.q is not None:
            if self.qmatrix is not None:
                print(
                    "WARNING: Both genotype_data.q and qmatrix are defined; "
                    "using local definition"
                )
                q = self.q_from_file(self.qmatrix)
            if self.qmatrix_iqtree is not None:
                print(
                    "WARNING: Both genotype_data.q and qmatrix are defined; "
                    "using local definition"
                )
                q = self.q_from_iqtree(self.qmatrix_iqtree)

        # read (optional) site-specific substitution rates
        site_rates = None
        if (
            genotype_data.site_rates is not None
            and self.siterates is None
            and self.siterates_iqtree is None
        ):
            site_rates = genotype_data.site_rates
        elif genotype_data.site_rates is None:
            if self.siterates is not None:
                site_rates = self.siterates_from_file(self.siterates)
            elif self.siterates_iqtree is not None:
                site_rates = self.siterates_from_iqtree(self.siterates_iqtree)

        elif genotype_data.site_rates is not None:
            if self.siterates is not None:
                print(
                    "WARNING: Both genotype_data.site_rates and siterates are defined; "
                    "using local definition"
                )
                site_rates = self.siterates_from_file(self.siterates)
            if self.siterates_iqtree is not None:
                print(
                    "WARNING: Both genotype_data.site_rates and siterates are defined; "
                    "using local definition"
                )
                site_rates = self.siterates_from_iqtree(self.siterates_iqtree)
        return (data, tree, q, site_rates)

    def _validate_arguments(self, genotype_data: Any) -> None:
        """Validate that the correct arguments were supplied.

        Args:
            genotype_data (GenotypeData object): Input GenotypeData object.

        Raises:
            TypeError: Cannot define both genotype_data and alnfile.
            TypeError: Must define either genotype_data or phylipfile.
            TypeError: Must define either genotype_data.tree or treefile.
            TypeError: filetype must be defined if genotype_data is None.
            TypeError: Q rate matrix must be defined.
            TypeError: qmatrix and qmatrix_iqtree cannot both be defined.
        """
        if genotype_data is not None and self.alnfile is not None:
            raise TypeError("genotype_data and alnfile cannot both be defined")

        if genotype_data is None and self.alnfile is None:
            raise TypeError(
                "Either genotype_data or phylipfle must be defined"
            )

        if genotype_data.tree is None and self.treefile is None:
            raise TypeError(
                "Either genotype_data.tree or treefile must be defined"
            )

        if genotype_data is None and self.filetype_ is None:
            raise TypeError(
                "filetype must be defined if genotype_data is None"
            )

        if (
            genotype_data is None
            and self.qmatrix is None
            and self.qmatrix_iqtree is None
        ):
            raise TypeError(
                "q matrix must be defined in either genotype_data, "
                "qmatrix_iqtree, or qmatrix"
            )

        if self.qmatrix is not None and self.qmatrix_iqtree is not None:
            raise TypeError(
                "qmatrix and qmatrix_iqtree cannot both be defined"
            )

    def _print_q(self, q: pd.DataFrame) -> None:
        """Print Rate Matrix Q.

        Args:
            q (pandas.DataFrame): Rate Matrix Q.
        """
        print("Rate matrix Q:")
        print("\tA\tC\tG\tT\t")
        for nuc1 in ["A", "C", "G", "T"]:
            print(nuc1, end="\t")
            for nuc2 in ["A", "C", "G", "T"]:
                print(q[nuc1][nuc2], end="\t")
            print("")

    def _is_int(self, val: Union[str, int]) -> bool:
        """Check if value is integer.

        Args:
            val (int or str): Value to check.

        Returns:
            bool: True if integer, False if string.
        """
        try:
            num = int(val)
        except ValueError:
            return False
        return True

    def _get_nuc_colors(self, nucs: List[str]) -> List[str]:
        """Get colors for each nucleotide when plotting.

        Args:
            nucs (List[str]): Nucleotides at current site.

        Returns:
            List[str]: Hex-code color values for each IUPAC nucleotide.
        """
        ret = list()
        for nuc in nucs:
            nuc = nuc.upper()
            if nuc == "A":
                ret.append("#0000FF")  # blue
            elif nuc == "C":
                ret.append("#FF0000")  # red
            elif nuc == "G":
                ret.append("#00FF00")  # green
            elif nuc == "T":
                ret.append("#FFFF00")  # yellow
            elif nuc == "R":
                ret.append("#0dbaa9")  # blue-green
            elif nuc == "Y":
                ret.append("#FFA500")  # orange
            elif nuc == "K":
                ret.append("#9acd32")  # yellow-green
            elif nuc == "M":
                ret.append("#800080")  # purple
            elif nuc == "S":
                ret.append("#964B00")
            elif nuc == "W":
                ret.append("#C0C0C0")
            else:
                ret.append("#000000")
        return ret

    def _label_bads(
        self, tips: List[str], labels: List[str], bads: List[str]
    ) -> List[str]:
        """Insert asterisks around bad nucleotides.

        Args:
            tips (List[str]): Tip labels (sample IDs).
            labels (List[str]): List of nucleotides at current site.
            bads (List[str]): List of tips that have missing data at current site.

        Returns:
            List[str]: IUPAC Nucleotides with "*" inserted around tips that had missing data.
        """
        for i, t in enumerate(tips):
            if t in bads:
                labels[i] = "*" + str(labels[i]) + "*"
        return labels

    def _draw_imputed_position(
        self,
        tree: tt.tree,
        bads: List[str],
        genotypes: Dict[str, List[str]],
        pos: int,
        out: str = "tree.pdf",
    ) -> None:
        """Draw nucleotides at phylogeny tip and saves to file on disk.

        Draws nucleotides as tip labels for the current SNP site. Imputed values have asterisk surrounding the nucleotide label. The tree is converted to a toyplot object and saved to file.

        Args:
            tree (toytree.tree): Input tree object.
            bads (List[str]): List of sampleIDs that have missing data at the current SNP site.
            genotypes (Dict[str, List[str]]): Genotypes at all SNP sites.
            pos (int): Current SNP index.
            out (str, optional): Output filename for toyplot object.
        """

        # print(tree.get_tip_labels())
        colors = [genotypes[i][pos] for i in tree.get_tip_labels()]
        labels = colors

        labels = self._label_bads(tree.get_tip_labels(), labels, bads)

        colors = self._get_nuc_colors(colors)

        mystyle = {
            "edge_type": "p",
            "edge_style": {
                "stroke": tt.colors[0],
                "stroke-width": 1,
            },
            "tip_labels_align": True,
            "tip_labels_style": {"font-size": "5px"},
            "node_labels": False,
        }

        canvas, _, __ = tree.draw(
            tip_labels_colors=colors,
            tip_labels=labels,
            width=400,
            height=600,
            **mystyle,
        )

        toyplot.pdf.render(canvas, out)

    def _all_missing(
        self,
        tree: tt.tree,
        node_index: int,
        snp_index: int,
        genotypes: Dict[str, List[str]],
    ) -> bool:
        """Check if all descendants of a clade have missing data at SNP site.

        Args:
            tree (toytree.tree): Input guide tree object.

            node_index (int): Parent node to determine if all descendants have missing data.

            snp_index (int): Index of current SNP site.

            genotypes (Dict[str, List[str]]): Genotypes at all SNP sites.

        Returns:
            bool: True if all descendants have missing data, otherwise False.
        """
        for des in tree.get_tip_labels(idx=node_index):
            if genotypes[des][snp_index].upper() not in ["N", "-"]:
                return False
        return True

    def _get_internal_lik(
        self, pt: pd.DataFrame, lik_arr: List[float]
    ) -> List[float]:
        """Get ancestral state likelihoods for internal nodes of the tree.

        Postorder traversal to calculate internal ancestral state likelihoods (tips -> root).

        Args:
            pt (pandas.DataFrame): Transition probabilities calculated from Rate Matrix Q.
            lik_arr (List[float]): Likelihoods for nodes or leaves.

        Returns:
            List[float]: Internal likelihoods.
        """
        ret = list()
        for i, val in enumerate(lik_arr):
            col = list(pt.iloc[:, i])
            sum = 0.0
            for v in col:
                sum += v * val
            ret.append(sum)
        return ret

    def _transition_probs(self, Q: pd.DataFrame, t: float) -> pd.DataFrame:
        """Get transition probabilities for tree.

        Args:
            Q (pd.DataFrame): Rate Matrix Q.
            t (float): Tree distance of child.

        Returns:
            pd.DataFrame: Transition probabilities.
        """
        ret = Q.copy(deep=True)
        m = Q.to_numpy()
        pt = scipy.linalg.expm(m * t)
        ret[:] = pt
        return ret

    def _str2iupac(
        self, genotypes: Dict[str, List[str]], str_encodings: Dict[str, int]
    ) -> Dict[str, List[str]]:
        """Convert STRUCTURE-format encodings to IUPAC bases.

        Args:
            genotypes (Dict[str, List[str]]): Genotypes at all sites.
            str_encodings (Dict[str, int]): Dictionary that maps IUPAC bases (keys) to integer encodings (values).

        Returns:
            Dict[str, List[str]]: Genotypes converted to IUPAC format.
        """
        a = str_encodings["A"]
        c = str_encodings["C"]
        g = str_encodings["G"]
        t = str_encodings["T"]
        n = str_encodings["N"]
        nuc = {
            f"{a}/{a}": "A",
            f"{c}/{c}": "C",
            f"{g}/{g}": "G",
            f"{t}/{t}": "T",
            f"{n}/{n}": "N",
            f"{a}/{c}": "M",
            f"{c}/{a}": "M",
            f"{a}/{g}": "R",
            f"{g}/{a}": "R",
            f"{a}/{t}": "W",
            f"{t}/{a}": "W",
            f"{c}/{g}": "S",
            f"{g}/{c}": "S",
            f"{c}/{t}": "Y",
            f"{t}/{c}": "Y",
            f"{g}/{t}": "K",
            f"{t}/{g}": "K",
        }

        for k, v in genotypes.items():
            for i, gt in enumerate(v):
                v[i] = nuc[gt]

        return genotypes

    def _get_iupac_full(self, char: str) -> List[str]:
        """Map nucleotide to list of expanded IUPAC encodings.

        Args:
            char (str): Current nucleotide.

        Returns:
            List[str]: List of nucleotides in ``char`` expanded IUPAC.
        """
        char = char.upper()
        iupac = {
            "A": ["A"],
            "G": ["G"],
            "C": ["C"],
            "T": ["T"],
            "N": ["A", "C", "T", "G"],
            "-": ["A", "C", "T", "G"],
            "R": ["A", "G"],
            "Y": ["C", "T"],
            "S": ["G", "C"],
            "W": ["A", "T"],
            "K": ["G", "T"],
            "M": ["A", "C"],
            "B": ["C", "G", "T"],
            "D": ["A", "G", "T"],
            "H": ["A", "C", "T"],
            "V": ["A", "C", "G"],
        }

        ret = iupac[char]
        return ret


class ImputeAlleleFreqTransformer(
    GenotypeData, BaseEstimator, TransformerMixin
):
    """Impute missing data by global or by-population allele frequency. Population IDs can be sepcified with the pops argument. if pops is None, then imputation is by global allele frequency. If pops is not None, then imputation is by population-wise allele frequency. A list of population IDs in the appropriate format can be obtained from the GenotypeData object as GenotypeData.populations.

    Args:
        by_populations (bool, optional): Whether or not to impute by population or globally. Defaults to False (global allele frequency).

        diploid (bool, optional): When diploid=True, function assumes 0=homozygous ref; 1=heterozygous; 2=homozygous alt. 0-1-2 genotypes are decomposed to compute p (=frequency of ref) and q (=frequency of alt). In this case, p and q alleles are sampled to generate either 0 (hom-p), 1 (het), or 2 (hom-q) genotypes. When diploid=FALSE, 0-1-2 are sampled according to their observed frequency. Defaults to True.

        default (int, optional): Value to set if no alleles sampled at a locus. Defaults to 0.

        missing (int, optional): Missing data value. Defaults to -9.

        prefix (str, optional): Prefix for writing output files. Defaults to "output".

        output_format (str, optional): Format of transformed imputed matrix. Possible values include: "df" for a pandas.DataFrame object, "array" for a numpy.ndarray object, and "list" for a 2D list. Defaults to "df".

        verbose (bool, optional): Whether to print status updates. Set to False for no status updates. Defaults to True.

    Attributes:
        pops_ (List[Union[str, int]]): List of population IDs from GenotypeData object.

        iterative_mode_ (bool): True if using IterativeImputer, False otherwise. Determines whether output dtype is "float32" or "Int8". Fetched from kwargs.

        imputed_ (GenotypeData): Imputed 012-encoded data.

    Example:
        >>>data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID",
        >>>    guidetree="test.tre",
        >>>    qmatrix_iqtree="test.iqtree"
        >>>)
        >>>
        >>>afpop = ImputeAlleleFreqTransformer(
        >>>     genotype_data=data,
        >>>     by_populations=True,
        >>>)
        >>>
        >>>afpop_gtdata = afpop.fit_transform(data)
    """

    def __init__(
        self,
        *,
        by_populations: bool = False,
        diploid: bool = True,
        default: int = 0,
        missing: int = -9,
        prefix: str = "output",
        output_format: str = "df",
        verbose: bool = True,
        **kwargs: Dict[str, Any],
    ) -> None:
        GenotypeData.__init__(self)

        self.by_populations = by_populations
        self.diploid = diploid
        self.default = default
        self.missing = missing
        self.prefix = prefix
        self.output_format = output_format
        self.verbose = verbose
        self.kwargs = kwargs

    def fit(self, X):
        """Fit imputer to input data X.

        Args:
            X (GenotypeData): Input GenotypeData instance.
        """
        gt_list = X.genotypes012_list
        self.pops_ = X.populations
        self.iterative_mode_ = self.kwargs.get("iterative_mode", False)
        if self.pops_ is None:
            self.imputed_ = self._global_impute(gt_list)
        else:
            self.imputed_ = self._impute(gt_list)
        return self

    def transform(self, X):
        """Transform imputed data and return desired output format.

        Args:
            X (GenotypeData): Instantiated GenotypeData object.

        Returns:
            pandas.DataFrame, numpy.ndarray, or List[List[int]]: Imputed 012-encoded data.
        """
        return self.imputed_

    def _global_impute(
        self, X: List[List[int]]
    ) -> Union[pd.DataFrame, np.ndarray, List[List[Union[int, float]]]]:
        if self.verbose:
            print("\nImputing by global allele frequency...")

        df = misc.validate_input_type(X, return_type="df")
        df.replace(self.missing, np.nan, inplace=True)
        imp = SimpleImputer(strategy="most_frequent")
        imp_arr = imp.fit_transform(df)

        if self.iterative_mode_:
            data = data.astype(dtype="float32")
        else:
            data = data.astype(dtype="Int8")

        if self.verbose:
            print("Done!")

        if self.output_format == "df":
            return data

        elif self.output_format == "array":
            return data.to_numpy()

        elif self.output_format == "list":
            return data.values.tolist()
        else:
            raise ValueError(
                f"Unsupported output_format provided. Valid options include "
                f"'df', 'array', or 'list', but got {self.output_format}"
            )

    def _impute(
        self, X: List[List[int]]
    ) -> Union[pd.DataFrame, np.ndarray, List[List[Union[int, float]]]]:
        """Impute missing genotypes using allele frequencies.

        Impute using global or by_population allele frequencies. Missing alleles are primarily coded as negative; usually -9.

        Args:
            X (List[List[int]], numpy.ndarray, or pandas.DataFrame): 012-encoded genotypes obtained from the GenotypeData object.

        Returns:
            pandas.DataFrame, numpy.ndarray, or List[List[Union[int, float]]]: Imputed genotypes of same shape as data.

            List[int]: Column indexes that were retained.

        Raises:
            TypeError: X must be either 2D list, numpy.ndarray, or pandas.DataFrame.

            ValueError: Unknown output_format type specified.
        """
        if self.pops_ is not None and self.verbose:
            print("\nImputing by population allele frequencies...")
        elif self.pops_ is None and self.verbose:
            print("\nImputing by global allele frequency...")

        if isinstance(X, (list, np.ndarray)):
            df = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise TypeError(
                f"X must be of type list(list(int)), numpy.ndarray, "
                f"or pandas.DataFrame, but got {type(X)}"
            )

        df.replace(self.missing, np.nan, inplace=True)

        data = pd.DataFrame()
        bad_cnt = 0
        if self.pops_ is not None:
            # Impute per-population mode.
            # Loop method is faster (by 2X) than no-loop transform.
            df["pops"] = self.pops_
            groups = df.groupby(["pops"], sort=False)
            mode_func = lambda x: x.fillna(x.mode().iloc[0])

            for col in df.columns:
                try:
                    data[col] = groups[col].transform(mode_func)

                except IndexError as e:
                    if str(e).lower().startswith("single positional indexer"):
                        # One or more populations had all missing values, so
                        # impute with global mode at the offending site.
                        bad_cnt += 1
                        # Impute with global mode
                        data[col] = df[col].fillna(df[col].mode().iloc[0])
                    else:
                        raise

            if bad_cnt > 0:
                print(
                    f"Warning: {bad_cnt} columns were imputed with the global "
                    f"mode because at least one of the populations "
                    f"contained only missing values."
                )

            data.drop("pops", axis=1, inplace=True)

        else:
            # Impute global mode.
            # No-loop method was faster for global.
            data = df.apply(lambda x: x.fillna(x.mode().iloc[0]), axis=1)

        if self.iterative_mode_:
            data = data.astype(dtype="float32")
        else:
            data = data.astype(dtype="Int8")

        if self.verbose:
            print("Done!")

        if self.output_format == "df":
            return data

        elif self.output_format == "array":
            return data.to_numpy()

        elif self.output_format == "list":
            return data.values.tolist()
        else:
            raise ValueError(
                f"Unsupported output_format provided. Valid options include "
                f"'df', 'array', or 'list', but got {self.output_format}"
            )


class ImputeNMFTransformer(BaseEstimator, TransformerMixin):
    """Impute missing data using matrix factorization. Population IDs can be sepcified with the pops argument. if pops is None, then imputation is by global allele frequency. If pops is not None, then imputation is by population-wise allele frequency. A list of population IDs in the appropriate format can be obtained from the GenotypeData object as GenotypeData.populations.

    Args:
        latent_features (float, optional): The number of latent variables used to reduce dimensionality of the data. Defaults to 2.

        learning_rate (float, optional): The learning rate for the optimizers. Adjust if the loss is learning too slowly. Defaults to 0.1.

        tol (float, optional): Tolerance of the stopping condition. Defaults to 1e-3.

        missing (int, optional): Missing data value. Defaults to -9.

        prefix (str, optional): Prefix for writing output files. Defaults to "output".

        output_format (str, optional): Format of output imputed matrix. Possible values include: "df" for a pandas.DataFrame object, "array" for a numpy.ndarray object, and "list" for a 2D list. Defaults to "df".

        verbose (bool, optional): Whether to print status updates. Set to False for no status updates. Defaults to True.

        **kwargs (Dict[str, Any]): Additional keyword arguments to supply. Primarily for internal purposes. Options include: {"iterative_mode": bool}. "iterative_mode" determines whether ``ImputeNMF`` is being used as the initial imputer in ``IterativeImputer``.

    Attributes:
        imputed_ (pandas.DataFrame, numpy.ndarray, or List[List[int]]): Imputed 012-encoded data.

        accuracy_ (float): Accuracy of predicted imputations for known genotypes.

    Example:
        >>>data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID"
        >>>)
        >>>
        >>>nmf = ImputeNMF(
        >>>     genotype_data=data,
        >>>     learning_rate=0.1,
        >>>)
        >>>
        >>>nmf_gtdata = nmf.fit_transform(data)
    """

    def __init__(
        self,
        *,
        latent_features: int = 2,
        max_iter: int = 100,
        learning_rate: float = 0.0002,
        regularization_param: float = 0.02,
        tol: float = 0.1,
        n_fail: int = 20,
        missing: int = -9,
        prefix: str = "output",
        output_format="df",
        verbose: bool = True,
        **kwargs: Dict[str, Any],
    ) -> None:
        self.latent_features = latent_features
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.regularization_param = regularization_param
        self.tol = tol
        self.n_fail = n_fail
        self.missing = missing
        self.prefix = prefix
        self.output_format = output_format
        self.verbose = verbose
        self.kwargs = kwargs

    def fit(self, X):
        """Fit imputer to input data X.

        Args:
            X (GenotypeData): Input GenotypeData instance.
        """
        self.iterative_mode_ = self.kwargs.get("iterative_mode", False)
        self.imputed_, self.accuracy_ = self._impute(X.genotypes012_array)

        if self.verbose:
            print(f"NMF imputation accuracy: {round(self.accuracy_, 2)}")

        return self

    def transform(self, X):
        """Transform imputed data and return desired output format.

        Args:
            X (GenotypeData): Instantiated GenotypeData object.

        Returns:
            pandas.DataFrame, numpy.ndarray, or List[List[int]]: Imputed 012-encoded data.
        """
        if self.output_format == "df":
            return self.imputed_
        elif self.output_format == "array":
            return self.imputed_.to_numpy()
        elif self.output_format == "list":
            return self.imputed_.values.tolist()
        else:
            raise ValueError(
                f"Unsupported output_format provided. Valid options include "
                f"'df', 'array', or 'list', but got {self.output_format}"
            )

    def _impute(self, X):
        """Do the imputation."""
        if self.verbose:
            print(f"Doing NMF imputation...")
        R = X.copy()
        R[R == self.missing] = -9
        R = R + 1
        R[R < 0] = 0
        n_row = len(R)
        n_col = len(R[0])
        p = np.random.rand(n_row, self.latent_features)
        q = np.random.rand(n_col, self.latent_features)
        q_t = q.T
        fails = 0
        e_current = None
        for step in range(self.max_iter):
            for i in range(n_row):
                for j in range(n_col):
                    if R[i][j] > 0:
                        eij = R[i][j] - np.dot(p[i, :], q_t[:, j])
                        for k in range(self.latent_features):
                            p[i][k] = p[i][k] + self.learning_rate * (
                                2 * eij * q_t[k][j]
                                - self.regularization_param * p[i][k]
                            )
                            q_t[k][j] = q_t[k][j] + self.learning_rate * (
                                2 * eij * p[i][k]
                                - self.regularization_param * q_t[k][j]
                            )
            e = 0
            for i in range(n_row):
                for j in range(len(R[i])):
                    if R[i][j] > 0:
                        e = e + pow(R[i][j] - np.dot(p[i, :], q_t[:, j]), 2)
                        for k in range(self.latent_features):
                            e = e + (self.regularization_param / 2) * (
                                pow(p[i][k], 2) + pow(q_t[k][j], 2)
                            )
            if e_current is None:
                e_current = e
            else:
                if abs(e_current - e) < self.tol:
                    fails += 1
                else:
                    fails = 0
                e_current = e
            if fails >= self.n_fail:
                break
        nR = np.dot(p, q_t)

        # transform values per-column
        # (i.e., only allowing values found in original)
        tR = self._fill(R, nR)

        # get accuracy of re-constructing non-missing genotypes
        acc = self.accuracy(X, tR)

        # insert imputed values for missing genotypes
        fR = X.copy()
        fR[X < 0] = tR[X < 0]

        if self.verbose:
            print("Done!")

        return pd.DataFrame(fR), acc

    def _fill(self, original, predicted):
        """Fill missing values values per-column, only allowing values found in original."""
        n_row = len(original)
        n_col = len(original[0])
        tR = predicted
        for j in range(n_col):
            observed = predicted[:, j]
            expected = original[:, j]
            options = np.unique(expected[expected != 0])
            for i in range(n_row):
                transform = min(
                    options, key=lambda x: abs(x - predicted[i, j])
                )
                tR[i, j] = transform
        tR = tR - 1
        tR[tR < 0] = -9
        return tR

    def accuracy(self, expected, predicted):
        """get accuracy of re-constructing non-missing genotypes."""

        prop_same = np.sum(expected[expected >= 0] == predicted[expected >= 0])
        tot = expected[expected >= 0].size
        return prop_same / tot


class SimGenotypeDataTransformer(BaseEstimator, TransformerMixin):
    """Simulate missing data on genotypes read/ encoded in a GenotypeData object.

    Copies metadata from a GenotypeData object and simulates user-specified proportion of missing data

    Args:
            genotype_data (GenotypeData): GenotypeData object. Assumes no missing data already present. Defaults to None.

            prop_missing (float, optional): Proportion of missing data desired in output. Defaults to 0.10

            strategy (str, optional): Strategy for simulating missing data. May be one of: \"nonrandom\", \"nonrandom_weighted\", or \"random\". When set to \"nonrandom\", branches from GenotypeData.guidetree will be randomly sampled to generate missing data on descendant nodes. For \"nonrandom_weighted\", missing data will be placed on nodes proportionally to their branch lengths (e.g., to generate data distributed as might be the case with mutation-disruption of RAD sites). Defaults to \"random\"

            missing_val (int, optional): Value that represents missing data. Defaults to -9.

            mask_missing (bool, optional): True if you want to skip original missing values when simulating new missing data, False otherwise. Defaults to True.

            verbose (bool, optional): Verbosity level. Defaults to 0.

            tol (float): Tolerance to reach proportion specified in self.prop_missing. Defaults to 1/num_snps*num_inds

            max_tries (int): Maximum number of tries to reach targeted missing data proportion within specified tol. Defaults to num_inds.

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
        masked_sites = np.sum(self.sim_missing_mask_)
        num_correct = np.sum(X_true[self.sim_missing_mask_] == X_pred[self.sim_missing_mask_])
        return num_correct / masked_sites

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
        for i, column in enumerate(self.mask_.T):
            if mask:
                miss_mask = self.original_missing_mask_[:, i]
                col = column[~miss_mask]
                obs_idx = np.where(~miss_mask)
                idx = obs_idx[np.random.choice(np.arange(len(obs_idx)))]
                mask_subset = self.mask_[~self.original_missing_mask_[:, i], i]
            else:
                col = column
                idx = np.random.choice(np.arange(col.shape[0]))
                mask_subset = self.mask_[:, i]
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
        mask_boolean = (self.mask_ != 0)
        Xt[mask_boolean] = mask_val
        # for i, row in enumerate(self.mask_):
        #     for j in row.nonzero()[0]:
        #         Xt[i][j] = mask_val
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
