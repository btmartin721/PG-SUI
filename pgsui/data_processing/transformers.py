import gc
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

# Custom Modules
try:
    from .simple_imputers import ImputeAlleleFreq, ImputePhylo, ImputeNMF
    from .neural_network_methods import NeuralNetworkMethods
except (ModuleNotFoundError, ValueError):
    from impute.simple_imputers import ImputeAlleleFreq, ImputePhylo, ImputeNMF
    from impute.neural_network_methods import NeuralNetworkMethods


def encode_onehot(X):
    """Convert 012-encoded data to one-hot encodings.

    Args:
        X (numpy.ndarray): Input array with 012-encoded data and -9 as the missing data value.

    Returns:
        pandas.DataFrame: One-hot encoded data, ignoring missing values (np.nan).
    """

    df = encode_categorical(X)
    df_incomplete = df.copy()
    missing_encoded = pd.get_dummies(df_incomplete)

    for col in df.columns:
        missing_cols = missing_encoded.columns.str.startswith(str(col) + "_")
        missing_encoded.loc[df_incomplete[col].isnull(), missing_cols] = np.nan
    return missing_encoded


def encode_categorical(X):
    """Encode -9 encoded missing values as np.nan.

    Args:
        X (numpy.ndarray): 012-encoded genotypes with -9 as missing values.

    Returns:
        pandas.DataFrame: DataFrame with missing values encoded as np.nan.
    """
    np.nan_to_num(X, copy=False, nan=-9.0)
    X = X.astype(str)
    X[(X == "-9.0") | (X == "-9")] = "none"

    df = pd.DataFrame(X)
    df_incomplete = df.copy()

    # Replace 'none' with np.nan
    for row in df.index:
        for col in df.columns:
            if df_incomplete.iat[row, col] == "none":
                df_incomplete.iat[row, col] = np.nan

    return df_incomplete


def decode_onehot(df_dummies):
    """Decode one-hot format to 012-encoded genotypes.

    Args:
        df_dummies (pandas.DataFrame): One-hot encoded imputed data.

    Returns:
        pandas.DataFrame: 012-encoded imputed data.
    """
    pos = defaultdict(list)
    vals = defaultdict(list)

    for i, c in enumerate(df_dummies.columns):
        if "_" in c:
            k, v = c.split("_", 1)
            pos[k].append(i)
            vals[k].append(v)

        else:
            pos["_"].append(i)

    df = pd.DataFrame(
        {
            k: pd.Categorical.from_codes(
                np.argmax(df_dummies.iloc[:, pos[k]].values, axis=1),
                vals[k],
            )
            for k in vals
        }
    )

    df[df_dummies.columns[pos["_"]]] = df_dummies.iloc[:, pos["_"]]

    return df


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
        phase (int or None): current phase if doing UBP or None if doing NLPCA.
        n_components (int): Number of principal components currently being used in V.

        V (numpy.ndarray or Dict[str, Any]): If doing grid search, should be a dictionary with current_component: numpy.ndarray. If not doing grid search, then it should be a numpy.ndarray.
    """

    def __init__(self, phase, n_components, V):
        self.phase = phase
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
        """
        if self.phase == None or self.phase == 1:
            if not isinstance(self.V, dict):
                raise TypeError("V must be a dictionary if phase == None or 1.")
            return self.V[self.n_components]
        else:
            return self.V


class NNInputTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X):
        X = self._validate_input(X)
        df = encode_onehot(X)
        X_train = df.copy().values
        self.missing_mask_, self.observed_mask_ = self._get_masks(X_train)

        return self

    def transform(self, X):
        X = self._validate_input(X)
        df = encode_onehot(X)
        X_train = df.copy().values
        return self._fill(X_train, self.missing_mask_)

    def _fill(self, data, missing_mask, missing_value=-1, num_classes=1):
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

        Returns:
            numpy.ndarray(bool): Boolean mask of missing values, with True corresponding to a missing data point.
        """
        if data.dtype != "f" and data.dtype != "d":
            data = data.astype(float)
        return np.isnan(data)

    def _validate_input(self, input_data, out_type="numpy"):
        """Validate input data and ensure that it is of correct type.

        Args:
            input_data (List[List[int]], numpy.ndarray, or pandas.DataFrame): Input data to validate.

            out_type (str, optional): Type of object to convert input data to. Possible options include "numpy" and "pandas". Defaults to "numpy".

        Returns:
            numpy.ndarray: Input data as numpy array.

        Raises:
            TypeError: Must be of type pandas.DataFrame, numpy.ndarray, or List[List[int]].

            ValueError: astype must be either "numpy" or "pandas".
        """
        if out_type == "numpy":
            if isinstance(input_data, pd.DataFrame):
                X = input_data.to_numpy()
            elif isinstance(input_data, list):
                X = np.array(input_data)
            elif isinstance(input_data, np.ndarray):
                X = input_data.copy()
            else:
                raise TypeError(
                    f"input_data must be of type pd.DataFrame, np.ndarray, or "
                    f"list(list(int)), but got {type(input_data)}"
                )

        elif out_type == "pandas":
            if isinstance(input_data, pd.DataFrame):
                X = input_data.copy()
            elif isinstance(input_data, (list, np.ndarray)):
                X = pd.DataFrame(input_data)
            else:
                raise TypeError(
                    f"input_data must be of type pd.DataFrame, np.ndarray, or "
                    f"list(list(int)), but got {type(input_data)}"
                )
        else:
            raise ValueError("astype must be either 'numpy' or 'pandas'.")

        return X


class NNOutputTransformer(BaseEstimator, TransformerMixin):
    """Transformer to format target data both before and after model fitting.

    Args:
        y_decoded (numpy.ndarray): Original input data that is 012-encoded.
    """

    def __init__(self, y_decoded):
        self.y_decoded = y_decoded

    def fit(self, y):
        """Fit to target data.

        Args:
            y (numpy.ndarray): Target data that is one-hot encoded.

        Returns:
            self: Class instance.
        """
        self.n_outputs_expected_ = 1
        return self

    def transform(self, y):
        """Transform y_true. Here to accomodate multiclass-multioutput targets.

        Args:
            y (numpy.ndarray): One-hot encoded target data.

        Returns:
            numpy.ndarray: y_true target data. No formatting necessary, but here to accomodate multiclass-multioutput targets.
        """
        return y

    def inverse_transform(self, y):
        """Transform y_pred to same format as y_true.

        This allows sklearn.metrics to be used.

        Args:
            y (numpy.ndarray): Predicted probabilities after model fitting.

        Returns:
            numpy.ndarray: y predictions in same format as y_true.
        """
        y_pred_proba = y  # Rename for clarity, Keras gives probs
        imputed_enc, dummy_df = self._predict(self.y_decoded, y_pred_proba)

        y_pred_df = decode_onehot(
            pd.DataFrame(data=imputed_enc, columns=dummy_df.columns)
        )

        return y_pred_df.to_numpy()

    def _predict(self, X, complete_encoded):
        """Evaluate VAE predictions by calculating the highest predicted value.

        Calucalates highest predicted value for each row vector and each class, setting the most likely class to 1.0.

        Args:
            X (numpy.ndarray): Input one-hot encoded data.

            complete_encoded (numpy.ndarray): Output one-hot encoded data with the maximum predicted values for each class set to 1.0.

        Returns:
            numpy.ndarray: Imputed one-hot encoded values.

            pandas.DataFrame: One-hot encoded pandas DataFrame with no missing values.
        """

        df = encode_categorical(X)

        # Had to add dropna() to count unique classes while ignoring np.nan
        col_classes = [len(df[c].dropna().unique()) for c in df.columns]
        df_dummies = pd.get_dummies(df)
        mle_complete = None
        for i, cnt in enumerate(col_classes):
            start_idx = int(sum(col_classes[0:i]))
            col_completed = complete_encoded[:, start_idx : start_idx + cnt]
            mle_completed = np.apply_along_axis(mle, axis=1, arr=col_completed)

            if mle_complete is None:
                mle_complete = mle_completed

            else:
                mle_complete = np.hstack([mle_complete, mle_completed])
        return mle_complete, df_dummies


class RandomizeMissingTransformer(BaseEstimator, TransformerMixin):
    """Replace random columns and rows with np.nan.

    Function to select ``col_selection_rate * columns`` columns in a pandas DataFrame and change anywhere from 15% to 50% of the values in each of those columns to np.nan (missing data). Since we know the true values of the ones changed to np.nan, we can assess the accuracy of the classifier and do a grid search.

    Args:
        initial_strategy (str): Initial strategy to simple impute with.

        genotype_data (GenotypeData): Initialized GenotypeData object.

        str_encodings (Dict[str, Any]): STRUCTURE file encodings.

        pops (List[Union[str, int]]): List of population IDs.

        col_selection_rate (float): Number of columns for which to randomly introduce missing data. Defaults to 1.0 (all columns).

    Attributes:
        cols_ (numpy.ndarray): Columns to introduce missing data to. The number of columns with missing data are determined with the ``col_selection_rate`` argument.

        simple_imputer_ (ImputePhylo, ImputeAlleleFreq, ImputeNMF, or SimpleImputer): Simple imputer instance. The type depends on ``initial_strategy``\.

        input_params_ (Dict[str, Any]): Parameters to input into self.simple_imputer_.

        df_filled_ (pandas.DataFrame): DataFrame with values imputed initially with sklearn.impute.SimpleImputer, ImputeAlleleFreq (by populations) or ImputePhylo.
    """

    def __init__(
        self,
        initial_strategy,
        genotype_data,
        str_encodings,
        pops,
        col_selection_rate=1.0,
        min_missing_prop=0.15,
        max_missing_prop=0.5,
    ):
        self.initial_strategy = initial_strategy
        self.genotype_data = genotype_data
        self.str_encodings = str_encodings
        self.pops = pops
        self.col_selection_rate = col_selection_rate
        self.min_missing_prop = min_missing_prop
        self.max_missing_prop = max_missing_prop

    def fit(self, X):
        """Fit to input data.

        Args:
            X (pandas.DataFrame): 012-encoded genotypes to extract columns from.

        Returns:
            self: Class instance.

        Raises:
            TypeError: X must be a pandas.DataFrame object.
        """
        # Code adapted from: https://medium.com/analytics-vidhya/using-scikit-learns-iterative-imputer-694c3cca34de

        if not isinstance(X, pd.DataFrame):
            df = pd.DataFrame(X)
        else:
            df = X.copy()

        self.cols_ = np.random.choice(
            df.columns,
            int(len(df.columns) * self.col_selection_rate),
            replace=False,
        )

        if self.initial_strategy == "populations":
            simple_imputer = ImputeAlleleFreq
            self.input_params_ = dict(
                genotype_data=self.genotype_data,
                pops=self.pops,
                by_populations=True,
                missing=-9,
                write_output=False,
                verbose=False,
                validation_mode=True,
            )

        elif self.initial_strategy == "phylogeny":
            simple_imputer = ImputePhylo
            self.input_params_ = dict(
                genotype_data=self.genotype_data,
                str_encodings=self.str_encodings,
                write_output=False,
                disable_progressbar=True,
                validation_mode=True,
            )

        elif self.initial_strategy == "nmf":
            simple_imputer = ImputeNMF
            self.input_params_ = dict(
                gt=df.fillna(-9).to_numpy(),
                missing=-9,
                write_output=False,
                verbose=False,
                validation_mode=True,
            )

        else:
            # Fill in unknown values with sklearn.impute.SimpleImputer
            simple_imputer = SimpleImputer
            self.input_params_ = dict(strategy=self.initial_strategy_)

        self.simple_imputer_ = simple_imputer(**self.input_params_)

        # initialize.
        self.df_filled_ = None

        return self

    def transform(self, X):
        """Transform input data X.

        Randomly introduces missing data into the self.df_filled_ DataFrame.

        Args:
            X (pandas.DataFrame): 012-encoded DataFrame.

        Returns:
            pandas.DataFrame: DataFrame with randomly introduced missing values.
        """
        if not isinstance(X, pd.DataFrame):
            df = pd.DataFrame(X)
        else:
            df = X.copy()

        if self.initial_strategy == "most_frequent":
            self.df_filled_ = pd.DataFrame(
                simple_imputer.fit_transform(df.fillna(-9).values)
            )

        else:
            self.df_filled_ = self.simple_imputer_.imputed
            if self.initial_strategy == "populations":
                # For some reason the df gets converted to integers with
                # pd.NA values instead of np.nan. Bandaid fix here.
                self.df_filled_ = self.df_filled_.astype(np.float)
                self.df_filled_.replace(pd.NA, np.nan, inplace=True)

        # Randomly choose rows (samples) to introduce missing data to
        df_defiled = self.df_filled_.copy()

        for col in self.cols_:
            data_drop_rate = np.random.choice(
                np.arange(self.min_missing_prop, self.max_missing_prop, 0.02), 1
            )[0]

            drop_ind = np.random.choice(
                np.arange(len(self.df_filled_[col])),
                size=int(len(self.df_filled_[col]) * data_drop_rate),
                replace=False,
            )

            # Introduce random np.nan values
            df_defiled.loc[drop_ind, col] = np.nan

        return df_defiled
