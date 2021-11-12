# Standard Library Imports

import gc
import os
import random
import sys

from collections import defaultdict
from statistics import mean, median

# Third-party Imports
import numpy as np
import pandas as pd

from scipy import stats as st

from memory_profiler import memory_usage

import matplotlib.pylab as plt
import seaborn as sns

# import theano
# import theano.tensor as T
# import theano.tensor.extra_ops
# import theano.tensor.nnet as nnet
from timeit import default_timer

import tensorflow as tf
from tensorflow.keras import initializers
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.objectives import mse
from keras.models import Sequential
from keras.layers.core import Dropout, Dense
from keras.regularizers import l1_l2

from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Custom Modules
from impute.impute import Impute
from read_input.read_input import GenotypeData
from utils.misc import generate_012_genotypes
from utils.misc import timer
from utils.misc import isnotebook

is_notebook = isnotebook()

if is_notebook:
    from tqdm.notebook import tqdm as progressbar
else:
    from tqdm import tqdm as progressbar


def make_reconstruction_loss(n_features):
    def reconstruction_loss(input_and_mask, y_pred):

        X_values = input_and_mask[:, :n_features]
        missing_mask = input_and_mask[:, n_features:]
        observed_mask = 1 - missing_mask
        X_values_observed = X_values * observed_mask
        pred_observed = y_pred * observed_mask

        return mse(y_true=X_values_observed, y_pred=pred_observed)

    return reconstruction_loss


def masked_mae(X_true, X_pred, mask):

    masked_diff = X_true[mask] - X_pred[mask]
    return np.mean(np.abs(masked_diff))


def masked_rmse(X_true, X_pred, mask):
    masked_diff = X_true[mask] - X_pred[mask]
    return np.sqrt(np.mean(masked_diff ** 2))


def masked_rmse(X_true, X_pred, mask):
    masked_diff = X_true[mask] - X_pred[mask]
    return np.sqrt(np.mean(masked_diff ** 2))


def get_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def create_weights(sz):
    theta = theano.shared(
        np.array(np.random.rand(sz[0], sz[1]), dtype=theano.config.floatX)
    )
    return theta


# def grad_desc(cost, theta, alpha):
#     return theta - (alpha * T.grad(cost, wrt=theta))


class ImputeVAE(Impute):
    def __init__(
        self,
        *,
        genotype_data=None,
        gt=None,
        prefix="output",
        cv=5,
        initial_strategy="populations",
        validation_only=0.2,
        disable_progressbar=False,
        train_epochs=100,
        batch_size=64,
        recurrent_weight=0.5,
        optimizer="adam",
        dropout_probability=0.2,
        hidden_activation="relu",
        output_activation="sigmoid",
        kernel_initializer="glorot_normal",
        l1_penalty=0,
        l2_penalty=0,
    ):

        self.clf = "VAE"
        self.clf_type = "classifier"
        self.imp_kwargs = {
            "initial_strategy": initial_strategy,
            "genotype_data": genotype_data,
            "str_encodings": {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9},
        }

        super().__init__(self.clf, self.clf_type, self.imp_kwargs)

        self.prefix = prefix

        self.train_epochs = train_epochs
        self.recurrent_weight = recurrent_weight
        self.optimizer = optimizer
        self.dropout_probability = dropout_probability
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.kernel_initializer = kernel_initializer
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty

        self.cv = cv
        self.validation_only = validation_only
        self.disable_progressbar = disable_progressbar

        self.df = None
        self.data = None

        if genotype_data is None and gt is None:
            raise TypeError("genotype_data and gt cannot both be NoneType")

        if genotype_data is not None and gt is not None:
            raise TypeError("genotype_data and gt cannot both be used")

        if genotype_data is not None:
            X = genotype_data.genotypes_nparray

        elif gt is not None:
            X = gt

        if batch_size > X.shape[0]:
            while batch_size > X.shape[0]:
                print(
                    "Batch size is larger than the number of samples. "
                    "Dividing batch_size by 2."
                )
                batch_size = batch_size // 2

        self.batch_size = batch_size

        if self.validation_only is not None:
            print("\nEstimating validation scores...")

            self.df_scores = self._imputer_validation(pd.DataFrame(X), self.clf)

            print("\nDone!\n")

        else:
            self.df_scores = None

        print("\nImputing full dataset...")

        # mem_usage = memory_usage((self._impute_single, (X,)))
        # with open(f"profiling_results/memUsage_{self.prefix}.txt", "w") as fout:
        # fout.write(f"{max(mem_usage)}")
        # sys.exit()

        self.imputed_df = self.fit_predict(X)
        print("\nDone!\n")

        self.imputed_df = self.imputed_df.astype(np.float)
        self.imputed_df = self.imputed_df.astype("Int8")

        self._validate_imputed(self.imputed_df)

        self.write_imputed(self.imputed_df)

        if self.df_scores is not None:
            print(self.df_scores)

    @timer
    def fit_predict(self, X):
        self.data = self._encode_onehot(X)
        # self.data = self.df.copy().values
        print(self.data.shape)
        sys.exit()

        imputed_enc = self.train(
            train_epochs=self.train_epochs, batch_size=self.batch_size
        )

        imputed_enc, dummy_df = self._eval_predictions(X, imputed_enc)

        imputed_df = self._decode_onehot(
            pd.DataFrame(data=imputed_enc, columns=dummy_df.columns)
        )

        return imputed_df

    @property
    def imputed(self):
        return self.imputed_df

    def _read_example_data(self):
        df = pd.read_csv("mushrooms_test_2.csv", header=None)

        df_incomplete = df.copy()

        df_incomplete.iat[1, 0] = np.nan
        df_incomplete.iat[2, 1] = np.nan

        missing_encoded = pd.get_dummies(df_incomplete)

        for col in df.columns:
            missing_cols = missing_encoded.columns.str.startswith(
                str(col) + "_"
            )

            missing_encoded.loc[
                df_incomplete[col].isnull(), missing_cols
            ] = np.nan

        return missing_encoded

    def _encode_categorical(self, X):
        """[Encode -9 encoded missing values as np.nan]

        Args:
            X ([numpy.ndarray]): [012-encoded genotypes with -9 as missing values]

        Returns:
            [pandas.DataFrame]: [DataFrame with missing values encoded as np.nan]
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

    def _encode_onehot(self, X):
        """[Convert 012-encoded data to one-hot encodings]

        Args:
            X ([numpy.ndarray]): [Input array with 012-encoded data and -9 as the missing data value]

        Returns:
            [pandas.DataFrame]: [One-hot encoded data, ignoring missing values (np.nan)]
        """

        df = self._encode_categorical(X)
        df_incomplete = df.copy()

        missing_encoded = pd.get_dummies(df_incomplete)

        for col in df.columns:
            missing_cols = missing_encoded.columns.str.startswith(
                str(col) + "_"
            )

            missing_encoded.loc[
                df_incomplete[col].isnull(), missing_cols
            ] = np.nan

        return missing_encoded

    def _mle(self, row):
        """[Get the Maximum Likelihood Estimation for the best prediction. Basically, it sets the index of the maxiumum value in a vector (row) to 1.0, since it is one-hot encoded]

        Args:
            row ([numpy.ndarray(float)]): [Row vector with predicted values as floating points]

        Returns:
            [numpy.ndarray(float)]: [Row vector with the highest prediction set to 1.0 and the others set to 0.0]
        """
        res = np.zeros(row.shape[0])
        res[np.argmax(row)] = 1
        return res

    def _eval_predictions(self, X, complete_encoded):
        """[Evaluate VAE predictions by calculating the highest predicted value for each row vector for each class and setting it to 1.0]

        Args:
            X ([numpy.ndarray]): [Input one-hot encoded data]
            complete_encoded ([numpy.ndarray]): [Output one-hot encoded data with the maximum predicted values for each class set to 1.0]

        Returns:
            [numpy.ndarray]: [Imputed one-hot encoded values]
            [pandas.DataFrame]: [One-hot encoded pandas DataFrame with no missing values]
        """

        df = self._encode_categorical(X)

        # Had to add dropna() to count unique classes while ignoring np.nan
        col_classes = [len(df[c].dropna().unique()) for c in df.columns]

        df_dummies = pd.get_dummies(df)

        mle_complete = None

        for i, cnt in enumerate(col_classes):
            start_idx = int(sum(col_classes[0:i]))
            # col_true = dummy_df.values[:, start_idx : start_idx + cnt]
            col_completed = complete_encoded[:, start_idx : start_idx + cnt]

            mle_completed = np.apply_along_axis(
                self._mle, axis=1, arr=col_completed
            )

            if mle_complete is None:
                mle_complete = mle_completed

            else:
                mle_complete = np.hstack([mle_complete, mle_completed])

        return mle_complete, df_dummies

    def _decode_onehot(self, df_dummies):
        """[Decode one-hot format to 012-encoded genotypes]

        Args:
            df_dummies ([pandas.DataFrame]): [One-hot encoded imputed data]

        Returns:
            [pandas.DataFrame]: [012-encoded imputed data]
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

    def _get_hidden_layer_sizes(self):
        """[Get dimensions of hidden layers]

        Returns:
            [int, int, int]: [Number of dimensions in hidden layers]
        """
        n_dims = self.data.shape[1]
        return [
            min(2000, 8 * n_dims),
            min(500, 2 * n_dims),
            int(np.ceil(0.5 * n_dims)),
        ]

    def _create_model(self):
        """[Create a variational autoencoder model with the following items: InputLayer -> DenseLayer1 -> Dropout1 -> DenseLayer2 -> Dropout2 -> DenseLayer3 -> Dropout3 -> DenseLayer4 -> OutputLayer]

        Returns:
            [keras model object]: [Compiled Keras model]
        """
        hidden_layer_sizes = self._get_hidden_layer_sizes()
        first_layer_size = hidden_layer_sizes[0]
        n_dims = self.data.shape[1]

        model = Sequential()

        model.add(
            Dense(
                first_layer_size,
                input_dim=2 * n_dims,
                activation=self.hidden_activation,
                kernel_regularizer=l1_l2(self.l1_penalty, self.l2_penalty),
                kernel_initializer=self.kernel_initializer,
            )
        )

        model.add(Dropout(self.dropout_probability))

        for layer_size in hidden_layer_sizes[1:]:
            model.add(
                Dense(
                    layer_size,
                    activation=self.hidden_activation,
                    kernel_regularizer=l1_l2(self.l1_penalty, self.l2_penalty),
                    kernel_initializer=self.kernel_initializer,
                )
            )

            model.add(Dropout(self.dropout_probability))

        model.add(
            Dense(
                n_dims,
                activation=self.output_activation,
                kernel_regularizer=l1_l2(self.l1_penalty, self.l2_penalty),
                kernel_initializer=self.kernel_initializer,
            )
        )

        loss_function = make_reconstruction_loss(n_dims)

        model.compile(optimizer=self.optimizer, loss=loss_function)

        return model

    def fill(self, missing_mask):
        """[Mask missing data as -1]

        Args:
            missing_mask ([np.ndarray(bool)]): [Missing data mask with True corresponding to a missing value]
        """
        self.data[missing_mask] = -1

    def _create_missing_mask(self):
        """[Creates a missing data mask with boolean values]

        Returns:
            [numpy.ndarray(bool)]: [Boolean mask of missing values, with True corresponding to a missing data point]
        """
        if self.data.dtype != "f" and self.data.dtype != "d":
            self.data = self.data.astype(float)
        return np.isnan(self.data)

    def _train_epoch(self, model, missing_mask, batch_size):
        """[Train one cycle (epoch) of a variational autoencoder model]

        Args:
            model ([Keras model object]): [VAE model object implemented in Keras]

            missing_mask ([numpy.ndarray(bool)]): [Missing data boolean mask, with True corresponding to a missing value]

            batch_size ([int]): [Batch size for one epoch]

        Returns:
            [numpy.ndarray]: [VAE model predictions of the current epoch]
        """

        input_with_mask = np.hstack([self.data, missing_mask])
        n_samples = len(input_with_mask)

        n_batches = int(np.ceil(n_samples / batch_size))
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_shuffled = input_with_mask[indices]

        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx + 1) * batch_size
            batch_data = X_shuffled[batch_start:batch_end, :]
            model.train_on_batch(batch_data, batch_data)

        return model.predict(input_with_mask)

    def train(self, batch_size=256, train_epochs=100):
        """[Train a variational autoencoder model]

        Args:
            batch_size (int, optional): [Number of data splits to train on per epoch]. Defaults to 256.

            train_epochs (int, optional): [Number of epochs (cycles through the data) to use]. Defaults to 100.

        Returns:
            [numpy.ndarray(float)]: [Predicted values as numpy array]
        """

        missing_mask = self._create_missing_mask()
        self.fill(missing_mask)
        self.model = self._create_model()

        observed_mask = ~missing_mask

        for epoch in range(train_epochs):
            X_pred = self._train_epoch(self.model, missing_mask, batch_size)
            observed_mae = masked_mae(
                X_true=self.data, X_pred=X_pred, mask=observed_mask
            )

            if epoch == 0:
                print(f"Initial MAE: {observed_mae}")

            elif epoch % 50 == 0:
                print(
                    f"Observed MAE ({epoch}/{train_epochs} epochs): "
                    f"{observed_mae}"
                )

            old_weight = 1.0 - self.recurrent_weight
            self.data[missing_mask] *= old_weight
            pred_missing = X_pred[missing_mask]
            self.data[missing_mask] += self.recurrent_weight * pred_missing

        return self.data.copy()


class ImputeUBP(Impute):
    def __init__(
        self,
        *,
        genotype_data=None,
        prefix="output",
        cv=5,
        initial_strategy="populations",
        validation_only=0.2,
        disable_progressbar=False,
        reduced_dimensions=2,
        num_hidden_layers=3,
        hidden_layer_sizes=100,
        recurrent_weight=0.5,
        optimizer="adam",
        hidden_activation="relu",
        output_activation="sigmoid",
        kernel_initializer="glorot_normal",
        l1_penalty=0,
        l2_penalty=0,
        **kwargs,
    ):
        if isinstance(hidden_layer_sizes, int):
            hidden_layer_sizes = [hidden_layer_sizes] * num_hidden_layers

        assert (
            num_hidden_layers == len(hidden_layer_sizes)
            and num_hidden_layers > 0
        )

        self.clf = "UBP"
        self.clf_type = "classifier"
        self.imp_kwargs = {
            "initial_strategy": initial_strategy,
            "genotype_data": genotype_data,
        }

        test_cat = kwargs.get("test_categorical", None)

        super().__init__(self.clf, self.clf_type, self.imp_kwargs)

        self.prefix = prefix

        self.cv = cv
        self.validation_only = validation_only
        self.disable_progressbar = disable_progressbar
        self.reduced_dimensions = reduced_dimensions
        self.recurrent_weight = recurrent_weight
        self.optimizer = optimizer
        self.kernel_initializer = kernel_initializer
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty

        self.df = None
        self.data = None

        if genotype_data is None:
            raise TypeError(
                "genotype_data was not supplied and is a required argument"
            )

        if test_cat is not None:
            X = test_cat.copy()
        else:
            X = genotype_data.genotypes_nparray

        self.Xt = None

        # Get missing data mask
        self.mask = np.isnan(X)

        # Get indices where data is not missing.
        self.valid = np.where(~self.mask)

        # Get number of hidden layers
        self.l = num_hidden_layers

        # Get initial reduced-dimension weights.
        # num_classes = 3
        # self.V = np.zeros((X.shape[0], reduced_dimensions, num_classes))
        # for r in range(X.shape[0]):
        self.V = self._init_weights(X.shape[0], reduced_dimensions)

        # Get initial weights for single layer perceptron.
        # self.T = self._init_weights(X.shape[0], reduced_dimensions)

        self.num_total_epochs = 0

        self.single_layer = None

        # if self.validation_only is not None:
        #     print("\nEstimating validation scores...")

        #     self.df_scores = self._imputer_validation(pd.DataFrame(X), self.clf)

        #     print("\nDone!\n")

        # else:
        #     self.df_scores = None

        print("\nImputing full dataset...")

        # mem_usage = memory_usage((self._impute_single, (X,)))
        # with open(f"profiling_results/memUsage_{self.prefix}.txt", "w") as fout:
        # fout.write(f"{max(mem_usage)}")
        # sys.exit()

        self.imputed_df = self.fit_predict(X)
        print("\nDone!\n")

        self.imputed_df = self.imputed_df.astype(np.float)
        self.imputed_df = self.imputed_df.astype("Int8")

        self._validate_imputed(self.imputed_df)

        self.write_imputed(self.imputed_df)

        if self.df_scores is not None:
            print(self.df_scores)

    @timer
    def fit_predict(self, X):
        # self.data = self._encode_onehot(X)
        # self.data = self.df.copy().values
        self.data = X.copy()
        imputed_enc = self._train()

        # imputed_enc, dummy_df = self._eval_predictions(X, imputed_enc)

        # imputed_df = self._decode_onehot(
        #     pd.DataFrame(data=imputed_enc, columns=dummy_df.columns)
        # )

        # return imputed_df

    def _train(self):
        """Train an unsupervised backpropagation model.

        Returns:
            numpy.ndarray(float): Predicted values as numpy array.
        """

        # from sklearn.datasets import make_blobs

        # # Configuration options
        # num_samples_total = 1000
        # training_split = 250
        # cluster_centers = [(15, 0), (15, 15), (0, 15), (30, 15)]
        # num_classes = len(cluster_centers)
        # loss_function_used = "categorical_crossentropy"

        # # Generate data
        # X, targets = make_blobs(
        #     n_samples=num_samples_total,
        #     centers=cluster_centers,
        #     n_features=num_classes,
        #     center_box=(0, 1),
        #     cluster_std=1.5,
        # )

        # categorical_targets = to_categorical(targets)
        # X_training = X[training_split:, :]
        # X_testing = X[:training_split, :]
        # Targets_training = categorical_targets[training_split:]
        # Targets_testing = categorical_targets[:training_split].astype(
        #     np.integer
        # )

        # # Set shape based on data
        # feature_vector_length = len(X_training[0])
        # input_shape = (feature_vector_length,)
        # print(f"Feature shape: {input_shape}")

        # # Create the model
        # model = Sequential()
        # model.add(
        #     Dense(
        #         12,
        #         input_shape=input_shape,
        #         activation="relu",
        #         kernel_initializer="he_uniform",
        #     )
        # )
        # model.add(Dense(8, activation="relu", kernel_initializer="he_uniform"))
        # model.add(Dense(num_classes, activation="softmax"))

        # # Configure the model and start training
        # model.compile(
        #     loss=loss_function_used,
        #     optimizer="adam",
        #     metrics=["accuracy"],
        # )
        # history = model.fit(
        #     X_training,
        #     Targets_training,
        #     epochs=30,
        #     batch_size=5,
        #     verbose=1,
        #     validation_split=0.2,
        # )

        # test = model.predict(X_testing)
        # print(test)
        # print(test.shape)
        # print(X_testing.shape)
        # sys.exit()

        # X, y = make_blobs(
        #     n_samples=1000,
        #     centers=3,
        #     n_features=2,
        #     cluster_std=2,
        #     random_state=2,
        # )
        # y = to_categorical(y)
        # n_train = 500

        # trainX, testX = X[:n_train, :], X[n_train:, :]
        # trainy, testy = y[:n_train], y[n_train:]

        # print(trainX.shape)
        # print(trainy.shape)
        # sys.exit()

        missing_mask = self._create_missing_mask_row()
        observed_mask = ~missing_mask
        self._fill(missing_mask)

        # Define single layer perceptron model.
        single_layer_model = self._model_phase1()
        self.initialise_parameters()

        while self.current_eta > self.target_eta:
            self._train_epoch(
                self.data, self.V, observed_mask, single_layer_model, phase=1
            )

        # self.single_layer.fit()

        # self._model_phase1(v)
        # sys.exit()

        # while self.current_eta > self.target_eta:
        #     s = self._train_epoch()
        # X_pred = self._train_epoch(self.model, missing_mask, batch_size)
        # observed_rmse = masked_mae(
        #     X_true=self.data, X_pred=X_pred, mask=observed_mask
        # )

        # if epoch == 0:
        #     print(f"Initial MAE: {observed_mae}")

        # elif epoch % 50 == 0:
        #     print(
        #         f"Observed MAE ({epoch}/{train_epochs} epochs): "
        #         f"{observed_mae}"
        #     )

        # old_weight = 1.0 - self.recurrent_weight
        # self.data[missing_mask] *= old_weight
        # pred_missing = X_pred[missing_mask]
        # self.data[missing_mask] += self.recurrent_weight * pred_missing

    def _train_epoch(self, X, V, valid_mask, model, phase=1, num_classes=3):
        rows = np.random.choice(X.shape[0], X.shape[0], replace=False)

        V_pred = np.zeros((X.shape[0], self.reduced_dimensions, num_classes))
        for r in rows:
            print(r)
            # Loop through each row in random order,
            # then slice the indexes in the row where
            # the data is not missing.
            # The shape of Xknown should be (X.shape[1], X.shape[2])
            X_train = V[r, :]
            target = self.data[r, np.where(valid_mask[r, :])][0]
            y_train = to_categorical(target)
            print(X_train)
            print(y_train)
            print(X_train.shape)
            print(y_train.shape)
            sys.exit()
        # for r in rows:
        #     # Loop through each row in random order,
        #     # then slice the indexes in the row where
        #     # the data is not missing.
        #     # The shape of Xknown should be (X.shape[1], X.shape[2])
        #     y_true = X[r, np.where(valid_mask[r, :])]

        #     for vc, xc in zip(
        #         np.arange(V.shape[1]), np.arange(y_true.shape[1])
        #     ):
        #         x_train = V[r, vc].reshape(-1, num_classes)
        #         y_train = y_true[0][xc, :].reshape(-1, num_classes)

        #         model.fit(
        #             x=x_train,
        #             y=y_train,
        #             epochs=50,
        #         )

        #         y_pred = model.predict(x_train)

        #         # error = get_rmse(y_true, y_pred)

        #         # V_pred[r] = y_pred
        #         print(y_pred)
        #         sys.exit()
        # print(error)

        # for c in X.shape[1]:
        #     if phase == 1:
        #         self.single_layer.fit(self.V[r, :], X[r, c])

    @property
    def imputed(self):
        return self.imputed_df

    def _init_weights(self, dim1, dim2, w_mean=0, w_stddev=0.01):
        # Get reduced-dimension dataset.
        return np.random.normal(loc=w_mean, scale=w_stddev, size=(dim1, dim2))

    def _encode_onehot(self, X):
        """[Convert 012-encoded data to one-hot encodings]

        Args:
            X ([numpy.ndarray]): [Input array with 012-encoded data and -9 as the missing data value]

        Returns:
            [pandas.DataFrame]: [One-hot encoded data, ignoring missing values (np.nan)]
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

        # df = self._encode_categorical(X)
        # df_incomplete = df.copy()

        # missing_encoded = pd.get_dummies(df_incomplete)

        # for col in df.columns:
        #     missing_cols = missing_encoded.columns.str.startswith(
        #         str(col) + "_"
        #     )

        #     missing_encoded.loc[
        #         df_incomplete[col].isnull(), missing_cols
        #     ] = np.nan

        # return missing_encoded

    def _decode_onehot(self, df_dummies):
        """[Decode one-hot format to 012-encoded genotypes]

        Args:
            df_dummies ([pandas.DataFrame]): [One-hot encoded imputed data]

        Returns:
            [pandas.DataFrame]: [012-encoded imputed data]
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

    def _get_hidden_layer_sizes(self, n_dims):
        """[Get dimensions of hidden layers]

        Args:
            n_dims (int): The number of feature dimensions (columns).

        Returns:
            [int, int, int]: [Number of dimensions in hidden layers]
        """
        return [
            min(2000, 8 * n_dims),
            min(500, 2 * n_dims),
            int(np.ceil(0.5 * n_dims)),
        ]

    def _model_phase1(self, num_classes=3):
        """Create a single layer perceptron for UBP model.

        Creates a network with the following structure:

        InputLayer -> DenseLayer1 -> ActivationFunction -> OutputLayer

        This layer is temporary and is only used to refine the intrinsic vector, V.

        Args:
            num_classes: The number of classes in the vector.

        Returns:
            keras model object: Compiled Keras model.
        """
        input_shape = (num_classes,)
        output_dim = num_classes

        model = Sequential()

        model.add(
            Dense(
                output_dim,
                input_shape=input_shape,
                activation="softmax",
                kernel_initializer=self.kernel_initializer,
                use_bias=False,
            )
        )

        model.compile(optimizer=self.optimizer, loss="categorical_crossentropy")
        return model

    def _fill(self, missing_mask):
        """Mask missing data as [0, 0, 0].

        Args:
            missing_mask ([np.ndarray(bool)]): [Missing data mask with True corresponding to a missing value]
        """
        self.data[missing_mask] = -1

    def _create_missing_mask(self):
        """[Creates a missing data mask with boolean values]

        Returns:
            [numpy.ndarray(bool)]: [Boolean mask of missing values, with True corresponding to a missing data point]
        """
        return np.isnan(self.data).all(axis=2)

    def _create_missing_mask_row(self):
        """[Creates a missing data mask with boolean values]

        Returns:
            [numpy.ndarray(bool)]: [Boolean mask of missing values, with True corresponding to a missing data point]
        """
        a = self.data.copy()
        a = a.astype("float32")
        a[(self.data == -9) | (self.data == -9.0)] = np.nan
        return np.isnan(a)

    def initialise_parameters(self):
        self.initial_eta = 0.1
        self.current_eta = self.initial_eta
        self.target_eta = 0.0001
        self.gamma = 0.00001
        self.lmda = 0.0001
        self.s = 0
        self.s_prime = np.inf
        self.num_epochs = 0

    # def _train_epoch(self, model, missing_mask, batch_size):
    #     """[Train one cycle (epoch) of a variational autoencoder model]

    #     Args:
    #         model ([Keras model object]): [VAE model object implemented in Keras]

    #         missing_mask ([numpy.ndarray(bool)]): [Missing data boolean mask, with True corresponding to a missing value]

    #         batch_size ([int]): [Batch size for one epoch]

    #     Returns:
    #         [numpy.ndarray]: [VAE model predictions of the current epoch]
    #     """

    #     input_with_mask = np.hstack([self.data, missing_mask])
    #     n_samples = len(input_with_mask)

    #     n_batches = int(np.ceil(n_samples / batch_size))
    #     indices = np.arange(n_samples)
    #     np.random.shuffle(indices)
    #     X_shuffled = input_with_mask[indices]

    #     for batch_idx in range(n_batches):
    #         batch_start = batch_idx * batch_size
    #         batch_end = (batch_idx + 1) * batch_size
    #         batch_data = X_shuffled[batch_start:batch_end, :]
    #         model.train_on_batch(batch_data, batch_data)

    #     return model.predict(input_with_mask)

    # def train(self, batch_size=256, train_epochs=100):
    #     """[Train a variational autoencoder model]

    #     Args:
    #         batch_size (int, optional): [Number of data splits to train on per epoch]. Defaults to 256.

    #         train_epochs (int, optional): [Number of epochs (cycles through the data) to use]. Defaults to 100.

    #     Returns:
    #         [numpy.ndarray(float)]: [Predicted values as numpy array]
    #     """

    #     missing_mask = self._create_missing_mask()
    #     self.fill(missing_mask)
    #     self.model = self._create_model()

    #     observed_mask = ~missing_mask

    #     for epoch in range(train_epochs):
    #         X_pred = self._train_epoch(self.model, missing_mask, batch_size)
    #         observed_mae = masked_mae(
    #             X_true=self.data, X_pred=X_pred, mask=observed_mask
    #         )

    #         if epoch == 0:
    #             print(f"Initial MAE: {observed_mae}")

    #         elif epoch % 50 == 0:
    #             print(
    #                 f"Observed MAE ({epoch}/{train_epochs} epochs): "
    #                 f"{observed_mae}"
    #             )

    #         old_weight = 1.0 - self.recurrent_weight
    #         self.data[missing_mask] *= old_weight
    #         pred_missing = X_pred[missing_mask]
    #         self.data[missing_mask] += self.recurrent_weight * pred_missing

    #     return self.data.copy()
