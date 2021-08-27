# Standard Library Imports

import os
import random
import sys

from collections import defaultdict

# Third-party Imports
import numpy as np
import pandas as pd

import matplotlib.pylab as plt
import seaborn as sns

from keras.utils import np_utils
from keras.objectives import mse
from keras.models import Sequential
from keras.layers.core import Dropout, Dense
from keras.regularizers import l1_l2 as l1l2

from sklearn.preprocessing import OneHotEncoder

# Custom Modules
from read_input.read_input import GenotypeData
from utils.misc import generate_012_genotypes


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


class ImputeVAE(GenotypeData):
    def __init__(
        self,
        *,
        genotype_data=None,
        gt=None,
        recurrent_weight=0.5,
        optimizer="adam",
        dropout_probability=0.5,
        hidden_activation="relu",
        output_activation="sigmoid",
        kernel_initializer="glorot_normal",
        l1_penalty=0,
        l2_penalty=0,
    ):

        super().__init__()

        self.recurrent_weight = recurrent_weight
        self.optimizer = optimizer
        self.dropout_probability = dropout_probability
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.kernel_initializer = kernel_initializer
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty

        if genotype_data is None and gt is None:
            raise TypeError("genotype_data and gt cannot both be NoneType")

        if genotype_data is not None and gt is not None:
            raise TypeError("genotype_data and gt cannot both be used")

        if genotype_data is not None:
            X = genotype_data.genotypes_nparray

        elif gt is not None:
            X = gt

        # imputed = self._read_example_data()

        self.data = self._encode_onehot(X).values

        imputed = self.train(train_epochs=300, batch_size=256)
        imputed = self._eval_predictions(X, imputed)

        print(imputed)

    def _get_imputed_data(self):
        return self.imputed

    def _read_example_data(self):
        df = pd.read_csv("mushrooms.csv", header=None)

        prob_missing = 0.1
        df_incomplete = df.copy()
        ix = [
            (row, col)
            for row in range(df.shape[0])
            for col in range(df.shape[1])
        ]

        for row, col in random.sample(ix, int(round(prob_missing * len(ix)))):
            df_incomplete.iat[row, col] = np.nan

        print(df_incomplete)

        missing_encoded = pd.get_dummies(df_incomplete)

        print(missing_encoded)

        for col in df.columns:
            missing_cols = missing_encoded.columns.str.startswith(
                str(col) + "_"
            )

            missing_encoded.loc[
                df_incomplete[col].isnull(), missing_cols
            ] = np.nan

        print(missing_encoded)

        return missing_encoded

    def _encode_categorical(self, X):
        np.nan_to_num(X, copy=False, nan=-9.0)
        X = X.astype(str)
        X[X == "-9.0"] = "none"
        return X

    def _encode_onehot(self, X):

        df = pd.DataFrame(self._encode_categorical(X))

        df_incomplete = df.copy()

        for row in df.index:
            for col in df.columns:
                if df_incomplete.iat[row, col] == "none":
                    df_incomplete.iat[row, col] = np.nan

        missing_encoded = pd.get_dummies(df_incomplete)

        for col in df.columns:
            missing_cols = missing_encoded.columns.str.startswith(
                str(col) + "_"
            )

            missing_encoded.loc[
                df_incomplete[col].isnull(), missing_cols
            ] = np.nan

        return missing_encoded

    def _encode_onehot_mask(self, X):
        ohe = OneHotEncoder()
        Xt = ohe.fit_transform(X).toarray()

        ncat = np.array([len(x) for x in ohe.categories_])

        missing_mask = np.where(X == "none", 1.0, 0.0)

        Xmiss = list()
        for row in range(X.shape[0]):
            row_mask = list()
            for col in range(X.shape[1]):
                if missing_mask[row, col] == 1.0:
                    ohe = [1.0] * ncat[col]
                else:
                    ohe = [0.0] * ncat[col]
                row_mask.extend(ohe)
            Xmiss.append(row_mask)

        return np.concatenate((Xt, np.array(Xmiss)), axis=1)

    def _get_hidden_layer_sizes(self):
        n_dims = self.data.shape[1]
        return [
            min(2000, 8 * n_dims),
            min(500, 2 * n_dims),
            int(np.ceil(0.5 * n_dims)),
        ]

    def _mle(self, row):
        res = np.zeros(row.shape[0])
        res[np.argmax(row)] = 1
        return res

    def _eval_predictions(self, X, complete_encoded):

        df = pd.DataFrame(self._encode_categorical(X))

        dummy_df = pd.get_dummies(df)

        col_classes = [len(df[c].unique()) for c in df.columns]

        mle_complete = None

        for i, cnt in enumerate(col_classes):
            start_idx = int(sum(col_classes[0:i]))
            col_true = dummy_df.values[:, start_idx : start_idx + cnt]
            print(f"col_true: {col_true}")
            col_completed = complete_encoded[:, start_idx : start_idx + cnt]
            print(f"col_completed: {col_completed}")
            if i == 1:
                sys.exit()
            mle_completed = np.apply_along_axis(
                self._mle, axis=1, arr=col_completed
            )

            if mle_complete is None:
                mle_complete = mle_completed

            else:
                mle_complete = np.hstack([mle_complete, mle_completed])

        return mle_complete

    def _create_model(self):

        hidden_layer_sizes = self._get_hidden_layer_sizes()
        first_layer_size = hidden_layer_sizes[0]
        n_dims = self.data.shape[1]

        model = Sequential()

        model.add(
            Dense(
                first_layer_size,
                input_dim=2 * n_dims,
                activation=self.hidden_activation,
                kernel_regularizer=l1l2(self.l1_penalty, self.l2_penalty),
                kernel_initializer=self.kernel_initializer,
            )
        )

        model.add(Dropout(self.dropout_probability))

        for layer_size in hidden_layer_sizes[1:]:
            model.add(
                Dense(
                    layer_size,
                    activation=self.hidden_activation,
                    kernel_regularizer=l1l2(self.l1_penalty, self.l2_penalty),
                    kernel_initializer=self.kernel_initializer,
                )
            )

            model.add(Dropout(self.dropout_probability))

        model.add(
            Dense(
                n_dims,
                activation=self.output_activation,
                kernel_regularizer=l1l2(self.l1_penalty, self.l2_penalty),
                kernel_initializer=self.kernel_initializer,
            )
        )

        loss_function = make_reconstruction_loss(n_dims)

        model.compile(optimizer=self.optimizer, loss=loss_function)

        return model

    def fill(self, missing_mask):
        self.data[missing_mask] = -1

    def _create_missing_mask(self):

        if self.data.dtype != "f" and self.data.dtype != "d":
            self.data = self.data.astype(float)
        return np.isnan(self.data)

    def _train_epoch(self, model, missing_mask, batch_size):

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

        missing_mask = self._create_missing_mask()
        self.fill(missing_mask)
        self.model = self._create_model()

        observed_mask = ~missing_mask

        for epoch in range(train_epochs):
            X_pred = self._train_epoch(self.model, missing_mask, batch_size)
            observed_mae = masked_mae(
                X_true=self.data, X_pred=X_pred, mask=observed_mask
            )

            if epoch % 50 == 0:
                print(f"Observed MAE: {observed_mae}")

            old_weight = 1.0 - self.recurrent_weight
            self.data[missing_mask] *= old_weight
            pred_missing = X_pred[missing_mask]
            self.data[missing_mask] += self.recurrent_weight * pred_missing

        return self.data.copy()
