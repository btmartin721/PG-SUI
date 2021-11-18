# Standard Library Imports

import gc
import os
import random
import sys
import math

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


def get_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def validate_batch_size(X, batch_size):
    if batch_size > X.shape[0]:
        while batch_size > X.shape[0]:
            print(
                "Batch size is larger than the number of samples. "
                "Dividing batch_size by 2."
            )
            batch_size //= 2
    return batch_size


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

        self.batch_size = validate_batch_size(X, batch_size)

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
        batch_size=32,
        n_components=3,
        num_hidden_layers=3,
        hidden_layer_sizes="midpoint",
        optimizer="adam",
        hidden_activation="elu",
        kernel_initializer="glorot_normal",
        **kwargs,
    ):

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
        self.n_components = n_components
        self.hidden_layer_sizes = hidden_layer_sizes
        self.optimizer = optimizer
        self.hidden_activation = hidden_activation
        self.kernel_initializer = kernel_initializer

        # Get number of hidden layers
        self.df = None
        self.data = None

        if genotype_data is None:
            raise TypeError(
                "genotype_data is a required argument, but was not supplied"
            )

        if test_cat is not None:
            self.X = test_cat.copy()
        else:
            self.X = genotype_data.genotypes_nparray

        (
            self.hidden_layer_sizes,
            num_hidden_layers,
        ) = self._validate_hidden_layers(hidden_layer_sizes, num_hidden_layers)

        self.hidden_layer_sizes = self._get_hidden_layer_sizes(
            self.X.shape[1], self.n_components, self.hidden_layer_sizes
        )

        self.batch_size = validate_batch_size(self.X, batch_size)

        self.V_latent = self._init_weights(self.X.shape[0], self.n_components)

        self.v = tf.Variable(
            tf.random.normal([self.batch_size, self.n_components], stddev=0.01),
        )
        self.v_bias = tf.Variable(tf.zeros([1, self.n_components]))

        self.vb = tf.Variable(tf.zeros([1, self.n_components]))

        self.w2 = tf.Variable(
            tf.random.normal(
                [self.n_components, self.hidden_layer_sizes[0]],
                stddev=0.01,
            ),
            name="w2",
        )

        self.b2 = tf.Variable(tf.zeros([1, self.hidden_layer_sizes[0]]))

        self.w3 = tf.Variable(
            tf.random.normal([self.hidden_layer_sizes[0], 3], stddev=0.01),
            name="w3",
        )

        self.b3 = tf.Variable(tf.zeros([1, 3]))

        # Collect all initialized weights and biases in self.params
        self.params = [self.v, self.vb, self.w2, self.b2, self.w3, self.b3]

        self.observed_mask = None

        # self.U = self._init_weights(reduced_dimensions, X.shape[1])

        # Get initial weights for single layer perceptron.
        # self.T = self._init_weights(X.shape[0], reduced_dimensions)

        self.num_total_epochs = 0

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

        self.imputed_df = self.fit_predict(self.X)
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
        # self.data = X.copy()
        imputed_enc = self._train()

        # imputed_enc, dummy_df = self._eval_predictions(X, imputed_enc)

        # imputed_df = self._decode_onehot(
        #     pd.DataFrame(data=imputed_enc, columns=dummy_df.columns)
        # )

        # return imputed_df

    def forward(self, X):
        """Forward pass of the network

        Args:
            X (np.ndarray): Input data.

        Returns:
            (np.ndarray): Predicted labels.
        """
        X_tf = tf.cast(X, dtype=tf.float32)
        Z1 = tf.matmul(X_tf, self.v) + self.vb
        Z1 = tf.nn.relu(Z1)
        Z2 = tf.matmul(Z1, self.w2) + self.b2
        Z2 = tf.nn.relu(Z2)
        Z3 = tf.matmul(Z2, self.w3) + self.b3
        return Z3

    def loss(self, y_true, logits):
        """Calculate loss during gradient descent.

        Args:
            y_true (tf.Tensor): Tensor of shape (batch_size, size_output).
            logits ([tf.Tensor]): Tensor of shape (batch_size, size_output).
        """
        y_true_tf = tf.cast(tf.reshape(y_true, (-1, 1)), dtype=tf.float32)
        logits_tf = tf.cast(tf.reshape(y_pred, (-1, 1)), dtype=tf.float32)
        return tf.compat.v1.losses.softmax_cross_entropy(y_true_tf, logits_tf)
        # return tf.reduce_mean(loss)

    def backpropagate(self, x, y, eta):
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(
            learning_rate=eta
        )
        with tf.GradientTape() as tape:
            predicted = self.forward(x)
            current_loss = self.loss(y, predicted)
        grads = tape.gradient(current_loss, self.params)
        optimizer.apply_gradients(
            zip(grads, self.params),
            global_step=tf.compat.v1.train.get_or_create_global_step(),
        )

        return self.v.assign(grads[0]), current_loss.numpy()

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

        missing_mask = self._create_missing_mask()
        self.observed_mask = ~missing_mask
        self._fill(missing_mask)

        model = self._build_ubp()
        # Define neural network model.
        # model = self._create_ubp_model()

        self.initialise_parameters()

        # Number of batches based on rows in X and V_latent.
        n_batches = int(np.ceil(self.data.shape[0] / self.batch_size))

        while self.current_eta > self.target_eta:
            # While stopping criterion not met.

            epoch_train_loss = 0

            s = self._train_epoch(model, n_batches, epoch_train_loss)

            self.num_epochs += 1

            if self.num_epochs % 50 == 0:
                print(f"Epoch {self.num_epochs}...")
                print(f"Current MSE: {s}")
                print(f"Current Learning Rate: {self.current_eta}")

            if self.num_epochs == 1:
                print(f"Beginning UBP training...\n")
                print(f"Initial MSE: {s}")
                print(f"Initial Learning Rate: {self.current_eta}")

            self.current_eta /= 2

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

    def _train_epoch(self, model, n_batches, epoch_train_loss, num_classes=3):

        # Randomize the order the of the samples in the batches.
        indices = np.arange(self.data.shape[0])
        np.random.shuffle(indices)

        for batch_idx in range(n_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = (batch_idx + 1) * self.batch_size
            x_batch = self.data[batch_start:batch_end, :]
            v_batch = self.V_latent[batch_start:batch_end, :]

            self.v.assign(v_batch)

            opt = tf.keras.optimizers.Adam(learning_rate=self.current_eta)
            with tf.GradientTape() as tape:
                pred = model(self.v)
                loss = tf.keras.losses.categorical_crossentropy(x_batch, pred)

            # calculate the gradients using our tape and then update the
            # model weights
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

            self.V_latent[batch_start:batch_end, :] = model.layers[
                0
            ].get_weights()[0]
            sys.exit()

            # if batch_idx == 0:
            #     model.get_layer("v").set_weights([v_batch, self.v_bias])

            # self.v.assign(v_batch)

            # v_tmp = model.get_layer("v").get_weights()[0]
            # print(v_tmp)
            # print(v_tmp.shape)
            # sys.exit()

            # model.train_on_batch(x_batch, x_batch)

            # self.V_latent[batch_start:batch_end, :] = np.reshape(
            #     v_tmp, (self.batch_size, self.n_components)
            # )

        x_pred = model.predict(self.data)
        x_true = self.data

        s = mse(y_true=x_true, y_pred=x_pred)
        return s

        # (
        #     self.V_latent[batch_start:batch_end, :],
        #     batch_loss,
        # ) = self.backpropagate(v_batch, x_batch, self.current_eta)

        # self.v.assign(v_batch)
        # model.trainable_variables[0] = self.v
        # model.train_on_batch(x_batch, x_batch)
        # self.v.assign(model.layers[0].get_weights()[0])

    def _build_ubp(self, num_classes=3):
        """Create and train a UBP neural network model.

        Creates a network with the following structure:

        InputLayer -> DenseLayer1 -> ActivationFunction1 -> ... -> DenseLayerN -> SoftmaxActivation -> OutputLayer

        Args:
            num_classes (int, optional): The number of classes in the vector. Defaults to 3.

        Returns:
            keras model object: Compiled Keras model.
        """
        # Build the model
        model = Sequential(
            [
                tf.keras.layers.Dense(self.hidden_layer_sizes[0]),
                tf.keras.layers.Activation("elu"),
                tf.keras.layers.Dense(self.data.shape[1]),
                tf.keras.layers.Softmax(axis=1),
            ]
        )
        return model
        # We need some layers. If we are implementing matrix
        # factorization, we want exactly one dense layer.
        # So in that case, X_hat = v * w, where w is the weights of that one
        # dense layer. If we are implementing NLPCA or UBP, then we should add
        # more layers.

    def _convert_tensor(self, outputs):
        return outputs

    def _create_ubp_model(self, num_classes=3):
        """Create and train a UBP neural network model.

        Creates a network with the following structure:

        InputLayer -> DenseLayer1 -> ActivationFunction1 -> ... -> DenseLayerN -> SoftmaxActivation -> OutputLayer

        Args:
            num_classes (int, optional): The number of classes in the vector. Defaults to 3.

        Returns:
            keras model object: Compiled Keras model.
        """
        # We need some layers. If we are implementing matrix
        # factorization, we want exactly one dense layer.
        # So in that case, X_hat = v * w, where w is the weights of that one
        # dense layer. If we are implementing NLPCA or UBP, then we should add
        # more layers.

        # # Specify the input layer.
        # v_in = tf.keras.Input((self.V_latent.shape[1],))
        inp = tf.keras.Input(shape=(self.data.shape[1], num_classes))
        flat = tf.keras.layers.Flatten()(inp)
        v = tf.keras.layers.Dense(self.n_components, name="v")(flat)
        v1 = tf.keras.layers.Activation(self.hidden_activation)(v)
        h1 = tf.keras.layers.Dense(self.data.shape[1], name="h1")(v1)
        h2 = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1))(h1)
        x_hat = tf.keras.layers.Dense(num_classes, name="x_hat")(h2)
        out = tf.keras.layers.Softmax(axis=1)(x_hat)

        model = tf.keras.Model(inputs=inp, outputs=out)
        model.compile(loss="categorical_crossentropy", optimizer=self.optimizer)
        return model

        # v1 = tf.keras.layers.Dense(self.V_latent.shape[1], name="v")(v_in)
        # v2 = tf.keras.layers.Activation(self.hidden_activation)(v1)

        # # Here we loop through and dynamically add hidden layers.
        # # This allows users to set a varying number of hidden layers
        # # for tuning.
        # h1 = tf.keras.layers.Dense(self.hidden_layer_sizes[0], name="Hidden2")(
        #     v2
        # )

        # h2 = tf.keras.layers.Activation(self.hidden_activation)(h1)
        # x_hat = tf.keras.layers.Dense(num_classes, name="x_hat")(h2)
        # output = tf.keras.layers.Softmax(axis=1)(x_hat)

        # model = tf.keras.Model(inputs=v_in, outputs=output)

        # model.compile(optimizer=self.optimizer, loss="categorical_crossentropy")

        # return model

        # In this case, x = f(v, w), where f is the multi-layer perceptron
        # with weights "w". "hidden_units" is some value I picked arbitrarily.
        # Will need to do some experimenting with this value. As a general rule
        # of thumb, it should probably be about halfway between t and d.

    @property
    def imputed(self):
        return self.imputed_df

    def _validate_hidden_layers(self, hidden_layer_sizes, num_hidden_layers):
        if isinstance(hidden_layer_sizes, (str, int)):
            hidden_layer_sizes = [hidden_layer_sizes] * num_hidden_layers

        # If not all integers
        elif isinstance(hidden_layer_sizes, list):
            if not all([isinstance(x, (str, int)) for x in hidden_layer_sizes]):
                ls = list(set([type(item) for item in hidden_layer_sizes]))
                raise TypeError(
                    f"Variable hidden_layer_sizes must either be None, "
                    f"an integer or string, or a list of integers or "
                    f"strings, but got the following type(s): {ls}"
                )

        else:
            raise TypeError(
                f"Variable hidden_layer_sizes must either be, "
                f"an integer, a string, or a list of integers or strings, "
                f"but got the following type: {type(hidden_layer_sizes)}"
            )

        assert (
            num_hidden_layers == len(hidden_layer_sizes)
            and num_hidden_layers > 0
        ), "num_hidden_layers must be of the same length as hidden_layer_sizes."

        return hidden_layer_sizes, num_hidden_layers

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

    def _get_hidden_layer_sizes(self, n_dims, n_components, hl_func):
        """Get dimensions of hidden layers.

        Args:
            n_dims (int): The number of feature dimensions (columns) (d).
            n_components (int): The number of reduced dimensions (t).
            hl_func (str): The function to use to calculate the hidden layer sizes. Possible options: "midpoint", "sqrt", "log2".

        Returns:
            [int, int, int]: [Number of dimensions in hidden layers]
        """
        layers = list()
        if not isinstance(hl_func, list):
            raise TypeError("hl_func must be of type list.")

        for func in hl_func:
            if func == "midpoint":
                units = round((n_dims + n_components) / 2)
            elif func == "sqrt":
                units = round(math.sqrt(n_dims))
            elif func == "log2":
                units = round(math.log(n_dims, 2))
            elif isinstance(func, int):
                units = func
            else:
                raise ValueError(
                    f"hidden_layer_sizes must be either integers or any of "
                    f"the following strings: 'midpoint', "
                    f"'sqrt', or 'log2', but got {func} of type {type(func)}"
                )

            assert units > 0 and units < n_dims, (
                f"The hidden layer sizes must be > 0 and < the number of "
                f"features (i.e., columns) in the dataset, but size was {units}"
            )

            layers.append(units)
        return layers

    def _fill(self, missing_mask):
        """Mask missing data as [0, 0, 0].

        Args:
            missing_mask ([np.ndarray(bool)]): [Missing data mask with True corresponding to a missing value]
        """
        self.data[missing_mask] = [0, 0, 0]

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


class VLayer(tf.keras.layers.Layer):
    def __init__(self, batch_size, n_dims):
        self.batch_size = batch_size
        self.n_dims = n_dims
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(batch_size, n_dims), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros.initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(n_dims,), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def compute_output_shape(self):
        return self.output_dim
