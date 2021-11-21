# Standard Library Imports
import sys
import math
from collections import defaultdict

# Third-party Imports
import numpy as np
import pandas as pd

# For development purposes
from memory_profiler import memory_usage

# Plotting modules
import matplotlib.pylab as plt
import seaborn as sns

# Neural network imports
import tensorflow as tf
from keras import backend as K
from keras.utils import to_categorical
from keras.objectives import mse
from keras.models import Sequential
from keras.layers.core import Dropout, Dense, Lambda
from keras.regularizers import l1_l2

# Custom Modules
from impute.impute import Impute
from read_input.read_input import GenotypeData
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


def categorical_crossentropy_masked(y_true, y_pred):
    y_true_masked = tf.boolean_mask(
        y_true, tf.reduce_any(tf.not_equal(y_true, -1), axis=2)
    )
    y_pred_masked = tf.boolean_mask(
        y_pred, tf.reduce_any(tf.not_equal(y_true, -1), axis=2)
    )
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    return loss_fn(y_true_masked, y_pred_masked)


def validate_batch_size(X, batch_size):
    if batch_size > X.shape[0]:
        while batch_size > X.shape[0]:
            print(
                "Batch size is larger than the number of samples. "
                "Dividing batch_size by 2."
            )
            batch_size //= 2
    return batch_size


def _mle(row):
    """Get the Maximum Likelihood Estimation for the best prediction. Basically, it sets the index of the maxiumum value in a vector (row) to 1.0, since it is one-hot encoded.

    Args:
        row (numpy.ndarray(float)): Row vector with predicted values as floating points.

    Returns:
        numpy.ndarray(float): Row vector with the highest prediction set to 1.0 and the others set to 0.0.
    """
    res = np.zeros(row.shape[0])
    res[np.argmax(row)] = 1
    return res


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
        self.df = self._encode_onehot(X)
        self.data = self.df.copy().values

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

            mle_completed = np.apply_along_axis(_mle, axis=1, arr=col_completed)

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
        validation_only=0.3,
        disable_progressbar=False,
        batch_size=32,
        n_components=3,
        early_stopping_gen=50,
        num_hidden_layers=3,
        hidden_layer_sizes="midpoint",
        optimizer="adam",
        hidden_activation="elu",
        learning_rate=0.1,
        max_epochs=1000,
        tol=1e-3,
        weights_initializer="glorot_normal",
        l1_penalty=0.01,
        l2_penalty=0.01,
        dropout_probability=0.2,
        **kwargs,
    ):

        self.clf = "UBP"
        self.clf_type = "classifier"
        self.imp_kwargs = {
            "initial_strategy": initial_strategy,
            "genotype_data": genotype_data,
            "str_encodings": {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9},
        }

        supported_kwargs = ["test_categorical"]
        if kwargs is not None:
            for k in kwargs.keys():
                if k not in supported_kwargs:
                    raise ValueError(f"{k} is not a valid argument.")

        test_cat = kwargs.get("test_categorical", None)

        super().__init__(self.clf, self.clf_type, self.imp_kwargs)

        self.prefix = prefix

        self.cv = cv
        self.validation_only = validation_only
        self.disable_progressbar = disable_progressbar
        self.n_components = n_components
        self.early_stopping_gen = early_stopping_gen
        self.hidden_layer_sizes = hidden_layer_sizes
        self.optimizer = optimizer
        self.hidden_activation = hidden_activation
        self.initial_eta = learning_rate
        self.max_epochs = max_epochs
        self.tol = tol
        self.weights_initializer = weights_initializer
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.dropout_probability = dropout_probability

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
        self.phase2_model = list()
        self.observed_mask = None
        self.num_total_epochs = 0
        self.model = None
        self.opt = self.set_optimizer()
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()

        if self.validation_only is not None:
            print("\nEstimating validation scores...")

            self.df_scores = self._imputer_validation(
                pd.DataFrame(self.X), self.clf
            )

            print("\nDone!\n")

        else:
            self.df_scores = None

        print("\nImputing full dataset...")

        # mem_usage = memory_usage((self._impute_single, (X,)))
        # with open(f"profiling_results/memUsage_{self.prefix}.txt", "w") as fout:
        # fout.write(f"{max(mem_usage)}")
        # sys.exit()

        self.imputed_df = self.fit_predict(self.X)

        self.imputed_df = self.imputed_df.astype(np.float)
        self.imputed_df = self.imputed_df.astype("Int8")

        self._validate_imputed(self.imputed_df)

        self.write_imputed(self.imputed_df)

        if self.df_scores is not None:
            print(self.df_scores)

    def set_optimizer(self):
        if self.optimizer == "adam":
            return tf.keras.optimizers.Adam(learning_rate=self.initial_eta)
        elif self.optimizer == "sgd":
            return tf.keras.optimizers.SGD(learning_rate=self.initial_eta)
        elif self.optimizer == "adagrad":
            return tf.keras.optimizers.Adagrad(learning_rate=self.initial_eta)
        else:
            raise ValueError(
                f"Only 'adam', 'sgd', and 'adagrad' optimizers are supported, "
                f"but got {self.optimizer}."
            )

    @timer
    def fit_predict(self, input_data):
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

        self.data = self._encode_onehot(X)
        self._train()
        Xpred = self.predict(self.V_latent)
        return pd.DataFrame(Xpred)

    def predict(self, V):
        pred = self.model(self.V_latent, training=False)
        Xprob = pred.numpy()
        Xt = np.apply_along_axis(_mle, axis=2, arr=Xprob)
        Xdecoded = np.argmax(Xt, axis=2)
        Xpred = np.zeros((Xdecoded.shape[0], Xdecoded.shape[1]))
        for idx, row in enumerate(Xpred):
            imputed_vals = np.zeros(len(row))
            known_vals = np.zeros(len(row))
            imputed_idx = np.where(self.observed_mask[idx] == 0)
            known_idx = np.nonzero(self.observed_mask[idx])
            Xpred[idx, imputed_idx] = Xdecoded[idx, imputed_idx]
            Xpred[idx, known_idx] = self.X[idx, known_idx]
        return Xpred

    def _train(self):
        """Train an unsupervised backpropagation model.

        Returns:
            numpy.ndarray(float): Predicted values as numpy array.
        """
        missing_mask = self._create_missing_mask()
        self.observed_mask = ~missing_mask
        self._fill(missing_mask)

        # Define neural network models.
        model_single_layer = self._build_ubp(phase=1)
        model_mlp_phase2 = self._build_ubp(phase=2)
        model_mlp_phase3 = self._build_ubp(phase=3)

        models = [model_single_layer, model_mlp_phase2, model_mlp_phase3]

        self.initialise_parameters()

        # Number of batches based on rows in X and V_latent.
        n_batches = int(np.ceil(self.data.shape[0] / self.batch_size))

        counter = 0
        criterion_met = False
        s_delta = None

        for phase in range(1, 4):
            self.num_epochs = 0
            while (
                counter < self.early_stopping_gen
                and self.num_epochs <= self.max_epochs
            ):
                # While stopping criterion not met.

                epoch_train_loss = 0

                s = self._train_epoch(
                    models, n_batches, epoch_train_loss, phase=phase
                )

                self.num_epochs += 1

                if self.num_epochs % 50 == 0:
                    print(f"Epoch {self.num_epochs}...")
                    print(f"Current MSE: {s}")
                    print(f"Current Learning Rate: {self.current_eta}")

                if self.num_epochs == 1:
                    self.s_prime = s
                    print(f"\nBeginning UBP Phase {phase} training...\n")
                    print(f"Initial MSE: {s}")
                    print(f"Initial Learning Rate: {self.current_eta}")

                if not criterion_met and self.num_epochs > 1:
                    if s < self.s_prime:
                        s_delta = abs(self.s_prime - s)
                        if s_delta <= self.tol:
                            criterion_met = True
                        else:
                            counter = 0
                            self.s_prime = s

                    else:
                        criterion_met = True

                elif criterion_met and self.num_epochs > 1:
                    if s < self.s_prime:
                        s_delta = abs(self.s_prime - s)
                        if s_delta > self.tol:
                            criterion_met = False
                            self.s_prime = s
                            counter = 0
                        else:
                            counter += 1
                            if counter == self.early_stopping_gen:
                                counter = 0
                                self.s = s
                                break
                    else:
                        counter += 1
                        if counter == self.early_stopping_gen:
                            counter = 0
                            self.s = s
                            break

            print(f"Number of epochs used to train: {self.num_epochs}")
            print(f"Final MSE: {self.s}")
            print(f"s_delta: {s_delta}")

        self.model = models[2]

    def _train_epoch(
        self, models, n_batches, epoch_train_loss, phase=3, num_classes=3
    ):

        # Randomize the order the of the samples in the batches.
        indices = np.arange(self.data.shape[0])
        np.random.shuffle(indices)

        model = models[phase - 1]

        losses = list()
        for batch_idx in range(n_batches):
            if phase == 3:
                # Set the refined weights from model 2.
                model.set_weights(self.phase2_model[batch_idx])

            batch_start = batch_idx * self.batch_size
            batch_end = (batch_idx + 1) * self.batch_size
            x_batch = self.data[batch_start:batch_end, :]
            v_batch = self.V_latent[batch_start:batch_end, :]

            if phase != 2:
                self.v = tf.Variable(
                    tf.zeros([x_batch.shape[0], self.n_components]),
                    trainable=True,
                    dtype=tf.float32,
                )

            elif phase == 2:
                self.v = tf.Variable(
                    tf.zeros([x_batch.shape[0], self.n_components]),
                    trainable=False,
                    dtype=tf.float32,
                )

            self.v.assign(v_batch)

            loss, refined = self.train_on_batch(self.v, x_batch, model, phase)
            losses.append(loss.numpy())

            if phase != 2:
                self.V_latent[batch_start:batch_end, :] = refined.numpy()
            else:
                self.phase2_model.append(refined)

        return np.mean(losses)

    def train_on_batch(self, x, y, model, phase):
        """Custom training loop for neural network.

        GradientTape records the weights and watched
        variables (usually tf.Variable objects), which
        in this case are the weights and the input (x)
        (if not phase 2), during the forward pass.
        This allows us to run gradient descent during
        backpropagation to refine the watched variables.

        This function will train on a batch of samples (rows).

        Args:
            x (tf.Variable): Input tensorflow variable of shape (batch_size, n_features).

            y (tf.Variable): Target variable to calculate loss.

            model (keras.models.Sequential): Keras model to use.

            phase (int): UBP phase to run.

        Returns:
            tf.Tensor: Calculated loss of current batch.
            tf.Variable, conditional: Input refined by gradient descent. Only returned if phase != 2.
            keras.models.Sequential, conditional: Keras model with refined weights. Only returned if phase == 2.

        """
        if phase != 2:
            src = [x]

        with tf.GradientTape() as tape:
            # Forward pass
            if phase != 2:
                tape.watch(x)
            pred = model(x, training=True)
            loss = categorical_crossentropy_masked(
                tf.convert_to_tensor(y, dtype=tf.float32), pred
            )

        if phase != 2:
            # Phase == 1 or 3.
            src.extend(model.trainable_variables)
        elif phase == 2:
            # Phase == 2.
            src = model.trainable_variables

        # Refine the watched variables with
        # gradient descent backpropagation
        gradients = tape.gradient(loss, src)
        self.opt.apply_gradients(zip(gradients, src))

        if phase != 2:
            return loss, x
        elif phase == 2:
            return loss, model.get_weights()

    def _build_ubp(self, phase=3, num_classes=3):
        """Create and train a UBP neural network model.

        If we are implementing a single layer perceptron, we want exactly one dense layer. So in that case, X_hat = v * w, where w is the weights of that one dense layer. If we are implementing NLPCA or UBP, then we should add more layers and x = f(v, w) in a multi-layer perceptron (MLP).

        Creates a network with the following structure:

        If phase > 1:
            InputLayer (V) -> DenseLayer1 -> ActivationFunction1 ... HiddenLayerN -> ActivationFunctionN ... DenseLayerN+1 -> Lambda (to expand shape) -> DenseOutputLayer -> Softmax

        If Phase == 1:
            InputLayer (V) -> DenseLayer1 -> Lambda (to expand shape) -> DenseOutputLayer -> Softmax

        Args:
            num_classes (int, optional): The number of classes in the vector. Defaults to 3.

        Returns:
            tf.keras.Model object: Compiled Keras model.
        """

        if phase == 1 or phase == 2:
            kernel_regularizer = l1_l2(self.l1_penalty, self.l2_penalty)
        elif phase == 3:
            kernel_regularizer = None
        else:
            raise ValueError(f"Phase must equal 1, 2, or 3, but got {phase}")

        if phase == 3:
            # Phase 3 uses weights from phase 2.
            kernel_initializer = None
        else:
            kernel_initializer = self.weights_initializer

        model = Sequential()

        model.add(tf.keras.Input(shape=(self.n_components,)))

        if phase > 1:
            # Construct multi-layer perceptron.
            # Add hidden layers dynamically.
            for layer_size in self.hidden_layer_sizes:
                model.add(
                    Dense(
                        layer_size,
                        activation=self.hidden_activation,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer,
                    )
                )

            model.add(
                Dense(
                    self.data.shape[1],
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                )
            )

            model.add(Lambda(lambda x: tf.expand_dims(x, -1)))
            model.add(Dense(num_classes, activation="softmax"))

        else:
            # phase == 1.
            # Construct single-layer perceptron.
            model.add(
                Dense(
                    self.data.shape[1],
                    input_shape=(self.n_components,),
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                )
            )
            model.add(Lambda(lambda x: tf.expand_dims(x, -1)))
            model.add(Dense(num_classes, activation="softmax"))

        return model

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
        ), "num_hidden_layers must be the length of hidden_layer_sizes."

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
        """Mask missing data as [-1, -1, -1].

        Args:
            missing_mask ([np.ndarray(bool)]): [Missing data mask with True corresponding to a missing value]
        """
        self.data[missing_mask] = [-1, -1, -1]

    def _create_missing_mask(self):
        """[Creates a missing data mask with boolean values]

        Returns:
            [numpy.ndarray(bool)]: [Boolean mask of missing values, with True corresponding to a missing data point]
        """
        return np.isnan(self.data).all(axis=2)

    def initialise_parameters(self):
        self.current_eta = self.initial_eta
        self.s = 0
        self.s_prime = np.inf
        self.num_epochs = 0
