# Standard Library Imports

import os
import random
import sys
import gc

from collections import defaultdict

# Third-party Imports
import numpy as np
import pandas as pd
from scipy import stats as st

import matplotlib.pylab as plt
import seaborn as sns

from keras.utils import np_utils
from keras.objectives import mse
from keras.models import Sequential
from keras.layers.core import Dropout, Dense
from keras.regularizers import l1_l2 as l1l2

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


class ImputeVAE(GenotypeData, Impute):
    def __init__(
        self,
        *,
        genotype_data=None,
        gt=None,
        prefix="imputed_VAE",
        recurrent_weight=0.5,
        optimizer="adam",
        dropout_probability=0.5,
        hidden_activation="relu",
        output_activation="sigmoid",
        kernel_initializer="glorot_normal",
        l1_penalty=0,
        l2_penalty=0,
        cv=5,
        validation_only=0.4,
        scoring_metric="accuracy",
        disable_progressbar=False,
    ):

        super().__init__()

        self.clf = "VAE"
        self.clf_type = "classifier"

        self.prefix = prefix

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
        self.scoring_metric = scoring_metric
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

        if self.validation_only is not None:
            print("Estimating validation scores...")
            self.df_scores = self._imputer_validation(pd.DataFrame(X))
            print("\nDone!\n")

        else:
            self.df_scores = None

        print("Imputing full dataset...")
        self.imputed_df = self.fit_predict(X)
        print("Done!")

        self.write_imputed(imputed_df)

        if df_scores is not None:
            print(df_scores)

    @timer
    def fit_predict(self, X):
        self.df = self._encode_onehot(X)
        self.data = self.df.copy().values

        imputed_enc = self.train(train_epochs=300, batch_size=2)
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

        # prob_missing = 0.1
        # df_incomplete = df.copy()
        # ix = [
        #     (row, col)
        #     for row in range(df.shape[0])
        #     for col in range(df.shape[1])
        # ]

        # for row, col in random.sample(ix, int(round(prob_missing * len(ix)))):
        #     df_incomplete.iat[row, col] = np.nan

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
        np.nan_to_num(X, copy=False, nan=-9.0)
        X = X.astype(str)
        X[X == "-9.0"] = "none"

        df = pd.DataFrame(X)
        df_incomplete = df.copy()

        # Replace 'none' with np.nan
        for row in df.index:
            for col in df.columns:
                if df_incomplete.iat[row, col] == "none":
                    df_incomplete.iat[row, col] = np.nan

        return df_incomplete

    def _encode_onehot(self, X):

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

    def _imputer_validation(self, df):

        reps = defaultdict(list)
        for cnt, rep in enumerate(
            progressbar(
                range(self.cv),
                desc="Validation replicates: ",
                leave=True,
                disable=self.disable_progressbar,
            ),
            start=1,
        ):

            if self.disable_progressbar:
                perc = int((cnt / self.cv) * 100)
                print(f"Validation replicate {cnt}/{self.cv} ({perc}%)")

            df_known, df_valid, cols = self._defile_dataset(
                df, col_selection_rate=self.validation_only
            )

            df_known_slice = df_known[cols]
            df_valid_slice = df_valid[cols]

            df_stg = df_valid.copy()
            df_imp = self.fit_predict(df_stg.to_numpy())
            df_imp = df_imp.astype(np.float)

            # Get score of each column
            scores = defaultdict(list)
            for i in range(len(df_known_slice.columns)):
                # Adapted from: https://medium.com/analytics-vidhya/using-scikit-learns-iterative-imputer-694c3cca34de

                y_true = df_known[df_known.columns[i]]
                y_pred = df_imp[df_imp.columns[i]]

                scores["accuracy"].append(
                    metrics.accuracy_score(y_true, y_pred)
                )

                scores["precision"].append(
                    metrics.precision_score(
                        y_true, y_pred, average="macro", zero_division=0
                    )
                )

                scores["f1"].append(
                    metrics.f1_score(
                        y_true, y_pred, average="macro", zero_division=0
                    )
                )

                scores["recall"].append(
                    metrics.recall_score(
                        y_true, y_pred, average="macro", zero_division=0
                    )
                )

                scores["jaccard"].append(
                    metrics.jaccard_score(
                        y_true, y_pred, average="macro", zero_division=0
                    )
                )

            lst2del = [
                df_stg,
                df_imp,
                df_known,
                df_known_slice,
                df_valid_slice,
                df_valid,
            ]
            del lst2del
            del cols
            gc.collect()

            for k, score_list in scores.items():
                score_list_filtered = filter(lambda x: x != -9, score_list)

                if score_list_filtered:
                    reps[k].append(score_list_filtered)
                else:
                    continue

        if not reps:
            raise ValueError("None of the features could be validated!")

        ci_lower = dict()
        ci_upper = dict()
        for k, v in reps.items():
            reps_t = np.array(v).T.tolist()

            cis = list()
            if len(reps_t) > 1:
                for rep in reps_t:
                    rep = [abs(x) for x in rep]

                    cis.append(
                        st.t.interval(
                            alpha=0.95,
                            df=len(rep) - 1,
                            loc=np.mean(rep),
                            scale=st.sem(rep),
                        )
                    )

                ci_lower[k] = mean(x[0] for x in cis)
                ci_upper[k] = mean(x[1] for x in cis)
            else:
                print(
                    "Warning: Only one replicate was useful; skipping "
                    "calculation of 95% CI"
                )

                ci_lower[k] = np.nan
                ci_upper[k] = np.nan

        results_list = list()
        for k, score_list in scores.items():
            avg_score = mean(abs(x) for x in score_list if x != -9)
            median_score = median(abs(x) for x in score_list if x != -9)
            max_score = max(abs(x) for x in score_list if x != -9)
            min_score = min(abs(x) for x in score_list if x != -9)

            results_list.append(
                {
                    "Metric": k,
                    "Mean": avg_score,
                    "Median": median_score,
                    "Min": min_score,
                    "Max": max_score,
                    "Lower 95% CI": ci_lower[k],
                    "Upper 95% CI": ci_upper[k],
                }
            )

        df_scores = pd.DataFrame(results_list)

        columns_list = [
            "Mean",
            "Median",
            "Min",
            "Max",
            "Lower 95% CI",
            "Upper 95% CI",
        ]

        df_scores = df_scores.round(2)

        outfile = f"{self.prefix}_imputed_best_score.csv"
        df_scores.to_csv(outfile, header=True, index=False)

        del results_list
        gc.collect()

        return df_scores

    def _defile_dataset(self, df, col_selection_rate=0.5):
        """[Function to select ``col_selection_rate * columns`` columns in a pandas DataFrame and change anywhere from 15% to 50% of the values in each of those columns to np.nan (missing data). Since we know the true values of the ones changed to np.nan, we can assess the accuracy of the classifier and do a grid search]

        Args:
            df ([pandas.DataFrame]): [012-encoded genotypes to extract columns from.]

            col_selection_rate (float, optional): [Proportion of the DataFrame to extract]. Defaults to 0.40.

        Returns:
            [pandas.DataFrame]: [DataFrame with newly missing values]
            [numpy.ndarray]: [Columns that were extracted via random sampling]
        """
        # Code adapted from:
        # https://medium.com/analytics-vidhya/using-scikit-learns-iterative-imputer-694c3cca34de

        cols = np.random.choice(
            df.columns, int(len(df.columns) * col_selection_rate)
        )

        # Fill in unknown values with simple imputations
        simple_imputer = SimpleImputer(strategy="most_frequent")

        df_validation = pd.DataFrame(simple_imputer.fit_transform(df))
        df_filled = df_validation.copy()

        del simple_imputer
        gc.collect()

        for col in cols:
            data_drop_rate = np.random.choice(np.arange(0.15, 0.5, 0.02), 1)[0]

            drop_ind = np.random.choice(
                np.arange(len(df_validation[col])),
                size=int(len(df_validation[col]) * data_drop_rate),
                replace=False,
            )

            # Introduce random np.nan values
            df_validation[col].iloc[drop_ind] = np.nan

        return df_filled, df_validation, cols

        prob_missing = n
        df_incomplete = df.copy()
        ix = [
            (row, col)
            for row in range(df.shape[0])
            for col in range(df.shape[1])
        ]

        for row, col in random.sample(ix, int(round(prob_missing * len(ix)))):
            df_incomplete.iat[row, col] = np.nan

    def write_imputed(self, data):
        """[Save imputed data to a CSV file]

        Args:
            data ([pandas.DataFrame, numpy.array, or list(list)]): [Object returned from impute_missing()]

        Raises:
            TypeError: [Must be of type pandas.DataFrame, numpy.array, or list]
        """
        outfile = f"{self.prefix}_imputed_012.csv"

        if isinstance(data, pd.DataFrame):
            data.to_csv(outfile, header=False, index=False)

        elif isinstance(data, np.ndarray):
            np.savetxt(outfile, data, delimiter=",")

        elif isinstance(data, list):
            with open(outfile, "w") as fout:
                fout.writelines(
                    ",".join(str(j) for j in i) + "\n" for i in data
                )
        else:
            raise TypeError(
                "'write_imputed()' takes either a pandas.DataFrame,"
                " numpy.ndarray, or 2-dimensional list"
            )

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
