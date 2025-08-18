from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.decomposition import PCA
import numpy as np


class FastIterativeImputer(IterativeImputer):
    def __init__(
        self,
        estimator,
        *,
        max_iter=10,
        tol=1e-3,
        verbose=0,
        skip_complete=True,
        keep_empty_features=False,
        dynamic_estimators=True,
        **kwargs,
    ):
        super().__init__(
            estimator=estimator, max_iter=max_iter, tol=tol, verbose=verbose, **kwargs
        )

        self.skip_complete = skip_complete
        self.keep_empty_features = keep_empty_features
        self.dynamic_estimators = dynamic_estimators

    def _fit_transform_one_column(self, X, missing_mask, column_idx, estimator):
        observed_idx = np.where(~missing_mask[:, column_idx])[0]
        missing_idx = np.where(missing_mask[:, column_idx])[0]

        if len(missing_idx) == 0:
            return X[:, column_idx]  # No missing values to impute

        if len(observed_idx) == 0:  # Handle columns with no observed values
            if self.keep_empty_features:
                return np.full(X.shape[0], np.nan)
            raise ValueError(
                f"Column {column_idx} has no observed samples for imputation."
            )

        X_train = np.delete(X, column_idx, axis=1)[observed_idx]
        y_train = X[observed_idx, column_idx]
        X_test = np.delete(X, column_idx, axis=1)[missing_idx]

        # Dynamically adjust n_estimators to speed up early iterations
        if self.dynamic_estimators:
            # Gradually increase number of trees with each iteration.
            n_estimators = min(50, 10 + 2 * column_idx)
            estimator.set_params(n_estimators=n_estimators)

        # Fit the estimator and predict missing values
        estimator.fit(X_train, y_train)
        imputed_values = estimator.predict(X_test)

        # Impute the values directly into X
        X[missing_idx, column_idx] = imputed_values
        return X[:, column_idx]

    def fit_transform(self, X, y=None):
        X = np.asarray(X, dtype=float)
        missing_mask = np.isnan(X)

        for iteration in range(self.max_iter):
            if self.verbose:
                print(f"Iteration {iteration + 1}/{self.max_iter}")

            X_last = X.copy()

            # Sequentially impute each column, skipping complete columns if specified
            for col in range(X.shape[1]):
                if self.skip_complete and not np.any(missing_mask[:, col]):
                    continue
                X[:, col] = self._fit_transform_one_column(
                    X, missing_mask, col, self.estimator
                )

            # Check for convergence
            if np.linalg.norm(X - X_last, ord="fro") < self.tol:
                if self.verbose:
                    print("Convergence reached.")
                break

        return X
