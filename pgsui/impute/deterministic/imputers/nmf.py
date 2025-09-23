from pathlib import Path
from typing import Dict, List

# Third-party imports
import numpy as np
import pandas as pd


class ImputeNMF:
    """Impute missing data using matrix factorization. If ``by_populations=False`` then imputation is by global allele frequency. If ``by_populations=True`` then imputation is by population-wise allele frequency.

    Args:
        genotype_data (GenotypeData object or None, optional): GenotypeData instance.
        latent_features (float, optional): The number of latent variables used to reduce dimensionality of the data. Defaults to 2.
        learning_rate (float, optional): The learning rate for the optimizers. Adjust if the loss is learning too slowly. Defaults to 0.1.
        tol (float, optional): Tolerance of the stopping condition. Defaults to 1e-3.
        missing (int, optional): Missing data value. Defaults to -9.
        prefix (str, optional): Prefix for writing output files. Defaults to "output".
        verbose (bool, optional): Whether to print status updates. Set to False for no status updates. Defaults to True.
        **kwargs (Dict[str, bool | List[List[int]] | None | float | int | str]): Additional keyword arguments to supply. Primarily for internal purposes. Options include: {"iterative_mode": bool, "validation_mode": bool, "gt": List[List[int]]}. "iterative_mode" determines whether ``ImputeAlleleFreq`` is being used as the initial imputer in ``IterativeImputer``. "gt" is used internally for the simple imputers during grid searches and validation. If ``genotype_data is None`` then ``gt`` cannot also be None, and vice versa. Only one of ``gt`` or ``genotype_data`` can be set.

    Attributes:
        imputed (GenotypeData): New GenotypeData instance with imputed data.

    Example:
        >>>data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure",
        >>>    popmapfile="test.popmap",
        >>>)
        >>>
        >>>nmf = ImputeMF(
        >>>    genotype_data=data,
        >>>    by_populations=True,
        >>>)
        >>>
        >>> # Get GenotypeData instance.
        >>>gd_nmf = nmf.imputed

    Raises:
        TypeError: genotype_data cannot be NoneType.
    """

    def __init__(
        self,
        genotype_data,
        *,
        latent_features: int = 2,
        max_iter: int = 100,
        learning_rate: float = 0.0002,
        regularization_param: float = 0.02,
        tol: float = 0.1,
        n_fail: int = 20,
        missing: int = -9,
        prefix: str = "imputer",
        verbose: bool = True,
        **kwargs: Dict[str, bool | List[List[int]] | None | float | int | str],
    ) -> None:
        self.max_iter = max_iter
        self.latent_features = latent_features
        self.n_fail = n_fail
        self.learning_rate = learning_rate
        self.tol = tol
        self.regularization_param = regularization_param
        self.missing = missing
        self.prefix = prefix
        self.verbose = verbose
        self.iterative_mode = kwargs.get("iterative_mode", False)
        self.validation_mode = kwargs.get("validation_mode", False)

        gt = kwargs.get("gt", None)

        if genotype_data is None and gt is None:
            raise TypeError("GenotypeData and gt cannot both be NoneType.")

        if gt is None:
            X = genotype_data.genotypes_012(fmt="numpy")
        else:
            X = gt.copy()
        imputed012 = pd.DataFrame(self.fit_predict(X))
        genotype_data = genotype_data.copy()
        genotype_data.snp_data = genotype_data.decode_012(
            imputed012, prefix=prefix, write_output=False
        )

        if self.validation_mode:
            self.imputed = imputed012.to_numpy()
        else:
            self.imputed = genotype_data

    @property
    def genotypes_012(self):
        return self.imputed.genotypes012

    @property
    def snp_data(self):
        return self.imputed.snp_data

    @property
    def alignment(self):
        return self.imputed.alignment

    def fit_predict(self, X):
        # imputation
        if self.verbose:
            print(f"Doing MF imputation...")
        R = X
        R = R.astype(int)
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

        # transform values per-column (i.e., only allowing values found in original)
        tR = self.transform(R, nR)

        # get accuracy of re-constructing non-missing genotypes
        accuracy = self.accuracy(X, tR)

        # insert imputed values for missing genotypes
        fR = X
        fR[X < 0] = tR[X < 0]

        if self.verbose:
            print("Done!")

        return fR

    def transform(self, original, predicted):
        n_row = len(original)
        n_col = len(original[0])
        tR = predicted
        for j in range(n_col):
            observed = predicted[:, j]
            expected = original[:, j]
            options = np.unique(expected[expected != 0])
            for i in range(n_row):
                transform = min(options, key=lambda x: abs(x - predicted[i, j]))
                tR[i, j] = transform
        tR = tR - 1
        tR[tR < 0] = -9
        return tR

    def accuracy(self, expected, predicted):
        prop_same = np.sum(expected[expected >= 0] == predicted[expected >= 0])
        tot = expected[expected >= 0].size
        accuracy = prop_same / tot
        return accuracy

    def write2file(
        self, X: pd.DataFrame | np.ndarray | List[List[int | float]]
    ) -> None:
        """Write imputed data to file on disk.

        Args:
            X (pandas.DataFrame | numpy.ndarray | List[List[int | float]]): Imputed data to write to file.

        Raises:
            TypeError: If X is of unsupported type.
        """
        outfile = Path(
            f"{self.prefix}_output",
            "alignments",
            "Deterministic",
            "ImputeMF",
        )

        Path(outfile).mkdir(parents=True, exist_ok=True)
        outfile = Path(outfile) / "imputed_012.csv"

        if isinstance(X, pd.DataFrame):
            df = X
        elif isinstance(X, (np.ndarray, list)):
            df = pd.DataFrame(X)
        else:
            raise TypeError(
                f"Could not write imputed data because it is of incorrect "
                f"type. Got {type(X)}"
            )

        df.to_csv(outfile, header=False, index=False)
