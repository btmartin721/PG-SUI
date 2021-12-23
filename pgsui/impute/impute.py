# Standard library imports
import errno
import gc
import math
import os
import sys
from collections import Counter
from collections import defaultdict
from operator import itemgetter
from pathlib import Path
from statistics import mean, median
from contextlib import redirect_stdout
from typing import Optional, Union, List, Dict, Tuple, Any, Callable

# Third party imports
import numpy as np
import pandas as pd

from scipy import stats as st

# from memory_profiler import memory_usage

# Scikit-learn imports
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

import xgboost as xgb
import lightgbm as lgbm

from sklearn_genetic.space import Continuous, Categorical, Integer

# Custom module imports
try:
    from .iterative_imputer_gridsearch import IterativeImputerGridSearch
    from .iterative_imputer_fixedparams import IterativeImputerFixedParams
    from .neural_network_imputers import VAE, UBP

    from ..read_input.read_input import GenotypeData
    from . import simple_imputers
    from ..utils.misc import get_processor_name
    from ..utils.misc import isnotebook
    from ..utils.misc import timer
except (ModuleNotFoundError, ValueError):
    from impute.iterative_imputer_gridsearch import IterativeImputerGridSearch
    from impute.iterative_imputer_fixedparams import IterativeImputerFixedParams
    from impute.neural_network_imputers import VAE, UBP

    from read_input.read_input import GenotypeData
    from impute import simple_imputers
    from utils.misc import get_processor_name
    from utils.misc import get_processor_name
    from utils.misc import isnotebook
    from utils.misc import timer

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
    except ImportError:
        print(
            "Warning: Intel CPU detected but scikit-learn-intelex is not "
            "installed. We recommend installing it to speed up computation."
        )
        intelex = False
else:
    intelex = False


class Impute:
    """Class to impute missing data from the provided classifier.

    The Impute class will either run a variational autoencoder or IterativeImputer with the provided estimator. The settings for the provided estimator should be provided as the ``kwargs`` argument as a dictionary object with the estimator's keyword arguments as the keys and the corresponding values. E.g., ``kwargs={"n_jobs", 4, "initial_strategy": "populations"}``\. ``clf_type`` just specifies either "classifier" or "regressor". "regressor" is primarily just for quick and dirty testing.

    Once the Impute class is initialized, the imputation should be performed with ``fit_predict()``\.

    The imputed data can then be written to a file with ``write_imputed()``

    Args:
        clf (str or Callable estimator object): The estimator object to use. If using a variational autoencoder, the provided value should be "VAE". Otherwise, it should be a callable estimator object that is compatible with scikit-learn's IterativeImputer.

        clf_type (str): Specify whether to use a "classifier" or "regressor". The "regressor" option is just for quick and dirty testing, and "classifier" should almost always be used.

        kwargs (Dict[str, Any]): Settings to use with the estimator. The keys should be the estimator's keywords, and the values should be their corresponding settings.

    Raises:
        TypeError: Check whether the ``gridparams`` values are of the correct format if ``ga=True`` or ``ga=False``\.

    Examples:
        # Don't use parentheses after estimator object.
        >>>imputer = Impute(sklearn.ensemble.RandomForestClassifier,
        "classifier",
        {"n_jobs": 4, "initial_strategy": "populations", "max_iter": 25, "n_estimators": 100, "ga": True})
        >>>self.imputed, self.best_params = imputer.fit_predict(df)
        >>>imputer.write_imputed(self.imputed)
        >>>print(self.imputed)
        [[0, 1, 1, 2],
        [0, 1, 1, 2],
        [0, 2, 2, 2],
        [2, 2, 2, 2]]
    """

    def __init__(
        self, clf: Union[str, Callable], clf_type: str, kwargs: Dict[str, Any]
    ) -> None:
        self.clf = clf
        self.clf_type = clf_type
        self.original_num_cols = None

        try:
            self.pops = kwargs["genotype_data"].populations
        except AttributeError:
            self.pops = None

        self.genotype_data = kwargs["genotype_data"]

        # Separate local variables into settings objects
        (
            self.gridparams,
            self.imp_kwargs,
            self.clf_kwargs,
            self.ga_kwargs,
            self.grid_iter,
            self.cv,
            self.n_jobs,
            self.prefix,
            self.column_subset,
            self.ga,
            self.disable_progressbar,
            self.progress_update_percent,
            self.scoring_metric,
            self.early_stop_gen,
            self.chunk_size,
            self.validation_only,
        ) = self._gather_impute_settings(kwargs)

        if self.gridparams is not None:
            for v in self.gridparams.values():
                if (
                    isinstance(v, (Categorical, Integer, Continuous))
                    and not self.ga
                ):
                    raise TypeError(
                        "ga argument must be True if gridparams are of type sklearn_genetic.space"
                    )

        self.logfilepath = f"{self.prefix}_imputer_progress_log.txt"

        self.invalid_indexes = None

        # Remove logfile if exists
        try:
            os.remove(self.logfilepath)
        except OSError:
            pass

    @timer
    def fit_predict(
        self, X: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fit and predict imputations with IterativeImputer(estimator).

        Fits and predicts imputed 012-encoded genotypes using IterativeImputer with any of the supported estimator objects. If ``gridparams=None``\, then a grid search is not performed. If ``gridparams!=None``\, then a RandomizedSearchCV is performed on a subset of the data and a final imputation is done on the whole dataset using the best found parameters.

        Args:
            X (pandas.DataFrame): DataFrame with 012-encoded genotypes.

        Returns:
            GenotypeData: GenotypeData object with missing genotypes imputed.
            Dict[str, Any]: Best parameters found during grid search.
        """

        # Test if output file can be written to
        try:
            outfile = f"{self.prefix}_imputed_012.csv"
            with open(outfile, "w") as fout:
                pass
        except IOError as e:
            print(f"Error: {e.errno}, {e.strerror}")
            if e.errno == errno.EACCES:
                sys.exit(f"Permission denied: Cannot write to {outfile}")
            elif e.errno == errno.EISDIR:
                sys.exit(f"Could not write to {outfile}; It is a directory")

        # mem_usage = memory_usage((self._impute_single, (X,)))
        # with open(f"profiling_results/memUsage_{self.prefix}.txt", "w") as fout:
        # fout.write(f"{max(mem_usage)}")
        # sys.exit()

        # Don't do a grid search
        if self.gridparams is None:
            imputed_df, df_scores, best_params = self._impute_single(X)

            if df_scores is not None:
                self._print_scores(df_scores)

        # Do a grid search and get the transformed data with the best parameters
        else:
            imputed_df, df_scores, best_params = self._impute_gridsearch(X)

            print("Grid Search Results:")
            if self.clf_type == "regressor":
                for col in df_scores:
                    print(
                        f"{self.scoring_metric} {col} (best parameters): "
                        f"{df_scores[col].iloc[0]}"
                    )
            else:
                for col in df_scores:
                    print(
                        f"{self.scoring_metric} {col} (best parameters): "
                        f"{df_scores[col].iloc[0]}"
                    )

            print(f"Best Parameters: {best_params}\n")

        imp_data = self._imputed2genotypedata(imputed_df, self.genotype_data)

        print("\nDone!\n")
        return imp_data, best_params

    def write_imputed(
        self, data: Union[pd.DataFrame, np.ndarray, List[List[int]]]
    ) -> None:
        """Save imputed data to disk as a CSV file.

        Args:
            data (pandas.DataFrame, numpy.ndarray, or List[List[int]]): Object returned from ``fit_predict()``\.

        Raises:
            TypeError: Must be of type pandas.DataFrame, numpy.array, or List[List[int]].
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

    def read_imputed(self, filename: str) -> pd.DataFrame:
        """Read in imputed CSV file as formatted by write_imputed.

        Args:
            filename (str): Name of imputed CSV file to be read.

        Returns:
            pandas.DataFrame: Imputed data as DataFrame of 8-bit integers.
        """

        return pd.read_csv(filename, dtype="Int8", header=None)

    def _df2chunks(
        self, df: pd.DataFrame, chunk_size: Union[int, float]
    ) -> List[pd.DataFrame]:
        """Break up pandas.DataFrame into chunks and impute chunks.

        If set to 1.0 of type float, then returns only one chunk containing all the data.

        Args:
            df (pandas.DataFrame): DataFrame to split into chunks.

            chunk_size (int or float): If type is integer, then breaks DataFrame into ``chunk_size`` chunks. If type is float, breaks DataFrame up into ``chunk_size * len(df.columns)`` chunks.

        Returns:
            List[pandas.DataFrame]: List of pandas DataFrames of shape (n_samples, n_features_in_chunk).

        Raises:
            ValueError: ``chunk_size`` must be of type int or float.
        """

        if (
            self.imp_kwargs["initial_strategy"] == "phylogeny"
            and chunk_size != 1.0
        ):
            print(
                "WARNING: Chunking is not supported with initial_strategy == "
                "'phylogeny'; Setting chunk_size to 1.0 and imputing entire "
                "dataset"
            )

            chunk_size = 1.0

        if self.imp_kwargs["initial_strategy"] == "nmf" and chunk_size != 1.0:
            print(
                "WARNING: Chunking is not supported with initial_strategy == "
                "'nmf'; Setting chunk_size to 1.0 and imputing entire "
                "dataset"
            )

            chunk_size = 1.0

        if isinstance(chunk_size, (int, float)):
            chunks = list()
            df_cp = df.copy()

            if isinstance(chunk_size, float):
                if chunk_size > 1.0:
                    raise ValueError(
                        f"If chunk_size is of type float, must be "
                        f"between 0.0 and 1.0; Value supplied was {chunk_size}"
                    )

                elif chunk_size == 1.0:
                    # All data in one chunk
                    chunks.append(df_cp)
                    print(
                        "Imputing all features at once since chunk_size is "
                        "set to 1.0"
                    )

                    return chunks

                tmp = chunk_size
                chunk_size = None
                chunk_size = math.ceil(len(df.columns) * tmp)

        else:
            raise ValueError(
                f"chunk_size must be of type float or integer, "
                f"but type {type(chunk_size)} was passed"
            )

        chunk_len_list = list()
        num_chunks = math.ceil(len(df.columns) / chunk_size)
        for i in range(num_chunks):
            chunks.append(df_cp.iloc[:, i * chunk_size : (i + 1) * chunk_size])
            chunk_len_list.append(len(chunks[i].columns))

        chunk_len = ",".join([str(x) for x in chunk_len_list])

        print(f"Data split into {num_chunks} chunks with {chunk_len} features")

        return chunks

    def _imputed2genotypedata(self, imp012, genotype_data):
        """Create new instance of GenotypeData object from imputed DataFrame.

        The imputed, decoded DataFrame gets written to file and re-loaded to instantiate a new GenotypeData object.

        Args:
            imp012 (pandas.DataFrame): Imputed 012-encoded DataFrame.

            genotype_data (GenotypeData): Original GenotypeData object to load attributes from.

        Returns:
            GenotypeData: GenotypeData object with imputed data.
        """
        imputed_filename = genotype_data.decode_imputed(
            imp012, write_output=True, prefix=self.prefix
        )

        ft = genotype_data.filetype

        if ft.lower().startswith("structure") and ft.lower().endswith("row"):
            ft += "PopID"

        return GenotypeData(
            filename=imputed_filename,
            filetype=ft,
            popmapfile=genotype_data.popmapfile,
            guidetree=genotype_data.guidetree,
            qmatrix_iqtree=genotype_data.qmatrix_iqtree,
            qmatrix=genotype_data.qmatrix,
            siterates=genotype_data.siterates,
            siterates_iqtree=genotype_data.siterates_iqtree,
            verbose=False,
        )

    def _subset_data_for_gridsearch(
        self,
        df: pd.DataFrame,
        columns_to_subset: Union[int, float],
        original_num_cols: int,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Randomly subsets pandas.DataFrame.

        Subset pandas DataFrame with ``column_percent`` fraction of the data. Allows for faster validation.

        Args:
            df (pandas.DataFrame): DataFrame with 012-encoded genotypes.

            columns_to_subset (int or float): If float, proportion of DataFrame to randomly subset should be between 0 and 1. if integer, subsets ``columns_to_subset`` random columns.

            original_num_cols (int): Number of columns in original DataFrame.

        Returns:
            pandas.DataFrame: New DataFrame with random subset of features.
            numpy.ndarray: Sorted numpy array of column indices to keep.

        Raises:
            TypeError: column_subset must be of type float or int.
        """

        # Get a random numpy arrray of column names to select
        if isinstance(columns_to_subset, float):
            n = int(original_num_cols * columns_to_subset)
        elif isinstance(columns_to_subset, int):
            n = columns_to_subset
        else:
            raise TypeError(
                f"column_subset must be of type float or int, "
                f"but got {type(columns_to_subset)}"
            )

        col_arr = np.array(df.columns)

        if n > len(df.columns):
            print(
                "Warning: Column_subset is greater than remaining columns following filtering. Using all columns"
            )

            df_sub = df.copy()
            cols = col_arr.copy()
        else:
            cols = np.random.choice(col_arr, n, replace=False)
            df_sub = df.loc[:, np.sort(cols)]
            # df_sub = df.sample(n=n, axis="columns", replace=False)

        df_sub.columns = df_sub.columns.astype(str)

        return df_sub, np.sort(cols)

    def _print_scores(self, df_scores: pd.DataFrame) -> None:
        """Print validation scores as pandas.DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame with score statistics.
        """

        print("Validation scores:")
        print(df_scores)

    def _write_imputed_params_score(
        self, df_scores: pd.DataFrame, best_params: Dict[str, Any]
    ) -> None:
        """Save best_score and best_params to files on disk.

        Args:
            best_score (float): Best RMSE or accuracy score for the regressor or classifier, respectively.

            best_params (dict): Best parameters found in grid search.
        """

        best_score_outfile = f"{self.prefix}_imputed_best_score.csv"
        best_params_outfile = f"{self.prefix}_imputed_best_params.csv"

        df_scores.to_csv(
            best_score_outfile, header=True, index=False, float_format="%.2f"
        )

        with open(best_params_outfile, "w") as fout:
            fout.write("parameter,best_value\n")
            for k, v in best_params.items():
                fout.write(f"{k},{v}\n")

    def _impute_single(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, None]:
        """Run IterativeImputer without a grid search.

        Will do a different type of validation if ``validation_only != None``\.

        Args:
            df (pandas.DataFrame): DataFrame of 012-encoded genotypes.

        Returns:
            pandas.DataFrame: Imputed DataFrame of 012-encoded genotypes.
            pandas.DataFrame: DataFrame with validation scores.
            NoneType: Only used with _impute_gridsearch. Set to None here for compatibility.
        """
        print(f"Doing {self.clf.__name__} imputation without grid search...")

        if self.clf == VAE and self.clf == UBP:
            clf = None

        else:
            clf = self.clf(**self.clf_kwargs)

        if self.validation_only is not None:
            print(f"Estimating {self.clf.__name__} validation scores...")

            if self.disable_progressbar:
                with open(self.logfilepath, "a") as fout:
                    # Redirect to progress logfile
                    with redirect_stdout(fout):
                        print(
                            f"Doing {self.clf.__name__} imputation "
                            f"without grid search...\n"
                        )

                        print(
                            f"Estimating {self.clf.__name__} "
                            f"validation scores...\n"
                        )

            df_scores = self._imputer_validation(df, clf)

            print(f"\nDone with {self.clf.__name__} validation!\n")

            if self.disable_progressbar:
                with open(self.logfilepath, "a") as fout:
                    # Redirect to progress logfile
                    with redirect_stdout(fout):
                        print(f"\nDone with {self.clf.__name__} validation!\n")

        else:
            df_scores = None

        if self.clf == VAE or self.clf == UBP:
            imputer = None

        else:
            imputer = self._define_iterative_imputer(
                clf,
                self.logfilepath,
                clf_kwargs=self.clf_kwargs,
                prefix=self.prefix,
                disable_progressbar=self.disable_progressbar,
                progress_update_percent=self.progress_update_percent,
                pops=self.pops,
            )

        if self.original_num_cols is None:
            self.original_num_cols = len(df.columns)

        # Remove non-biallelic loci
        # Only used if initial_strategy == 'phylogeny'
        if self.invalid_indexes is not None:
            df.drop(
                labels=self.invalid_indexes,
                axis=1,
                inplace=True,
            )

        if self.disable_progressbar:
            with open(self.logfilepath, "a") as fout:
                # Redirect to progress logfile
                with redirect_stdout(fout):
                    print(f"Doing {self.clf.__name__} imputation...\n")

        df_chunks = self._df2chunks(df, self.chunk_size)
        imputed_df = self._impute_df(df_chunks, imputer)

        if self.disable_progressbar:
            with open(self.logfilepath, "a") as fout:
                # Redirect to progress logfile
                with redirect_stdout(fout):
                    print(f"\nDone with {self.clf.__name__} imputation!\n")

        lst2del = [df_chunks]
        del lst2del
        gc.collect()

        self._validate_imputed(imputed_df)

        print(f"\nDone with {self.clf.__name__} imputation!\n")
        return imputed_df, df_scores, None

    def _impute_gridsearch(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Do IterativeImputer with RandomizedSearchCV or GASearchCV.

        Args:
            df (pandas.DataFrame): DataFrame with 012-encoded genotypes.

        Returns:
            pandas.DataFrame: DataFrame with 012-encoded genotypes imputed using the best parameters found with the grid search.
            float: Absolute value of best score found during the grid search.
            dict: Best parameters found during the grid search.
        """
        original_num_cols = len(df.columns)
        df_subset, cols_to_keep = self._subset_data_for_gridsearch(
            df, self.column_subset, original_num_cols
        )

        print(f"Validation dataset size: {len(df_subset.columns)}\n")

        clf = self.clf()

        print(f"Doing {self.clf.__name__} grid search...")

        if self.disable_progressbar:
            with open(self.logfilepath, "a") as fout:
                # Redirect to progress logfile
                with redirect_stdout(fout):
                    print(f"Doing {self.clf.__name__} grid search...\n")

        imputer = self._define_iterative_imputer(
            clf,
            self.logfilepath,
            clf_kwargs=self.clf_kwargs,
            ga_kwargs=self.ga_kwargs,
            prefix=self.prefix,
            n_jobs=self.n_jobs,
            n_iter=self.grid_iter,
            cv=self.cv,
            clf_type=self.clf_type,
            ga=self.ga,
            search_space=self.gridparams,
            disable_progressbar=self.disable_progressbar,
            progress_update_percent=self.progress_update_percent,
            scoring_metric=self.scoring_metric,
            early_stop_gen=self.early_stop_gen,
            pops=self.pops,
        )

        if len(cols_to_keep) == original_num_cols:
            cols_to_keep = None

        Xt, params_list, score_list = imputer.fit_transform(
            df_subset, cols_to_keep
        )

        print(f"\nDone with {self.clf.__name__} grid search!")

        if self.disable_progressbar:
            with open(self.logfilepath, "a") as fout:
                # Redirect to progress logfile
                with redirect_stdout(fout):
                    print(f"\nDone with {self.clf.__name__} grid search!")

        del imputer
        del Xt

        # Average or mode of best parameters
        # and write them to a file
        best_params = self._get_best_params(params_list)

        avg_score = mean(abs(x) for x in score_list if x != -9)
        median_score = median(abs(x) for x in score_list if x != -9)
        max_score = max(abs(x) for x in score_list if x != -9)
        min_score = min(abs(x) for x in score_list if x != -9)

        df_scores = pd.DataFrame(
            {
                "Mean": avg_score,
                "Median": median_score,
                "Min": min_score,
                "Max": max_score,
            },
            index=[0],
        )

        del avg_score
        del median_score
        del max_score
        del min_score
        gc.collect()

        if self.clf_type == "classifier":
            df_scores = df_scores.apply(lambda x: x * 100)

        df_scores = df_scores.round(2)

        self._write_imputed_params_score(df_scores, best_params)

        # Change values to the ones in best_params
        self.clf_kwargs.update(best_params)

        test = self.clf()
        if hasattr(test, "n_jobs"):
            best_clf = self.clf(n_jobs=self.n_jobs, **self.clf_kwargs)
        else:
            best_clf = self.clf(**self.clf_kwargs)

        del test
        gc.collect()

        print(
            f"\nDoing {self.clf.__name__} imputation "
            f"with best found parameters...\n"
        )

        if self.disable_progressbar:
            with open(self.logfilepath, "a") as fout:
                # Redirect to progress logfile
                with redirect_stdout(fout):
                    print(
                        f"\nDoing {self.clf.__name__} imputation "
                        f"with best found parameters...\n"
                    )

        best_imputer = self._define_iterative_imputer(
            best_clf,
            self.logfilepath,
            clf_kwargs=self.clf_kwargs,
            prefix=self.prefix,
            disable_progressbar=self.disable_progressbar,
            progress_update_percent=self.progress_update_percent,
            pops=self.pops,
        )

        final_cols = None
        if len(df.columns) < original_num_cols:
            final_cols = np.array(df.columns)

        df_chunks = self._df2chunks(df, self.chunk_size)
        imputed_df = self._impute_df(
            df_chunks, best_imputer, cols_to_keep=final_cols
        )

        lst2del = [df_chunks, df]
        del lst2del
        gc.collect()

        self._validate_imputed(imputed_df)

        print(f"Done with {self.clf.__name__} imputation!\n")

        if self.disable_progressbar:
            with open(self.logfilepath, "a") as fout:
                # Redirect to progress logfile
                with redirect_stdout(fout):
                    print(f"Done with {self.clf.__name__} imputation!\n")

        return imputed_df, df_scores, best_params

    def _imputer_validation(
        self, df: pd.DataFrame, clf: Optional[Callable]
    ) -> pd.DataFrame:
        """Validate imputation with a validation test set.

        Validation imputation by running it on a validation test set ``cv`` times. Actual missing values are imputed with sklearn.impute.SimpleImputer, and then missing values are randomly introduced to known genotypes. The dataset with no missing data is compared to the dataset with known missing data to obtain validation scores.

        Args:
            df (pandas.DataFrame): 012-encoded genotypes to impute.

            clf (sklearn classifier instance or None): sklearn classifier instance with which to run the imputation.

        Raises:
            ValueError: If none of the scores were able to be estimated and reps variable is empty.

        Returns:
            pandas.DataFrame: Validation scores in a pandas DataFrame object. Contains the scoring metric, mean, median, minimum, and maximum validation scores among all features, and the lower and upper 95% confidence interval among the replicates.
        """
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

                with open(self.logfilepath, "a") as fout:
                    # Redirect to progress logfile
                    with redirect_stdout(fout):
                        print(f"Validation replicate {cnt}/{self.cv} ({perc}%)")

            scores = self._impute_eval(df, clf)

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
                    "Warning: There was no variance among replicates; "
                    "the 95% CI could not be calculated"
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

        if self.clf_type == "classifier":
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

    def _impute_df(
        self,
        df_chunks: List[pd.DataFrame],
        imputer: Optional[
            Union[IterativeImputerFixedParams, IterativeImputerGridSearch]
        ],
        cols_to_keep: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Impute list of pandas.DataFrame objects using custom IterativeImputer class.

        The DataFrames are chunks of the whole input data, with each chunk correspoding to ``chunk_size`` features from ``_df2chunks()``\.

        Args:
            df_chunks (List[pandas.DataFrame]): List of Dataframes of shape(n_samples, n_features_in_chunk).

            imputer (IterativeImputerFixedParams instance or None): Defined IterativeImputerFixedParams instance to perform the imputation.

            cols_to_keep (numpy.ndarray): Final bi-allelic columns to keep. If some columns were non-biallelic, it will be a subset of columns.

        Returns:
            pandas.DataFrame: Single DataFrame object, with all the imputed chunks concatenated together.
        """
        imputed_chunks = list()
        num_chunks = len(df_chunks)
        for i, Xchunk in enumerate(df_chunks, start=1):
            if self.clf_type == "classifier":
                if self.clf == VAE or self.clf == UBP:
                    imputer = self.clf(**self.clf_kwargs)
                    if self.clf == UBP:
                        df_imp = pd.DataFrame(
                            imputer.fit_transform(Xchunk),
                            dtype="Int8",
                        )
                    else:
                        df_imp = pd.DataFrame(imputer.fit_transform(Xchunk))
                        df_imp = df_imp.astype("float")
                        df_imp = df_imp.astype("Int8")

                else:
                    df_imp = pd.DataFrame(
                        imputer.fit_transform(Xchunk, valid_cols=cols_to_keep),
                        dtype="Int8",
                    )

                imputed_chunks.append(df_imp)

            else:
                # Regressor. Needs to be rounded to integer first.
                df_imp = pd.DataFrame(
                    imputer.fit_transform(
                        Xchunk,
                        valid_cols=cols_to_keep,
                    )
                )
                df_imp = df_imp.round(0).astype("Int8")

                imputed_chunks.append(df_imp)

        concat_df = pd.concat(imputed_chunks, axis=1)

        del imputed_chunks
        gc.collect()

        return concat_df

    def _validate_imputed(self, df: pd.DataFrame) -> None:
        """Asserts that there is no missing data left in the imputed DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame with imputed 012-encoded genotypes.

        Raises:
            AssertionError: Error if missing values are still found in the dataset after imputation.
        """
        assert (
            not df.isnull().values.any()
        ), "Imputation failed...Missing values found in the imputed dataset"

    def _get_best_params(
        self, params_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """[Gets the best parameters from the grid search. Determines the parameter types and either gets the mean or mode if the type is numeric or string/ boolean]

        Args:
            params_list (List[dict]): List of grid search parameter values.

        Returns:
            Dict[str, Any]: Dictionary with parameters as keys and their best values.
        """
        best_params = dict()
        keys = list(params_list[0].keys())
        first_key = keys[0]

        params_list = list(filter(lambda i: i[first_key] != -9, params_list))

        for k in keys:
            if all(isinstance(x[k], (int, float)) for x in params_list if x[k]):
                if all(isinstance(y[k], int) for y in params_list):
                    best_params[k] = self._average_list_of_dicts(
                        params_list, k, is_int=True
                    )

                elif all(isinstance(z[k], float) for z in params_list):
                    best_params[k] = self._average_list_of_dicts(params_list, k)

            elif all(isinstance(x[k], (str, bool)) for x in params_list):
                best_params[k] = self._mode_list_of_dicts(params_list, k)

            else:
                best_params[k] = self._mode_list_of_dicts(params_list, k)

        return best_params

    def _mode_list_of_dicts(
        self, l: List[Dict[str, Union[str, bool]]], k: str
    ) -> str:
        """Get mode for key k in a list of dictionaries.

        Args:
            l (list(dict)): List of dictionaries.
            k (str): Key to find the mode across all dictionaries in l.

        Returns:
            str or bool: Most common value across list of dictionaries for one key.
        """
        k_count = Counter(map(itemgetter(k), l))
        return k_count.most_common()[0][0]

    def _average_list_of_dicts(
        self,
        l: List[Dict[str, Union[int, float]]],
        k: str,
        is_int: bool = False,
    ) -> Union[int, float]:
        """Get average of a given key in a list of dictionaries.

        Args:
            l (List[Dict[str, Union[int, float]]]): List of dictionaries.

            k (str): Key to find average across list of dictionaries.

            is_int (bool, optional): Whether or not the value for key k is an integer. If False, it is expected to be of type float. Defaults to False.

        Returns:
            int or float: average of given key across list of dictionaries.
        """
        if is_int:
            return int(sum(d[k] for d in l) / len(l))
        else:
            return sum(d[k] for d in l) / len(l)

    def _remove_nonbiallelic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove non-biallelic sites from pandas.DataFrame.

        Remove sites that do not have both 0 and 2 encoded values in a column and if any of the allele counts is less than the number of cross-validation folds.

        Args:
            df (pandas.DataFrame): DataFrame with 012-encoded genotypes.

        Returns:
            pandas.DataFrame: DataFrame with non-biallelic sites dropped.
        """
        df_cp = df.copy()
        bad_cols = list()
        if pd.__version__[0] == 0:
            for col in df_cp.columns:
                if (
                    not df_cp[col].isin([0.0]).any()
                    or not df_cp[col].isin([2.0]).any()
                ):
                    bad_cols.append(col)

                elif len(df_cp[df_cp[col] == 0.0]) < self.cv:
                    bad_cols.append(col)

                elif df_cp[col].isin([1.0]).any():
                    if len(df_cp[df_cp[col] == 1]) < self.cv:
                        bad_cols.append(col)

                elif len(df_cp[df_cp[col] == 2.0]) < self.cv:
                    bad_cols.append(col)

        # pandas 1.X.X
        else:
            for col in df_cp.columns:
                if 0.0 not in df[col].unique() and 2.0 not in df[col].unique():
                    bad_cols.append(col)

                elif len(df_cp[df_cp[col] == 0.0]) < self.cv:
                    bad_cols.append(col)

                elif 1.0 in df_cp[col].unique():
                    if len(df_cp[df_cp[col] == 1.0]) < self.cv:
                        bad_cols.append(col)

                elif len(df_cp[df_cp[col] == 2.0]) < self.cv:
                    bad_cols.append(col)

        if bad_cols:
            df_cp.drop(bad_cols, axis=1, inplace=True)

            print(
                f"{len(bad_cols)} columns removed for being non-biallelic or "
                f"having genotype counts < number of cross-validation "
                f"folds\nSubsetting from {len(df_cp.columns)} remaining columns\n"
            )

        return df_cp

    def _gather_impute_settings(
        self, kwargs: Dict[str, Any]
    ) -> Tuple[
        Optional[Dict[str, Any]],
        Optional[Dict[str, Any]],
        Optional[Dict[str, Any]],
        Optional[Dict[str, Any]],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[str],
        Optional[Union[int, float]],
        Optional[bool],
        Optional[bool],
        Optional[Optional[int]],
        Optional[str],
        Optional[int],
        Optional[Union[int, float]],
        Optional[float],
    ]:
        """Gather impute settings from kwargs object.

        Gather impute settings from the various imputation classes and IterativeImputer. Gathers them for use with the ``Impute`` class. Returns dictionary with keys as keyword arguments and the values as the settings. The imputation can then be run by specifying IterativeImputer(imp_kwargs).

        Args:
            kwargs (Dict[str, Any]): Dictionary with keys as the keyword arguments and their corresponding values.

        Returns:
            Dict[str, Any]: Parameters with distributions to run with grid search.
            Dict[str, Any]: IterativeImputer keyword arguments.
            Dict[str, Any]: Classifier keyword arguments.
            Dict[str, Any]: Genetic algorithm keyword arguments.
            int: Number of iterations to run with grid search.
            int: Number of cross-validation folds to use with grid search.
            int: Number of processors to use with grid search.
            str or None: Prefix for output files.
            int or float: Proportion of dataset (if float) or number of columns (if int) to use for grid search.
            bool: True if using GASearchCV, False if using RandomSearchCV.
            bool: If True, disables the tqdm progress bar and just prints status updates to a file. If False, uses tqdm progress bar.
            int or None: Percent in which to print progress updates for features. If None, then does not print progress through features.
            str: Scoring metric to use with grid search. Can be any of the sklearn classifier metrics in string format.
            int: Number of generations without improvement before Early Stopping criterion is called.
            int or float: Chunk sizes for doing full imputation following grid search. If int, then splits into chunks of ``chunk_size``\. If float, then splits into chunks of ``n_features * chunk_size``\.
            float or None: Proportion of loci to use for validation if grid search is not used. If None, then doesn't do validation.
        """
        gridparams = kwargs.pop("gridparams", None)
        cv = kwargs.pop("cv", None)
        n_jobs = kwargs.pop("n_jobs", None)
        prefix = kwargs.pop("prefix", None)
        grid_iter = kwargs.pop("grid_iter", None)
        column_subset = kwargs.pop("column_subset", None)
        ga = kwargs.pop("ga", None)
        disable_progressbar = kwargs.pop("disable_progressbar", None)
        scoring_metric = kwargs.pop("scoring_metric", None)
        early_stop_gen = kwargs.pop("early_stop_gen", None)
        chunk_size = kwargs.pop("chunk_size", None)
        validation_only = kwargs.pop("validation_only", None)
        progress_update_percent = kwargs.pop("progress_update_percent", None)

        if prefix is None:
            prefix = "output"

        imp_kwargs = kwargs.copy()
        clf_kwargs = kwargs.copy()
        ga_kwargs = kwargs.copy()

        imp_keys = [
            "n_nearest_features",
            "max_iter",
            "tol",
            "initial_strategy",
            "imputation_order",
            "skip_complete",
            "random_state",
            "verbose",
            "sample_posterior",
            "genotype_data",
            "str_encodings",
        ]

        ga_keys = [
            "population_size",
            "tournament_size",
            "elitism",
            "crossover_probability",
            "mutation_probability",
            "ga_algorithm",
        ]

        to_remove = ["self", "__class__"]

        for k, v in clf_kwargs.copy().items():
            if k in to_remove:
                clf_kwargs.pop(k)
            if k in imp_keys:
                clf_kwargs.pop(k)
            if k in ga_keys:
                clf_kwargs.pop(k)

        if "clf_random_state" in clf_kwargs:
            clf_kwargs["random_state"] = clf_kwargs.pop("clf_random_state")

        if "clf_tol" in clf_kwargs:
            clf_kwargs["tol"] = clf_kwargs.pop("clf_tol")

        for k, v in imp_kwargs.copy().items():
            if k not in imp_keys:
                imp_kwargs.pop(k)

        for k, v in ga_kwargs.copy().items():
            if k not in ga_keys:
                ga_kwargs.pop(k)

        if "ga_algorithm" in ga_kwargs:
            ga_kwargs["algorithm"] = ga_kwargs.pop("ga_algorithm")

        if self.clf_type == "regressor":
            ga_kwargs["criteria"] = "min"

        elif self.clf_type == "classifier":
            ga_kwargs["criteria"] = "max"

        return (
            gridparams,
            imp_kwargs,
            clf_kwargs,
            ga_kwargs,
            grid_iter,
            cv,
            n_jobs,
            prefix,
            column_subset,
            ga,
            disable_progressbar,
            progress_update_percent,
            scoring_metric,
            early_stop_gen,
            chunk_size,
            validation_only,
        )

    def _format_features(
        self, df: pd.DataFrame, missing_val: int = -9
    ) -> pd.DataFrame:
        """Format a 2D list for input into iterative imputer.

        Args:
            df (pandas.DataFrame): DataFrame of features with shape (n_samples, n_features).

            missing_val (int, optional): Missing value to replace with numpy.nan. Defaults to -9.

        Returns:
            pandas.DataFrame: Formatted pandas.DataFrame for input into IterativeImputer.
        """
        # Replace missing data with NaN
        X = df.replace(missing_val, np.nan)
        return X

    def _defile_dataset(
        self, df: pd.DataFrame, col_selection_rate: float = 0.40
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """Replace random columns and rows with np.nan.

        Function to select ``col_selection_rate * columns`` columns in a pandas DataFrame and change anywhere from 15% to 50% of the values in each of those columns to np.nan (missing data). Since we know the true values of the ones changed to np.nan, we can assess the accuracy of the classifier and do a grid search.

        Args:
            df (pandas.DataFrame): 012-encoded genotypes to extract columns from.

            col_selection_rate (float, optional): Proportion of the DataFrame to extract. Defaults to 0.40.

        Returns:
            pandas.DataFrame: DataFrame with values imputed initially with sklearn.impute.SimpleImputer, ImputeAlleleFreq (by populations) or ImputePhylo.
            pandas.DataFrame: DataFrame with randomly missing values.
            numpy.ndarray: Columns that were extracted via random sampling.
        """
        # Code adapted from: https://medium.com/analytics-vidhya/using-scikit-learns-iterative-imputer-694c3cca34de
        cols = np.random.choice(
            df.columns,
            int(len(df.columns) * col_selection_rate),
            replace=False,
        )

        initial_strategy = self.imp_kwargs["initial_strategy"]
        gt_data = self.imp_kwargs["genotype_data"]
        str_enc = self.imp_kwargs["str_encodings"]

        if initial_strategy == "populations":
            simple_imputer = simple_imputers.ImputeAlleleFreq(
                genotype_data=self.genotype_data,
                pops=self.pops,
                by_populations=True,
                missing=-9,
                write_output=False,
                verbose=False,
                validation_mode=True,
            )

            df_defiled = simple_imputer.imputed
            valid_cols = cols.copy()

        elif initial_strategy == "phylogeny":
            print(
                "Doing initial imputation with initial_strategy == "
                "'phylogeny'..."
            )

            # NOTE: non-biallelic sites are removed with ImputePhylo
            simple_imputer = simple_imputers.ImputePhylo(
                genotype_data=gt_data,
                str_encodings=str_enc,
                write_output=False,
                disable_progressbar=True,
                validation_mode=True,
            )

            valid_mask = np.flatnonzero(
                np.logical_not(np.isnan(simple_imputer.valid_sites))
            )

            # ImputePhylo resets the column names
            # So here I switch them back
            # and remove any columns that were non-biallelic
            valid_cols = np.sort(self._remove_invalid_cols(cols, valid_mask))
            new_colnames = self._remove_invalid_cols(
                np.array(df.columns), valid_mask
            )

            self.invalid_indexes = self._remove_invalid_cols(
                np.array(df.columns), valid_mask, validation_mode=False
            )

            df_defiled = simple_imputer.imputed
            old_colnames = list(df_defiled.columns)
            df_defiled.rename(
                columns={i: j for i, j in zip(old_colnames, new_colnames)},
                inplace=True,
            )

        elif initial_strategy == "nmf":
            simple_imputer = simple_imputers.ImputeNMF(
                gt=df.fillna(-9).to_numpy(),
                missing=-9,
                write_output=False,
                verbose=False,
                validation_mode=True,
            )
            df_defiled = simple_imputer.imputed
            valid_cols = cols.copy()

        else:
            # Fill in unknown values with sklearn.impute.SimpleImputer
            simple_imputer = SimpleImputer(
                strategy=self.imp_kwargs["initial_strategy"]
            )

            df_defiled = pd.DataFrame(
                simple_imputer.fit_transform(df.fillna(-9).values)
            )
            valid_cols = cols.copy()

        del simple_imputer
        del cols
        gc.collect()

        self.original_num_cols = len(df_defiled.columns)
        df_filled = df_defiled.copy()

        # Randomly choose rows (samples) to introduce missing data to
        for col in valid_cols:
            data_drop_rate = np.random.choice(np.arange(0.15, 0.5, 0.02), 1)[0]

            drop_ind = np.random.choice(
                np.arange(len(df_defiled[col])),
                size=int(len(df_defiled[col]) * data_drop_rate),
                replace=False,
            )

            # Introduce random np.nan values
            df_defiled.loc[drop_ind, col] = np.nan

        return df_filled, df_defiled, valid_cols

    def _remove_invalid_cols(
        self, arr1: np.ndarray, arr2: np.ndarray, validation_mode: bool = True
    ) -> Optional[Union[List[int], np.ndarray]]:
        """Finds set difference between two numpy arrays and returns the values unique to arr1 that are not found in arr2.

        Args:
            arr1 (numpy.ndarray): 1D Array that contains input values.
            arr2 (numpy.ndarray): 1D Comparison array.
            validation_mode (bool, optional): If False, returns indexes to remove. If True, returns numpy.ndarray of remaining column labels.

        Returns:
            List[int], np.ndarray, or None: List of Bad indexes to remove, numpy.ndarray with invalid column labels removed, or None if no bad columns were found.
        """
        bad_idx = list()
        for i, item in enumerate(arr1):
            if not np.any(arr2 == item):
                bad_idx.append(i)

        if not validation_mode:
            if bad_idx:
                return bad_idx
            else:
                return None

        if bad_idx:
            return np.delete(arr1, bad_idx)
        else:
            return arr1.copy()

    def _impute_eval(
        self, df: pd.DataFrame, clf: Optional[Callable]
    ) -> Dict[str, List[Union[float, int]]]:
        """Function to run IterativeImputer on a pandas.DataFrame.

        The dataframe columns are randomly subset and a fraction of the known, true values are converted to missing data to allow evalutation of the model with either accuracy or mean_squared_error scores.

        Args:
            df (pandas.DataFrame): Original DataFrame with 012-encoded genotypes.

            clf (sklearn Classifier or None): Classifier instance to use with IterativeImputer.

        Returns:
            Dict[List[float or int]]: Validation scores for the current imputation.
        """
        # Code adapted from:
        # https://medium.com/analytics-vidhya/using-scikit-learns-iterative-imputer-694c3cca34de
        df_known, df_unknown, cols = self._defile_dataset(
            df, col_selection_rate=self.validation_only
        )

        df_known_slice = df_known[cols]
        df_unknown_slice = df_unknown[cols]
        df_missing_mask = df_unknown_slice.isnull()

        # Neural networks
        if self.clf == VAE or self.clf == UBP:
            df_stg = df_unknown_slice.copy()

            for col in df_stg.columns:
                df_stg[col] = df_stg[col].replace({pd.NA: np.nan})
            df_stg.fillna(-9, inplace=True)

            imputer = self.clf(**self.clf_kwargs)

            if self.clf == VAE:
                df_imp = pd.DataFrame(
                    imputer.fit_transform(df_stg.to_numpy()),
                    columns=cols,
                )

                df_imp = df_imp.astype("float")
                df_imp = df_imp.astype("int64")

            else:
                df_imp = pd.DataFrame(
                    imputer.fit_transform(df_stg.to_numpy()),
                    columns=cols,
                    dtype="int64",
                )

        else:
            # Using IterativeImputer
            df_stg = df_unknown.copy()

            imputer = self._define_iterative_imputer(
                clf,
                self.logfilepath,
                clf_kwargs=self.clf_kwargs,
                prefix=self.prefix,
                disable_progressbar=self.disable_progressbar,
                progress_update_percent=self.progress_update_percent,
                pops=self.pops,
            )

            imp_arr = imputer.fit_transform(df_stg)

            # Get only subset of validation columns
            # get_loc returns the index of the value
            df_imp = pd.DataFrame(
                imp_arr[:, [df_unknown.columns.get_loc(i) for i in cols]],
                columns=cols,
            )

        # Get score of each column
        scores = defaultdict(list)
        for i in range(len(df_known_slice.columns)):
            # Adapted from: https://medium.com/analytics-vidhya/using-scikit-learns-iterative-imputer-694c3cca34de

            mask = df_missing_mask[df_known_slice.columns[i]]
            y_true = df_known[df_known_slice.columns[i]]
            y_true = y_true[mask]

            y_pred = df_imp[df_known_slice.columns[i]]
            y_pred = y_pred[mask]

            if self.clf_type == "classifier":

                # Had to do this because get incompatible type error if using
                # initial_imputation="populations"
                if y_true.dtypes != "int64":
                    y_true = y_true.astype("int64")
                if y_pred.dtypes != "int64":
                    y_pred = y_pred.astype("int64")

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

            else:
                scores["explained_var"].append(
                    metrics.explained_variance_score(y_true, y_pred)
                )

                scores["rmse"].append(
                    metrics.mean_squared_error(y_true, y_pred, squared=False)
                )

        lst2del = [
            df_stg,
            df_imp,
            df_known,
            df_known_slice,
            df_unknown,
        ]

        if self.clf == VAE or self.clf == UBP:
            del lst2del
            del cols
        else:
            del lst2del
            del imp_arr
            del imputer
            del cols

        gc.collect()

        return scores

    def _define_iterative_imputer(
        self,
        clf: Callable,
        logfilepath: str,
        clf_kwargs: Optional[Dict[str, Any]] = None,
        ga_kwargs: Optional[Dict[str, Any]] = None,
        prefix: str = "out",
        n_jobs: Optional[int] = None,
        n_iter: Optional[int] = None,
        cv: Optional[int] = None,
        clf_type: Optional[str] = None,
        ga: bool = False,
        search_space: Optional[Dict[str, Any]] = None,
        disable_progressbar: bool = False,
        progress_update_percent: Optional[int] = None,
        scoring_metric: Optional[str] = None,
        early_stop_gen: Optional[int] = None,
        pops: Optional[List[Union[int, str]]] = None,
    ) -> Union[IterativeImputerGridSearch, IterativeImputerFixedParams]:
        """Define an IterativeImputer instance.

        The instances are of custom, overloaded IterativeImputer classes.

        Args:
            clf (sklearn Classifier instance): Estimator to use with IterativeImputer.

            logfilepath (str): Path to progress log file.

            clf_kwargs (dict, optional): Keyword arguments for classifier. Defaults to None.

            ga_kwargs (dict, optional): Keyword arguments for genetic algorithm grid search. Defaults to None.

            prefix (str, optional): Prefix for saving GA plots to PDF file. Defaults to None.

            n_jobs (int, optional): Number of parallel jobs to use with the IterativeImputer grid search. Ignored if ``search_space=None``\. Defaults to None.

            n_iter (int, optional): [Number of iterations for grid search. Ignored if ``search_space=None``]. Defaults to None.

            cv (int, optional): Number of cross-validation folds to use with grid search. Ignored if ``search_space=None``\. Defaults to None

            clf_type (str, optional): Type of estimator. Valid options are "classifier" or "regressor". Ignored if ``search_space=None``\. Defaults to None.

            ga (bool, optional): Whether to use genetic algorithm for the grid search. If False, uses RandomizedSearchCV instead. Defaults to False.

            search_space (dict, optional): gridparams dictionary with search space to explore in random grid search. Defaults to None.

            disable_progressbar (bool, optional): Whether or not to disable the tqdm progress bar. Defaults to False.

            progress_update_percent (int, optional): Print progress updates every ``progress_update_percent`` percent. Defaults to None.

            scoring_metric (str, optional): Scoring metric to use with grid search. Defaults to None.

            early_stop_gen (int, optional): Number of consecutive generations without improvement to raise early stopping callback. Defaults to None.

            pops (List[Union[int, str]], optional): Population IDs as 1d-list in order of sampleID.

        Returns:
            sklearn.impute.IterativeImputer: IterativeImputer instance.
        """
        if search_space is None:
            imp = IterativeImputerFixedParams(
                logfilepath,
                clf_kwargs,
                prefix,
                estimator=clf,
                disable_progressbar=disable_progressbar,
                progress_update_percent=progress_update_percent,
                pops=pops,
                **self.imp_kwargs,
            )

        else:
            # Create iterative imputer
            imp = IterativeImputerGridSearch(
                logfilepath,
                search_space,
                clf_kwargs,
                ga_kwargs,
                prefix,
                estimator=clf,
                grid_n_jobs=n_jobs,
                grid_n_iter=n_iter,
                grid_cv=cv,
                clf_type=clf_type,
                ga=ga,
                disable_progressbar=disable_progressbar,
                progress_update_percent=progress_update_percent,
                scoring_metric=scoring_metric,
                early_stop_gen=early_stop_gen,
                pops=pops,
                **self.imp_kwargs,
            )

        return imp
