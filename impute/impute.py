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
from timeit import default_timer

# Third party imports
import numpy as np
import pandas as pd
import scipy.linalg
import toyplot.pdf
import toyplot as tp
import toytree as tt

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

import theano
import theano.tensor as T
import theano.tensor.nnet as nnet

# Custom module imports
from impute.iterative_imputer_custom import (
    IterativeImputerGridSearch,
    IterativeImputerAllData,
)

from read_input.read_input import GenotypeData

from utils import sequence_tools
from utils import misc
from utils.misc import get_processor_name
from utils.misc import isnotebook
from utils.misc import StreamToLogger
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
            "Warning: Intel CPU detected but scikit-learn-intelex is not installed. We recommend installing it to speed up computation."
        )
        intelex = False
else:
    intelex = False


class Impute:
    def __init__(self, clf, clf_type, kwargs):
        self.clf = clf
        self.clf_type = clf_type

        # Separate local variables into separate settings objects
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

        self.logfilepath = f"{self.prefix}_imputer_progress_log.txt"

        # Remove logfile if exists
        try:
            os.remove(self.logfilepath)
        except OSError:
            pass

    @timer
    def fit_predict(self, X):
        """[Fit and predict with IterativeImputer(clf). if ```gridparams is None```, then a grid search is not performed. If gridparams is not None, then a RandomizedSearchCV is performed on a subset of the data and a final imputation is done on the whole dataset using the best found parameters]

        Args:
            X ([pandas.DataFrame]): [DataFrame with 012-encoded genotypes]

        Returns:
            [pandas.DataFrame]: [DataFrame with 012-encoded genotypes and missing data imputed]

            [float]: [Best score found from grid search. If self.clf is a regressor, will be the lowest root mean squared error. If self.clf is a classifier, will be the highest percent accuracy]

            [dict]: [Best parameters found using grid search]
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

        print("\nDone!\n")
        return imputed_df, best_params

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

    def read_imputed(self, filename):
        """[Read in imputed CSV file as formatted by write_imputed]

        Args:
            filename ([str]): [Name of imputed CSV file to be read]

        Returns:
            [pandas.DataFrame]: [Imputed data as DataFrame of 8-bit integers]
        """
        return pd.read_csv(filename, dtype="Int8", header=None)

    def df2chunks(self, df, chunk_size):
        """[Break up pandas.DataFrame into chunks. If set to 1.0 of type float, then returns only one chunk containing all the data]

        Args:
            df ([pandas.DataFrame]): [DataFrame to split into chunks]

            chunk_size ([int or float]): [If integer, then breaks DataFrame into ```chunk_size``` chunks. If float, breaks DataFrame up into ```chunk_size * len(df.columns)``` chunks]

        Returns:
            [list(pandas.DataFrame)]: [List of pandas DataFrames (one element per chunk)]

        Raises:
            [ValueError]: [chunk_size must be of type int or float]
        """
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

    def subset_data_for_gridsearch(
        self, df, columns_to_subset, original_num_cols
    ):
        """[Randomly subsets pandas.DataFrame with ```column_percent``` fraction of the data. Allows faster testing]

        Args:
            df ([pandas.DataFrame]): [DataFrame with 012-encoded genotypes]

            columns_to_subset ([int or float]): [If float, proportion of DataFrame to randomly subset, which should be between 0 and 1. if integer, subsets ```columns_to_subset``` random columns]

            original_num_cols ([int]): [Number of columns in original DataFrame]

        Returns:
            [pandas.DataFrame]: [Randomly subset DataFrame]
        """
        # Get a random numpy arrray of column names to select
        if isinstance(columns_to_subset, float):
            n = int(original_num_cols * columns_to_subset)
            if n > len(df.columns):
                print(
                    "Warning: column_subset is greater than remaining columns following filtering. Using all columns"
                )
                all_cols = True
            else:
                all_cols = False

            if all_cols:
                df_sub = df.copy()
            else:
                df_sub = df.sample(n=n, axis="columns", replace=False)

        elif isinstance(columns_to_subset, int):
            if columns_to_subset > len(df.columns):
                print(
                    "Warning: column_subset is greater than remaining columns following filtering. Using all columns"
                )
                all_cols = True
            else:
                all_cols = False

            if all_cols:
                df_sub = df.copy()
            else:
                sub = columns_to_subset
                df_sub = df.sample(n=sub, axis="columns", replace=False)

        df_sub.columns = df_sub.columns.astype(str)

        return df_sub

    def _print_scores(self, df_scores):
        """[Print pandas Dataframe with validation scores]

        Args:
            df ([pandas.DataFrame]): [DataFrame with score statistics]
        """
        print("Validation scores")
        print(df_scores)

    def _write_imputed_params_score(self, df_scores, best_params):
        """[Save best_score and best_params to files]

        Args:
            best_score ([float]): [Best RMSE or accuracy score for the regressor or classifier, respectively]

            best_params ([dict]): [Best parameters found in grid search]
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

    def _impute_single(self, df):
        """[Run IterativeImputer without a grid search]

        Args:
            df ([pandas.DataFrame]): [DataFrame of 012-encoded genotypes]

        Returns:
            [pandas.DataFrame]: [Imputed 012-encoded genotypes]
        """
        print(f"Doing {self.clf.__name__} imputation without grid search...")

        clf = self.clf(**self.clf_kwargs)

        if self.validation_only is not None:
            print("Estimating validation scores...")

            df_scores = self._imputer_validation(df, clf)

            print("\nDone!\n")

        else:
            df_scores = None

        imputer = self._define_iterative_imputer(
            clf,
            self.logfilepath,
            clf_kwargs=self.clf_kwargs,
            prefix=self.prefix,
            disable_progressbar=self.disable_progressbar,
            progress_update_percent=self.progress_update_percent,
        )

        df_chunks = self.df2chunks(df, self.chunk_size)
        imputed_df = self._impute_df(df_chunks, imputer, len(df.columns))

        lst2del = [df_chunks]
        del lst2del
        gc.collect()

        self._validate_imputed(imputed_df)

        print(f"\nDone with {self.clf.__name__} imputation!\n")
        return imputed_df, df_scores, None

    def _impute_gridsearch(self, df):
        """[Do IterativeImputer with RandomizedSearchCV]

        Args:
            df ([pandas.DataFrame]): [DataFrame with 012-encoded genotypes]

        Returns:
            [pandas.DataFrame]: [DataFrame with 012-encoded genotypes imputed using the best parameters found from the grid search]

            [float]: [Absolute value of the best score found during the grid search]

            [dict]: [Best parameters found during the grid search]
        """
        original_num_cols = len(df.columns)
        df_tmp = self._remove_nonbiallelic(df)
        df_subset = self.subset_data_for_gridsearch(
            df_tmp, self.column_subset, original_num_cols
        )

        lst2del = [df_tmp]
        del lst2del
        gc.collect()

        print(f"Validation dataset size: {len(df_subset.columns)}\n")
        print("Doing grid search...")

        clf = self.clf()

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
        )

        Xt, params_list, score_list = imputer.fit_transform(df_subset)

        print("\nDone!")

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

        print("\nDoing imputation with best found parameters...")

        best_imputer = self._define_iterative_imputer(
            best_clf,
            self.logfilepath,
            clf_kwargs=self.clf_kwargs,
            prefix=self.prefix,
            disable_progressbar=self.disable_progressbar,
            progress_update_percent=self.progress_update_percent,
        )

        df_chunks = self.df2chunks(df, self.chunk_size)
        imputed_df = self._impute_df(df_chunks, best_imputer, original_num_cols)

        lst2del = [df_chunks]
        del df_chunks
        gc.collect()

        self._validate_imputed(imputed_df)

        print(f"\nDone with {self.clf.__name__} imputation!\n")
        return imputed_df, df_scores, best_params

    def _imputer_validation(self, df, clf):
        """[Validate imputation by running it on a validation test set ``cv`` times. Actual missing values are imputed with sklearn.impute.SimpleImputer, and then missing values are randomly introduced to known genotypes. The dataset with no missing data is compared to the dataset with known missing data to obtain validation scores]

        Args:
            df ([pandas.DataFrame]): [012-encoded genotypes to impute]

            clf ([sklearn classifier instance]): [sklearn classifier instance with which to run the imputation]

        Raises:
            ValueError: [If none of the scores were able to be estimated and reps is empty]

        Returns:
            [pandas.DataFrame]: [Validation scores in a pandas DataFrame object. Contains the scoring metric, mean, median, minimum, and maximum validation scores among all features, and the lower and upper 95% confidence interval among the replicates]
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

    def _impute_df(self, df_chunks, imputer, original_num_cols):
        """[Impute list of pandas.DataFrame objects using IterativeImputer. The DataFrames correspond to each chunk of features]

        Args:
            df_chunks ([list(pandas.DataFrame)]): [Dataframe with 012-encoded genotypes]

            imputer ([IterativeImputerAllData instance]): [Defined IterativeImputerAllData instance to perform the imputation]

            original_num_cols ([int]): [Number of columns in original dataset]

        Returns:
            [pandas.DataFrame]: [Single DataFrame object, with all the imputed chunks concatenated together]
        """
        imputed_chunks = list()
        num_chunks = len(df_chunks)
        for i, Xchunk in enumerate(df_chunks, start=1):
            if self.clf_type == "classifier":
                df_imp = pd.DataFrame(
                    imputer.fit_transform(Xchunk), dtype="Int8"
                )

                imputed_chunks.append(df_imp)

            else:
                # Regressor. Needs to be rounded to integer first.
                df_imp = pd.DataFrame(imputer.fit_transform(Xchunk))
                df_imp = df_imp.round(0).astype("Int8")

                imputed_chunks.append(df_imp)

        concat_df = pd.concat(imputed_chunks, axis=1)
        assert (
            len(concat_df.columns) == original_num_cols
        ), "Failed merge operation: Could not merge chunks back together"

        del imputed_chunks
        gc.collect()

        return concat_df

    def _validate_imputed(self, df):
        """[Asserts that there is no missing data left in the imputed DataFrame]

        Args:
            df ([pandas.DataFrame]): [DataFrame with imputed 012-encoded genotypes]
        """
        assert (
            not df.isnull().values.any()
        ), "Imputation failed...Missing values found in the imputed dataset"

    def _get_best_params(self, params_list):
        """[Gets the best parameters from the grid search. Determines the parameter types and either gets the mean or mode if the type is numeric or string/ boolean]

        Args:
            params_list ([list(dict)]): [List of grid search parameter values]

        Returns:
            [dict]: [Dictionary with parameters as keys and their best values]
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

    def _mode_list_of_dicts(self, l, k):
        """[Get mode for key k in a list of dictionaries]

        Args:
            l ([list(dict)]): [List of dictionaries]
            k ([str]): [Key to find the mode across all dictionaries in l]

        Returns:
            [str]: [Most common value across list of dictionaries for one key]
        """
        k_count = Counter(map(itemgetter(k), l))
        return k_count.most_common()[0][0]

    def _average_list_of_dicts(self, l, k, is_int=False):
        """[Get average of a given key in a list of dictionaries]

        Args:
            l ([list(dict)]): [List of dictionaries]

            k ([str]): [Key to find average across list of dictionaries]

            is_int (bool, optional): [Whether or not the value for key k is an integer. If False, it is expected to be of type float]. Defaults to False.

        Returns:
            [int or float]: [average of given key across list of dictionaries]
        """
        if is_int:
            return int(sum(d[k] for d in l) / len(l))
        else:
            return sum(d[k] for d in l) / len(l)

    def _remove_nonbiallelic(self, df):
        """[Remove sites that do not have both 0 and 2 encoded values in a column and if any of the allele counts is less than the number of cross-validation folds]

        Args:
            df ([pandas.DataFrame]): [DataFrame with 012-encoded genotypes]

        Returns:
            [pandas.DataFrame]: [DataFrame with sites removed]
        """
        cv2 = self.cv * 2
        df_cp = df.copy()
        bad_cols = list()
        if pd.__version__[0] == 0:
            for col in df_cp.columns:
                if (
                    not df_cp[col].isin([0.0]).any()
                    or not df_cp[col].isin([2.0]).any()
                ):
                    bad_cols.append(col)

                if len(df_cp[df_cp[col] == 0.0]) < cv2:
                    bad_cols.append(col)

                if df_cp[col].isin([1.0]).any():
                    if len(df_cp[df_cp[col] == 1]) < cv2:
                        bad_cols.append(col)

                if len(df_cp[df_cp[col] == 2.0]) < cv2:
                    bad_cols.append(col)

        # pandas 1.X.X
        else:
            for col in df_cp.columns:
                if 0.0 not in df[col].unique() and 2.0 not in df[col].unique():
                    bad_cols.append(col)

                elif len(df_cp[df_cp[col] == 0.0]) < cv2:
                    bad_cols.append(col)

                elif 1.0 in df_cp[col].unique():
                    if len(df_cp[df_cp[col] == 1.0]) < cv2:
                        bad_cols.append(col)

                elif len(df_cp[df_cp[col] == 2.0]) < cv2:
                    bad_cols.append(col)

        df_cp.drop(bad_cols, axis=1, inplace=True)

        print(
            f"{len(bad_cols)} columns removed for being non-biallelic or having genotype counts < number of cross-validation folds\nSubsetting from {len(df_cp.columns)} remaining columns\n"
        )

        return df_cp

    def _gather_impute_settings(self, kwargs):
        """[Gather impute settings from the various imputation classes and IterativeImputer. Gathers them for use with the Impute() class. Returns dictionary with keys as keyword arguments and the values as the settings. The imputation can then be run by specifying e.g. IterativeImputer(**imp_kwargs)]

        Args:
            kwargs ([dict]): [Dictionary with keys as the keyword arguments and their corresponding values]

        Returns:
            [dict]: [Parameters distributions to run with grid search]
            [dict]: [IterativeImputer keyword arguments]
            [dict]: [Classifier keyword arguments]
            [dict]: [Genetic algorithm keyword arguments]
            [int]: [Number of iterations to run with grid search]
            [int]: [Number of cross-validation folds to use]
            [int]: [Number of processors to use with grid search]
            [str]: [Prefix for output files]
            [int or float]: [Proportion of dataset to use for grid search]
            [bool]: [If True, disables the tqdm progress bar and just prints status updates to a file]
            [int]: [Percent in which to print progress updates for features]
            [str]: [Scoring metric to use with grid search. Can be any of the sklearn classifier metrics in string format]
            [int]: [Number of generations without improvement before Early Stopping criterion is called]
            [int or float]: [Chunk sizes for doing full imputation following grid search. If int, then splits into chunks of ``chunk_size``. If float, then splits into chunks of ``n_features * chunk_size``]
            [float or None]: [Proportion of loci to use for validation if grid search is not used. If None, then doesn't do validation]
        """
        gridparams = kwargs.pop("gridparams")
        cv = kwargs.pop("cv")
        n_jobs = kwargs.pop("n_jobs")
        prefix = kwargs.pop("prefix")
        grid_iter = kwargs.pop("grid_iter")
        column_subset = kwargs.pop("column_subset")
        ga = kwargs.pop("ga")
        disable_progressbar = kwargs.pop("disable_progressbar")
        scoring_metric = kwargs.pop("scoring_metric")
        early_stop_gen = kwargs.pop("early_stop_gen")
        chunk_size = kwargs.pop("chunk_size")
        validation_only = kwargs.pop("validation_only")

        progress_update_percent = kwargs.pop("progress_update_percent")

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
        ]

        ga_keys = [
            "population_size",
            "tournament_size",
            "elitism",
            "crossover_probability",
            "mutation_probability",
            "ga_algorithm",
        ]

        to_remove = ["genotype_data", "self", "__class__"]

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

    def _format_features(self, df, missing_val=-9):
        """[Format a 2D list for input into iterative imputer]

        Args:
            df ([pandas.DataFrame]): [DataFrame of features with shape(n_samples, n_features)]

            missing_val (int, optional): [Missing value to replace with numpy.nan]. Defaults to -9.

        Returns:
            [pandas.DataFrame]: [Formatted pandas.DataFrame for input into IterativeImputer]
        """
        # Replace missing data with NaN
        X = df.replace(missing_val, np.nan)
        return X

    def _defile_dataset(self, df, col_selection_rate=0.40):
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
        simple_imputer = SimpleImputer(
            strategy=self.imp_kwargs["initial_strategy"]
        )

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

    def _impute_eval(self, df, clf):
        """[Function to run IterativeImputer on a DataFrame. The dataframe columns will be randomly subset and a fraction of the known, true values are converted to missing data to allow evalutation of the model with either accuracy or mean_squared_error scores]

            Args:
                df ([pandas.DataFrame]): [Original DataFrame with 012-encoded genotypes]

                clf ([sklearn Classifier]): [Classifier instance to use with IterativeImputer]

        Returns:
                [dict(list)]: [Validation scores for the current imputation]
        """
        # Code adapted from:
        # https://medium.com/analytics-vidhya/using-scikit-learns-iterative-imputer-694c3cca34de

        # Subset the DataFrame randomly and replace known values with np.nan
        df_known, df_valid, cols = self._defile_dataset(df)
        df_known_slice = df_known[cols]
        df_valid_slice = df_valid[cols]

        # Slice the original DataFrame to the same columns as df_miss
        # df_orig_slice = df_orig[cols]

        imputer = self._define_iterative_imputer(
            clf,
            self.logfilepath,
            clf_kwargs=self.clf_kwargs,
            prefix=self.prefix,
            disable_progressbar=self.disable_progressbar,
            progress_update_percent=self.progress_update_percent,
        )

        df_stg = df_valid.copy()
        imp_arr = imputer.fit_transform(df_stg)

        df_imp = pd.DataFrame(
            imp_arr[:, [df_known.columns.get_loc(i) for i in cols]]
        )

        # Get score of each column
        scores = defaultdict(list)
        for i in range(len(df_known_slice.columns)):
            # Adapted from: https://medium.com/analytics-vidhya/using-scikit-learns-iterative-imputer-694c3cca34de

            y_true = df_known[df_known.columns[i]]
            y_pred = df_imp[df_imp.columns[i]]

            if self.clf_type == "classifier":

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
            df_valid_slice,
            df_valid,
        ]
        del lst2del
        del imp_arr
        del imputer
        del cols
        gc.collect()

        return scores

    def _define_iterative_imputer(
        self,
        clf,
        logfilepath,
        clf_kwargs=None,
        ga_kwargs=None,
        prefix="out",
        n_jobs=None,
        n_iter=None,
        cv=None,
        clf_type=None,
        ga=False,
        search_space=None,
        disable_progressbar=False,
        progress_update_percent=None,
        scoring_metric=None,
        early_stop_gen=None,
    ):
        """[Define an IterativeImputer instance]

            Args:
                clf ([sklearn Classifier]): [Classifier to use with IterativeImputer]

                logfilepath [str]: [Path to progress log file]

                clf_kwargs (dict, optional): [Keyword arguments for classifier]. Defaults to None.

                ga_kwargs (dict, optional): [Keyword arguments for genetic algorithm grid search]. Defaults to None.

                prefix (str, optional): [Prefix for saving GA plots to PDF file]. Defaults to None.

                n_jobs (int, optional): [Number of parallel jobs to use with the IterativeImputer grid search. Ignored if search_space=None]. Defaults to None.

                n_iter (int, optional): [Number of iterations for grid search. Ignored if search_space=None]. Defaults to None.

                cv (int, optional): [Number of cross-validation folds to use with grid search. Ignored if search_space=None]. Defaults to None

                clf_type (str, optional): [Type of estimator. Valid options are "classifier" or "regressor". Ignored if search_space=None]. Defaults to None.

                ga (bool, optional): [Whether to use genetic algorithm for the grid search. If False, uses RandomizedSearchCV instead]. Defaults to False.

                search_space (dict, optional): [gridparams dictionary with search space to explore in random grid search]. Defaults to None.

                disable_progressbar (bool, optional): [Whether or not to disable the tqdm progress bar]. Defaults to False.

                progress_update_percent (int, optional): [Print progress updates every ```progress_update_percent``` percent]. Defaults to None.

                scoring_metric (str, optional): [Scoring metric to use with grid search]. Defaults to None.

                early_stop_gen (int, optional): [Number of consecutive generations without improvement to raise early stopping callback]. Defaults to None.

        Returns:
                [sklearn.impute.IterativeImputer]: [IterativeImputer instance]
        """
        if search_space is None:
            imp = IterativeImputerAllData(
                logfilepath,
                clf_kwargs,
                prefix,
                estimator=clf,
                disable_progressbar=disable_progressbar,
                progress_update_percent=progress_update_percent,
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
                **self.imp_kwargs,
            )

        return imp


class ImputeKNN:
    """[Does K-Nearest Neighbors Iterative imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process]

    Args:
        genotype_data ([GenotypeData]): [GenotypeData instance that was used to read in the sequence data]

        prefix ([str]): [Prefix for imputed data's output filename]

        gridparams (dict, optional): [Dictionary with lists as values or distributions of parameters. Distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). ```gridparams``` will be used for a randomized grid search. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). **NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. The full imputation can be performed by setting ```gridparams=None``` (default)]. Defaults to None.

        grid_iter (int, optional): [Number of iterations for randomized grid search]. Defaults to 50.

        cv (int, optional): [Number of folds for cross-validation during randomized grid search]. Defaults to 5.

        validation_only (float, optional): [Validates the imputation without doing a grid search. The validation method randomly replaces 15% to 50% of the known, non-missing genotypes in ``n_features * validation_only`` of the features. It then imputes the newly missing genotypes for which we know the true values and calculates validation scores. This procedure is replicated ``cv`` times and a mean, median, minimum, maximum, lower 95% confidence interval (CI) of the mean, and the upper 95% CI are calculated and saved to a CSV file. ``gridparams`` must be set to None (default) for ``validation_only`` to work. Calculating a validation score can be turned off altogether by setting ``validation_only`` to None]. Defaults to 0.4.

        ga (bool, optional): [Whether to use a genetic algorithm for the grid search. If False, a RandomizedSearchCV is done instead]. Defaults to False.

        population_size (int, optional): [For genetic algorithm grid search: Size of the initial population to sample randomly generated individuals. See GASearchCV documentation]. Defaults to 10.

        tournament_size (int, optional): [For genetic algorithm grid search: Number of individuals to perform tournament selection. See GASearchCV documentation]. Defaults to 3.

        elitism (bool, optional): [For genetic algorithm grid search:     If True takes the tournament_size best solution to the next generation. See GASearchCV documentation]. Defaults to True.

        crossover_probability (float, optional): [For genetic algorithm grid search: Probability of crossover operation between two individuals. See GASearchCV documentation]. Defaults to 0.8.

        mutation_probability (float, optional): [For genetic algorithm grid search: Probability of child mutation. See GASearchCV documentation]. Defaults to 0.1.

        ga_algorithm (str, optional): [For genetic algorithm grid search: Evolutionary algorithm to use. See more details in the deap algorithms documentation]. Defaults to "eaMuPlusLambda".

        early_stop_gen (int, optional): [If the genetic algorithm sees ```early_stop_gen``` consecutive generations without improvement in the scoring metric, an early stopping callback is implemented. This saves time by reducing the number of generations the genetic algorithm has to perform]. Defaults to 5.

        scoring_metric (str, optional): [Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options]. Defaults to "accuracy".

        column_subset (int or float, optional): [If float, proportion of the dataset to randomly subset for the grid search. Should be between 0 and 1, and should also be small, because the grid search takes a long time. If int, subset ```column_subset``` columns]. Defaults to 0.1.

        chunk_size (int or float, optional): [Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ```chunk_size``` loci at a time. If a float is specified, selects ```total_loci * chunk_size``` loci at a time]. Defaults to 1.0 (all features).

        disable_progressbar (bool, optional): [Whether or not to disable the tqdm progress bar when doing the imputation. If True, progress bar is disabled, which is useful when running the imputation on e.g. an HPC cluster. If the bar is disabled, a status update will be printed to standard output for each iteration and feature instead. If False, the tqdm progress bar will be used]. Defaults to False.

        progress_update_percent (int, optional): [Print status updates for features every ```progress_update_percent```%. IterativeImputer iterations will always be printed, but ```progress_update_percent``` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features]. Defaults to None.

        n_jobs (int, optional): [Number of parallel jobs to use. If ```gridparams``` is not None, n_jobs is used for the grid search. Otherwise it is used for the classifier. -1 means using all available processors]. Defaults to 1.

        n_neighbors (int, optional): [Number of neighbors to use by default for K-Nearest Neighbors queries]. Defaults to 5.

        weights (str, optional): [Weight function used in prediction. Possible values: 'Uniform': Uniform weights with all points in each neighborhood weighted equally; 'distance': Weight points by the inverse of their distance, in this case closer neighbors of a query point will have  a greater influence than neighbors that are further away; 'callable': A user-defined function that accepts an array of distances and returns an array of the same shape containing the weights]. Defaults to "distance".

        algorithm (str, optional): [Algorithm used to compute the nearest neighbors. Possible values: 'ball_tree', 'kd_tree', 'brute', 'auto']. Defaults to "auto".

        leaf_size (int, optional): [Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem]. Defaults to 30.

        p (int, optional): [Power parameter for the Minkowski metric. When p=1, this is equivalent to using manhattan_distance (l1), and if p=2 it is equivalent to using euclidean distance (l2). For arbitrary p, minkowski_distance (l_p) is used]. Defaults to 2.

        metric (str, optional): [The distance metric to use for the tree. The default metric is minkowski, and with p=2 this is equivalent to the standard Euclidean metric. See the documentation of sklearn.DistanceMetric for a list of available metrics. If metric is 'precomputed', X is assumed to be a distance matrix and must be square during fit]. Defaults to "minkowski".

        max_iter (int, optional): [Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values]. Defaults to 10.

        tol ([type], optional): [Tolerance of the stopping condition]. Defaults to 1e-3.

        n_nearest_features (int, optional): [Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge]. Defaults to 10.

        initial_strategy (str, optional): [Which strategy to use to initialize the missing values. Same as the strategy parameter in sklearn.impute.SimpleImputer Valid values: {“mean”, “median”, or “most_frequent”}. Defaults to "most_frequent".

        imputation_order (str, optional): [The order in which the features will be imputed. Possible values: 'ascending' (from features with fewest missing values to most), 'descending' (from features with most missing values to fewest), 'roman' (left to right), 'arabic' (right to left), 'random' (a random order for each round). ]. Defaults to "ascending".

        skip_complete (bool, optional): [If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time]. Defaults to False.

        random_state (int, optional): [The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if n_nearest_features is not None or the imputation_order is 'random'. Use an integer for determinism. If None, then uses a different random seed each time]. Defaults to None.

        verbose (int, optional): [Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2]. Defaults to 0.
    """

    def __init__(
        self,
        genotype_data,
        *,
        prefix=None,
        gridparams=None,
        grid_iter=50,
        cv=5,
        validation_only=0.4,
        ga=False,
        population_size=10,
        tournament_size=3,
        elitism=True,
        crossover_probability=0.8,
        mutation_probability=0.1,
        ga_algorithm="eaMuPlusLambda",
        early_stop_gen=5,
        scoring_metric="accuracy",
        column_subset=0.1,
        chunk_size=1.0,
        disable_progressbar=False,
        progress_update_percent=None,
        n_jobs=1,
        n_neighbors=5,
        weights="distance",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        max_iter=10,
        tol=1e-3,
        n_nearest_features=10,
        initial_strategy="most_frequent",
        imputation_order="ascending",
        skip_complete=False,
        random_state=None,
        verbose=0,
    ):
        # Get local variables into dictionary object
        kwargs = locals()

        self.clf_type = "classifier"
        self.clf = KNeighborsClassifier

        imputer = Impute(self.clf, self.clf_type, kwargs)

        self.imputed, self.best_params = imputer.fit_predict(
            genotype_data.genotypes_df
        )

        imputer.write_imputed(self.imputed)


class ImputeRandomForest:
    """[Does Random Forest Iterative imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process]

    Args:
        genotype_data ([GenotypeData]): [GenotypeData instance that was used to read in the sequence data]

        prefix ([str]): [Prefix for imputed data's output filename]

        gridparams (dict, optional): [Dictionary with lists as values or distributions of parameters. Distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). ```gridparams``` will be used for a randomized grid search. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). **NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. The full imputation can be performed by setting ```gridparams=None``` (default)]. Defaults to None.

        grid_iter (int, optional): [Number of iterations for randomized grid search]. Defaults to 50.

        cv (int, optional): [Number of folds for cross-validation during randomized grid search]. Defaults to 5.

        validation_only (float, optional): [Validates the imputation without doing a grid search. The validation method randomly replaces 15% to 50% of the known, non-missing genotypes in ``n_features * validation_only`` of the features. It then imputes the newly missing genotypes for which we know the true values and calculates validation scores. This procedure is replicated ``cv`` times and a mean, median, minimum, maximum, lower 95% confidence interval (CI) of the mean, and the upper 95% CI are calculated and saved to a CSV file. ``gridparams`` must be set to None (default) for ``validation_only`` to work. Calculating a validation score can be turned off altogether by setting ``validation_only`` to None]. Defaults to 0.4.

        ga (bool, optional): [Whether to use a genetic algorithm for the grid search. If False, a RandomizedSearchCV is done instead]. Defaults to False.

        population_size (int, optional): [For genetic algorithm grid search: Size of the initial population to sample randomly generated individuals. See GASearchCV documentation]. Defaults to 10.

        tournament_size (int, optional): [For genetic algorithm grid search: Number of individuals to perform tournament selection. See GASearchCV documentation]. Defaults to 3.

        elitism (bool, optional): [For genetic algorithm grid search:     If True takes the tournament_size best solution to the next generation. See GASearchCV documentation]. Defaults to True.

        crossover_probability (float, optional): [For genetic algorithm grid search: Probability of crossover operation between two individuals. See GASearchCV documentation]. Defaults to 0.8.

        mutation_probability (float, optional): [For genetic algorithm grid search: Probability of child mutation. See GASearchCV documentation]. Defaults to 0.1.

        ga_algorithm (str, optional): [For genetic algorithm grid search: Evolutionary algorithm to use. See more details in the deap algorithms documentation]. Defaults to "eaMuPlusLambda".

        early_stop_gen (int, optional): [If the genetic algorithm sees ```early_stop_gen``` consecutive generations without improvement in the scoring metric, an early stopping callback is implemented. This saves time by reducing the number of generations the genetic algorithm has to perform]. Defaults to 5.

        scoring_metric (str, optional): [Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options]. Defaults to "accuracy".

        column_subset (int or float, optional): [If float, proportion of the dataset to randomly subset for the grid search. Should be between 0 and 1, and should also be small, because the grid search takes a long time. If int, subset ```column_subset``` columns]. Defaults to 0.1.

        chunk_size (int or float, optional): [Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ```chunk_size``` loci at a time. If a float is specified, selects ```total_loci * chunk_size``` loci at a time]. Defaults to 1.0 (all features).

        disable_progressbar (bool, optional): [Whether or not to disable the tqdm progress bar when doing the imputation. If True, progress bar is disabled, which is useful when running the imputation on e.g. an HPC cluster. If the bar is disabled, a status update will be printed to standard output for each iteration and feature instead. If False, the tqdm progress bar will be used]. Defaults to False.

        progress_update_percent (int, optional): [Print status updates for features every ```progress_update_percent```%. IterativeImputer iterations will always be printed, but ```progress_update_percent``` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features]. Defaults to None.

        n_jobs (int, optional): [Number of parallel jobs to use. If ```gridparams``` is not None, n_jobs is used for the grid search. Otherwise it is used for the classifier. -1 means using all available processors]. Defaults to 1.

        extra_trees (bool, optional): [Whether to use ExtraTreesClassifier (If True) instead of RandomForestClassifier (If False). ExtraTreesClassifier is faster, but is not supported by the scikit-learn-intelex patch and RandomForestClassifier is. If using an Intel CPU, the optimizations provided by the scikit-learn-intelex patch might make setting ```extratrees=False``` worthwhile. If you are not using an Intel CPU, the scikit-learn-intelex library is not supported and ExtraTreesClassifier will be faster with similar performance. NOTE: If using scikit-learn-intelex, ```criterion``` must be set to "gini" and ```oob_score``` to False, as those parameters are not currently supported]. Defaults to True.

        n_estimators (int, optional): [The number of trees in the forest. Increasing this value can improves the fit, but at the cost of compute time]. Defaults to 100.

        criterion (str, optional): [The function to measure the quality of a split. Supported values are 'gini' for the Gini impurity and 'entropy' for the information gain]. Defaults to "gini".

        max_depth (int, optional): [The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples]. Defaults to None.

        min_samples_split (int or float, optional): [The minimum number of samples required to split an internal node. If value is an integer, then considers min_samples_split as the minimum number. If value is a floating point, then min_samples_split is a fraction and (min_samples_split * n_samples), rounded up to the nearest integer, are the minimum number of samples for each split]. Defaults to 2.

        min_samples_leaf (int or float, optional): [The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression. If value is an integer, then min_samples_leaf is the minimum number. If value is floating point, then min_samples_leaf is a fraction and (min_samples_leaf * n_samples) rounded up to the nearest integer is the minimum number of samples for each node]. Defaults to 1.

        min_weight_fraction_leaf (float, optional): [The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.]. Defaults to 0.0.

        max_features (int or float, optional): [The number of features to consider when looking for the best split. If int, then consider 'max_features' features at each split. If float, then 'max_features' is a fraction and (max_features * n_samples) features, rounded to the nearest integer, are considered at each split. If 'auto', then max_features=sqrt(n_features). If 'sqrt', then max_features=sqrt(n_features). If 'log2', then max_features=log2(n_features). If None, then max_features=n_features]. Defaults to "auto".

        max_leaf_nodes (int, optional): [Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes]. Defaults to None.

        min_impurity_decrease (float, optional): [A node will be split if this split induces a decrease of the impurity greater than or equal to this value. See sklearn.ensemble.ExtraTreesClassifier documentation for more information]. Defaults to 0.0.

        bootstrap (bool, optional): [Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree]. Defaults to False.

        oob_score (bool, optional): [Whether to use out-of-bag samples to estimate the generalization score. Only available if bootstrap=True]. Defaults to False.

        max_samples (int or float, optional): [If bootstrap is True, the number of samples to draw from X to train each base estimator. If None (default), then draws X.shape[0] samples. if int, then draw 'max_samples' samples. If float, then draw (max_samples * X.shape[0] samples) with max_samples in the interval (0, 1)]. Defaults to None.

        clf_random_state (int, optional): [Controls three sources of randomness for sklearn.ensemble.ExtraTreesClassifier: The bootstrapping of the samples used when building trees (if bootstrap=True), the sampling of the features to consider when looking for the best split at each node (if max_features < n_features), and the draw of the splits for each of the max_features. If None, then uses a different random seed each time the imputation is run]. Defaults to None.

        max_iter (int, optional): [Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values]. Defaults to 10.

        tol ([type], optional): [Tolerance of the stopping condition]. Defaults to 1e-3.

        n_nearest_features (int, optional): [Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge]. Defaults to 10.

        initial_strategy (str, optional): [Which strategy to use to initialize the missing values. Same as the strategy parameter in sklearn.impute.SimpleImputer Valid values: {“mean”, “median”, or “most_frequent”}]. Defaults to "most_frequent".

        imputation_order (str, optional): [The order in which the features will be imputed. Possible values: 'ascending' (from features with fewest missing values to most), 'descending' (from features with most missing values to fewest), 'roman' (left to right), 'arabic' (right to left), 'random' (a random order for each round). ]. Defaults to "ascending".

        skip_complete (bool, optional): [If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time]. Defaults to False.

        random_state (int, optional): [The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if n_nearest_features is not None or the imputation_order is 'random'. Use an integer for determinism. If None, then uses a different random seed each time]. Defaults to None.

        verbose (int, optional): [Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2]. Defaults to 0.
    """

    def __init__(
        self,
        genotype_data,
        *,
        prefix=None,
        gridparams=None,
        grid_iter=50,
        cv=5,
        validation_only=0.4,
        ga=False,
        population_size=10,
        tournament_size=3,
        elitism=True,
        crossover_probability=0.8,
        mutation_probability=0.1,
        ga_algorithm="eaMuPlusLambda",
        early_stop_gen=5,
        scoring_metric="accuracy",
        column_subset=0.1,
        chunk_size=1.0,
        disable_progressbar=False,
        progress_update_percent=None,
        n_jobs=1,
        extratrees=True,
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,
        oob_score=False,
        max_samples=None,
        clf_random_state=None,
        max_iter=10,
        tol=1e-3,
        n_nearest_features=10,
        initial_strategy="most_frequent",
        imputation_order="ascending",
        skip_complete=False,
        random_state=None,
        verbose=0,
    ):
        # Get local variables into dictionary object
        kwargs = locals()

        self.extratrees = kwargs.pop("extratrees")

        if self.extratrees:
            self.clf = ExtraTreesClassifier

        elif intelex and not self.extratrees:
            self.clf = RandomForestClassifier

            if kwargs["criterion"] != "gini":
                raise ValueError(
                    "criterion must be set to 'gini' if using the RandomForestClassifier with scikit-learn-intelex"
                )
            if kwargs["oob_score"]:
                raise ValueError(
                    "oob_score must be set to False if using the RandomForestClassifier with scikit-learn-intelex"
                )
        else:
            self.clf = RandomForestClassifier

        self.clf_type = "classifier"

        imputer = Impute(self.clf, self.clf_type, kwargs)

        self.imputed, self.best_params = imputer.fit_predict(
            genotype_data.genotypes_df
        )

        imputer.write_imputed(self.imputed)


class ImputeGradientBoosting:
    """[Does Gradient Boosting Iterative Imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process]

    Args:
        genotype_data ([GenotypeData]): [GenotypeData instance that was used to read in the sequence data]

        prefix ([str]): [Prefix for imputed data's output filename]

        gridparams (dict, optional): [Dictionary with lists as values or distributions of parameters. Distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). ```gridparams``` will be used for a randomized grid search. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). **NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. The full imputation can be performed by setting ```gridparams=None``` (default)]. Defaults to None.

        grid_iter (int, optional): [Number of iterations for randomized grid search]. Defaults to 50.

        cv (int, optional): [Number of folds for cross-validation during randomized grid search]. Defaults to 5.

        validation_only (float, optional): [Validates the imputation without doing a grid search. The validation method randomly replaces 15% to 50% of the known, non-missing genotypes in ``n_features * validation_only`` of the features. It then imputes the newly missing genotypes for which we know the true values and calculates validation scores. This procedure is replicated ``cv`` times and a mean, median, minimum, maximum, lower 95% confidence interval (CI) of the mean, and the upper 95% CI are calculated and saved to a CSV file. ``gridparams`` must be set to None (default) for ``validation_only`` to work. Calculating a validation score can be turned off altogether by setting ``validation_only`` to None]. Defaults to 0.4.

        ga (bool, optional): [Whether to use a genetic algorithm for the grid search. If False, a RandomizedSearchCV is done instead]. Defaults to False.

        population_size (int, optional): [For genetic algorithm grid search: Size of the initial population to sample randomly generated individuals. See GASearchCV documentation]. Defaults to 10.

        tournament_size (int, optional): [For genetic algorithm grid search: Number of individuals to perform tournament selection. See GASearchCV documentation]. Defaults to 3.

        elitism (bool, optional): [For genetic algorithm grid search:     If True takes the tournament_size best solution to the next generation. See GASearchCV documentation]. Defaults to True.

        crossover_probability (float, optional): [For genetic algorithm grid search: Probability of crossover operation between two individuals. See GASearchCV documentation]. Defaults to 0.8.

        mutation_probability (float, optional): [For genetic algorithm grid search: Probability of child mutation. See GASearchCV documentation]. Defaults to 0.1.

        ga_algorithm (str, optional): [For genetic algorithm grid search: Evolutionary algorithm to use. See more details in the deap algorithms documentation]. Defaults to "eaMuPlusLambda".

        early_stop_gen (int, optional): [If the genetic algorithm sees ```early_stop_gen``` consecutive generations without improvement in the scoring metric, an early stopping callback is implemented. This saves time by reducing the number of generations the genetic algorithm has to perform]. Defaults to 5.

        scoring_metric (str, optional): [Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options]. Defaults to "accuracy".

        column_subset (int or float, optional): [If float, proportion of the dataset to randomly subset for the grid search. Should be between 0 and 1, and should also be small, because the grid search takes a long time. If int, subset ```column_subset``` columns]. Defaults to 0.1.

        chunk_size (int or float, optional): [Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ```chunk_size``` loci at a time. If a float is specified, selects ```total_loci * chunk_size``` loci at a time]. Defaults to 1.0 (all features).

        disable_progressbar (bool, optional): [Whether or not to disable the tqdm progress bar when doing the imputation. If True, progress bar is disabled, which is useful when running the imputation on e.g. an HPC cluster. If the bar is disabled, a status update will be printed to standard output for each iteration and feature instead. If False, the tqdm progress bar will be used]. Defaults to False.

        progress_update_percent (int, optional): [Print status updates for features every ```progress_update_percent```%. IterativeImputer iterations will always be printed, but ```progress_update_percent``` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features]. Defaults to None.

        n_jobs (int, optional): [Number of parallel jobs to use. If ```gridparams``` is not None, n_jobs is used for the grid search. Otherwise it is used for the classifier. -1 means using all available processors]. Defaults to 1.

        n_estimators (int, optional): [The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance]. Defaults to 100.

        loss (str, optional): [The loss function to be optimized. ‘deviance’ refers to deviance (=logistic regression) for classification with probabilistic outputs. For loss ‘exponential’ gradient boosting recovers the AdaBoost algorithm]. Defaults to "deviance".

        learning_rate (float, optional): [Learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators]. Defaults to 0.1.

        subsample (float, optional): [The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting. subsample interacts with the parameter n_estimators. Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias]. Defaults to 1.0.

        criterion (str, optional): [The function to measure the quality of a split. Supported criteria are 'friedman_mse' for the mean squared error with improvement score by Friedman and 'mse' for mean squared error. The default value of 'friedman_mse' is generally the best as it can provide a better approximation in some cases]. Defaults to "friedman_mse".

        min_samples_split (int or float, optional): [The minimum number of samples required to split an internal node. If value is an integer, then consider min_samples_split as the minimum number. If value is a floating point, then min_samples_split is a fraction and (min_samples_split * n_samples) is rounded up to the nearest integer and used as the number of samples per split]. Defaults to 2.

        min_samples_leaf (int or float, optional): [The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression. If value is an integer, consider min_samples_leaf as the minimum number. If value is a floating point, then min_samples_leaf is a fraction and (min_samples_leaf * n_samples) rounded up to the nearest integer is the minimum number of samples per node]. Defaults to 1.

        min_weight_fraction_leaf (float, optional): [The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided]. Defaults to 0.0.

        max_depth (int, optional): [The maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables.]. Defaults to 3.

        min_impurity_decrease (float, optional): [A node will be split if this split induces a decrease of the impurity greater than or equal to this value]. Defaults to 0.0. See sklearn.ensemble.GradientBoostingClassifier documentation for more information]. Defaults to 0.0.

        max_features (int, float, or str, optional): [The number of features to consider when looking for the best split. If value is an integer, then consider 'max_features' features at each split. If value is a floating point, then 'max_features' is a fraction and (max_features * n_features) is rounded to the nearest integer and considered as the number of features per split. If 'auto', then max_features=sqrt(n_features). If 'sqrt', then max_features=sqrt(n_features). If 'log2', then max_features=log2(n_features). If None, then max_features=n_features]. Defaults to None.

        max_leaf_nodes (int, optional): [Grow trees with 'max_leaf_nodes' in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then uses an unlimited number of leaf nodes]. Defaults to None.

        clf_random_state (int, optional): [Controls the random seed given to each Tree estimator at each boosting iteration. In addition, it controls the random permutation of the features at each split. Pass an int for reproducible output across multiple function calls. If None, then uses a different random seed for each function call]. Defaults to None.

        max_iter (int, optional): [Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values]. Defaults to 10.

        tol ([type], optional): [Tolerance of the stopping condition]. Defaults to 1e-3.

        n_nearest_features (int, optional): [Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge]. Defaults to 10.

        initial_strategy (str, optional): [Which strategy to use to initialize the missing values. Same as the strategy parameter in sklearn.impute.SimpleImputer Valid values: {“mean”, “median”, or “most_frequent”}]. Defaults to "populations".

        imputation_order (str, optional): [The order in which the features will be imputed. Possible values: 'ascending' (from features with fewest missing values to most), 'descending' (from features with most missing values to fewest), 'roman' (left to right), 'arabic' (right to left), 'random' (a random order for each round). ]. Defaults to "ascending".

        skip_complete (bool, optional): [If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time]. Defaults to False.

        random_state (int, optional): [The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if n_nearest_features is not None or the imputation_order is 'random'. Use an integer for determinism. If None, then uses a different random seed each time]. Defaults to None.

        verbose (int, optional): [Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2]. Defaults to 0.
    """

    def __init__(
        self,
        genotype_data,
        *,
        prefix=None,
        gridparams=None,
        grid_iter=50,
        cv=5,
        validation_only=0.4,
        ga=False,
        population_size=10,
        tournament_size=3,
        elitism=True,
        crossover_probability=0.8,
        mutation_probability=0.1,
        ga_algorithm="eaMuPlusLambda",
        early_stop_gen=5,
        scoring_metric="accuracy",
        column_subset=0.1,
        chunk_size=1.0,
        disable_progressbar=False,
        progress_update_percent=None,
        n_jobs=1,
        n_estimators=100,
        loss="deviance",
        learning_rate=0.1,
        subsample=1.0,
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        max_features=None,
        max_leaf_nodes=None,
        clf_random_state=None,
        max_iter=10,
        tol=1e-3,
        n_nearest_features=10,
        initial_strategy="most_frequent",
        imputation_order="ascending",
        skip_complete=False,
        random_state=None,
        verbose=0,
    ):
        # Get local variables into dictionary object
        kwargs = locals()

        self.clf_type = "classifier"
        self.clf = GradientBoostingClassifier

        imputer = Impute(self.clf, self.clf_type, kwargs)

        self.imputed, self.best_params = imputer.fit_predict(
            genotype_data.genotypes_df
        )

        imputer.write_imputed(self.imputed)


class ImputeBayesianRidge:
    """[NOTE: This is a regressor estimator and is only intended for testing purposes, as it is faster than the classifiers. Does Bayesian Ridge Iterative Imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process]

    Args:
        genotype_data ([GenotypeData]): [GenotypeData instance that was used to read in the sequence data]

        prefix ([str]): [Prefix for imputed data's output filename]

        gridparams (dict, optional): [Dictionary with lists as values or distributions of parameters. Distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). ```gridparams``` will be used for a randomized grid search. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). **NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. The full imputation can be performed by setting ```gridparams=None``` (default)]. Defaults to None.

        grid_iter (int, optional): [Number of iterations for randomized grid search]. Defaults to 50.

        cv (int, optional): [Number of folds for cross-validation during randomized grid search]. Defaults to 5.

        validation_only (float, optional): [Validates the imputation without doing a grid search. The validation method randomly replaces 15% to 50% of the known, non-missing genotypes in ``n_features * validation_only`` of the features. It then imputes the newly missing genotypes for which we know the true values and calculates validation scores. This procedure is replicated ``cv`` times and a mean, median, minimum, maximum, lower 95% confidence interval (CI) of the mean, and the upper 95% CI are calculated and saved to a CSV file. ``gridparams`` must be set to None (default) for ``validation_only`` to work. Calculating a validation score can be turned off altogether by setting ``validation_only`` to None]. Defaults to 0.4.

        ga (bool, optional): [Whether to use a genetic algorithm for the grid search. If False, a RandomizedSearchCV is done instead]. Defaults to False.

        population_size (int, optional): [For genetic algorithm grid search: Size of the initial population to sample randomly generated individuals. See GASearchCV documentation]. Defaults to 10.

        tournament_size (int, optional): [For genetic algorithm grid search: Number of individuals to perform tournament selection. See GASearchCV documentation]. Defaults to 3.

        elitism (bool, optional): [For genetic algorithm grid search:     If True takes the tournament_size best solution to the next generation. See GASearchCV documentation]. Defaults to True.

        crossover_probability (float, optional): [For genetic algorithm grid search: Probability of crossover operation between two individuals. See GASearchCV documentation]. Defaults to 0.8.

        mutation_probability (float, optional): [For genetic algorithm grid search: Probability of child mutation. See GASearchCV documentation]. Defaults to 0.1.

        ga_algorithm (str, optional): [For genetic algorithm grid search: Evolutionary algorithm to use. See more details in the deap algorithms documentation]. Defaults to "eaMuPlusLambda".

        early_stop_gen (int, optional): [If the genetic algorithm sees ```early_stop_gen``` consecutive generations without improvement in the scoring metric, an early stopping callback is implemented. This saves time by reducing the number of generations the genetic algorithm has to perform]. Defaults to 5.

        scoring_metric (str, optional): [Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options]. Defaults to "accuracy".

        column_subset (int or float, optional): [If float, proportion of the dataset to randomly subset for the grid search. Should be between 0 and 1, and should also be small, because the grid search takes a long time. If int, subset ```column_subset``` columns]. Defaults to 0.1.

        chunk_size (int or float, optional): [Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ```chunk_size``` loci at a time. If a float is specified, selects ```total_loci * chunk_size``` loci at a time]. Defaults to 1.0 (all features).

        disable_progressbar (bool, optional): [Whether or not to disable the tqdm progress bar when doing the imputation. If True, progress bar is disabled, which is useful when running the imputation on e.g. an HPC cluster. If the bar is disabled, a status update will be printed to standard output for each iteration and feature instead. If False, the tqdm progress bar will be used]

        progress_update_percent (int, optional): [Print status updates for features every ```progress_update_percent```%. IterativeImputer iterations will always be printed, but ```progress_update_percent``` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features]. Defaults to None.

        n_jobs (int, optional): [Number of parallel jobs to use. If ```gridparams``` is not None, n_jobs is used for the grid search. Otherwise it is used for the regressor. -1 means using all available processors]. Defaults to 1.

        n_iter (int, optional): [Maximum number of iterations. Should be greater than or equal to 1]. Defaults to 300.

        clf_tol (float, optional): [Stop the algorithm if w has converged]. Defaults to 1e-3.

        alpha_1 (float, optional): [Hyper-parameter: shape parameter for the Gamma distribution prior over the alpha parameter]. Defaults to 1e-6.

        alpha_2 (float, optional): [Hyper-parameter: inverse scale parameter (rate parameter) for the Gamma distribution prior over the alpha parameter]. Defaults to 1e-6.

        lambda_1 (float, optional): [Hyper-parameter: shape parameter for the Gamma distribution prior over the lambda parameter]. Defaults to 1e-6.

        lambda_2 (float, optional): [Hyper-parameter: inverse scale parameter (rate parameter) for the Gamma distribution prior over the lambda parameter]. Defaults to 1e-6.

        alpha_init (float, optional): [Initial value for alpha (precision of the noise). If None, alpha_init is 1/Var(y).]. Defaults to None.

        lambda_init (float, optional): [Initial value for lambda (precision of the weights). If None, lambda_init is 1]. Defaults to None.

        max_iter (int, optional): [Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values]. Defaults to 10.

        tol ([type], optional): [Tolerance of the stopping condition]. Defaults to 1e-3.

        n_nearest_features (int, optional): [Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge]. Defaults to 10.

        initial_strategy (str, optional): [Which strategy to use to initialize the missing values. Same as the strategy parameter in sklearn.impute.SimpleImputer Valid values: {“mean”, “median”, or “most_frequent”}]. Defaults to "most_frequent".

        imputation_order (str, optional): [The order in which the features will be imputed. Possible values: 'ascending' (from features with fewest missing values to most), 'descending' (from features with most missing values to fewest), 'roman' (left to right), 'arabic' (right to left), 'random' (a random order for each round). ]. Defaults to "ascending".

        skip_complete (bool, optional): [If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time]. Defaults to False.

        random_state (int, optional): [The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if n_nearest_features is not None or the imputation_order is 'random'. Use an integer for determinism. If None, then uses a different random seed each time]. Defaults to None.

        verbose (int, optional): [Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2]. Defaults to 0.
    """

    def __init__(
        self,
        genotype_data,
        *,
        prefix=None,
        gridparams=None,
        grid_iter=50,
        cv=5,
        validation_only=0.4,
        ga=False,
        population_size=10,
        tournament_size=3,
        elitism=True,
        crossover_probability=0.8,
        mutation_probability=0.1,
        ga_algorithm="eaMuPlusLambda",
        early_stop_gen=5,
        scoring_metric="neg_root_mean_squared_error",
        column_subset=0.1,
        chunk_size=1.0,
        disable_progressbar=False,
        progress_update_percent=None,
        n_jobs=1,
        n_iter=300,
        clf_tol=1e-3,
        alpha_1=1e-6,
        alpha_2=1e-6,
        lambda_1=1e-6,
        lambda_2=1e-6,
        alpha_init=None,
        lambda_init=None,
        max_iter=10,
        tol=1e-3,
        n_nearest_features=10,
        initial_strategy="most_frequent",
        imputation_order="ascending",
        skip_complete=False,
        random_state=None,
        verbose=0,
    ):
        # Get local variables into dictionary object
        kwargs = locals()
        kwargs["normalize"] = True
        kwargs["sample_posterior"] = False

        self.clf_type = "regressor"
        self.clf = BayesianRidge

        imputer = Impute(self.clf, self.clf_type, kwargs)

        self.imputed, self.best_params = imputer.fit_predict(
            genotype_data.genotypes_df
        )

        imputer.write_imputed(self.imputed)


class ImputeXGBoost:
    """[Does XGBoost (Extreme Gradient Boosting) Iterative imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process. The grid searches are not compatible with XGBoost, but validation scores can still be calculated without a grid search. In addition, ImputeLightGBM is a similar algorithm and is compatible with grid searches, so use ImputeLightGBM if you want a grid search]

    Args:
        genotype_data ([GenotypeData]): [GenotypeData instance that was used to read in the sequence data]

        prefix ([str]): [Prefix for imputed data's output filename]

        cv (int, optional): [Number of folds for cross-validation during randomized grid search]. Defaults to 5.

        validation_only (float, optional): [Validates the imputation without doing a grid search. The validation method randomly replaces 15% to 50% of the known, non-missing genotypes in ``n_features * validation_only`` of the features. It then imputes the newly missing genotypes for which we know the true values and calculates validation scores. This procedure is replicated ``cv`` times and a mean, median, minimum, maximum, lower 95% confidence interval (CI) of the mean, and the upper 95% CI are calculated and saved to a CSV file. ``gridparams`` must be set to None (default) for ``validation_only`` to work. Calculating a validation score can be turned off altogether by setting ``validation_only`` to None]. Defaults to 0.4.

        scoring_metric (str, optional): [Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options]. Defaults to "accuracy".

        chunk_size (int or float, optional): [Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ```chunk_size``` loci at a time. If a float is specified, selects ```total_loci * chunk_size``` loci at a time]. Defaults to 1.0 (all features).

        disable_progressbar (bool, optional): [Whether or not to disable the tqdm progress bar when doing the imputation. If True, progress bar is disabled, which is useful when running the imputation on e.g. an HPC cluster. If the bar is disabled, a status update will be printed to standard output for each iteration and feature instead. If False, the tqdm progress bar will be used]. Defaults to False.

        progress_update_percent (int, optional): [Print status updates for features every ```progress_update_percent```%. IterativeImputer iterations will always be printed, but ```progress_update_percent``` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features]. Defaults to None.

        n_jobs (int, optional): [Number of parallel jobs to use. If ```gridparams``` is not None, n_jobs is used for the grid search. Otherwise it is used for the classifier. -1 means using all available processors]. Defaults to 1.

        n_estimators (int, optional): [The number of boosting rounds. Increasing this value can improve the fit, but at the cost of compute time and RAM usage]. Defaults to 100.

        max_depth (int, optional): [Maximum tree depth for base learners]. Defaults to 3.

        learning_rate (float, optional): [Boosting learning rate (eta). Basically, it serves as a weighting factor for correcting new trees when they are added to the model. Typical values are between 0.1 and 0.3. Lower learning rates generally find the best optimum at the cost of requiring far more compute time and resources]. Defaults to 0.1.

        booster (str, optional): [Specify which booster to use. Possible values include "gbtree", "gblinear", and "dart"]. Defaults to "gbtree".

        gamma (float, optional): [Minimum loss reduction required to make a further partition on a leaf node of the tree]. Defaults to 0.0.

        min_child_weight (float, optional): [Minimum sum of instance weight(hessian) needed in a child]. Defaults to 1.0.

        max_delta_step (float, optional): [Maximum delta step we allow each tree's weight estimation to be]. Defaults to 0.0.

        subsample (float, optional): [Subsample ratio of the training instance]. Defaults to 1.0.

        colsample_bytree (float, optional): [Subsample ratio of columns when constructing each tree]. Defaults to 1.0.

        reg_lambda (float, optional): [L2 regularization term on weights (xgb's lambda parameter)]. Defaults to 1.0.

        reg_alpha (float, optional): [L1 regularization term on weights (xgb's alpha parameter)]. Defaults to 1.0.

        clf_random_state (int or numpy.random.RandomState object, optional): [Controls three sources of randomness for sklearn.ensemble.ExtraTreesClassifier: The bootstrapping of the samples used when building trees (if bootstrap=True), the sampling of the features to consider when looking for the best split at each node (if max_features < n_features), and the draw of the splits for each of the max_features. If None, then uses a different random seed each time the imputation is run]. Defaults to None.

        max_iter (int, optional): [Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values]. Defaults to 10.

        tol ([type], optional): [Tolerance of the stopping condition]. Defaults to 1e-3.

        n_nearest_features (int, optional): [Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge]. Defaults to 10.

        initial_strategy (str, optional): [Which strategy to use to initialize the missing values. Same as the strategy parameter in sklearn.impute.SimpleImputer Valid values: {“mean”, “median”, or “most_frequent”}]. Defaults to "most_frequent".

        imputation_order (str, optional): [The order in which the features will be imputed. Possible values: 'ascending' (from features with fewest missing values to most), 'descending' (from features with most missing values to fewest), 'roman' (left to right), 'arabic' (right to left), 'random' (a random order for each round). ]. Defaults to "ascending".

        skip_complete (bool, optional): [If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time]. Defaults to False.

        random_state (int, optional): [The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if n_nearest_features is not None or the imputation_order is 'random'. Use an integer for determinism. If None, then uses a different random seed each time]. Defaults to None.

        verbose (int, optional): [Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2]. Defaults to 0.
    """

    def __init__(
        self,
        genotype_data,
        *,
        prefix=None,
        cv=5,
        validation_only=0.4,
        scoring_metric="accuracy",
        chunk_size=1.0,
        disable_progressbar=False,
        progress_update_percent=None,
        n_jobs=1,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        booster="gbtree",
        gamma=0,
        min_child_weight=1,
        max_delta_step=0,
        subsample=1,
        colsample_bytree=1,
        reg_lambda=1,
        reg_alpha=0,
        clf_random_state=None,
        n_nearest_features=10,
        max_iter=10,
        tol=1e-3,
        initial_strategy="most_frequent",
        imputation_order="ascending",
        skip_complete=False,
        random_state=None,
        verbose=0,
    ):
        # Get local variables into dictionary object
        kwargs = locals()
        kwargs["gridparams"] = None
        # kwargs["num_class"] = 3
        # kwargs["use_label_encoder"] = False

        self.clf_type = "classifier"
        self.clf = xgb.XGBClassifier
        kwargs["verbosity"] = verbose

        imputer = Impute(self.clf, self.clf_type, kwargs)

        self.imputed, self.best_params = imputer.fit_predict(
            genotype_data.genotypes_df
        )

        imputer.write_imputed(self.imputed)


class ImputeLightGBM:
    """[Does LightGBM (Light Gradient Boosting) Iterative imputation of missing data. LightGBM is an alternative to XGBoost that is around 7X faster and uses less memory, while still maintaining high accuracy. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process]

    Args:
        genotype_data ([GenotypeData]): [GenotypeData instance that was used to read in the sequence data]

        prefix ([str]): [Prefix for imputed data's output filename]

        gridparams (dict, optional): [Dictionary with lists as values or distributions of parameters. Distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). ```gridparams``` will be used for a randomized grid search. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). **NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. The full imputation can be performed by setting ```gridparams=None``` (default)]. Defaults to None.

        grid_iter (int, optional): [Number of iterations for randomized grid search]. Defaults to 50.

        cv (int, optional): [Number of folds for cross-validation during randomized grid search]. Defaults to 5.

        validation_only (float, optional): [Validates the imputation without doing a grid search. The validation method randomly replaces 15% to 50% of the known, non-missing genotypes in ``n_features * validation_only`` of the features. It then imputes the newly missing genotypes for which we know the true values and calculates validation scores. This procedure is replicated ``cv`` times and a mean, median, minimum, maximum, lower 95% confidence interval (CI) of the mean, and the upper 95% CI are calculated and saved to a CSV file. ``gridparams`` must be set to None (default) for ``validation_only`` to work. Calculating a validation score can be turned off altogether by setting ``validation_only`` to None]. Defaults to 0.4.

        ga (bool, optional): [Whether to use a genetic algorithm for the grid search. If False, a RandomizedSearchCV is done instead]. Defaults to False.

        population_size (int, optional): [For genetic algorithm grid search: Size of the initial population to sample randomly generated individuals. See GASearchCV documentation]. Defaults to 10.

        tournament_size (int, optional): [For genetic algorithm grid search: Number of individuals to perform tournament selection. See GASearchCV documentation]. Defaults to 3.

        elitism (bool, optional): [For genetic algorithm grid search: If True takes the tournament_size best solution to the next generation. See GASearchCV documentation]. Defaults to True.

        crossover_probability (float, optional): [For genetic algorithm grid search: Probability of crossover operation between two individuals. See GASearchCV documentation]. Defaults to 0.8.

        mutation_probability (float, optional): [For genetic algorithm grid search: Probability of child mutation. See GASearchCV documentation]. Defaults to 0.1.

        ga_algorithm (str, optional): [For genetic algorithm grid search: Evolutionary algorithm to use. See more details in the deap algorithms documentation]. Defaults to "eaMuPlusLambda".

        early_stop_gen (int, optional): [If the genetic algorithm sees ```early_stop_gen``` consecutive generations without improvement in the scoring metric, an early stopping callback is implemented. This saves time by reducing the number of generations the genetic algorithm has to perform]. Defaults to 5.

        scoring_metric (str, optional): [Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options]. Defaults to "accuracy".

        column_subset (int or float, optional): [If float, proportion of the dataset to randomly subset for the grid search. Should be between 0 and 1, and should also be small, because the grid search takes a long time. If int, subset ```column_subset``` columns]. Defaults to 0.1.

        chunk_size (int or float, optional): [Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ```chunk_size``` loci at a time. If a float is specified, selects ```total_loci * chunk_size``` loci at a time]. Defaults to 1.0 (all features).

        disable_progressbar (bool, optional): [Whether or not to disable the tqdm progress bar when doing the imputation. If True, progress bar is disabled, which is useful when running the imputation on e.g. an HPC cluster. If the bar is disabled, a status update will be printed to standard output for each iteration and feature instead. If False, the tqdm progress bar will be used]. Defaults to False.

        progress_update_percent (int, optional): [Print status updates for features every ```progress_update_percent```%. IterativeImputer iterations will always be printed, but ```progress_update_percent``` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features]. Defaults to None.

        n_jobs (int, optional): [Number of parallel jobs to use. If ```gridparams``` is not None, n_jobs is used for the grid search. Otherwise it is used for the classifier. -1 means using all available processors]. Defaults to 1.

        n_estimators (int, optional): [The number of boosting rounds. Increasing this value can improve the fit, but at the cost of compute time and RAM usage]. Defaults to 100.

        max_depth (int, optional): [Maximum tree depth for base learners]. Defaults to 3.

        learning_rate (float, optional): [Boosting learning rate (eta). Basically, it serves as a weighting factor for correcting new trees when they are added to the model. Typical values are between 0.1 and 0.3. Lower learning rates generally find the best optimum at the cost of requiring far more compute time and resources]. Defaults to 0.1.

        boosting_type (str, optional): [Possible values: "gbdt", traditional Gradient Boosting Decision Tree. "dart", Dropouts meet Multiple Additive Regression Trees. "goss", Gradient-based One-Side Sampling. The "rf" option is not currently supported]. Defaults to "gbdt".

        num_leaves (int, optional): [Maximum tree leaves for base learners]. Defaults to 31.

        subsample_for_bin (int, optional): [Number of samples for constructing bins]. Defaults to 200000.

        min_split_gain (float, optional): [Minimum loss reduction required to make a further partition on a leaf node of the tree]. Defaults to 0.0.

        min_child_weight (float, optional): [Minimum sum of instance weight (hessian) needed in a child (leaf)]. Defaults to 1e-3.

        min_child_samples (int, optional): [Minimum number of data needed in a child (leaf)]. Defaults to 20.

        subsample (float, optional): [Subsample ratio of the training instance]. Defaults to 1.0.

        subsample_freq (int, optional): [Frequency of subsample, <=0 means no enable]. Defaults to 0.

        colsample_bytree (float, optional): [Subsample ratio of columns when constructing each tree]. Defaults to 1.0.

        reg_lambda (float, optional): [L2 regularization term on weights]. Defaults to 0.

        reg_alpha (float, optional): [L1 regularization term on weights]. Defaults to 0.

        clf_random_state (int, numpy.random.RandomState object, or None): [Random number seed. If int, this number is used to seed the C++ code. If RandomState object (numpy), a random integer is picked based on its state to seed the C++ code. If None, default seeds in C++ code are used]. Defaults to None.

        silent (bool, optional): [Whether to print messages while running boosting]. Defaults to True.

        max_iter (int, optional): [Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values]. Defaults to 10.

        tol ([type], optional): [Tolerance of the stopping condition]. Defaults to 1e-3.

        n_nearest_features (int, optional): [Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge]. Defaults to 10.

        initial_strategy (str, optional): [Which strategy to use to initialize the missing values. Same as the strategy parameter in sklearn.impute.SimpleImputer Valid values: {“mean”, “median”, or “most_frequent”}]. Defaults to "most_frequent".

        imputation_order (str, optional): [The order in which the features will be imputed. Possible values: 'ascending' (from features with fewest missing values to most), 'descending' (from features with most missing values to fewest), 'roman' (left to right), 'arabic' (right to left), 'random' (a random order for each round). ]. Defaults to "ascending".

        skip_complete (bool, optional): [If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time]. Defaults to False.

        random_state (int, optional): [The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if n_nearest_features is not None or the imputation_order is 'random'. Use an integer for determinism. If None, then uses a different random seed each time]. Defaults to None.

        verbose (int, optional): [Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2]. Defaults to 0.
    """

    def __init__(
        self,
        genotype_data,
        *,
        prefix=None,
        gridparams=None,
        grid_iter=50,
        cv=5,
        validation_only=0.4,
        ga=False,
        population_size=10,
        tournament_size=3,
        elitism=True,
        crossover_probability=0.8,
        mutation_probability=0.1,
        ga_algorithm="eaMuPlusLambda",
        early_stop_gen=5,
        scoring_metric="accuracy",
        column_subset=0.1,
        chunk_size=1.0,
        disable_progressbar=False,
        progress_update_percent=None,
        n_jobs=1,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        boosting_type="gbdt",
        num_leaves=31,
        subsample_for_bin=200000,
        min_split_gain=0.0,
        min_child_weight=1e-3,
        min_child_samples=20,
        subsample=1.0,
        subsample_freq=0,
        colsample_bytree=1.0,
        reg_lambda=0,
        reg_alpha=0,
        clf_random_state=None,
        silent=True,
        n_nearest_features=10,
        max_iter=10,
        tol=1e-3,
        initial_strategy="most_frequent",
        imputation_order="ascending",
        skip_complete=False,
        random_state=None,
        verbose=0,
    ):

        # Get local variables into dictionary object
        kwargs = locals()

        if kwargs["boosting_type"] == "rf":
            raise ValueError("boosting_type 'rf' is not supported!")

        self.clf_type = "classifier"
        self.clf = lgbm.LGBMClassifier

        imputer = Impute(self.clf, self.clf_type, kwargs)

        self.imputed, self.best_params = imputer.fit_predict(
            genotype_data.genotypes_df
        )

        imputer.write_imputed(self.imputed)


class ImputePhylo(GenotypeData):
    """[Impute missing data using a phylogenetic tree to inform the imputation]

    Args:
        phylipfile ([str]): [Path to PHYLIP-formatted file to impute]

        treefile ([str]): [Path to Newick-formatted phylogenetic tree file]

        qmatrix ([str]): [Path to *.iqtree file containing Rate Matrix Q table]

        prefix (str, optional): [Prefix to use with output files]

        save_plots (str, optional): [Whether to save PDF files with genotype imputations for each site. It makes one PDF file per locus, so if you have a lot of loci it will make a lot of PDF files]. Defaults to False.
    """

    def __init__(
        self, phylipfile, treefile, qmatrix, prefix="phylo", save_plots=False
    ):
        self.phylipfile = phylipfile
        self.treefile = treefile
        self.qmatrix = qmatrix
        self.prefix = prefix
        self.save_plots = save_plots

        super().__init__()

        data = self.read_phylip_tree_imputation(self.phylipfile)
        tree = self.read_tree(self.treefile)
        q = self.q_from_iqtree(self.qmatrix)

        imputed = self.impute_phylo(tree, data, q)

        outfile = f"{prefix}_imputed_012.csv"
        imputed.to_csv(outfile, header=False, index=False)

    def q_from_file(self, fname, label=True):
        q = self.blank_q_matrix()

        if not label:
            print(
                "Warning: Assuming the following nucleotide order: A, C, G, T"
            )

        with open(fname, "r") as fin:
            header = True
            qlines = list()
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                if header:
                    if label:
                        order = line.split()
                        header = False
                    else:
                        order = ["A", "C", "G", "T"]
                    continue
                else:
                    qlines.append(line.split())
        fin.close()

        for l in qlines:
            for index in range(0, 4):
                q[l[0]][order[index]] = float(l[index + 1])
        qdf = pd.DataFrame(q)
        return qdf.T

    def print_q(self, q):
        print("Rate matrix Q:")
        print("\tA\tC\tG\tT\t")
        for nuc1 in ["A", "C", "G", "T"]:
            print(nuc1, end="\t")
            for nuc2 in ["A", "C", "G", "T"]:
                print(q[nuc1][nuc2], end="\t")
            print("")

    def blank_q_matrix(self, default=0.0):
        q = dict()
        for nuc1 in ["A", "C", "G", "T"]:
            q[nuc1] = dict()
            for nuc2 in ["A", "C", "G", "T"]:
                q[nuc1][nuc2] = default
        return q

    def impute_phylo(
        self, tree, genotypes, Q, site_rates=None, exclude_N=False
    ):
        """[Imputes genotype values by using a provided guide
        tree to inform the imputation, assuming maximum parsimony]

        Sketch:
            For each SNP:
            1) if site_rates, get site-transformated Q matrix

            2) Postorder traversal of tree to compute ancestral
            state likelihoods for internal nodes (tips -> root)
            If exclude_N==True, then ignore N tips for this step

            3) Preorder traversal of tree to populate missing genotypes
            with the maximum likelihood state (root -> tips)

        Args:
            tree ([toytree object]): [Toytree object]

            genotypes ([dict(list)]): [Dictionary with key=sampleids, value=sequences]

            Q ([pandas.DataFrame]): [Rate Matrix Q from .iqtree file]
        """
        # For each SNP:
        nsites = list(set([len(v) for v in genotypes.values()]))
        assert len(nsites) == 1, "Some sites have different lengths!"

        outdir = f"{self.prefix}_imputation_plots"

        if self.save_plots:
            Path(outdir).mkdir(parents=True, exist_ok=True)

        for snp_index in progressbar(
            range(nsites[0]), desc="Feature Progress: ", leave=True
        ):
            node_lik = dict()

            # LATER: Need to get site rates
            rate = 1.0

            site_Q = Q.copy(deep=True) * rate

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
                    pt = self.transition_probs(site_Q, child.dist)
                    if child.is_leaf():
                        if child.name in genotypes:
                            # get genotype
                            sum = None

                            for allele in self.get_iupac_full(
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
                        l = self.get_internal_lik(pt, node_lik[child.idx])
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
                        if self.allMissing(
                            tree, node.idx, snp_index, genotypes
                        ):
                            dist += node.dist
                            node = node.up

                        else:
                            pt = self.transition_probs(site_Q, dist)
                            lik = self.get_internal_lik(pt, node_lik[node.idx])
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
                self.draw_imputed_position(
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

        gt = df.to_numpy().tolist()
        return pd.DataFrame.from_records(self.convert_012(gt, impute_mode=True))

    def get_nuc_colors(self, nucs):
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

    def label_bads(self, tips, labels, bads):
        for i, t in enumerate(tips):
            if t in bads:
                labels[i] = "*" + str(labels[i]) + "*"
        return labels

    def draw_imputed_position(self, tree, bads, genotypes, pos, out="tree.pdf"):
        # print(tree.get_tip_labels())
        sizes = [8 if i in bads else 0 for i in tree.get_tip_labels()]
        colors = [genotypes[i][pos] for i in tree.get_tip_labels()]
        labels = colors

        labels = self.label_bads(tree.get_tip_labels(), labels, bads)

        colors = self.get_nuc_colors(colors)

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

        canvas, axes, mark = tree.draw(
            tip_labels_colors=colors,
            # node_sizes = sizes,
            tip_labels=labels,
            width=400,
            height=600,
            **mystyle,
        )

        toyplot.pdf.render(canvas, out)

    def allMissing(self, tree, node_index, snp_index, genotypes):
        for des in tree.get_tip_labels(idx=node_index):
            if genotypes[des][snp_index].upper() not in ["N", "-"]:
                return False
        return True

    def get_internal_lik(self, pt, lik_arr):
        ret = list()
        for i, val in enumerate(lik_arr):

            col = list(pt.iloc[:, i])
            sum = 0.0
            for v in col:
                sum += v * val
            ret.append(sum)
        return ret

    def transition_probs(self, Q, t):
        ret = Q.copy(deep=True)
        m = Q.to_numpy()
        pt = scipy.linalg.expm(m * t)
        ret[:] = pt
        return ret

    def get_iupac_full(self, char):
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


class ImputeBackPropogation:
    def __init__(
        self,
        genotype_data,
        *,
        num_reduced_dims=3,
        hidden_layers=3,
        hidden_layer_sizes=list(),
    ):

        self.X = genotype_data.genotypes_nparray

        assert hidden_layers == len(hidden_layer_sizes) and hidden_layers > 0, (
            f"Hidden layers must be greater than 0 and of the same length as "
            f"hidden_layer_sizes, but got hidden_layers={hidden_layers} and "
            f"len(hidden_layer_sizes) == {len(hidden_layer_sizes)}"
        )

        self.Xt = None
        self.invalid_mask = np.isnan(self.X)
        self.valid_mask = np.where(~self.invalid_mask)
        self.l = hidden_layers
        self.V = np.random.randn(self.X.shape[0], num_reduced_dims)
        self.total_epochs = 0
        self.x_r = T.vector()
        self.learning_rate = T.scalar("eta")
        self.c = T.iscalar()
        self.r = T.iscalar()

        self.V = theano.shared(
            np.array(
                np.random.rand(self.X.shape[0], num_reduced_dims),
                dtype=theano.config.floatX,
            )
        )

        self.weights = list()

        self.U = theano.shared(
            np.array(
                np.random.rand(num_reduced_dims, self.X.shape[1]),
                dtype=theano.config.floatX,
            )
        )

        self.single_layer = nnet.sigmoid(T.dot(self.U.T, self.V[self.r, :]))

        self.layers = list()

        for i in range(hidden_layers):
            if i == 0:
                self.weights.append(
                    self.initialize_weights(
                        (num_reduced_dims, hidden_layer_sizes[0])
                    )
                )

            else:
                self.weights.append(
                    self.initialize_weights(
                        (hidden_layer_sizes[i - 1], hidden_layer_sizes[i])
                    )
                )

        self.weights.append(
            self.initialize_weights((hidden_layer_sizes[-1], self.X.shape[1]))
        )

        for i in range(hidden_layers):
            if i == 0:
                self.layers.append(
                    nnet.sigmoid(T.dot(self.weights[i].T, self.V[self.r, :]))
                )

            else:
                self.layers.append(
                    nnet.sigmoid(T.dot(self.weights[i].T, self.layers[-1]))
                )

        self.layers.append(
            nnet.sigmoid(T.dot(self.weights[-1].T, self.layers[-1]))
        )

        self.fc1 = ((self.single_layer - self.x_r) ** 2)[self.c]
        self.fc = ((self.layers[-1] - self.x_r) ** 2)[self.c]

        self.phases = list()

        self.phases.append(
            theano.function(
                inputs=[
                    self.x_r,
                    self.r,
                    self.c,
                    theano.In(self.learning_rate, value=0.1),
                ],
                outputs=self.fc1,
                updates=[
                    (
                        self.U,
                        self.loss_func(self.fc1, self.U, self.learning_rate),
                    ),
                    (
                        self.V,
                        self.loss_func(self.fc1, self.V, self.learning_rate),
                    ),
                ],
            )
        )

        self.phases.append(
            theano.function(
                inputs=[
                    self.x_r,
                    self.r,
                    self.c,
                    theano.In(self.learning_rate, value=0.1),
                ],
                outputs=self.fc,
                updates=[
                    (theta, self.loss_func(self.fc, theta, self.learning_rate))
                    for theta in self.weights
                ],
            )
        )

        self.phases.append(
            theano.function(
                inputs=[
                    self.x_r,
                    self.r,
                    self.c,
                    theano.In(self.learning_rate, value=0.1),
                ],
                outputs=self.fc,
                updates=[
                    (theta, self.loss_func(self.fc, theta, self.learning_rate))
                    for theta in self.weights
                ]
                + [
                    (
                        self.V,
                        self.loss_func(self.fc, self.V, self.learning_rate),
                    )
                ],
            )
        )

        self.run_phase1 = theano.function(
            inputs=[self.r], outputs=self.single_layer
        )

        self.run = theano.function(inputs=[self.r], outputs=self.layers[-1])

        self.fit_predict()
        imputed = self.get_Xt()
        print(imputed)

    @timer
    def fit_predict(self):
        print("Doing Unsupervised Back-Propogation Imputation...")

        self.fit(phase=2)

        print(f"Initial RMSE: {self.get_rmse()}")

        for i in range(1):
            print(f"Phase {(i + 1)}")

            self.initialize_params()

            while self.current_eta > self.target_eta:
                self.s = self.train_epoch(phase=i)

                if 1 - self.s / self.s_ < self.gamma:
                    self.current_eta = self.current_eta / 2
                    print(f"Reduced eta to {self.current_eta}")

                self.s_ = self.s
                self.num_epochs += 1
                self.total_epochs += 1
                self.print_num_epochs()

    def fit(self, phase=2):
        # Initialize numpy array
        self.Xt = np.zeros(self.X.shape)

        if phase == 2 or phase == 3:
            for r in range(self.X.shape[0]):
                self.Xt[r, :] = self.run(r)

        elif phase == 1:
            for r in range(self.X.shape[0]):
                self.Xt[r, :] = self.run_phase1(r)

        else:
            raise Exception("Wrong phase provided!")

    def train_epoch(self, phase=1):
        start = default_timer()

        arr_rand = np.random.choice(
            len(self.valid_mask[0]), len(self.valid_mask[0]), replace=False
        )

        for r, c in zip(
            self.valid_mask[0][arr_rand], self.valid_mask[1][arr_rand]
        ):
            self.phases[phase](self.X[r, :], r, c, self.current_eta)

        end = default_timer()

        print(f"Epoch Training Time: {str((end-start) / 60)} minutes")

        self.fit()

        return self.get_rmse()

    def print_num_epochs(self, interval=10):
        if self.num_epochs % interval == 0:
            print(f"Epochs: {self.num_epochs}\tRMSE: {self.s}")

    def initialize_params(self):
        self.initial_eta = 0.1
        self.target_eta = 1e-4
        self.s = 0
        self.s_ = np.inf
        self.current_eta = self.initial_eta
        self.gamma = 1e-5
        self.lambd = 1e-4
        self.num_epochs = 0

    def get_rmse(self):
        return np.sqrt(
            np.mean(
                (self.Xt[~self.invalid_mask] - self.X[~self.invalid_mask]) ** 2
            )
        )

    def initialize_weights(self, sz):
        theta = theano.shared(
            np.array(np.random.rand(sz[0], sz[1]), dtype=theano.config.floatX)
        )

        return theta

    def loss_func(self, cost, theta, alpha):
        """[Loss function for neural network. Uses gradient descent]"""
        return theta - (alpha * T.grad(cost, wrt=theta))

    def get_Xt(self):
        return self.Xt


class ImputeAlleleFreq(GenotypeData):
    """
    [Impute missing data by global allele frequency. Population IDs can be sepcified with the pops argument. if pops is None, then imputation is by global allele frequency. If pops is not None, then imputation is by population-wise allele frequency. A list of population IDs in the appropriate format can be obtained from the GenotypeData object as GenotypeData.populations]

    Args:
        genotype_data ([GenotypeData]): [GenotypeData instance. If ``genotype_data`` is not defined, then ``genotypes`` must be defined instead, and they cannot both be defined]. Defaults to None.

        by_populations (bool, optional): [Whether or not to impute by population or globally]. Defaults to False (globally).

        diploid (bool, optional): [When diploid=True, function assumes 0=homozygous ref; 1=heterozygous; 2=homozygous alt. 0-1-2 genotypes are decomposed to compute p (=frequency of ref) and q (=frequency of alt). In this case, p and q alleles are sampled to generate either 0 (hom-p), 1 (het), or 2 (hom-q) genotypes. When diploid=FALSE, 0-1-2 are sampled according to their observed frequency]. Defaults to True.

        default (int, optional): [Value to set if no alleles sampled at a locus]. Defaults to 0.

        missing (int, optional): [Missing data value]. Defaults to -9.

        prefix (str, optional): [Prefix for writing output files]

        write_output (bool, optional): [Whether to save imputed output to a file. If ``write_output`` is False, then just returns the imputed values as a pandas.DataFrame object. If ``write_output`` is True, then it saves the imputed data as a CSV file called ``<prefix>_imputed_012.csv``]
    """

    def __init__(
        self,
        genotype_data,
        *,
        by_populations=False,
        diploid=True,
        default=0,
        missing=-9,
        prefix="out",
        write_output=True,
    ):

        super().__init__()

        if by_populations:
            self.pops = genotype_data.populations
        else:
            self.pops = None

        self.diploid = diploid
        self.default = default
        self.missing = missing
        self.prefix = prefix

        self.imputed = self.fit_predict(genotype_data.genotypes_list)

        if write_output:
            self.write2file(self.imputed)

    @timer
    def fit_predict(self, X):
        """[Impute missing genotypes using global allele frequencies, with missing alleles coded as negative; usually -9]

        Args:
            X ([list(list(int))]): [012-encoded genotypes obtained from the GenotypeData object as GenotypeData.genotypes_list]

        Returns:
            [pandas.DataFrame]: [Imputed genotypes of same dimensions as data]
        """
        if self.pops is not None:
            print("\nImputing by population allele frequencies...")
        else:
            print("\nImputing by global allele frequency...")

        data = [item[:] for item in X]

        if self.pops is not None:
            pop_indices = misc.get_indices(self.pops)

        loc_index = 0
        for locus in data:
            if self.pops is None:
                allele_probs = self._get_allele_probs(locus, self.diploid)
                # print(allele_probs)
                if (
                    misc.all_zero(list(allele_probs.values()))
                    or not allele_probs
                ):
                    print(
                        "\nWarning: No alleles sampled at locus",
                        str(loc_index),
                        "setting all values to:",
                        str(self.default),
                    )
                    gen_index = 0
                    for geno in locus:
                        data[loc_index][gen_index] = self.default
                        gen_index += 1

                else:
                    gen_index = 0
                    for geno in locus:
                        if geno == self.missing:
                            data[loc_index][gen_index] = self._sample_allele(
                                allele_probs, diploid=True
                            )
                        gen_index += 1

            else:
                for pop in pop_indices.keys():
                    allele_probs = self._get_allele_probs(
                        locus,
                        self.diploid,
                        missing=self.missing,
                        indices=pop_indices[pop],
                    )

                    if (
                        misc.all_zero(list(allele_probs.values()))
                        or not allele_probs
                    ):
                        print(
                            "\nWarning: No alleles sampled at locus",
                            str(loc_index),
                            "setting all values to:",
                            str(self.default),
                        )
                        gen_index = 0
                        for geno in locus:
                            data[loc_index][gen_index] = self.default
                            gen_index += 1
                    else:
                        gen_index = 0
                        for geno in locus:
                            if geno == self.missing:
                                data[loc_index][
                                    gen_index
                                ] = self._sample_allele(
                                    allele_probs, diploid=True
                                )
                            gen_index += 1

            loc_index += 1

        df = pd.DataFrame(data)

        print("Done!")
        return df

    def _sample_allele(self, allele_probs, diploid=True):
        if diploid:
            alleles = misc.weighted_draw(allele_probs, 2)
            if alleles[0] == alleles[1]:
                return alleles[0]
            else:
                return 1
        else:
            return misc.weighted_draw(allele_probs, 1)[0]

    def _get_allele_probs(
        self, genotypes, diploid=True, missing=-9, indices=None
    ):
        data = genotypes
        length = len(genotypes)

        if indices is not None:
            data = [genotypes[index] for index in indices]
            length = len(data)

        if len(set(data)) == 1:
            if data[0] == missing:
                ret = dict()
                return ret
            else:
                ret = dict()
                ret[data[0]] = 1.0
                return ret

        if diploid:
            length = length * 2
            ret = {0: 0.0, 2: 0.0}
            for g in data:
                if g == 0:
                    ret[0] += 2
                elif g == 2:
                    ret[2] += 2
                elif g == 1:
                    ret[0] += 1
                    ret[2] += 1
                elif g == missing:
                    length -= 2
                else:
                    print(
                        "\nWarning: Ignoring unrecognized allele",
                        str(g),
                        "in get_allele_probs\n",
                    )
            for allele in ret.keys():
                ret[allele] = ret[allele] / float(length)
            return ret
        else:
            ret = dict()
            for key in set(data):
                if key != missing:
                    ret[key] = 0.0
            for g in data:
                if g == missing:
                    length -= 1
                else:
                    ret[g] += 1
            for allele in ret.keys():
                ret[allele] = ret[allele] / float(length)
            return ret

    def get_imputed_genotypes(self):
        """[Getter for the imputed genotypes]

        Returns:
            [pandas.DataFrame]: [Imputed 012-encoded genotypes as DataFrame]
        """
        return self.imputed

    def write2file(self, df):
        outfile = f"{self.prefix}_imputed_012.csv"
        df.to_csv(outfile, header=False, index=False)
