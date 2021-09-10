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

# Custom module imports
from impute.iterative_imputer_custom import (
    IterativeImputerGridSearch,
    IterativeImputerAllData,
)

import impute.estimators

from utils import misc
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
            "Warning: Intel CPU detected but scikit-learn-intelex is not installed. We recommend installing it to speed up computation."
        )
        intelex = False
else:
    intelex = False


class Impute:
    def __init__(self, clf, clf_type, kwargs):
        self.clf = clf
        self.clf_type = clf_type

        self.pops = kwargs["genotype_data"].populations

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
            pops=self.pops,
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
            pops=self.pops,
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
            pops=self.pops,
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

        df_defiled = pd.DataFrame(simple_imputer.fit_transform(df))
        df_filled = df_defiled.copy()

        del simple_imputer
        gc.collect()

        for col in cols:
            data_drop_rate = np.random.choice(np.arange(0.15, 0.5, 0.02), 1)[0]

            drop_ind = np.random.choice(
                np.arange(len(df_defiled[col])),
                size=int(len(df_defiled[col]) * data_drop_rate),
                replace=False,
            )

            # Introduce random np.nan values
            df_defiled.iloc[drop_ind, col] = np.nan

        return df_filled, df_defiled, cols

    def _defile_dataset_groups(self, df, col_selection_rate=0.40):
        cols = np.random.choice(
            df.columns, int(len(df.columns) * col_selection_rate)
        )

        if self.imp_kwargs["initial_strategy"] == "most_frequent_populations":
            simple_imputer = impute.estimators.ImputeAlleleFreq(
                gt=df.fillna(-9).values.tolist(),
                pops=self.pops,
                by_populations=True,
                missing=-9,
                write_output=False,
                verbose=False,
            )

        df_defiled = simple_imputer.imputed
        df_filled = df_defiled.copy()

        for col in cols:
            data_drop_rate = np.random.choice(np.arange(0.15, 0.5, 0.02), 1)[0]

            drop_ind = np.random.choice(
                np.arange(len(df_defiled[col])),
                size=int(len(df_defiled[col]) * data_drop_rate),
                replace=False,
            )

            # Introduce random np.nan values
            df_defiled.iloc[drop_ind, col] = np.nan

        return df_filled, df_defiled, cols

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

        if self.imp_kwargs["initial_strategy"] == "most_frequent_populations":
            df_known, df_unknown, cols = self._defile_dataset_groups(
                df, col_selection_rate=self.validation_only
            )

        else:

            df_known, df_unknown, cols = self._defile_dataset(
                df, col_selection_rate=self.validation_only
            )

        df_known_slice = df_known[cols]
        df_unknown_slice = df_unknown[cols]

        df_stg = df_unknown.copy()

        # Variational Autoencoder Neural Network
        if self.clf == "VAE":
            df_imp = self.fit_predict(df_stg.to_numpy())
            df_imp = df_imp.astype(np.float)

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
            df_unknown_slice,
            df_unknown,
        ]

        if self.clf == "VAE":
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
        pops=None,
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

                pops [list]: [Population IDs as 1d-list in order of sampleID]

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
