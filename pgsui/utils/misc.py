import sys
import os
import functools
import time
import datetime
import platform
import subprocess
import re
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.utils import disp_len, _unicode  # for overriding status_print


# from skopt import BayesSearchCV


def validate_input_type(X, return_type="array"):
    """Validate input type and return as numpy array.

    Args:
        X (pandas.DataFrame, numpy.ndarray, or List[List[int]]): Input data.

        return_type (str): Type of returned object. Supported options include: "df", "array", and "list". Defaults to "array".

    Returns:
        pandas.DataFrame, numpy.ndarray, or List[List[int]]: Input data desired return_type.

    Raises:
        TypeError: X must be of type pandas.DataFrame, numpy.ndarray, or List[List[int]].

        ValueError: Unsupported return_type provided. Supported types are "df", "array", and "list".

    """
    if not isinstance(X, (pd.DataFrame, np.ndarray, list)):
        raise TypeError(
            f"X must be of type pandas.DataFrame, numpy.ndarray, "
            f"or List[List[int]], but got {type(X)}"
        )

    if return_type == "array":
        if isinstance(X, pd.DataFrame):
            return X.to_numpy()
        elif isinstance(X, list):
            return np.array(X)
        elif isinstance(X, np.ndarray):
            return X.copy()

    elif return_type == "df":
        if isinstance(X, pd.DataFrame):
            return X.copy()
        elif isinstance(X, (np.ndarray, list)):
            return pd.DataFrame(X)

    elif return_type == "list":
        if isinstance(X, list):
            return X
        elif isinstance(X, np.ndarray):
            return X.tolist()
        elif isinstance(X, pd.DataFrame):
            return X.values.tolist()

    else:
        raise ValueError(
            f"Unsupported return type provided: {return_type}. Supported types "
            f"are 'df', 'array', and 'list'"
        )


def generate_random_dataset(
    min_value=0,
    max_value=2,
    nrows=35,
    ncols=20,
    min_missing_rate=0.15,
    max_missing_rate=0.5,
):
    """Generate a random integer dataset that can be used for testing.

    Will also add randomly missing values of random proportions between ``min_missing_rate`` and ``max_missing_rate``.

    Args:
        min_value (int, optional): Minimum value to use. Defaults to 0.

        max_value (int, optional): Maxiumum value to use. Defaults to 2.

        nrows (int, optional): Number of rows to use. Defaults to 35.

        ncols (int, optional): Number of columns to use. Defaults to 20.

        min_missing_rate (float, optional): Minimum proportion of missing data per column. Defaults to 0.15.

        max_missing_rate (float, optional): Maximum proportion of missing data per column.

    Returns:
        numpy.ndarray: Numpy array with randomly generated dataset.
    """
    assert (
        min_missing_rate >= 0 and min_missing_rate < 1.0
    ), f"min_missing_rate must be >= 0 and < 1.0, but got {min_missing_rate}"

    assert (
        max_missing_rate > 0 and max_missing_rate < 1.0
    ), f"max_missing_rate must be > 0 and < 1.0, but got {max_missing_rate}"

    assert nrows > 1, f"nrows must be > 1, but got {nrows}"
    assert ncols > 1, f"ncols must be > 1, but got {ncols}"

    try:
        min_missing_rate = float(min_missing_rate)
        max_missing_rate = float(max_missing_rate)
    except TypeError:
        sys.exit(
            "min_missing_rate and max_missing_rate must be of type float or "
            "must be cast-able to type float"
        )

    X = np.random.randint(
        min_value, max_value + 1, size=(nrows, ncols)
    ).astype(float)
    for i in range(X.shape[1]):
        drop_rate = int(
            np.random.choice(
                np.arange(min_missing_rate, max_missing_rate, 0.02), 1
            )[0]
            * X.shape[0]
        )

        rows = np.random.choice(np.arange(0, X.shape[0]), size=drop_rate)
        X[rows, i] = np.nan

    return X


def generate_012_genotypes(
    nrows=35,
    ncols=20,
    max_missing_rate=0.5,
    min_het_rate=0.001,
    max_het_rate=0.3,
    min_alt_rate=0.001,
    max_alt_rate=0.3,
):
    """Generate random 012-encoded genotypes.

    Allows users to control the rate of reference, heterozygote, and alternate alleles. Will insert a random proportion between ``min_het_rate`` and ``max_het_rate`` and ``min_alt_rate`` and ``max_alt_rate`` and from no misssing data to a proportion of ``max_missing_rate``.

    Args:
        nrows (int, optional): Number of rows to generate. Defaults to 35.

        ncols (int, optional): Number of columns to generate. Defaults to 20.

        max_missing_rate (float, optional): Maximum proportion of missing data to use. Defaults to 0.5.

        min_het_rate (float, optional): Minimum proportion of heterozygotes (1's) to insert. Defaults to 0.001.

        max_het_rate (float, optional): Maximum proportion of heterozygotes (1's) to insert. Defaults to 0.3.

        min_alt_rate (float, optional): Minimum proportion of alternate alleles (2's) to insert. Defaults to 0.001.

        max_alt_rate (float, optional): Maximum proportion of alternate alleles (2's) to insert. Defaults to 0.3.
    """
    assert (
        min_het_rate > 0 and min_het_rate <= 1.0
    ), f"min_het_rate must be > 0 and <= 1.0, but got {min_het_rate}"

    assert (
        max_het_rate > 0 and max_het_rate <= 1.0
    ), f"max_het_rate must be > 0 and <= 1.0, but got {max_het_rate}"

    assert (
        min_alt_rate > 0 and min_alt_rate <= 1.0
    ), f"min_alt_rate must be > 0 and <= 1.0, but got {min_alt_rate}"

    assert (
        max_alt_rate > 0 and max_alt_rate <= 1.0
    ), f"max_alt_rate must be > 0 and <= 1.0, but got {max_alt_rate}"

    assert nrows > 1, f"The number of rows must be > 1, but got {nrows}"

    assert ncols > 1, f"The number of columns must be > 1, but got {ncols}"

    assert (
        max_missing_rate > 0 and max_missing_rate < 1.0
    ), f"max_missing rate must be > 0 and < 1.0, but got {max_missing_rate}"

    try:
        min_het_rate = float(min_het_rate)
        max_het_rate = float(max_het_rate)
        min_alt_rate = float(min_alt_rate)
        max_alt_rate = float(max_alt_rate)
        max_missing_rate = float(max_missing_rate)
    except TypeError:
        sys.exit(
            "max_missing_rate, min_het_rate, max_het_rate, min_alt_rate, and "
            "max_alt_rate must be of type float, or must be cast-able to type "
            "float"
        )

    X = np.zeros((nrows, ncols))
    for i in range(X.shape[1]):
        het_rate = int(
            np.ceil(
                np.random.choice(
                    np.arange(min_het_rate, max_het_rate, 0.02), 1
                )[0]
                * X.shape[0]
            )
        )

        alt_rate = int(
            np.ceil(
                np.random.choice(
                    np.arange(min_alt_rate, max_alt_rate, 0.02), 1
                )[0]
                * X.shape[0]
            )
        )

        het = np.sort(
            np.random.choice(
                np.arange(0, X.shape[0]), size=het_rate, replace=False
            )
        )

        alt = np.sort(
            np.random.choice(
                np.arange(0, X.shape[0]), size=alt_rate, replace=False
            )
        )

        sidx = alt.argsort()
        idx = np.searchsorted(alt, het, sorter=sidx)
        idx[idx == len(alt)] = 0
        het_unique = het[alt[sidx[idx]] != het]

        X[alt, i] = 2
        X[het_unique, i] = 1

        drop_rate = int(
            np.random.choice(np.arange(0.15, max_missing_rate, 0.02), 1)[0]
            * X.shape[0]
        )

        missing = np.random.choice(np.arange(0, X.shape[0]), size=drop_rate)

        X[missing, i] = np.nan

    print(
        f"Created a dataset of shape {X.shape} with {np.isnan(X).sum()} total missing values"
    )

    return X


def unique2D_subarray(a):
    """Get unique subarrays for each column from numpy array."""
    dtype1 = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[1:])))
    b = np.ascontiguousarray(a.reshape(a.shape[0], -1)).view(dtype1)
    return a[np.unique(b, return_index=1, axis=-1)[1]]


def _remove_nonbiallelic(df, cv=5):
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

            elif len(df_cp[df_cp[col] == 0.0]) < cv:
                bad_cols.append(col)

            elif df_cp[col].isin([1.0]).any():
                if len(df_cp[df_cp[col] == 1]) < cv:
                    bad_cols.append(col)

            elif len(df_cp[df_cp[col] == 2.0]) < cv:
                bad_cols.append(col)

    # pandas 1.X.X
    else:
        for col in df_cp.columns:
            if 0.0 not in df[col].unique() and 2.0 not in df[col].unique():
                bad_cols.append(col)

            elif len(df_cp[df_cp[col] == 0.0]) < cv:
                bad_cols.append(col)

            elif 1.0 in df_cp[col].unique():
                if len(df_cp[df_cp[col] == 1.0]) < cv:
                    bad_cols.append(col)

            elif len(df_cp[df_cp[col] == 2.0]) < cv:
                bad_cols.append(col)

    if bad_cols:
        df_cp.drop(bad_cols, axis=1, inplace=True)

        print(
            f"{len(bad_cols)} columns removed for being non-biallelic or "
            f"having genotype counts < number of cross-validation "
            f"folds\nSubsetting from {len(df_cp.columns)} remaining columns\n"
        )

    return df_cp


def get_indices(l):
    """Takes a list and returns dict giving indices matching each possible
    list member.

    Example:
            Input [0, 1, 1, 0, 0]
            Output {0:[0,3,4], 1:[1,2]}
    """
    ret = dict()
    for member in set(l):
        ret[member] = list()
    i = 0
    for el in l:
        ret[el].append(i)
        i += 1
    return ret


def all_zero(l):
    """Check whether list consists of all zeros.

    Returns TRUE if supplied list contains all zeros
    Returns FALSE if list contains ANY non-zero values
    Returns FALSE if list is empty.

    Args:
        l (List[int]): List to check.

    Returns:
        bool: True if all zeros, False otherwise.
    """
    values = set(l)
    if len(values) > 1:
        return False
    elif len(values) == 1 and l[0] in [0, 0.0, "0", "0.0"]:
        return True
    else:
        return False


def weighted_draw(d, num_samples=1):
    choices = list(d.keys())
    weights = list(d.values())
    return np.random.choice(choices, num_samples, p=weights)


def timer(func):
    """print the runtime of the decorated function in the format HH:MM:SS."""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        final_runtime = str(datetime.timedelta(seconds=run_time))
        print(f"Finshed {func.__name__!r} in {final_runtime}\n")
        return value

    return wrapper_timer


def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)

    def show(j):
        x = int(size * j / count)
        file.write(
            "%s[%s%s] %i/%i\r" % (prefix, "#" * x, "." * (size - x), j, count)
        )
        file.flush()

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    file.write("\n")
    file.flush()


def isnotebook():
    """Checks whether in Jupyter notebook.

    Returns:
        bool: True if in Jupyter notebook, False otherwise.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            # Jupyter notebook or qtconsole
            return True
        elif shell == "TerminalInteractiveShell":
            # Terminal running IPython
            return False
        else:
            # Other type (?)
            return False
    except NameError:
        # Probably standard Python interpreter
        return False


def get_processor_name():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        # os.environ["PATH"] = os.environ["PATH"] + os.pathsep + "/usr/sbin"
        arch = platform.processor()
        if arch[0] == "i":
            return "Intel"
        else:
            return arch
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).strip()
        all_info = all_info.decode("utf-8")
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, 1)
    return ""


class tqdm_linux(tqdm):
    """Adds a dynamically updating progress bar.

    Decorate an iterable object, with a dynamically updating progressbar every time a value is requested.
    """

    @staticmethod
    def status_printer(self, file):
        """Manage the printing and in-place updating of a line of characters.

        NOTE: If the string is longer than a line, then in-place updating may not work (it will print a new line at each refresh).

        Overridden to work with linux HPC clusters. Replaced carriage return with linux newline in fp_write function.

        Args:
            file (str): Path of file to print status to.
        """

        fp = file
        fp_flush = getattr(fp, "flush", lambda: None)

        def fp_write(s):
            fp.write(_unicode(s))
            fp_flush()

        last_len = [0]

        def print_status(s):
            len_s = disp_len(s)
            fp_write("\n" + s + (" " * max(last_len[0] - len_s, 0)))
            last_len[0] = len_s

        return print_status


class HiddenPrints:
    """Class to supress printing within a with statement."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class StreamToLogger(object):
    """Fake file-like stream object that redirects writes to a logger instance."""

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ""
