import datetime
import functools
import logging
import os
import platform
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from tqdm.utils import _unicode, disp_len  # for overriding status_print

logger = logging.getLogger(__name__)


def validate_input_type(X, return_type="array"):
    """Validate input type and return as numpy array.

    This method validates the input type and returns the input data as a numpy array, pandas DataFrame, 2D list, or torch.Tensor.

    Args:
        X (Union[pandas.DataFrame, numpy.ndarray, List[List[int]]]): Input data. Supported types include: pandas.DataFrame, numpy.ndarray, List[List[int]], and torch.Tensor.

        return_type (str): Type of returned object. Supported options include: "df", "array", "list", and "tensor". Defaults to "array".

    Returns:
        Union[pandas.DataFrame, numpy.ndarray, List[List[int]]]: Input data as the desired return_type.

    Raises:
        TypeError: X must be of type pandas.DataFrame, numpy.ndarray, List[List[int]], or torch.Tensor.
        ValueError: Unsupported return_type provided. Supported types are "df", "array", "list", and "tensor".

    """
    if not isinstance(X, (pd.DataFrame, np.ndarray, list, torch.Tensor)):
        msg = f"X must be of type pandas.DataFrame, numpy.ndarray, List[List[int]], or torch.Tensor, but got {type(X)}"
        logger.error(msg)
        raise TypeError(msg)

    if return_type not in {"df", "array", "list", "tensor"}:
        msg = f"Unsupported return type provided: {return_type}. Supported types are 'df', 'array', 'list', and 'tensor'"
        logger.error(msg)
        raise ValueError(msg)

    if return_type == "array":
        if isinstance(X, pd.DataFrame):
            return X.to_numpy()
        elif isinstance(X, list):
            return np.array(X)
        elif isinstance(X, np.ndarray):
            return X.copy()
        elif isinstance(X, torch.Tensor):
            return X.cpu().detach().numpy()

    elif return_type == "df":
        if isinstance(X, pd.DataFrame):
            return X.copy()
        elif isinstance(X, (np.ndarray, list)):
            return pd.DataFrame(X)
        elif isinstance(X, torch.Tensor):
            return pd.DataFrame(X.cpu().detach().numpy())

    elif return_type == "list":
        if isinstance(X, list):
            return X
        elif isinstance(X, np.ndarray):
            return X.tolist()
        elif isinstance(X, pd.DataFrame):
            return X.values.tolist()
        elif isinstance(X, torch.Tensor):
            return X.cpu().detach().numpy().tolist()

    elif return_type == "tensor":
        if isinstance(X, torch.Tensor):
            return X
        elif isinstance(X, np.ndarray):
            return torch.tensor(X, dtype=torch.float32)
        elif isinstance(X, pd.DataFrame):
            return torch.tensor(X.to_numpy(), dtype=torch.float32)
        elif isinstance(X, list):
            return torch.tensor(X, dtype=torch.float32)


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

    X = np.random.randint(min_value, max_value + 1, size=(nrows, ncols)).astype(float)
    for i in range(X.shape[1]):
        drop_rate = int(
            np.random.choice(np.arange(min_missing_rate, max_missing_rate, 0.02), 1)[0]
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
                np.random.choice(np.arange(min_het_rate, max_het_rate, 0.02), 1)[0]
                * X.shape[0]
            )
        )

        alt_rate = int(
            np.ceil(
                np.random.choice(np.arange(min_alt_rate, max_alt_rate, 0.02), 1)[0]
                * X.shape[0]
            )
        )

        het = np.sort(
            np.random.choice(np.arange(0, X.shape[0]), size=het_rate, replace=False)
        )

        alt = np.sort(
            np.random.choice(np.arange(0, X.shape[0]), size=alt_rate, replace=False)
        )

        sidx = alt.argsort()
        idx = np.searchsorted(alt, het, sorter=sidx)
        idx[idx == len(alt)] = 0
        het_unique = het[alt[sidx[idx]] != het]

        X[alt, i] = 2
        X[het_unique, i] = 1

        drop_rate = int(
            np.random.choice(np.arange(0.15, max_missing_rate, 0.02), 1)[0] * X.shape[0]
        )

        missing = np.random.choice(np.arange(0, X.shape[0]), size=drop_rate)

        X[missing, i] = np.nan

    print(
        f"Created a dataset of shape {X.shape} with {np.isnan(X).sum()} total missing values"
    )

    return X


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
    def status_printer(self, f):
        """Manage the printing and in-place updating of a line of characters.

        NOTE: If the string is longer than a line, then in-place updating may not work (it will print a new line at each refresh).

        Overridden to work with linux HPC clusters. Replaced carriage return with linux newline in fp_write function.

        Args:
            f (str): Path of file to print status to.
        """

        fp = f
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
