from typing import Literal

import numpy as np
import pandas as pd
import torch


def validate_input_type(
    X: pd.DataFrame | np.ndarray | list | torch.Tensor,
    return_type: Literal["array", "df", "list", "tensor"] = "array",
):
    """Validate input type and return as numpy array.

    This method validates the input type and returns the input data as a numpy array, pandas DataFrame, 2D list, or torch.Tensor.

    Args:
        X (pandas.DataFrame | numpy.ndarray | list | torch.Tensor): Input data. Supported types include: pandas.DataFrame, numpy.ndarray, list, and torch.Tensor.

        return_type (Literal["array", "df", "list", "tensor"]): Type of returned object. Supported options include: "df", "array", "list", and "tensor". "df" corresponds to a pandas DataFrame. "array" corresponds to a numpy array. "list" corresponds to a 2D list. "tensor" corresponds to a torch.Tensor. Defaults to "array".

    Returns:
        pandas.DataFrame | numpy.ndarray | list | torch.Tensor: Input data as the desired return_type.

    Raises:
        TypeError: X must be of type pandas.DataFrame, numpy.ndarray, list, or torch.Tensor.
        ValueError: Unsupported return_type provided. Supported types are "df", "array", "list", and "tensor".

    """
    if not isinstance(X, (pd.DataFrame, np.ndarray, list, torch.Tensor)):
        msg = f"X must be of type pandas.DataFrame, numpy.ndarray, list, or torch.Tensor, but got {type(X)}"
        raise TypeError(msg)

    if return_type not in {"df", "array", "list", "tensor"}:
        msg = f"Unsupported return type provided: {return_type}. Supported types are 'df', 'array', 'list', and 'tensor'"
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
