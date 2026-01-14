from __future__ import annotations

from dataclasses import dataclass
from typing import Any, FrozenSet, Literal, Mapping

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True, slots=True)
class OptunaParamSpec:
    """Specification and validation for Optuna objective parameter keys.

    Attributes:
        keys: Canonical keys used in the Optuna objective `params` dict.
    """

    keys: FrozenSet[str]

    def __post_init__(self) -> None:
        if not isinstance(self.keys, frozenset):
            raise TypeError(
                f"`keys` must be a frozenset, got: {type(self.keys).__name__}"
            )
        if not self.keys:
            raise ValueError("`keys` must be non-empty.")
        if not all(isinstance(k, str) and k for k in self.keys):
            raise TypeError("`keys` must contain only non-empty strings.")

    def count(self) -> int:
        """Return the integer count of tuned parameters."""
        return len(self.keys)

    def validate(self, params: Mapping[str, Any], *, allow_extra: bool = False) -> None:
        """Validate that a params mapping matches this spec's keys.

        Args:
            params: Mapping of parameter names -> values (typically the objective `params` dict).
            allow_extra: If True, extra keys are allowed; missing keys still error.

        Raises:
            TypeError: If `params` is not a Mapping.
            KeyError: If required keys are missing (or extras exist when allow_extra=False).
        """
        if not isinstance(params, Mapping):
            raise TypeError(f"`params` must be a Mapping, got: {type(params).__name__}")

        got = frozenset(params.keys())
        missing = self.keys - got
        extra = got - self.keys

        if missing or (extra and not allow_extra):
            parts: list[str] = []
            if missing:
                parts.append(f"missing={sorted(missing)}")
            if extra and not allow_extra:
                parts.append(f"extra={sorted(extra)}")
            raise KeyError("Objective params keys mismatch: " + "; ".join(parts))


OBJECTIVE_SPEC_VAE = OptunaParamSpec(
    keys=frozenset(
        {
            "latent_dim",
            "learning_rate",
            "dropout_rate",
            "num_hidden_layers",
            "activation",
            "l1_penalty",
            "layer_scaling_factor",
            "layer_schedule",
            "power",
            "normalize",
            "inverse",
            "gamma",
            "kl_beta",
            "kl_beta_schedule",
            "gamma_schedule",
        }
    )
)

OBJECTIVE_SPEC_AE = OptunaParamSpec(
    keys=frozenset(
        {
            "latent_dim",
            "learning_rate",
            "dropout_rate",
            "num_hidden_layers",
            "activation",
            "l1_penalty",
            "layer_scaling_factor",
            "layer_schedule",
            "power",
            "normalize",
            "inverse",
            "gamma",
            "gamma_schedule",
        }
    )
)

OBJECTIVE_SPEC_UBP = OptunaParamSpec(
    keys=frozenset(
        {
            "latent_dim",
            "learning_rate",
            "dropout_rate",
            "num_hidden_layers",
            "activation",
            "l1_penalty",
            "layer_scaling_factor",
            "layer_schedule",
            "power",
            "normalize",
            "inverse",
            "gamma",
            "gamma_schedule",
        }
    )
)


OBJECTIVE_SPEC_NLPCA = OptunaParamSpec(
    keys=frozenset(
        {
            "latent_dim",
            "learning_rate",
            "dropout_rate",
            "num_hidden_layers",
            "activation",
            "l1_penalty",
            "layer_scaling_factor",
            "layer_schedule",
            "power",
            "normalize",
            "inverse",
            "gamma",
            "gamma_schedule",
        }
    )
)


def validate_input_type(
    X: pd.DataFrame | np.ndarray | list | torch.Tensor,
    return_type: Literal["array", "df", "list", "tensor"] = "array",
) -> pd.DataFrame | np.ndarray | list | torch.Tensor:
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
            return X.to_numpy().tolist()
        elif isinstance(X, torch.Tensor):
            return X.detach().cpu().numpy().tolist()

    elif return_type == "tensor":
        if isinstance(X, torch.Tensor):
            return X
        elif isinstance(X, np.ndarray):
            return torch.from_numpy(X)
        elif isinstance(X, pd.DataFrame):
            return torch.from_numpy(X.to_numpy())
        elif isinstance(X, list):
            return torch.tensor(X)


def detect_computing_device(
    *, force_cpu: bool = False, verbose: bool = False
) -> torch.device:
    """Detects and returns the best available PyTorch compute device.

    Prioritizes CUDA (NVIDIA) > MPS (Apple Silicon) > CPU.

    Args:
        force_cpu (bool): If True, forces the device to CPU regardless of available hardware. Defaults to False.
        verbose (bool): If True, prints the selected device to stdout. Defaults to False.

    Returns:
        torch.device: The selected computing device.
    """
    if force_cpu:
        device = torch.device("cpu")  # Forced to CPU
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")  # Fallback to CPU

    if verbose:
        print(f"Selected compute device: {device}")

    return device


def get_missing_mask(
    X: pd.DataFrame | pd.Series | np.ndarray | list | torch.Tensor,
) -> pd.DataFrame | pd.Series | np.ndarray | torch.Tensor:
    """Returns a boolean mask indicating missing values (NaN, None).

    Notes:
    Lists are converted to numpy arrays to compute the mask.

    Args:
        X: Input data.

    Returns:
        pd.DataFrame | pd.Series | np.ndarray | torch.Tensor: Boolean mask of the same shape as X (returned as DF, Array, or Tensor).

    Raises:
        TypeError: If input type is not supported.
    """
    if isinstance(X, pd.DataFrame):
        return X.isna()

    elif isinstance(X, pd.Series):
        return pd.isna(X)

    elif isinstance(X, np.ndarray):
        # np.isnan fails on object arrays (e.g. strings)
        # so we check generically first
        if X.dtype.kind in {"U", "S", "O"}:  # String/Object
            return pd.isnull(X)
        return np.isnan(X)

    elif isinstance(X, torch.Tensor):
        return torch.isnan(X)

    elif isinstance(X, list):
        arr = np.array(X)
        # Handle mixed types in lists
        if arr.dtype.kind in {"U", "S", "O"}:
            return pd.isnull(arr)
        return np.isnan(arr)

    else:
        raise TypeError(
            f"Unsupported type for missing value detection. Expected pandas.DataFrame, pandas.Series, numpy.ndarray, list, or torch.Tensor but got {type(X)}"
        )


def ensure_2d(
    X: pd.DataFrame | pd.Series | np.ndarray | list | torch.Tensor,
) -> pd.DataFrame | np.ndarray | list | torch.Tensor:
    """Ensures the input is at least 2-dimensional.

    If input is 1D (e.g., shape (N,)), it is reshaped to (N, 1). Already 2D+ inputs are returned unchanged.

    Args:
        X (pd.DataFrame | pd.Series | np.ndarray | list | torch.Tensor): Input data.

    Returns:
        pd.DataFrame | np.ndarray | list | torch.Tensor: Input data transformed to be at least 2D.

    Raises:
        TypeError: If input type is not supported.
    """
    if isinstance(X, pd.DataFrame):
        return X  # DataFrames are always 2D

    elif isinstance(X, pd.Series):
        return X.to_frame()  # Convert Series to DataFrame (2D)

    elif isinstance(X, np.ndarray):
        if X.ndim == 1:
            return X.reshape(-1, 1)
        return X

    elif isinstance(X, torch.Tensor):
        if X.dim() == 1:
            return X.unsqueeze(1)
        return X

    elif isinstance(X, list):
        # Check depth of list
        if not X:
            return X
        if not isinstance(X[0], list):
            return [[x] for x in X]
        return X

    else:
        msg = f"X must be of type pandas.DataFrame, pd.Series, numpy.ndarray, list, or torch.Tensor, but got {type(X)}"
        raise TypeError(msg)


def flatten_1d(
    y: pd.DataFrame | pd.Series | np.ndarray | list | torch.Tensor,
) -> pd.Series | np.ndarray | list | torch.Tensor:
    """
    Flattens input to a 1D structure.

    Args:
        y (pd.DataFrame | pd.Series | np.ndarray | list | torch.Tensor): Input data.

    Returns:
        pd.Series | np.ndarray | list | torch.Tensor: 1D representation of the input.

    Notes:
        Inputs with multiple columns (e.g., DataFrame with >1 column) are flattened into a single 1D structure.

    Raises:
        TypeError: If input type is not supported.
    """
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 1:
            return y.iloc[:, 0]
        else:
            return pd.Series(y.to_numpy().flatten())

    elif isinstance(y, np.ndarray):
        return y.flatten()

    elif isinstance(y, torch.Tensor):
        return y.view(-1)

    elif isinstance(y, list):
        # Recursively flatten list if needed, or simple comprehension if just 2D
        if not y:
            return y
        if isinstance(y[0], list):
            return [item for sublist in y for item in sublist]
        return y

    else:
        msg = f"Input must be of type pandas.DataFrame, pandas.Series, numpy.ndarray, list, or torch.Tensor, but got {type(y)}"
        raise TypeError(msg)


def safe_shape(
    X: pd.DataFrame | pd.Series | np.ndarray | list | torch.Tensor,
) -> tuple[int, ...]:
    """Returns the shape of the input container as a tuple.

    Args:
        X (pd.DataFrame | pd.Series | np.ndarray | list | torch.Tensor): Input data.

    Returns:
        tuple[int, ...]: Dimensions of the data (rows, cols, etc.).
    """
    if isinstance(X, (pd.DataFrame, np.ndarray)):
        return X.shape

    elif isinstance(X, pd.Series):
        return (X.shape[0],)

    elif isinstance(X, torch.Tensor):
        return tuple(X.shape)

    elif isinstance(X, list):
        if not X:
            return (0,)
        rows = len(X)

        # Check if 2D list
        if isinstance(X[0], list):
            return (rows, len(X[0]))
        return (rows,)

    else:
        msg = f"X must be of type pandas.DataFrame, pd.Series, numpy.ndarray, list, or torch.Tensor, but got {type(X)}"
        raise TypeError(msg)
