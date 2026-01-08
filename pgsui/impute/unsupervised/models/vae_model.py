from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Literal, Optional, Tuple, Union
import numpy as np

from snpio.utils.logging import LoggerManager
from pgsui.utils.logging_utils import configure_logger


class Sampling(nn.Module):
    """A layer that samples from a latent distribution using the reparameterization trick."""

    def forward(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        z_sigma = torch.exp(0.5 * z_log_var)
        epsilon = torch.randn_like(z_mean, device=z_mean.device)
        return z_mean + z_sigma * epsilon


class Encoder(nn.Module):
    """The Encoder module of a Variational Autoencoder (VAE)."""

    def __init__(
        self,
        n_features: int,
        num_classes: int,
        latent_dim: int,
        hidden_layer_sizes: List[int],
        dropout_rate: float,
        activation: torch.nn.Module,
    ):
        super().__init__()
        self.flatten = nn.Flatten()

        layers = []
        input_dim = n_features * num_classes

        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(copy.deepcopy(activation))
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_size

        self.hidden_layers = nn.Sequential(*layers)
        self.dense_z_mean = nn.Linear(input_dim, latent_dim)
        self.dense_z_log_var = nn.Linear(input_dim, latent_dim)
        self.sampling = Sampling()

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.flatten(x)
        x = self.hidden_layers(x)
        z_mean = self.dense_z_mean(x)
        z_log_var = self.dense_z_log_var(x)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z


class Decoder(nn.Module):
    """The Decoder module of a Variational Autoencoder (VAE)."""

    def __init__(
        self,
        n_features: int,
        num_classes: int,
        latent_dim: int,
        hidden_layer_sizes: List[int],
        dropout_rate: float,
        activation: torch.nn.Module,
    ) -> None:
        super().__init__()

        layers = []
        input_dim = latent_dim

        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(copy.deepcopy(activation))
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_size

        self.hidden_layers = nn.Sequential(*layers)
        output_dim = n_features * num_classes
        self.dense_output = nn.Linear(input_dim, output_dim)
        self.reshape = (n_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layers(x)
        x = self.dense_output(x)
        return x.view(-1, *self.reshape)


class VAEModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        prefix: str,
        *,
        num_classes: int = 4,
        hidden_layer_sizes: List[int] | np.ndarray = [128, 64],
        latent_dim: int = 2,
        dropout_rate: float = 0.2,
        activation: Literal["relu", "elu", "selu", "leaky_relu"] = "relu",
        kl_beta: float = 1.0,
        device: Literal["cpu", "gpu", "mps"] = "cpu",
        verbose: bool = False,
        debug: bool = False,
    ):
        """Variational Autoencoder (VAE) model for unsupervised imputation."""
        super().__init__()
        self.n_features = int(n_features)
        self.num_classes = int(num_classes)
        self.latent_dim = int(latent_dim)
        self.kl_beta = float(kl_beta)
        self.torch_device = device

        logman = LoggerManager(
            name=__name__, prefix=prefix, verbose=verbose, debug=debug
        )
        self.logger = configure_logger(
            logman.get_logger(), verbose=verbose, debug=debug
        )

        act = self._resolve_activation(activation)
        hls = (
            hidden_layer_sizes.tolist()
            if isinstance(hidden_layer_sizes, np.ndarray)
            else hidden_layer_sizes
        )

        self.encoder = Encoder(
            self.n_features, self.num_classes, self.latent_dim, hls, dropout_rate, act
        )
        self.decoder = Decoder(
            self.n_features,
            self.num_classes,
            self.latent_dim,
            list(reversed(hls)),
            dropout_rate,
            act,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var

    def _resolve_activation(self, activation: Union[str, nn.Module]) -> nn.Module:
        if isinstance(activation, nn.Module):
            return activation
        a = activation.lower()
        if a == "relu":
            return nn.ReLU()
        if a == "elu":
            return nn.ELU()
        if a in {"leaky_relu", "leakyrelu"}:
            return nn.LeakyReLU()
        if a == "selu":
            return nn.SELU()
        raise ValueError(f"Activation {activation} not supported.")
