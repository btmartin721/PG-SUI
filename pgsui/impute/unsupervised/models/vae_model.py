from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from snpio.utils.logging import LoggerManager
from torch.distributions import Normal

from pgsui.impute.unsupervised.loss_functions import MaskedFocalLoss


class Sampling(nn.Module):
    def forward(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        z_sigma = torch.exp(0.5 * z_log_var)  # Precompute outside

        # Ensure on GPU
        # rand_like takes random values from a normal distribution
        # of the same shape as z_mean.
        epsilon = torch.randn_like(z_mean, device=z_mean.device)
        return z_mean + z_sigma * epsilon


class Encoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        num_classes: int,
        latent_dim: int,
        hidden_layer_sizes: List[int],
        dropout_rate: float,
        activation: Union[str, torch.nn.Module],
    ):
        super(Encoder, self).__init__()
        self.flatten = nn.Flatten()
        self.activation = (
            getattr(F, activation) if isinstance(activation, str) else activation
        )

        layers = []
        input_dim = n_features * num_classes
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))

            # BatchNorm can lead to faster convergence.
            layers.append(nn.BatchNorm1d(hidden_size))

            layers.append(nn.Dropout(dropout_rate))
            layers.append(activation)
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
    def __init__(
        self,
        n_features: int,
        num_classes: int,
        latent_dim: int,
        hidden_layer_sizes: List[int],
        dropout_rate: float,
        activation: Union[str, torch.nn.Module],
    ) -> None:
        super(Decoder, self).__init__()

        layers = []
        input_dim = latent_dim
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))

            # BatchNorm can lead to faster convergence.
            layers.append(nn.BatchNorm1d(hidden_size))

            layers.append(nn.Dropout(dropout_rate))
            layers.append(activation)
            input_dim = hidden_size

        self.hidden_layers = nn.Sequential(*layers)
        self.dense_output = nn.Linear(input_dim, n_features * num_classes)
        self.reshape = (n_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder."""
        x = self.hidden_layers(x)
        x = self.dense_output(x)
        return x.view(-1, *self.reshape)


class VAEModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        prefix: str = "pgsui",
        num_classes: int = 3,
        latent_dim: int = 2,
        hidden_layer_sizes: List[int] = [128, 64],
        dropout_rate: float = 0.2,
        activation: str = "elu",
        gamma: float = 2.0,
        beta: float = 1.0,
        verbose: int = 0,
        debug: bool = False,
    ):
        """Variational Autoencoder (VAE) model for imputation.

        Args:
            n_features (int): Number of features in the input data.
            prefix (str): Prefix for the logger name. Default is 'pgsui'.
            num_classes (int): Number of classes in the input data. Default is 3.
            latent_dim (int): Dimensionality of the latent space. Default is 2.
            hidden_layer_sizes (List[int]): List of hidden layer sizes. Default is [128, 64].
            dropout_rate (float): Dropout rate for hidden layers. Default is 0.2.
            activation (str): Activation function for hidden layers. Default is 'elu'.
            gamma (float): Focal loss gamma parameter. Default is 2.0.
            beta (float): Weight for the KL divergence term. Default is 1.0.
            verbose (int): Verbosity level for logging messages. Default is 0.
            debug (bool): Debug mode for logging messages. Default is False.
        """
        super(VAEModel, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.beta = beta

        logman = LoggerManager(
            name=__name__, prefix=prefix, verbose=verbose >= 1, debug=debug
        )
        self.logger = logman.get_logger()

        activation = self._resolve_activation(activation)

        self.encoder = Encoder(
            n_features,
            num_classes,
            latent_dim,
            hidden_layer_sizes,
            dropout_rate,
            activation,
        )

        decoder_layer_sizes = list(reversed(hidden_layer_sizes))
        self.decoder = Decoder(
            n_features,
            num_classes,
            latent_dim,
            decoder_layer_sizes,
            dropout_rate,
            activation,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE model.

        Args:
            x (torch.Tensor): Input data of shape (batch_size, n_features, num_classes).
        """
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var

    def compute_loss(
        self,
        y: torch.Tensor,
        outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        mask: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reconstruction, z_mean, z_log_var = outputs

        # KL Divergence using torch.distributions. Normal is used as the prior.
        prior = Normal(0, 1)
        posterior = Normal(z_mean, torch.exp(0.5 * z_log_var))
        kl_loss = torch.distributions.kl.kl_divergence(posterior, prior).mean()

        if class_weights is None:
            class_weights = torch.ones(self.num_classes, device=y.device)

        if mask is None:
            mask = torch.ones_like(y, dtype=torch.bool)

        criterion = MaskedFocalLoss(alpha=class_weights, gamma=self.gamma)
        reconstruction_loss = criterion(reconstruction, y, valid_mask=mask)
        return reconstruction_loss + self.beta * kl_loss

    def _resolve_activation(
        self, activation: Union[str, torch.nn.Module]
    ) -> torch.nn.Module:
        """Resolve the activation function.

        Args:
            activation (Union[str, torch.nn.Module]): Activation function.

        Returns:
            torch.nn.Module: Resolved activation function.

        Raises:
            ValueError: If the activation function is not supported.
        """
        if isinstance(activation, str):
            activation = activation.lower()
            if activation == "relu":
                return nn.ReLU()
            elif activation == "elu":
                return nn.ELU()
            elif activation in ["leaky_relu", "leakyrelu"]:
                return nn.LeakyReLU()
            elif activation == "selu":
                return nn.SELU()
            else:
                msg = f"Activation {activation} not supported."
                self.logger.error(msg)
                raise ValueError(msg)

        return activation
