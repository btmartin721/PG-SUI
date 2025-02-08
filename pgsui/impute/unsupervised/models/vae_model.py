import logging
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from snpio.utils.logging import LoggerManager
from torch.distributions import Normal

from pgsui.impute.unsupervised.loss_functions import MaskedFocalLoss


class Sampling(nn.Module):
    """Sampling layer for the reparameterization trick.

    This layer is used to sample from the latent space using the reparameterization trick. The reparameterization trick is used to sample from a normal distribution with a given mean and variance. The trick is used to make the model differentiable and to allow for backpropagation through the sampling layer. The sampling layer is used in the VAE model to sample from the latent space.
    """

    def forward(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        """Forward pass through the sampling layer.

        Args:
            z_mean (torch.Tensor): Mean of the latent space.
            z_log_var (torch.Tensor): Log variance of the latent space.

        Returns:
            torch.Tensor: Sampled latent space.
        """
        z_sigma = torch.exp(0.5 * z_log_var)  # Precompute outside

        # Ensure on GPU
        # rand_like takes random values from a normal distribution
        # of the same shape as z_mean.
        epsilon = torch.randn_like(z_mean, device=z_mean.device)
        return z_mean + z_sigma * epsilon


class Encoder(nn.Module):
    """Encoder module for the VAE model.

    This module is used to encode the input data into the latent space. The encoder consists of a series of hidden layers followed by a dense layer to compute the mean and log variance of the latent space. The encoder is used to encode the input data into the latent space for the VAE model.
    """

    def __init__(
        self,
        n_features: int,
        num_classes: int,
        latent_dim: int,
        hidden_layer_sizes: List[int],
        dropout_rate: float,
        activation: Union[str, torch.nn.Module],
    ):
        """Encoder module for the VAE model.

        Args:
            n_features (int): Number of features in the input data.
            num_classes (int): Number of classes in the input data.
            latent_dim (int): Dimensionality of the latent space.
            hidden_layer_sizes (List[int]): List of hidden layer sizes.
            dropout_rate (float): Dropout rate for hidden layers.
            activation (Union[str, torch.nn.Module]): Activation function for hidden layers.
        """
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
        """Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input data of shape (batch_size, n_features, num_classes).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of z_mean, z_log_var, and z.
        """
        x = self.flatten(x)
        x = self.hidden_layers(x)
        z_mean = self.dense_z_mean(x)
        z_log_var = self.dense_z_log_var(x)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z


class Decoder(nn.Module):
    """Decoder module for the VAE model.

    This module is used to decode the latent space into the output data. The decoder consists of a series of hidden layers followed by a dense layer to compute the output data. The decoder is used to decode the latent space into the output data for the VAE model. The output data is reshaped to the original shape of the input data.
    """

    def __init__(
        self,
        n_features: int,
        num_classes: int,
        latent_dim: int,
        hidden_layer_sizes: List[int],
        dropout_rate: float,
        activation: Union[str, torch.nn.Module],
    ) -> None:
        """Decoder module for the VAE model.

        Args:
            n_features (int): Number of features in the input data.
            num_classes (int): Number of classes in the input data.
            latent_dim (int): Dimensionality of the latent space.
            hidden_layer_sizes (List[int]): List of hidden layer sizes.
            dropout_rate (float): Dropout rate for hidden layers.
            activation (Union[str, torch.nn.Module]): Activation function for hidden layers.
        """
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
        """Forward pass through the decoder.

        Args:
            x (torch.Tensor): Input data of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Output data of shape (batch_size, n_features, num_classes).
        """
        x = self.hidden_layers(x)
        x = self.dense_output(x)
        return x.view(-1, *self.reshape)


class VAEModel(nn.Module):
    """Variational Autoencoder (VAE) model for imputation.

    This model is used to impute missing data using a VAE architecture. The model consists of an encoder and a decoder. The encoder encodes the input data into the latent space, while the decoder decodes the latent space into the output data. The model uses a focal loss for the reconstruction loss and a KL divergence term for the latent space regularization. The model can be used to impute missing data in genotype data.
    """

    def __init__(
        self,
        *,
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
        logger: logging.Logger | None = None,
    ):
        """Variational Autoencoder (VAE) model for imputation.

        This model is used to impute missing data using a VAE architecture. The model consists of an encoder and a decoder. The encoder encodes the input data into the latent space, while the decoder decodes the latent space into the output data. The model uses a focal loss for the reconstruction loss and a KL divergence term for the latent space regularization. The model can be used to impute missing data in genotype data.

        Args:
            n_features (int): Number of features in the input data.
            prefix (str, optional): Prefix for the logger. Defaults to "pgsui".
            num_classes (int, optional): Number of classes in the input data. Defaults to 3.
            latent_dim (int, optional): Dimensionality of the latent space. Defaults to 2.
            hidden_layer_sizes (List[int], optional): List of hidden layer sizes. Defaults to [128, 64].
            dropout_rate (float, optional): Dropout rate for hidden layers. Defaults to 0.2.
            activation (str, optional): Activation function for hidden layers. Defaults to "elu".
            gamma (float, optional): Focal loss gamma parameter. Defaults to 2.0.
            beta (float, optional): Beta parameter for the KL divergence term. Defaults to 1.0.
            verbose (int, optional): Verbosity level for logging messages. Defaults to 0.
            debug (bool, optional): Debug mode for logging messages. Defaults to False.
            logger (logging.Logger, optional): Logger object. Defaults to None.
        """
        super(VAEModel, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.beta = beta

        if logger is None:
            prefix = "pgsui_output" if prefix == "pgsui" else prefix
            logman = LoggerManager(
                name=__name__, prefix=prefix, verbose=verbose >= 1, debug=debug
            )
            self.logger = logman.get_logger()
        else:
            self.logger = logger

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

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of the reconstruction, z_mean, and z_log_var
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
        """Compute the loss function for the VAE model.

        This method computes the loss function for the VAE model. The loss function consists of a reconstruction loss and a KL divergence term. The reconstruction loss is computed using a focal loss, while the KL divergence term is computed using the KL divergence between the posterior and the prior. The class weights and mask are used to weight the loss function and mask the loss values, respectively. The loss function is a weighted sum of the reconstruction loss and the KL divergence term. The class weights are used to weight the loss values for each class. The mask is used to mask the loss values for missing data. The loss function is used to optimize the model parameters using backpropagation. The loss function is used to compute the gradients of the model parameters with respect to the loss value.

        Args:
            y (torch.Tensor): Target data tensor.
            outputs (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple of the reconstruction, z_mean, and z_log_var.
            mask (torch.Tensor, optional): Mask tensor. Defaults to None.
            class_weights (torch.Tensor, optional): Class weights tensor. Defaults to None.

        Returns:
            torch.Tensor: Loss value.
        """
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

    def _resolve_activation(self, activation: str | torch.nn.Module) -> torch.nn.Module:
        """Resolve the activation function.

        This method resolves the activation function based on the input string.

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
