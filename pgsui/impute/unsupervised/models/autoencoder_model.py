from typing import List, Literal

import numpy as np
import torch
import torch.nn as nn
from snpio.utils.logging import LoggerManager

from pgsui.utils.logging_utils import configure_logger


class Encoder(nn.Module):
    """The Encoder module of a standard Autoencoder.

    This module defines the encoder network, which takes high-dimensional input data and maps it to a deterministic, low-dimensional latent representation. The architecture consists of a series of fully-connected hidden layers that progressively compress the flattened input data into a single latent vector, `z`.
    """

    def __init__(
        self,
        n_features: int,
        num_classes: int,
        latent_dim: int,
        hidden_layer_sizes: List[int],
        dropout_rate: float,
        activation: torch.nn.Module,
    ):
        """Initializes the Encoder module.

        This class defines the encoder network, which takes high-dimensional input data and maps it to a deterministic, low-dimensional latent representation. The architecture consists of a series of fully-connected hidden layers that progressively compress the flattened input data into a single latent vector, `z`.

        Args:
            n_features (int): The number of features in the input data (e.g., SNPs).
            num_classes (int): Number of genotype states per locus (2 for haploid, 3 for diploid in practice).
            latent_dim (int): The dimensionality of the output latent space.
            hidden_layer_sizes (List[int]): A list of integers specifying the size of each hidden layer.
            dropout_rate (float): The dropout rate for regularization in the hidden layers.
            activation (torch.nn.Module): An instantiated activation function module (e.g., `nn.ReLU()`) for the hidden layers.
        """
        super(Encoder, self).__init__()
        self.flatten = nn.Flatten()

        layers = []
        input_dim = n_features * num_classes
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation)
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_size

        self.hidden_layers = nn.Sequential(*layers)
        self.dense_z = nn.Linear(input_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the encoder.

        Args:
            x (torch.Tensor): The input data tensor of shape `(batch_size, n_features, num_classes)`.

        Returns:
            torch.Tensor: The latent representation `z` of shape `(batch_size, latent_dim)`.
        """
        x = self.flatten(x)
        x = self.hidden_layers(x)
        z = self.dense_z(x)
        return z


class Decoder(nn.Module):
    """The Decoder module of a standard Autoencoder.

    This module defines the decoder network, which takes a deterministic latent vector and maps it back to the high-dimensional data space, aiming to reconstruct the original input. The architecture typically mirrors the encoder, consisting of a series of fully-connected hidden layers that progressively expand the representation, followed by a final linear layer to produce the reconstructed data.
    """

    def __init__(
        self,
        n_features: int,
        num_classes: int,
        latent_dim: int,
        hidden_layer_sizes: List[int],
        dropout_rate: float,
        activation: torch.nn.Module,
    ) -> None:
        """Initializes the Decoder module.

        Args:
            n_features (int): The number of features in the output data (e.g., SNPs).
            num_classes (int): Number of genotype states per locus (2 or 3 in practice).
            latent_dim (int): The dimensionality of the input latent space.
            hidden_layer_sizes (List[int]): A list of integers specifying the size of each hidden layer (typically the reverse of the encoder's).
            dropout_rate (float): The dropout rate for regularization in the hidden layers.
            activation (torch.nn.Module): An instantiated activation function module (e.g., `nn.ReLU()`) for the hidden layers.
        """
        super(Decoder, self).__init__()

        layers = []
        input_dim = latent_dim
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation)
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_size

        self.hidden_layers = nn.Sequential(*layers)
        output_dim = n_features * num_classes
        self.dense_output = nn.Linear(input_dim, output_dim)
        self.reshape = (n_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the decoder.

        Args:
            x (torch.Tensor): The input latent tensor of shape `(batch_size, latent_dim)`.

        Returns:
            torch.Tensor: The reconstructed output data of shape `(batch_size, n_features, num_classes)`.
        """
        x = self.hidden_layers(x)
        x = self.dense_output(x)
        return x.view(-1, *self.reshape)


class AutoencoderModel(nn.Module):
    """A standard Autoencoder (AE) model for imputation.

    This class combines an `Encoder` and a `Decoder` to form a standard autoencoder. The model is trained to learn a compressed, low-dimensional representation of the input data and then reconstruct it as accurately as possible. It is particularly useful for unsupervised dimensionality reduction and data imputation.

    **Model Architecture and Objective:**

    The autoencoder consists of two parts: an encoder, $f_{\\theta}$, and a decoder, $g_{\\phi}$.
        1.  The **encoder** maps the input data $x$ to a latent representation $z$:
            $$
            z = f_{\theta}(x)
            $$
        2.  The **decoder** reconstructs the data $\\hat{x}$ from the latent representation:
            $$
            \\hat{x} = g_{\\phi}(z)
            $$

    The model is trained by minimizing a reconstruction loss, $L(x, \\hat{x})$, which measures the dissimilarity between the original input and the reconstructed output. This implementation uses a ``FocalCELoss`` to handle missing values and class imbalance effectively.
    """

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
        gamma: torch.Tensor = torch.tensor(2.0),
        device: Literal["cpu", "gpu", "mps"] = "cpu",
        verbose: bool = False,
        debug: bool = False,
    ):
        """Initializes the AutoencoderModel.

        Args:
            n_features (int): The number of features in the input data (e.g., SNPs).
            prefix (str): A prefix used for logging.
            num_classes (int): Number of genotype states per locus. Defaults to 4 for backward compatibility, but the genotype imputers pass 2 (haploid) or 3 (diploid).
            hidden_layer_sizes (List[int] | np.ndarray): A list of integers specifying the size of each hidden layer in the encoder. The decoder will use the reverse of this structure. Defaults to [128, 64].
            latent_dim (int): The dimensionality of the latent space (bottleneck). Defaults to 2.
            dropout_rate (float): The dropout rate for regularization in hidden layers. Defaults to 0.2.
            activation (Literal["relu", "elu", "selu", "leaky_relu"]): The name of the activation function for hidden layers. Defaults to "relu".
            gamma (float): The focusing parameter for the focal loss function. Defaults to 2.0.
            device (Literal["cpu", "gpu", "mps"]): The device to run the model on.
            verbose (bool): If True, enables detailed logging.
            debug (bool): If True, enables debug mode.
        """
        super(AutoencoderModel, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.device = device

        logman = LoggerManager(
            name=__name__, prefix=prefix, verbose=verbose, debug=debug
        )
        self.logger = configure_logger(
            logman.get_logger(), verbose=verbose, debug=debug
        )

        activation_module = self._resolve_activation(activation)

        if isinstance(hidden_layer_sizes, np.ndarray):
            hls = hidden_layer_sizes.tolist()
        else:
            hls = hidden_layer_sizes

        self.encoder = Encoder(
            n_features,
            self.num_classes,
            latent_dim,
            hls,
            dropout_rate,
            activation_module,
        )

        decoder_layer_sizes = list(reversed(hls))
        self.decoder = Decoder(
            n_features,
            self.num_classes,
            latent_dim,
            decoder_layer_sizes,
            dropout_rate,
            activation_module,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the full Autoencoder model.

        Args:
            x (torch.Tensor): The input data tensor of shape `(batch_size, n_features, num_classes)`.

        Returns:
            torch.Tensor: The reconstructed data tensor.
        """
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction

    def _resolve_activation(
        self, activation: Literal["relu", "elu", "leaky_relu", "selu"]
    ) -> torch.nn.Module:
        """Resolves an activation function module from a string name.

        Args:
            activation (Literal["relu", "elu", "leaky_relu", "selu"]): The name of the activation function.

        Returns:
            torch.nn.Module: The corresponding instantiated PyTorch activation function module.

        Raises:
            ValueError: If the provided activation name is not supported.
        """
        act: str = activation.lower()

        if act == "relu":
            return nn.ReLU()
        elif act == "elu":
            return nn.ELU()
        elif act in ("leaky_relu", "leakyrelu"):
            return nn.LeakyReLU()
        elif act == "selu":
            return nn.SELU()
        else:
            msg = f"Activation {activation} not supported."
            self.logger.error(msg)
            raise ValueError(msg)
