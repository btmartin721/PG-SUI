from typing import List, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from snpio.utils.logging import LoggerManager
from torch.distributions import Normal

from pgsui.impute.unsupervised.loss_functions import MaskedFocalLoss


class Sampling(nn.Module):
    """A layer that samples from a latent distribution using the reparameterization trick.

    This layer is a core component of a Variational Autoencoder (VAE). It takes the mean and log-variance of a latent distribution as input and generates a sample from that distribution. By using the reparameterization trick ($z = \mu + \sigma \cdot \epsilon$), it allows gradients to be backpropagated through the random sampling process, making the VAE trainable.
    """

    def forward(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass to generate a latent sample.

        Args:
            z_mean (torch.Tensor): The mean of the latent normal distribution.
            z_log_var (torch.Tensor): The log of the variance of the latent normal distribution.

        Returns:
            torch.Tensor: A sampled vector from the latent space.
        """
        z_sigma = torch.exp(0.5 * z_log_var)  # Precompute outside

        # Ensure on GPU
        # rand_like takes random values from a normal distribution
        # of the same shape as z_mean.
        epsilon = torch.randn_like(z_mean, device=z_mean.device)
        return z_mean + z_sigma * epsilon


class Encoder(nn.Module):
    """The Encoder module of a Variational Autoencoder (VAE).

    This module defines the encoder network, which takes high-dimensional input data and maps it to the parameters of a lower-dimensional latent distribution. The architecture consists of a series of fully-connected hidden layers that process the flattened input. The network culminates in two separate linear layers that output the mean (`z_mean`) and log-variance (`z_log_var`) of the approximate posterior distribution, $q(z|x)$.
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

        Args:
            n_features (int): The number of features in the input data (e.g., SNPs).
            num_classes (int): Number of genotype states per locus (2 for haploid, 3 for diploid in practice).
            latent_dim (int): The dimensionality of the latent space.
            hidden_layer_sizes (List[int]): A list of integers specifying the size of each hidden layer.
            dropout_rate (float): The dropout rate for regularization in the hidden layers.
            activation (torch.nn.Module): An instantiated activation function module (e.g., `nn.ReLU()`) for the hidden layers.
        """
        super(Encoder, self).__init__()
        self.flatten = nn.Flatten()
        self.activation = (
            getattr(F, activation) if isinstance(activation, str) else activation
        )

        layers = []
        # The input dimension accounts for channels
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
        """Performs the forward pass through the encoder.

        Args:
            x (torch.Tensor): The input data tensor of shape `(batch_size, n_features, num_classes)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the latent mean (`z_mean`), latent log-variance (`z_log_var`), and a sample from the latent distribution (`z`).
        """
        x = self.flatten(x)
        x = self.hidden_layers(x)
        z_mean = self.dense_z_mean(x)
        z_log_var = self.dense_z_log_var(x)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z


class Decoder(nn.Module):
    """The Decoder module of a Variational Autoencoder (VAE).

    This module defines the decoder network, which takes a sample from the low-dimensional latent space and maps it back to the high-dimensional data space. It aims to reconstruct the original input data. The architecture consists of a series of fully-connected hidden layers followed by a final linear layer that produces the reconstructed data, which is then reshaped to match the original input's dimensions.
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
            num_classes (int): Number of genotype states per locus (typically 2 or 3).
            latent_dim (int): The dimensionality of the input latent space.
            hidden_layer_sizes (List[int]): A list of integers specifying the size of each hidden layer.
            dropout_rate (float): The dropout rate for regularization in the hidden layers.
            activation (torch.nn.Module): An instantiated activation function module (e.g., `nn.ReLU()`) for the hidden layers.
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
        # UPDATED: Output dimension must account for channels
        output_dim = n_features * num_classes
        self.dense_output = nn.Linear(input_dim, output_dim)
        # UPDATED: Reshape must account for channels
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


class VAEModel(nn.Module):
    """A Variational Autoencoder (VAE) model for imputation.

    This class combines an `Encoder` and a `Decoder` to form a VAE, a generative model for learning complex data distributions. It is designed for imputing missing values in categorical data, such as genomic SNPs. The model is trained by maximizing the Evidence Lower Bound (ELBO), which is a lower bound on the log-likelihood of the data.

    **Objective Function (ELBO):**
    The VAE loss function is derived from the ELBO and consists of two main components: a reconstruction term and a regularization term.
    $$
    \\mathcal{L}(\\theta, \\phi; x) = \\underbrace{\\mathbb{E}_{q_{\\phi}(z|x)}[\\log p_{\\theta}(x|z)]}_{\\text{Reconstruction Loss}} - \\underbrace{D_{KL}(q_{\\phi}(z|x) || p(z))}_{\\text{KL Divergence}}
    $$
    -   The **Reconstruction Loss** encourages the decoder to accurately reconstruct the input data from its latent representation. This implementation uses a `MaskedFocalLoss`.
    -   The **KL Divergence** acts as a regularizer, forcing the approximate posterior distribution $q_{\\phi}(z|x)$ learned by the encoder to be close to a prior distribution $p(z)$ (typically a standard normal distribution).
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
        gamma: float = 2.0,
        beta: float = 1.0,
        device: Literal["cpu", "gpu", "mps"] = "cpu",
        verbose: bool = False,
        debug: bool = False,
    ):
        """Initializes the VAEModel.

        Args:
            n_features (int): The number of features in the input data (e.g., SNPs).
            prefix (str): A prefix used for logging.
            num_classes (int): Number of genotype states per locus. Defaults to 4 for backward compatibility, though the imputer passes 2 (haploid) or 3 (diploid).
            hidden_layer_sizes (List[int] | np.ndarray): A list of integers specifying the size of each hidden layer in the encoder and decoder. Defaults to [128, 64].
            latent_dim (int): The dimensionality of the latent space. Defaults to 2.
            dropout_rate (float): The dropout rate for regularization in the hidden layers. Defaults to 0.2.
            activation (str): The name of the activation function to use in hidden layers. Defaults to "relu".
            gamma (float): The focusing parameter for the focal loss component. Defaults to 2.0.
            beta (float): A weighting factor for the KL divergence term in the total loss ($\beta$-VAE). Defaults to 1.0.
            device (Literal["cpu", "gpu", "mps"]): The device to run the model on.
            verbose (bool): If True, enables detailed logging. Defaults to False.
            debug (bool): If True, enables debug mode. Defaults to False.
        """
        super(VAEModel, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.beta = beta
        self.device = device

        logman = LoggerManager(
            name=__name__, prefix=prefix, verbose=verbose, debug=debug
        )
        self.logger = logman.get_logger()

        activation = self._resolve_activation(activation)

        self.encoder = Encoder(
            n_features,
            self.num_classes,
            latent_dim,
            hidden_layer_sizes,
            dropout_rate,
            activation,
        )

        decoder_layer_sizes = list(reversed(hidden_layer_sizes))

        self.decoder = Decoder(
            n_features,
            self.num_classes,
            latent_dim,
            decoder_layer_sizes,
            dropout_rate,
            activation,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs the forward pass through the full VAE model.

        Args:
            x (torch.Tensor): The input data tensor of shape `(batch_size, n_features, num_classes)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the reconstructed output, the latent mean (`z_mean`), and the latent log-variance (`z_log_var`).
        """
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var

    def compute_loss(
        self,
        outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        y: torch.Tensor,
        mask: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the VAE loss function (negative ELBO).

        The loss is the sum of a reconstruction term and a regularizing KL divergence term. The reconstruction loss is calculated using a masked focal loss, and the KL divergence measures the difference between the learned latent distribution and a standard normal prior.

        Args:
            outputs (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): The tuple of (reconstruction, z_mean, z_log_var) from the model's forward pass.
            y (torch.Tensor): The target data tensor, expected to be one-hot encoded. This is converted to class indices internally for the loss function.
            mask (torch.Tensor | None): A boolean mask to exclude missing values from the reconstruction loss.
            class_weights (torch.Tensor | None): Weights to apply to each class in the reconstruction loss to handle imbalance.

        Returns:
            torch.Tensor: The computed scalar loss value.
        """
        reconstruction, z_mean, z_log_var = outputs

        # 1. KL Divergence Calculation
        prior = Normal(torch.zeros_like(z_mean), torch.ones_like(z_log_var))
        posterior = Normal(z_mean, torch.exp(0.5 * z_log_var))
        kl_loss = (
            torch.distributions.kl.kl_divergence(posterior, prior).sum(dim=1).mean()
        )

        if class_weights is None:
            class_weights = torch.ones(self.num_classes, device=y.device)

        # 2. Reconstruction Loss Calculation
        # Reverting to the robust method of flattening tensors and using the
        # custom loss function.
        n_classes = reconstruction.shape[-1]
        logits_flat = reconstruction.reshape(-1, n_classes)

        # Convert one-hot `y` to class indices for the loss function.
        targets_flat = torch.argmax(y, dim=-1).reshape(-1)

        if mask is None:
            # If no mask is provided, all targets are considered valid.
            mask_flat = torch.ones_like(targets_flat, dtype=torch.bool)
        else:
            # The mask needs to be reshaped to match the flattened targets.
            mask_flat = mask.reshape(-1)

        # Logits, class-index targets, and the valid mask.
        criterion = MaskedFocalLoss(alpha=class_weights, gamma=self.gamma)

        reconstruction_loss = criterion(
            logits_flat.to(self.device),
            targets_flat.to(self.device),
            valid_mask=mask_flat.to(self.device),
        )

        return reconstruction_loss + self.beta * kl_loss

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
