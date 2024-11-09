from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from snpio.utils.logging import LoggerManager


class Sampling(nn.Module):
    """Layer to calculate Z to sample from latent dimension.

    The Sampling layer takes in the mean and log variance of the latent space and samples from the distribution.

    Example:
        >>> sampling = Sampling()
        >>> z = sampling(z_mean, z_log_var)
    """

    def __init__(self) -> None:
        """Initialize the Sampling layer.

        This method initializes the Sampling layer. The Sampling layer is used to sample from the distribution using the 'reparameterization trick'.
        """
        super(Sampling, self).__init__()

    def forward(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        """Sampling during forward pass.

        The forward pass calculates the standard deviation from the log variance and samples from the distribution. The sampling is done using the 'reparameterization trick'.

        Args:
            z_mean (torch.Tensor): Mean of the latent space.
            z_log_var (torch.Tensor): Log variance of the latent space.

        Returns:
            torch.Tensor: Sampled z from the distribution.
        """
        z_sigma = torch.exp(0.5 * z_log_var)
        epsilon = torch.randn_like(z_mean)
        return z_mean + z_sigma * epsilon


class Encoder(nn.Module):
    """VAE encoder to Encode genotypes to (z_mean, z_log_var, z).

    Class to encode the input data into the latent space. The encoder consists of dense layers to encode the input data into the latent space. The latent space is represented by the mean and log variance of the distribution.

    Example:
        >>> encoder = Encoder(n_features, num_classes, latent_dim, hidden_layer_sizes, dropout_rate, activation)
        >>> z_mean, z_log_var, z = encoder(input, training=True)

    Attributes:
        flatten (nn.Flatten): Flatten the input data.
        activation (torch.nn.Module): Activation function.
        hidden_layers (nn.Sequential): Hidden layers of the encoder.
        dense_z_mean (nn.Linear): Dense layer to calculate z_mean.
        dense_z_log_var (nn.Linear): Dense layer to calculate z_log_var.
        sampling (Sampling): Sampling layer to sample from the latent space.
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
        """Initialize the Encoder.

        The Encoder initializes the layers of the encoder. The encoder encodes the input data into the latent space.

        Args:
            n_features (int): Number of features.
            num_classes (int): Number of classes.
            latent_dim (int): Dimension of the latent space.
            hidden_layer_sizes (List[int]): List of hidden layer sizes.
            dropout_rate (float): Dropout rate.
            activation (Union[str, torch.nn.Module]): Activation function.
        """
        super(Encoder, self).__init__()
        self.flatten = nn.Flatten()
        self.activation = (
            getattr(F, activation) if isinstance(activation, str) else activation
        )

        # Define dense layers based on hidden_layer_sizes
        layers = []
        input_dim = n_features * num_classes
        for i, hidden_size in enumerate(hidden_layer_sizes):
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.ReLU() if activation == "relu" else activation)
            input_dim = hidden_size

        self.hidden_layers = nn.Sequential(*layers)

        # Latent layers
        self.dense_z_mean = nn.Linear(input_dim, latent_dim)
        self.dense_z_log_var = nn.Linear(input_dim, latent_dim)
        self.sampling = Sampling()

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the encoder.

        This method performs the forward pass of the encoder, which encodes the input data into the latent space. The encoder consists of dense layers to encode the input data into the latent space.

        Args:
            x (torch.Tensor): Input data. This is the genotype data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: z_mean, z_log_var, z
        """
        x = self.flatten(x)
        x = self.hidden_layers(x)
        z_mean = self.dense_z_mean(x)
        z_log_var = self.dense_z_log_var(x)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z


class Decoder(nn.Module):
    """Converts z, the encoded vector, back into the reconstructed output.

    Class to decode the latent space into the reconstructed output. The decoder consists of dense layers to decode the latent space into the reconstructed output.

    Example:
        >>> decoder = Decoder(n_features, num_classes, latent_dim, hidden_layer_sizes, dropout_rate, activation)
        >>> output = decoder(z)

    Attributes:
        activation (torch.nn.Module): Activation function.
        hidden_layers (nn.Sequential): Hidden layers of the decoder.
        dense_output (nn.Linear): Dense layer to calculate the output.
        reshape (Tuple[int, int]): Reshape the output.
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
        """Initialize the Decoder.

        This method initializes the decoder with the specified parameters. The decoder decodes the latent space into the reconstructed output.

        Args:
            n_features (int): Number of features.
            num_classes (int): Number of classes.
            latent_dim (int): Dimension of the latent space.
            hidden_layer_sizes (List[int]): List of hidden layer sizes.
            dropout_rate (float): Dropout rate.
            activation (Union[str, torch.nn.Module]): Activation function.
        """
        super(Decoder, self).__init__()

        # Define dense layers based on hidden_layer_sizes
        layers = []
        input_dim = latent_dim
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.Dropout(dropout_rate))
            layers.append(activation)
            input_dim = hidden_size

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.dense_output = nn.Linear(input_dim, n_features * num_classes)
        self.reshape = (n_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the decoder.

        This method performs the forward pass of the decoder, which decodes the latent space into the reconstructed output.

        Args:
            x (torch.Tensor): Input data. This is the latent space.

        Returns:
            torch.Tensor: Reconstructed output. The output is reshaped to the original shape.
        """
        x = self.hidden_layers(x)
        x = self.dense_output(x)
        return x.view(-1, *self.reshape)


class VAEModel(nn.Module):
    """VAE PyTorch Model.

    This class defines the VAE model, which consists of an encoder and a decoder. The model is used to encode the input data into the latent space and decode it back into the reconstructed output.

    Example:
        >>> model = VAEModel(n_features, num_classes, latent_dim, hidden_layer_sizes, dropout_rate, activation)
        >>> reconstruction, z_mean, z_log_var = model(input)
        >>> loss = model.compute_loss(y, (reconstruction, z_mean, z_log_var))

    Attributes:
        encoder (Encoder): Encoder to encode the input data.
        decoder (Decoder): Decoder to decode the latent space.
        logger (LoggerManager): Logger for the model.
    """

    def __init__(
        self,
        n_features=None,
        num_classes=3,
        latent_dim=2,
        hidden_layer_sizes=[128, 64],
        dropout_rate=0.2,
        activation="elu",
    ):
        """Initialize the VAE model.

        This method initializes the VAE model with the specified parameters. The model consists of an encoder and a decoder. The encoder encodes the input data into the latent space, and the decoder decodes the latent space into the reconstructed output.

        Args:
            n_features (int): Number of features.
            num_classes (int): Number of classes.
            latent_dim (int): Dimension of the latent space.
            hidden_layer_sizes (List[int]): List of hidden layer sizes.
            dropout_rate (float): Dropout rate.
            activation (str or torch.nn.Module): Activation function.

        Raises:
            ValueError: If n_features is None.
        """

        logman = LoggerManager(name=__name__, prefix="VAEModel")
        self.logger = logman.get_logger()

        if n_features is None:
            msg = "n_features must be provided, but got 'n_features=None'"
            self.logger.error(msg)
            raise ValueError(msg)

        activation = self._resolve_activation(activation)

        super(VAEModel, self).__init__()

        self.encoder = Encoder(
            n_features,
            num_classes,
            latent_dim,
            hidden_layer_sizes,
            dropout_rate,
            activation,
        )
        hidden_layer_sizes.reverse()
        self.decoder = Decoder(
            n_features,
            num_classes,
            latent_dim,
            hidden_layer_sizes,
            dropout_rate,
            activation,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the VAE model.

        This method performs the forward pass of the VAE model, which consists of encoding the input data into the latent space and decoding it back into the reconstructed output.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Reconstruction, z_mean, z_log_var
        """
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var

    def compute_loss(
        self, y: torch.Tensor, outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Compute VAE loss (reconstruction + KL divergence).

        This method computes the VAE loss, which is the sum of the reconstruction loss and the KL divergence loss. The reconstruction loss is the mean squared error between the input data and the reconstructed output. The KL divergence loss is the Kullback-Leibler divergence between the latent space and the standard normal distribution.

        Args:
            y (torch.Tensor): Input data to be used as ground truth. This is the target tensor for the reconstruction loss.
            outputs (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing the reconstruction, z_mean, and z_log_var. The reconstruction is the output of the decoder, and z_mean and z_log_var are the mean and log variance of the latent space.

        Returns:
            torch.Tensor: VAE loss (reconstruction + KL divergence).
        """
        if not isinstance(outputs, tuple):
            msg = f"Expected outputs to be a 3-element tuple: {type(outputs)}"
            self.logger.error(msg)
            raise ValueError
        reconstruction, z_mean, z_log_var = outputs
        reconstruction_loss = F.mse_loss(reconstruction, y, reduction="mean")
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        return reconstruction_loss + kl_loss

    def _resolve_activation(
        self, activation: Union[str, torch.nn.Module]
    ) -> torch.nn.Module:
        """Resolve activation function.

        This method resolves the activation function based on the input string or torch.nn.Module. If the input is a string, it maps the string to the corresponding torch.nn.Module activation function.

        Args:
            activation (str or torch.nn.Module): Activation function.

        Returns:
            torch.nn.Module: Resolved activation function.

        Raises:
            ValueError: If the activation function is not supported.
            TypeError: If the activation function is not a string or torch.nn.Module.
        """
        if isinstance(activation, str):
            if activation == "relu":
                activation = nn.ReLU()
            elif activation == "elu":
                activation = nn.ELU()
            elif activation == "leaky_relu":
                activation = nn.LeakyReLU()
            elif activation == "selu":
                activation = nn.SELU()
            else:
                msg = f"Activation function {activation} not supported."
                self.logger.error(msg)
                raise ValueError(msg)

        if not isinstance(activation, nn.Module):
            msg = f"Activation function must be a string or torch.nn.Module, but got {type(activation)}"
            self.logger.error(msg)
            raise TypeError(msg)

        return activation
