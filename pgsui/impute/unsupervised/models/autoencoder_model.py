import logging
from typing import List

import torch
import torch.nn as nn
from snpio.utils.logging import LoggerManager

from pgsui.impute.unsupervised.loss_functions import MaskedFocalLoss


def get_activation(activation: str | nn.Module) -> nn.Module:
    """Resolve activation function by name.

    This method resolves the activation function by name and returns the corresponding module.

    Args:
        activation (str | nn.Module): Name of the activation function or the activation function module itself.

    Returns:
        nn.Module: Activation function module.

    Raises:
        ValueError: If the activation function is not supported
    """

    if isinstance(activation, str):
        act = activation.lower().strip()

        if act in {"leaky_relu", "leakyrelu"}:
            act = "leaky_relu"

        if act == "relu":
            return nn.ReLU()
        elif act == "elu":
            return nn.ELU()
        elif act == "selu":
            return nn.SELU()
        elif act == "leaky_relu":
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    elif isinstance(activation, nn.Module):
        return activation
    else:
        raise ValueError("activation must be a string or nn.Module")


class Encoder(nn.Module):
    """Encoder network for the autoencoder model.

    Encodes an input of shape (batch_size, n_features, num_classes) into a latent representation. This is the first part of the autoencoder. The encoder is a feedforward neural network with hidden layers and an output layer. The output layer is the latent representation. The encoder is used to compress the input data into a lower-dimensional latent space. The latent representation is then used by the decoder to reconstruct the input data. The encoder is trained to minimize the reconstruction error between the input and the output of the decoder. The encoder is used to learn a compact representation of the input data that captures the most important features. The encoder is used in unsupervised learning to learn a representation of the input data that can be used for other tasks, such as clustering or classification.
    """

    def __init__(
        self,
        n_features: int,
        num_classes: int = 3,
        hidden_layer_sizes: list[int] = [128, 64],
        latent_dim: int = 2,
        dropout_rate: float = 0.2,
        activation: str = "relu",
        gamma: float = 2.0,
    ):
        """Initialize the Encoder network.

        This module encodes an input tensor of shape (batch_size, n_features, num_classes) into a latent representation. The encoder is a feedforward neural network with hidden layers and an output layer. The output layer is the latent representation. The encoder is used to compress the input data into a lower-dimensional latent space. The latent representation is then used by the decoder to reconstruct the input data. The encoder is trained to minimize the reconstruction error between the input and the output of the decoder. The encoder is used to learn a compact representation of the input data that captures the most important features. The encoder is used in unsupervised learning to learn a representation of the input data that can be used for other tasks, such as clustering or classification.

        Args:
            n_features (int): Number of features.
            num_classes (int): Number of classes.
            latent_dim (int): Dimension of the latent representation.
            hidden_layer_sizes (List[int]): List of hidden layer sizes.
            dropout_rate (float): Dropout rate.
            activation (str or nn.Module): Activation function to use.
            gamma (float): Focal loss gamma parameter.
        """
        super(Encoder, self).__init__()

        self.n_features = n_features
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.gamma = gamma

        # Calculate the input dimension (flattened input)
        input_dim = n_features * num_classes

        # Build the hidden layers dynamically.
        layers = nn.ModuleList()
        for size in hidden_layer_sizes:
            layers.append(nn.Linear(input_dim, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Dropout(dropout_rate))
            layers.append(get_activation(activation))
            input_dim = size

        # Final latent layer.
        layers.append(nn.Linear(input_dim, latent_dim))
        layers.append(get_activation(activation))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_features, num_classes).

        Returns:
            torch.Tensor: Latent representation tensor.
        """
        x = x.view(x.size(0), self.num_classes * self.n_features)
        return self.encoder(x)


class Decoder(nn.Module):
    """Decoder network for the autoencoder model.

    Decodes the latent representation back to a tensor with shape (batch_size, n_features, num_classes). This is the reconstruction of the input. The decoder is the second part of the autoencoder. The decoder is a feedforward neural network with hidden layers and an output layer. The output layer is the reconstructed input. The decoder is trained to minimize the reconstruction error between the input and the output. The decoder is used to reconstruct the input data from the latent representation learned by the encoder. The decoder is used to generate new data points by sampling from the latent space. The decoder is used in unsupervised learning to generate new data points that are similar to the input data. The decoder is used in generative modeling to generate new data points that follow the same distribution as the input data.
    """

    def __init__(
        self,
        n_features: int,
        num_classes: int = 3,
        hidden_layer_sizes: list[int] = [128, 64],
        latent_dim: int = 2,
        dropout_rate: float = 0.2,
        activation: str = "relu",
        gamma: float = 2.0,
    ):
        """
        Initialize the Decoder network.

        This module decodes the latent representation back to a tensor with shape (batch_size, n_features, num_classes). This is the reconstruction of the input. The decoder is the second part of the autoencoder. The decoder is a feedforward neural network with hidden layers and an output layer. The output layer is the reconstructed input. The decoder is trained to minimize the reconstruction error between the input and the output. The decoder is used to reconstruct the input data from the latent representation learned by the encoder. The decoder is used to generate new data points by sampling from the latent space. The decoder is used in unsupervised learning to generate new data points that are similar to the input data. The decoder is used in generative modeling to generate new data points that follow the same distribution as the input data.

        Args:
            n_features (int): Number of features.
            num_classes (int): Number of classes.
            latent_dim (int): Dimension of the latent representation.
            hidden_layer_sizes (List[int]): List of hidden layer sizes for the decoder.
            dropout_rate (float): Dropout rate.
            activation (str or nn.Module): Activation function to use.
        """
        super(Decoder, self).__init__()

        self.n_features = n_features
        self.num_classes = num_classes
        self.hidden_layer_sizes = hidden_layer_sizes
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.gamma = gamma

        output_dim = n_features * num_classes

        # Build the hidden layers.
        layers = nn.ModuleList()
        input_dim = latent_dim
        for size in hidden_layer_sizes:
            layers.append(nn.Linear(input_dim, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Dropout(dropout_rate))
            layers.append(get_activation(activation))
            input_dim = size

        # Final output layer
        # No activation function here, as it gets applied later.
        layers.append(nn.Linear(hidden_layer_sizes[-1], output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the decoder.

        Args:
            x (torch.Tensor): Latent representation tensor.

        Returns:
            torch.Tensor: Reconstructed tensor of shape (batch_size, n_features, num_classes).
        """
        x = self.decoder(x)
        x = x.view(-1, self.n_features, self.num_classes)
        return x


class AutoencoderModel(nn.Module):
    """Autoencoder model for unsupervised learning.

    This autoencoder is intended to predict three classes (0, 1, and 2) and is  designed to help address class imbalance. The model consists of an encoder  and decoder, with an optional final activation. No final activation function is applied, so outputs must be passed through an activation function, such as a softmax. The model uses a focal loss for training. The encoder and decoder are built with the specified hidden layer sizes and latent dimension. The latent dimension is the dimension of the latent representation. The dropout rate is applied to the hidden layers. The activation function is applied to the hidden layers. The gamma parameter is used in the focal loss. The model is used for unsupervised learning to learn a representation of the input data that can be used for other tasks, such as clustering or classification.
    """

    def __init__(
        self,
        *,
        n_features: int,
        prefix: str = "psgui",
        num_classes: int = 3,
        hidden_layer_sizes: List[int] = [128, 64],
        latent_dim: int = 2,
        dropout_rate: float = 0.2,
        activation: str = "relu",
        gamma: float = 2.0,
        logger: logging.Logger | None = None,
        verbose: int = 0,
        debug: bool = False,
    ):
        """Initialize the Autoencoder model.

        This model is designed to predict three classes (0, 1, and 2) and is intended to help address class imbalance. The model consists of an encoder and decoder. No final activation function is applied, so outputs must be passed through an activation function, such as a softmax. The model uses a focal loss for training. The encoder and decoder are built with the specified hidden layer sizes and latent dimension. The latent dimension is the dimension of the latent representation. The dropout rate is applied to the hidden layers. The activation function is applied to the hidden layers. The gamma parameter is used in the focal loss. The model is used for unsupervised learning to learn a representation of the input data that can be used for other tasks, such as clustering or classification.

        Args:
            n_features (int): Number of features.
            num_classes (int): Number of classes. Default is 3.
            hidden_layer_sizes (List[int]): List of hidden layer sizes. Default is [128, 64].
            latent_dim (int): Dimension of the latent representation. Default is 2.
            dropout_rate (float): Dropout rate. Default is 0.2.
            activation (str): Activation function to use. Default is "relu".
            gamma (float): Focal loss gamma parameter. Default is 2.0.
        """
        super(AutoencoderModel, self).__init__()
        self.n_features = n_features
        self.num_classes = num_classes
        self.hidden_layer_sizes = hidden_layer_sizes
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.gamma = gamma

        if logger is None:
            prefix = "pgsui_output" if prefix == "pgsui" else prefix
            logman = LoggerManager(
                name=__name__, prefix=prefix, verbose=verbose >= 1, debug=debug
            )
            self.logger = logman.get_logger()
        else:
            self.logger = logger

        # For the encoder, multiply each hidden layer size by num_classes.
        encoder_hidden_sizes = [h * num_classes for h in hidden_layer_sizes]

        # For the decoder, reverse the encoder hidden sizes.
        decoder_hidden_sizes = list(reversed(encoder_hidden_sizes))

        self.encoder = Encoder(
            n_features=n_features,
            num_classes=num_classes,
            hidden_layer_sizes=encoder_hidden_sizes,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            activation=activation,
            gamma=gamma,
        )

        self.decoder = Decoder(
            n_features=n_features,
            num_classes=num_classes,
            hidden_layer_sizes=decoder_hidden_sizes,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            activation=activation,
            gamma=gamma,
        )

    def forward(self, x):
        """Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_features, num_classes).

        Returns:
            torch.Tensor: Reconstructed output tensor.
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def compute_loss(
        self,
        y: torch.Tensor,
        outputs: torch.Tensor,
        mask: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the masked focal loss between predictions and targets.

        This method computes the masked focal loss between the model predictions and the ground truth labels. The mask tensor is used to ignore certain values (< 0), and class weights can be provided to balance the loss. The focal loss is a modified version of the cross-entropy loss that focuses on hard-to-classify examples. The focal loss is used to address class imbalance and improve the performance of the model.

        Args:
            y (torch.Tensor): Ground truth labels of shape (batch, seq).
            outputs (torch.Tensor): Model outputs.
            mask (torch.Tensor, optional): Mask tensor to ignore certain values. Default is None.
            class_weights (torch.Tensor, optional): Class weights for the loss. Default is None.

        Returns:
            torch.Tensor: Computed focal loss value.
        """
        if class_weights is None:
            class_weights = torch.ones(self.num_classes, device=y.device)

        if mask is None:
            mask = torch.ones_like(y, dtype=torch.bool)

        criterion = MaskedFocalLoss(alpha=class_weights, gamma=self.gamma)
        reconstruction_loss = criterion(outputs, y, valid_mask=mask)
        return reconstruction_loss
