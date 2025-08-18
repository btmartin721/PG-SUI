import logging
from typing import List, Literal

import torch
import torch.nn as nn
from snpio.utils.logging import LoggerManager

from pgsui.impute.unsupervised.loss_functions import MaskedFocalLoss


class NLPCAModel(nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        prefix: str = "pgsui",
        num_classes: int = 3,
        hidden_layer_sizes: List[int] = [128, 64],
        latent_dim: int = 2,
        dropout_rate: float = 0.2,
        activation: str = "relu",
        gamma: float = 2.0,
        device: Literal["gpu", "cpu", "mps"] = "cpu",
        logger: logging.Logger | None = None,
        verbose: int = 0,
        debug: bool = False,
        use_convolution: bool = False,
    ):
        """Non-linear Principal Component Analysis (NLPCA) model for imputation.

        This model is used to impute missing values in genotype data using backpropagation to randomized inputs to fit the real data (targets). The model refines both the input latent dimension and the weights using backpropagation.

        Args:
            n_features (int): Number of features (SNPs) in the input data.
            prefix (str): Prefix for the logger. Default is 'pgsui'.
            num_classes (int): Number of classes for each feature. Default is 3.
            hidden_layer_sizes (list[int]): List of hidden layer sizes.
            latent_dim (int): Dimensionality of the latent space. Default is 64.
            dropout_rate (float): Dropout rate for regularization. Default is 0.2.
            activation (str): Activation function for hidden layers. Default is 'relu'.
            gamma (float): Focal loss gamma parameter. Default is 2.0.
            device (str): Device to run the model on ('gpu', 'cpu', or 'mps'). Default is 'cpu'.
            logger (logging.Logger, optional): Logger instance. Default is None.
            verbose (int): Verbosity level for logging. Default is 0.
            debug (bool): Debug mode flag. Default is False.
            use_convolution (bool): Whether to use convolutional layers. Ignored for this model. Default is False.
        """
        super(NLPCAModel, self).__init__()

        if logger is None:
            prefix = "pgsui_output" if prefix == "pgsui" else prefix
            logman = LoggerManager(
                name=__name__, prefix=prefix, verbose=verbose >= 1, debug=debug
            )
            self.logger = logman.get_logger()
        else:
            self.logger = logger

        self.n_features = n_features
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.device = device

        if hidden_layer_sizes[0] > hidden_layer_sizes[-1]:
            hidden_layer_sizes = list(reversed(hidden_layer_sizes))

        # Flexible deep network.
        layers = []
        input_dim = latent_dim
        for size in hidden_layer_sizes:
            layers.append(nn.Linear(input_dim, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Dropout(dropout_rate))
            layers.append(self._resolve_activation(activation))
            input_dim = size

        layers.append(nn.Linear(hidden_layer_sizes[-1], n_features * num_classes))

        self.phase23_decoder = nn.Sequential(*layers)
        self.reshape = (n_features, num_classes)

    def _resolve_activation(self, activation: str) -> nn.Module:
        """Resolve activation function by name.

        This method resolves the activation function by name and returns the corresponding module.

        Args:
            activation (str): Name of the activation function.

        Returns:
            nn.Module: Activation function module.

        Raises:
            ValueError: If the activation function is not supported.
        """
        activation = activation.lower()
        if activation == "relu":
            return nn.ReLU()
        elif activation == "elu":
            return nn.ELU()
        elif activation == "leaky_relu":
            return nn.LeakyReLU()
        elif activation == "selu":
            return nn.SELU()
        else:
            msg = f"Activation function {activation} not supported."
            self.logger.error(msg)
            raise ValueError(msg)

    def forward(self, x: torch.Tensor, phase: int = 1) -> torch.Tensor:
        """Forward pass through the model.

        This method performs a forward pass through the model. The phase parameter determines which decoder to use. Phase 1 uses a linear decoder, while phases 2 and 3 use a flexible deeper network. The output tensor is reshaped to the original target shape.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_features * num_classes).
            phase (int): Phase of training (1, 2, or 3).

        Returns:
            torch.Tensor: Output tensor of the same shape as the input.
        """
        x = self.phase23_decoder(x)
        return x.view(-1, *self.reshape)

    def compute_loss(
        self, y, outputs, mask=None, class_weights=None, gamma: float = 2.0
    ) -> torch.Tensor:
        """Compute the loss for the UBP model.

        This method computes the loss between the ground truth labels and the model outputs.

        Args:
            y (torch.Tensor): Ground truth labels.
            outputs (torch.Tensor): Model outputs.
            mask (torch.Tensor | None): Optional mask to apply to the loss.
            class_weights (torch.Tensor | None): Optional class weights.
            gamma (float): Focusing parameter for the focal loss.

        Returns:
            torch.Tensor: Computed loss.
        """
        if class_weights is None:
            class_weights = torch.ones(self.num_classes, device=outputs.device)

        if mask is None:
            mask = torch.ones_like(y, dtype=torch.bool)

        # Focal loss with weights
        criterion = MaskedFocalLoss(gamma=gamma, alpha=class_weights)
        return criterion(
            outputs.to(self.device), y.to(self.device), valid_mask=mask.to(self.device)
        )
