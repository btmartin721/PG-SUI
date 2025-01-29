from typing import Tuple

import torch
import torch.nn as nn
from pgsui.impute.unsupervised.loss_functions import MaskedFocalLoss


class UBPModel(nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        num_classes: int = 3,
        hidden_layer_sizes: list[int] = [128, 64],
        latent_dim: int = 64,
        dropout_rate: float = 0.2,
        activation: str = "relu",
        gamma: float = 2.0,
    ):
        """
        Unsupervised Backpropagation (UBP) model for imputation with three phases.

        Args:
            n_features (int): Number of features (SNPs) in the input data.
            num_classes (int): Number of classes for each feature. Default is 3.
            hidden_layer_sizes (list[int]): List of hidden layer sizes for phases 2 and 3.
            latent_dim (int): Dimensionality of the latent space. Default is 64.
            dropout_rate (float): Dropout rate for regularization. Default is 0.2.
            activation (str): Activation function for hidden layers. Default is 'relu'.
            gamma (float): Focal loss gamma parameter. Default is 2.0.
        """
        super(UBPModel, self).__init__()

        self.n_features = n_features
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.gamma = gamma

        # Phase 1: Linear model
        self.phase1_encoder = nn.Sequential(
            nn.Linear(n_features * num_classes, latent_dim),
        )

        self.phase1_decoder = nn.Sequential(
            nn.Linear(latent_dim, n_features * num_classes),
        )

        # Phase 2 & 3: Flexible deeper network
        layers = []
        input_dim = n_features * num_classes
        for size in hidden_layer_sizes:
            layers.append(nn.Linear(input_dim, size))
            layers.append(self._resolve_activation(activation))
            layers.append(nn.Dropout(dropout_rate))
            input_dim = size

        layers.append(nn.Linear(hidden_layer_sizes[-1], latent_dim))

        self.phase23_encoder = nn.Sequential(*layers)

        self.phase23_decoder = nn.Sequential(
            nn.Linear(latent_dim, n_features * num_classes),
        )

    def _resolve_activation(self, activation: str) -> nn.Module:
        """Resolve activation function by name."""
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
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x: torch.Tensor, phase: int = 1) -> torch.Tensor:
        """
        Forward pass through the UBP model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_features * num_classes).
            phase (int): Phase of training (1, 2, or 3).

        Returns:
            torch.Tensor: Output tensor of the same shape as the input.
        """
        # Reshape the input if it's not flattened
        if x.dim() == 3:
            x = x.view(x.size(0), -1)

        if phase == 1:
            z = self.phase1_encoder(x)
            reconstructed = self.phase1_decoder(z)
        else:  # Phase 2 or 3
            z = self.phase23_encoder(x)
            reconstructed = self.phase23_decoder(z)

        return reconstructed.view(x.size())

    def compute_loss(
        self,
        y: torch.Tensor,
        outputs: torch.Tensor,
        mask: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the masked focal loss between predictions and targets.

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
