import logging

import torch
import torch.nn as nn
from snpio.utils.logging import LoggerManager

from pgsui.impute.unsupervised.loss_functions import MaskedFocalLoss


class UBPModel(nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        prefix: str = "pgsui",
        num_classes: int = 3,
        hidden_layer_sizes: list[int] = [128, 64],
        latent_dim: int = 2,
        dropout_rate: float = 0.2,
        activation: str = "relu",
        gamma: float = 2.0,
        logger: logging.Logger | None = None,
        verbose: int = 0,
        debug: bool = False,
    ):
        """Unsupervised Backpropagation (UBP) model for imputation.

        This model is used to impute missing values in genotype data using unsupervised backpropagation (UBP). The model consists of three phases: 1) a linear decoder with latent space inputs that get refined to create a decent starting point for phase 2, 2) a flexible deeper network where only the weights get refined, and 3) the same deep network as phase 2 but with both the input latent dimension and the weights getting refined.

        Args:
            n_features (int): Number of features (SNPs) in the input data.
            prefix (str): Prefix for the logger. Default is 'pgsui'.
            num_classes (int): Number of classes for each feature. Default is 3.
            hidden_layer_sizes (list[int]): List of hidden layer sizes for phases 2 and 3.
            latent_dim (int): Dimensionality of the latent space. Default is 64.
            dropout_rate (float): Dropout rate for regularization. Default is 0.2.
            activation (str): Activation function for hidden layers. Default is 'relu'.
            gamma (float): Focal loss gamma parameter. Default is 2.0.
            logger (logging.Logger, optional): Logger instance. Default is None.
            verbose (int): Verbosity level for logging. Default is 0.
            debug (bool): Debug mode flag. Default is False.
        """
        super(UBPModel, self).__init__()

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

        self.phase1_decoder = nn.Sequential(
            nn.Linear(latent_dim, n_features * num_classes),
        )

        if hidden_layer_sizes[0] > hidden_layer_sizes[-1]:
            hidden_layer_sizes = list(reversed(hidden_layer_sizes))

        # Phase 2 & 3: Flexible deeper network
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
        """Forward pass through the UBP model.

        This method performs a forward pass through the model. The phase parameter determines which decoder to use. Phase 1 uses a linear decoder, while phases 2 and 3 use a flexible deeper network. The output tensor is reshaped to the original target shape.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_features * num_classes).
            phase (int): Phase of training (1, 2, or 3).

        Returns:
            torch.Tensor: Output tensor of the same shape as the input.
        """
        if phase == 1:
            x = self.phase1_decoder(x)
        else:  # Phase 2 or 3
            x = self.phase23_decoder(x)

        return x.view(-1, *self.reshape)

    def compute_loss(
        self,
        y: torch.Tensor,
        outputs: torch.Tensor,
        mask: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the masked focal loss between predictions and targets.

        This method computes the masked focal loss between the model predictions and the ground truth labels. The mask tensor is used to ignore certain values (< 0), and class weights can be provided to balance the loss. The focal loss is a variant of the cross-entropy loss that focuses on hard-to-classify examples. It is useful for imbalanced datasets. The loss is computed as: L = -α_t * (1 - p_t)^γ * log(p_t), where α_t is the class weight, p_t is the predicted probability, and γ is the focal loss gamma parameter.

        Args:
            y (torch.Tensor): Ground truth labels of shape (batch, seq).
            outputs (torch.Tensor): Model outputs.
            mask (torch.Tensor, optional): Mask tensor to ignore certain values. If None, all values are used. Default is None.
            class_weights (torch.Tensor, optional): Class weights for the loss. If None, all classes are weighted equally. Default is None.

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
