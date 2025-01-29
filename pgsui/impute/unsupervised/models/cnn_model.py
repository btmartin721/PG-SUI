from typing import List, Union

import torch
import torch.nn as nn
from snpio.utils.logging import LoggerManager
from torch.utils.checkpoint import checkpoint_sequential

from pgsui.impute.unsupervised.loss_functions import MaskedFocalLoss


class CNNModel(nn.Module):
    def __init__(
        self,
        n_features,
        num_classes=3,
        num_conv_layers=3,
        conv_out_channels=[32, 64, 128],
        kernel_size=3,
        pool_size=2,
        dropout_rate=0.2,
        activation="elu",
        gamma=2.0,
        verbose=0,
        debug=False,
    ):
        super(CNNModel, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.verbose = verbose
        self.debug = debug

        activation = self._resolve_activation(activation)

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        input_channels = num_classes
        for out_channels in conv_out_channels:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(input_channels, out_channels, kernel_size),
                    nn.BatchNorm1d(out_channels),
                    activation,
                    nn.MaxPool1d(kernel_size=pool_size),
                    nn.Dropout(dropout_rate),
                )
            )
            input_channels = out_channels

        # Dynamically compute the flattened size
        dummy_input = torch.zeros(1, num_classes, n_features)
        dummy_output = self._compute_conv_output_size(dummy_input)
        flattened_size = dummy_output.numel()

        self.fc = nn.Linear(flattened_size, n_features * num_classes)
        self.reshape = (n_features, num_classes)

    def _compute_conv_output_size(self, x):
        for conv in self.conv_layers:
            x = conv(x)
        return x

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(
                f"Input tensor must have shape (batch_size, n_features, num_classes). Got {x.shape}"
            )

        x = x.permute(0, 2, 1)
        x = checkpoint_sequential(self.conv_layers, 2, x, use_reentrant=False)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x.view(x.size(0), *self.reshape)

    def compute_loss(
        self,
        y: torch.Tensor,
        outputs: torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss for the CNN model.

        Method to compute the loss for the CNN model using the MaskedFocalLoss. The class weights are used for imbalanced data.

        Args:
            y (torch.Tensor): Ground truth labels of shape (batch_size, n_features, num_classes).
            outputs (torch.Tensor): Model predictions of shape (batch_size, n_features, num_classes).
            class_weights (torch.Tensor): Class weights for imbalanced data. If ``class_weights is None``, then all class weights will be set to 1.0. Default is None.

        Returns:
            torch.Tensor: Computed loss value.
        """
        if class_weights is None:
            class_weights = torch.ones(self.num_classes, device=y.device)

        criterion = MaskedFocalLoss(
            alpha=class_weights,
            gamma=self.gamma,
            reduction="mean",
            verbose=self.verbose,
            debug=self.debug,
        )
        return criterion(outputs, y)

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
            elif activation in {"leaky_relu", "leakyrelu"}:
                return nn.LeakyReLU()
            elif activation == "selu":
                return nn.SELU()
            else:
                msg = f"Activation {activation} not supported."
                self.logger.error(msg)
                raise ValueError(msg)

        return activation
