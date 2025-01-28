from typing import Union

import torch
import torch.nn as nn
from snpio.utils.logging import LoggerManager

from pgsui.impute.unsupervised.loss_functions import MaskedFocalLoss


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        num_classes: int = 3,
        lstm_hidden_size: int = 128,
        num_lstm_layers: int = 2,
        latent_dim: int = 64,  # Latent space dimensionality
        dropout_rate: float = 0.2,
        bidirectional: bool = False,
    ):
        super(LSTMAutoencoder, self).__init__()
        self.num_classes = num_classes

        # Encoder: LSTM compresses input to latent representation
        self.encoder = nn.LSTM(
            input_size=num_classes,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Bottleneck (latent space)
        lstm_output_size = lstm_hidden_size * (2 if bidirectional else 1)
        self.latent_fc = nn.Linear(lstm_output_size, latent_dim)

        # Decoder: LSTM reconstructs input from latent representation
        self.decoder_fc = nn.Linear(latent_dim, lstm_output_size)
        self.decoder = nn.LSTM(
            input_size=lstm_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Fully connected layer to reconstruct original input
        self.output_fc = nn.Linear(
            lstm_hidden_size * (2 if bidirectional else 1), num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder: Pass input through LSTM
        x, _ = self.encoder(x)
        x = self.latent_fc(x)

        # Decoder: Reconstruct from latent representation
        x = self.decoder_fc(x)
        x, _ = self.decoder(x)

        # Output: Map back to original shape
        x = self.output_fc(x)
        return x


class LSTMModel(nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        prefix: str = "pgsui",
        num_classes: int = 3,
        lstm_hidden_size: int = 128,
        num_lstm_layers: int = 2,
        dropout_rate: float = 0.2,
        bidirectional: bool = False,
        activation: str = "elu",
        gamma: float = 2.0,
        verbose: int = 0,
        debug: bool = False,
    ):
        """
        Long Short-Term Memory (LSTM) model for imputation.

        Args:
            n_features (int): Number of features (SNPs) in the input data.
            num_classes (int): Number of output classes. Default is 3.
            lstm_hidden_size (int): Hidden size of the LSTM layer. Default is 128.
            num_lstm_layers (int): Number of stacked LSTM layers. Default is 2.
            dropout_rate (float): Dropout rate for regularization. Default is 0.2.
            bidirectional (bool): Whether to use bidirectional LSTM. Default is False.
            activation (str): Activation function for the final dense layer. Default is 'elu'.
            gamma (float): Focal loss gamma parameter. Default is 2.0.
            verbose (int): Verbosity level for logging messages. Default is 0.
            debug (bool): Debug mode for logging messages. Default is False.
        """
        super(LSTMModel, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.verbose = verbose
        self.debug = debug
        logman = LoggerManager(
            __name__, prefix=prefix, debug=debug, verbose=verbose >= 1
        )
        self.logger = logman.get_logger()

        self.activation = self._resolve_activation(activation)

        self.lstm = nn.LSTM(
            input_size=num_classes,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Adjust the output size of the dense layer based on bidirectionality
        lstm_output_size = lstm_hidden_size * (2 if bidirectional else 1)

        self.fc = nn.Linear(lstm_output_size, num_classes)
        self.reshape = (n_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_features, num_classes).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_features, num_classes).
        """
        if x.dim() != 3:
            raise ValueError(
                f"Input tensor must have shape (batch_size, n_features, num_classes). Got {x.shape}."
            )

        # Pass through LSTM (batch_first=True ensures output is [batch_size, n_features, hidden_size])
        x, _ = self.lstm(x)

        # Apply the fully connected layer to each time step
        x = self.fc(x)

        return x.view(x.size(0), *self.reshape)

    def compute_loss(
        self,
        y: torch.Tensor,
        outputs: torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the loss for the LSTM model.

        Args:
            y (torch.Tensor): Ground truth labels of shape (batch_size, n_features, num_classes).
            outputs (torch.Tensor): Model predictions of shape (batch_size, n_features, num_classes).
            class_weights (torch.Tensor): Class weights for imbalanced data. If ``class_weights is None``,
                all class weights will be set to 1.0.

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
        """
        Resolve the activation function.

        Args:
            activation (Union[str, torch.nn.Module]): Activation function.

        Returns:
            torch.nn.Module: Resolved activation function.
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
                raise ValueError(f"Activation {activation} not supported.")

        return activation
