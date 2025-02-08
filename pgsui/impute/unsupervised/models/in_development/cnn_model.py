from typing import List

import torch
import torch.nn as nn

from pgsui.impute.unsupervised.loss_functions import MaskedFocalLoss


class CNNModel(nn.Module):
    """Convolutional Neural Network (CNN) model for SNP imputation.

        This model is designed for input tensors of shape (n_samples, num_features, num_classes) where the SNP loci are unordered. To avoid memory intensity, the model is fully convolutional: it uses only pointwise (kernel_size=1) convolutions (optionally, you can experiment with a small kernel size and appropriate padding) and does not flatten feature maps into a large fully connected layer. The final convolution maps features back to the number of classes.

    Attributes:
        conv_layers (nn.ModuleList): A list of convolutional blocks.
        final_conv (nn.Conv1d): Final pointwise convolution mapping to num_classes.
        logger (Logger): Logger instance.
    """

    def __init__(
        self,
        n_features: int,
        num_classes: int = 3,
        num_conv_layers: int = 3,
        conv_out_channels: List[int] = [32, 64, 128],
        kernel_size: int = 1,
        dropout_rate: float = 0.2,
        activation: str | nn.Module = "elu",
        gamma: float = 2.0,
        verbose: int = 0,
        debug: bool = False,
    ):
        """
        Args:
            n_features (int): Number of SNP features.
            num_classes (int, optional): Number of classes per SNP. Defaults to 3.
            num_conv_layers (int, optional): Number of convolutional layers. Defaults to 3.
            conv_out_channels (List[int], optional): Output channels for each conv layer.
                Must have at least num_conv_layers elements.
            kernel_size (int, optional): Convolution kernel size. Defaults to 1.
                For unordered SNP data a kernel size of 1 is recommended.
            dropout_rate (float, optional): Dropout rate for each conv block. Defaults to 0.2.
            activation (Union[str, nn.Module], optional): Activation function to use. Defaults to "elu".
            gamma (float, optional): Gamma parameter for focal loss. Defaults to 2.0.
            verbose (int, optional): Verbosity level. Defaults to 0.
            debug (bool, optional): Debug flag. Defaults to False.
        """
        super(CNNModel, self).__init__()

        self.n_features = n_features
        self.num_classes = num_classes
        self.gamma = gamma
        self.verbose = verbose
        self.debug = debug

        # Resolve the activation function.
        self.activation_fn = self._resolve_activation(activation)

        if len(conv_out_channels) < num_conv_layers:
            raise ValueError(
                "Length of conv_out_channels must be at least num_conv_layers"
            )

        # Build the convolutional layers.
        self.conv_layers = nn.ModuleList()
        in_channels = num_classes
        for i in range(num_conv_layers):
            out_channels = conv_out_channels[i]
            # Using kernel_size with appropriate padding to preserve length.
            # For unordered SNP data, kernel_size=1 is typically preferred.
            conv_block = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size // 2),
                ),
                nn.BatchNorm1d(out_channels),
                self.activation_fn,
                nn.Dropout(dropout_rate),
            )
            self.conv_layers.append(conv_block)
            in_channels = out_channels

        # Final convolution layer to map to the required number of classes.
        self.final_conv = nn.Conv1d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, num_classes).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_features, num_classes).
        """
        if x.dim() != 3:
            raise ValueError(
                f"Input tensor must have shape (batch_size, num_features, num_classes). Got {x.shape}"
            )
        # Permute input to (batch, channels, length) for Conv1d.
        x = x.permute(0, 2, 1)
        for conv in self.conv_layers:
            x = conv(x)
        x = self.final_conv(x)
        # Permute back to (batch, num_features, num_classes)
        x = x.permute(0, 2, 1)
        return x

    def compute_loss(
        self,
        y: torch.Tensor,
        outputs: torch.Tensor,
        mask: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the masked focal loss between predictions and targets.

        This method computes the masked focal loss between the model predictions and the ground truth labels. The mask tensor is used to ignore certain values (< 0), and class weights can be provided to balance the loss.

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

    def _resolve_activation(self, activation: str | nn.Module) -> nn.Module:
        """Resolve the activation function.

        Args:
            activation (str | nn.Module): Activation function identifier or module.

        Returns:
            nn.Module: The activation function module.

        Raises:
            ValueError: If the activation function is not supported.
        """
        if isinstance(activation, str):
            act = activation.lower()
            if act == "relu":
                return nn.ReLU()
            elif act == "elu":
                return nn.ELU()
            elif act in {"leaky_relu", "leakyrelu"}:
                return nn.LeakyReLU()
            elif act == "selu":
                return nn.SELU()
            else:
                msg = f"Activation {activation} not supported."
                raise ValueError(msg)
        return activation
