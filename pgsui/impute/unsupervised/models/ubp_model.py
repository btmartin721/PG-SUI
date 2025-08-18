import logging
from typing import List, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from snpio.utils.logging import LoggerManager

from pgsui.impute.unsupervised.loss_functions import MaskedFocalLoss


class ConvDecoder(nn.Module):
    """A 1D Convolutional Decoder module.

    This network takes a latent vector and progressively upsamples it using transposed convolutions to generate a sequence of the desired length. The output is a tensor of shape (batch_size, num_classes, n_features).

    The general steps are as follows:

        1. Project and reshape the latent vector.
        2. Pass the reshaped tensor through a series of transposed convolutional layers to progressively upsample the feature map.
        3. Apply a final 1D convolution to produce the output logits.
        4. Interpolate the output to match the desired feature length.
        5. Permute the output tensor to the correct shape.
    """

    def __init__(
        self,
        latent_dim: int,
        n_features: int,
        num_classes: int,
        channel_sizes: List[int],
        dropout_rate: float,
        activation_factory: callable,
        device: torch.device = torch.device("cpu"),
    ):
        super(ConvDecoder, self).__init__()
        self.n_features = n_features

        # The first channel size is used for the initial projection
        self.initial_channels = channel_sizes[0]

        ups = max(0, len(channel_sizes) - 1)

        self.initial_seq_len = self._choose_initial_seq_len(n_features, ups)

        self.initial_linear = nn.Linear(
            latent_dim, self.initial_channels * self.initial_seq_len, device=device
        )

        # Build the convolutional upsampling blocks
        blocks = []
        in_channels = self.initial_channels
        for out_channels in channel_sizes[1:]:
            # In the __init__ method's loop
            block = nn.Sequential(
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size=8,
                    stride=2,
                    padding=7,
                    device=device,
                    dilation=2,
                ),
                nn.BatchNorm1d(out_channels, device=device),
                activation_factory(),
                nn.Dropout(dropout_rate),
            )
            blocks.append(block)
            in_channels = out_channels

        self.decoder_blocks = nn.ModuleList(blocks)

        self.final_conv = nn.Conv1d(
            in_channels, num_classes, kernel_size=1, device=device
        )

    def _choose_initial_seq_len(self, n_features: int, n_upsamples: int) -> int:
        """Pick an initial length so that initial_len * 2**n_upsamples ~= n_features.

        This method calculates the initial sequence length for the decoder by ensuring that the length is appropriate given the number of upsampling layers.

        Args:
            n_features (int): Number of features (SNPs) in the input data.
            n_upsamples (int): Number of upsampling layers in the decoder.

        Returns:
            int: The initial sequence length.
        """
        target = max(4, int(np.round(n_features / (2**n_upsamples))))
        return target

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ConvDecoder.

        This method performs the forward pass through the convolutional decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes, n_features).
        """
        # 1. Project and reshape latent vector
        x = self.initial_linear(x)
        x = x.reshape(-1, self.initial_channels, self.initial_seq_len)

        # 2. Pass through convolutional upsampling blocks
        for block in self.decoder_blocks:
            x = block(x)

        # 3. Final convolution to get class logits
        x = self.final_conv(x)

        # 4. Interpolate to the exact number of features (n_features)
        # This makes the model flexible to any number of SNPs.
        x = F.interpolate(x, size=self.n_features, mode="linear", align_corners=False)

        # 5. Permute to get the final shape:
        # (batch_size, n_features, num_classes)
        x = x.permute(0, 2, 1)
        return x


class UBPModel(nn.Module):
    """Unsupervised Backpropagation (UBP) model with a Convolutional Decoder.

    This class implements the UBP model with a convolutional decoder for processing genomic data. The decoder is designed to handle the high dimensionality of SNP data and learn meaningful representations. The overall architecture consists of a simple linear layer for initial processing, followed by a series of convolutional layers that progressively refine the learned representations.

    The general steps are as follows:

        1. Input data is passed through the phase 1 decoder, which is a simple linear layer.
        2. The output from phase 1 is reshaped and passed through the phase 2 & 3 decoder, which is a convolutional decoder.
        3. The final output is produced by the phase 2 & 3 decoder.
    """

    def __init__(
        self,
        *,
        n_features: int,
        prefix: str = "pgsui",
        num_classes: int = 3,
        hidden_layer_sizes: list[int] = [256, 128, 64],
        latent_dim: int = 32,
        dropout_rate: float = 0.2,
        activation: str = "relu",
        gamma: float = 2.0,
        device: Literal["cpu", "gpu", "mps"] = "cpu",
        use_convolution: bool = False,
        logger: logging.Logger | None = None,
        verbose: int = 0,
        debug: bool = False,
    ):
        """Unsupervised Backpropagation (UBP) model with a Convolutional Decoder.

        This class implements the UBP model with a convolutional decoder for processing genomic data. The decoder is designed to handle the high dimensionality of SNP data and learn meaningful representations.

        The general steps of the UBP model are as follows:

            1. Input data is passed through the phase 1 decoder, which is a simple linear layer.
            2. The output from phase 1 is reshaped and passed through the phase 2 & 3 decoder, which is a convolutional decoder.
            3. The final output is produced by the phase 2 & 3 decoder.

        Args:
            n_features (int): Number of features (SNPs) in the input data.
            prefix (str): Prefix for the logger name.
            num_classes (int): Number of output classes.
            hidden_layer_sizes (list[int]): List of channel sizes for the ConvNet decoder.
            latent_dim (int): Dimensionality of the latent space.
            dropout_rate (float): Dropout rate for the decoder.
            activation (str): Activation function to use in the decoder.
            gamma (float): Focusing parameter for the focal loss.
            device (Literal["cpu", "gpu", "mps"]): Device to run the model on.
            use_convolution (bool): Whether to use the convolutional decoder in the model. If False, uses the original UBP model architecture.
            logger (logging.Logger | None): Logger for the model.
            verbose (int): Verbosity level.
            debug (bool): Whether to enable debug mode.
        """
        super(UBPModel, self).__init__()

        self.use_convolution = use_convolution

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

        # Phase 1 decoder remains a simple linear model
        self.phase1_decoder = nn.Sequential(
            nn.Linear(latent_dim, n_features * num_classes, device=device),
        )

        # Phase 2 & 3 uses the Convolutional Decoder
        act_factory = self._resolve_activation_factory(activation)

        if use_convolution:
            self.phase23_decoder = ConvDecoder(
                latent_dim=latent_dim,
                n_features=n_features,
                num_classes=num_classes,
                channel_sizes=hidden_layer_sizes,
                dropout_rate=dropout_rate,
                activation_factory=act_factory,
                device=device,
            )
        else:
            if hidden_layer_sizes[0] > hidden_layer_sizes[-1]:
                hidden_layer_sizes = list(reversed(hidden_layer_sizes))

            # Phase 2 & 3: Flexible deeper network
            layers = []
            input_dim = latent_dim
            for size in hidden_layer_sizes:
                layers.append(nn.Linear(input_dim, size))
                layers.append(nn.BatchNorm1d(size))
                layers.append(nn.Dropout(dropout_rate))
                layers.append(act_factory())
                input_dim = size

            layers.append(nn.Linear(hidden_layer_sizes[-1], n_features * num_classes))

            self.phase23_decoder = nn.Sequential(*layers)

    def _resolve_activation_factory(self, activation: str) -> callable:
        """Resolve the activation function factory based on the provided string.

        This method returns a callable that produces the desired activation function.

        Args:
            activation (str): Name of the activation function.

        Returns:
            callable: A callable that returns the activation function.

        Raises:
            ValueError: If the activation function is not supported.
        """
        a = activation.lower()
        if a == "relu":
            return lambda: nn.ReLU()
        if a == "elu":
            return lambda: nn.ELU()
        if a == "leaky_relu":
            return lambda: nn.LeakyReLU()
        if a == "selu":
            return lambda: nn.SELU()
        raise ValueError(f"Activation function {activation} not supported.")

    def forward(self, x: torch.Tensor, phase: int = 1) -> torch.Tensor:
        """Forward pass through the UBP model.

        This method performs the forward pass through the UBP model. The output shape depends on the phase. For phase 1, the output is a flattened tensor of shape (batch_size, n_features * num_classes). For phases 2 and 3, the output is a 4D tensor suitable for convolutional layers.

        Args:
            x (torch.Tensor): Input tensor.
            phase (int): Phase of the model (1, 2, or 3).

        Notes:
            The training loop calls the convolutional decoders directly.
        """
        if phase == 1:
            # Linear decoder for phase 1
            x = self.phase1_decoder(x)
            return x.reshape(-1, self.n_features, self.num_classes)
        elif phase in {2, 3}:
            if self.use_convolution:
                # Convolutional decoder for phases 2 and 3
                return self.phase23_decoder(x)
            else:
                # Fallback to linear decoder for phases 2 and 3
                x = self.phase23_decoder(x)
                return x.reshape(-1, self.n_features, self.num_classes)
        else:
            msg = f"Invalid phase: {phase}. Expected 1, 2, or 3."
            self.logger.error(msg)
            raise ValueError(msg)

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
