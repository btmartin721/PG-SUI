from typing import List, Literal

import numpy as np
import torch
import torch.nn as nn
from snpio.utils.logging import LoggerManager

from pgsui.impute.unsupervised.loss_functions import MaskedFocalLoss


class UBPModel(nn.Module):
    """An Unsupervised Backpropagation (UBP) model with a multi-phase decoder.

    This class implements a deep neural network that serves as the decoder component in an unsupervised imputation pipeline. It's designed to reconstruct high-dimensional genomic data from a low-dimensional latent representation. The model features a unique multi-phase architecture with two distinct decoding paths:

    1.  **Phase 1 Decoder:** A simple, shallow linear network.
    2.  **Phase 2 & 3 Decoder:** A deeper, multi-layered, fully-connected network with batch normalization and dropout for regularization.

    This phased approach allows for progressive training strategies. The model is tailored for two-channel allele data, where it learns to predict allele probabilities for each of the two channels at every SNP locus.

    **Model Architecture:**

    The model's forward pass maps a latent vector, $z$, to a reconstructed output, $\hat{x}$, via one of two paths.

    -   **Phase 1 Path (Shallow Decoder):**
        $$
        \hat{x}_{p1} = W_{p1} z + b_{p1}
        $$

    -   **Phase 2/3 Path (Deep Decoder):**
        This path uses a series of hidden layers with non-linear activations, $f(\cdot)$:
        $$
        h_1 = f(W_1 z + b_1)
        $$
        $$
        \dots
        $$
        $$
        h_L = f(W_L h_{L-1} + b_L)
        $$
        $$
        \hat{x}_{p23} = W_{L+1} h_L + b_{L+1}
        $$

    The final output from either path is reshaped into a tensor of shape `(batch_size, n_features, n_channels, n_classes)`. The model is optimized using a `MaskedFocalLoss` function, which effectively handles the missing data and class imbalance common in genomic datasets.
    """

    def __init__(
        self,
        n_features: int,
        prefix: str,
        *,
        num_classes: int = 3,
        hidden_layer_sizes: List[int] | np.ndarray = [128, 64],
        latent_dim: int = 2,
        dropout_rate: float = 0.2,
        activation: Literal["relu", "elu", "selu", "leaky_relu"] = "relu",
        gamma: float = 2.0,
        device: Literal["cpu", "gpu", "mps"] = "cpu",
        verbose: bool = False,
        debug: bool = False,
    ):
        """Initializes the UBPModel.

        Args:
            n_features (int): The number of features (SNPs) in the input data.
            prefix (str): A prefix used for logging.
            num_classes (int): The number of possible allele classes (e.g., 3 for 0, 1, 2). Defaults to 3.
            hidden_layer_sizes (list[int] | np.ndarray): A list of integers specifying the size of each hidden layer in the deep (Phase 2/3) decoder. Defaults to [128, 64].
            latent_dim (int): The dimensionality of the input latent space. Defaults to 2.
            dropout_rate (float): The dropout rate for regularization in the deep decoder. Defaults to 0.2.
            activation (str): The non-linear activation function to use in the deep decoder's hidden layers. Defaults to 'relu'.
            gamma (float): The focusing parameter for the focal loss function. Defaults to 2.0.
            device (Literal["cpu", "gpu", "mps"]): The PyTorch device to run the model on. Defaults to 'cpu'.
            verbose (bool): If True, enables detailed logging. Defaults to False.
            debug (bool): If True, enables debug mode. Defaults to False.
        """
        super(UBPModel, self).__init__()

        logman = LoggerManager(
            name=__name__, prefix=prefix, verbose=verbose, debug=debug
        )
        self.logger = logman.get_logger()

        self.n_features = n_features
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.device = device

        if isinstance(hidden_layer_sizes, np.ndarray):
            hidden_layer_sizes = hidden_layer_sizes.tolist()

        # Final layer output size is now n_features * num_classes
        final_output_size = n_features * num_classes

        # Phase 1 decoder: Simple linear model
        self.phase1_decoder = nn.Sequential(
            nn.Linear(latent_dim, final_output_size, device=device),
        )

        # Phase 2 & 3 uses the Convolutional Decoder
        act_factory = self._resolve_activation_factory(activation)

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

        layers.append(nn.Linear(hidden_layer_sizes[-1], final_output_size))

        self.phase23_decoder = nn.Sequential(*layers)
        self.reshape = (self.n_features, self.num_classes)

    def _resolve_activation_factory(
        self, activation: Literal["relu", "elu", "selu", "leaky_relu"]
    ) -> callable:
        """Resolves an activation function factory from a string name.

        This method acts as a factory, returning a callable (lambda function) that produces the desired PyTorch activation function module when called.

        Args:
            activation (Literal["relu", "elu", "selu", "leaky_relu"]): The name of the activation function.

        Returns:
            callable: A factory function that, when called, returns an instance of the specified activation layer.

        Raises:
            ValueError: If the provided activation name is not supported.
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

        msg = f"Activation function {activation} not supported."
        self.logger.error(msg)
        raise ValueError(msg)

    def forward(self, x: torch.Tensor, phase: int = 1) -> torch.Tensor:
        """Performs the forward pass through the UBP model.

        This method routes the input tensor through the appropriate decoder based on the specified training `phase`. The final output is reshaped to match the target data structure of `(batch_size, n_features, n_channels, n_classes)`.

        Args:
            x (torch.Tensor): The input latent tensor of shape `(batch_size, latent_dim)`.
            phase (int): The training phase (1, 2, or 3), which determines which decoder path to use.

        Returns:
            torch.Tensor: The reconstructed output tensor.

        Raises:
            ValueError: If an invalid phase is provided.
        """
        if phase == 1:
            # Linear decoder for phase 1
            x = self.phase1_decoder(x)
            return x.view(-1, *self.reshape)
        elif phase in {2, 3}:
            x = self.phase23_decoder(x)
            return x.view(-1, *self.reshape)
        else:
            msg = f"Invalid phase: {phase}. Expected 1, 2, or 3."
            self.logger.error(msg)
            raise ValueError(msg)

    def compute_loss(
        self,
        y: torch.Tensor,
        outputs: torch.Tensor,
        mask: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        """Computes the masked focal loss between model outputs and ground truth.

        This method calculates the loss value, handling class imbalance with weights and ignoring masked (missing) values in the ground truth tensor.

        Args:
            y (torch.Tensor): The ground truth tensor of shape `(batch_size, n_features, n_channels)`.
            outputs (torch.Tensor): The model's raw output (logits) of shape `(batch_size, n_features, n_channels, n_classes)`.
            mask (torch.Tensor | None): An optional boolean mask indicating which elements should be included in the loss calculation.
            class_weights (torch.Tensor | None): An optional tensor of weights for each class to address imbalance.
            gamma (float): The focusing parameter for the focal loss.

        Returns:
            torch.Tensor: The computed scalar loss value.
        """
        if class_weights is None:
            class_weights = torch.ones(self.num_classes, device=outputs.device)

        if mask is None:
            mask = torch.ones_like(y, dtype=torch.bool)

        # Explicitly flatten all tensors to the (N, C) and (N,) format.
        # This creates a clear contract with the new MaskedFocalLoss function.
        n_classes = outputs.shape[-1]
        logits_flat = outputs.reshape(-1, n_classes)
        targets_flat = y.reshape(-1)
        mask_flat = mask.reshape(-1)

        criterion = MaskedFocalLoss(gamma=gamma, alpha=class_weights)

        return criterion(
            logits_flat.to(self.device),
            targets_flat.to(self.device),
            valid_mask=mask_flat.to(self.device),
        )
