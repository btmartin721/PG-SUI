from typing import List, Literal

import numpy as np
import torch
import torch.nn as nn
from snpio.utils.logging import LoggerManager

from pgsui.impute.unsupervised.loss_functions import MaskedFocalLoss


class NLPCAModel(nn.Module):
    """A non-linear Principal Component Analysis (NLPCA) model.

    This model serves as a decoder for an autoencoder-based imputation strategy. It's a deep neural network that takes a low-dimensional latent vector as input and reconstructs the high-dimensional allele data. The architecture is a multi-layered, fully-connected network with optional batch normalization and dropout layers. The model is specifically designed for two-channel allele data, predicting allele probabilities for each of the two channels at every SNP.

    **Model Architecture:**
    The model's forward pass, from a latent representation, $z$, to the reconstructed input, $\hat{x}$, can be described as follows:

    Let $z \in \mathbb{R}^{d_{latent}}$ be the latent vector.
    The decoder consists of a series of fully-connected layers with activation functions:
    $$
    h_1 = f(W_1 z + b_1)
    $$
    $$
    h_2 = f(W_2 h_1 + b_2)
    $$
    $$
    \vdots
    $$
    $$
    h_L = f(W_L h_{L-1} + b_L)
    $$
    The final output layer produces a tensor of shape `(batch_size, n_features, n_channels, n_classes)`:
    $$
    \\hat{x} = W_{L+1} h_L + b_{L+1}
    $$
    where $f(\cdot)$ is the activation function (e.g., ReLU, ELU), and $W_i$ and $b_i$ are the weights and biases of each layer.

    **Loss Function:**
    The model is trained by minimizing the `MaskedFocalLoss`, which is an extension of the cross-entropy loss that focuses on hard-to-classify examples and handles missing values. The loss is computed on the reconstructed output and the ground truth, using a mask to only consider observed data.
    """

    def __init__(
        self,
        n_features: int,
        prefix: str,
        *,
        num_classes: int = 4,
        hidden_layer_sizes: List[int] | np.ndarray = [128, 64],
        latent_dim: int = 2,
        dropout_rate: float = 0.2,
        activation: Literal["relu", "elu", "selu", "leaky_relu"] = "relu",
        gamma: float = 2.0,
        device: Literal["gpu", "cpu", "mps"] = "cpu",
        verbose: bool = False,
        debug: bool = False,
    ):
        """Initializes the NLPCAModel.

        Args:
            n_features (int): The number of features (SNPs) in the input data.
            prefix (str): A prefix used for logging.
            num_classes (int): The number of possible allele classes (e.g., 4 for A, T, C, G). Defaults to 4.
            hidden_layer_sizes (list[int] | np.ndarray): A list of integers specifying the number of units in each hidden layer. Defaults to [128, 64].
            latent_dim (int): The dimensionality of the latent space (the size of the bottleneck layer). Defaults to 2.
            dropout_rate (float): The dropout rate applied to each hidden layer for regularization. Defaults to 0.2.
            activation (Literal["relu", "elu", "selu", "leaky_relu"]): The non-linear activation function to use in hidden layers. Defaults to 'relu'.
            gamma (float): The focusing parameter for the focal loss function, which down-weights well-classified examples. Defaults to 2.0.
            device (Literal["gpu", "cpu", "mps"]): The PyTorch device to run the model on. Defaults to 'cpu'.
            verbose (bool): If True, enables detailed logging. Defaults to False.
            debug (bool): If True, enables debug mode. Defaults to False.
        """
        super(NLPCAModel, self).__init__()

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

        layers = []
        input_dim = latent_dim
        for size in hidden_layer_sizes:
            layers.append(nn.Linear(input_dim, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Dropout(dropout_rate))
            layers.append(self._resolve_activation(activation))
            input_dim = size

        # Final layer output size is now n_features * num_classes
        final_output_size = self.n_features * self.num_classes
        layers.append(nn.Linear(hidden_layer_sizes[-1], final_output_size))

        self.phase23_decoder = nn.Sequential(*layers)

        # Reshape tuple reflects the output structure
        self.reshape = (self.n_features, self.num_classes)

    def _resolve_activation(
        self, activation: Literal["relu", "elu", "selu", "leaky_relu"]
    ) -> nn.Module:
        """Resolves an activation function from a string name.

        This method acts as a factory, returning the correct PyTorch activation function module based on the provided name.

        Args:
            activation (Literal["relu", "elu", "selu", "leaky_relu"]): The name of the activation function.

        Returns:
            nn.Module: The corresponding PyTorch activation function module.

        Raises:
            ValueError: If the provided activation name is not supported.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of the model.

        The input tensor is passed through the decoder network to produce the reconstructed output. The output is then reshaped to a 4D tensor representing batches, features, channels, and classes.

        Args:
            x (torch.Tensor): The input tensor, which should represent the latent space vector.

        Returns:
            torch.Tensor: The reconstructed output tensor of shape `(batch_size, n_features, n_channels, n_classes)`.
        """
        x = self.phase23_decoder(x)

        # Reshape to (batch, features, channels, classes)
        return x.view(-1, *self.reshape)

    def compute_loss(
        self,
        y: torch.Tensor,
        outputs: torch.Tensor,
        mask: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        """Computes the masked focal loss between model outputs and ground truth.

        This method calculates the loss value, handling class imbalance with weights and ignoring masked (missing) values.

        Args:
            y (torch.Tensor): The ground truth tensor of shape `(batch_size, n_features, n_channels)`.
            outputs (torch.Tensor): The model's raw output (logits) of shape `(batch_size, n_features, n_channels, n_classes)`.
            mask (torch.Tensor | None): An optional boolean mask indicating which elements should be included in the loss calculation. Defaults to None.
            class_weights (torch.Tensor | None): An optional tensor of weights for each class to address imbalance. Defaults to None.
            gamma (float): The focusing parameter for the focal loss. Defaults to 2.0.

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
