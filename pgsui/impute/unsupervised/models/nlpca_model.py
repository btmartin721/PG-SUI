from typing import List, Literal

import numpy as np
import torch
import torch.nn as nn
from snpio.utils.logging import LoggerManager

from pgsui.utils.logging_utils import configure_logger


class NLPCAModel(nn.Module):
    r"""A non-linear Principal Component Analysis (NLPCA) decoder for genotypes.

    This module maps a low-dimensional latent vector to logits over genotype states (two classes for haploids or three for diploids) at every locus. It is a fully connected network with optional batch normalization and dropout layers and is used as the decoder inside the NLPCA imputer.

    **Model Architecture**

    Let :math:`z \in \mathbb{R}^{d_{\text{latent}}}` be the latent vector. For a
    network with :math:`L` hidden layers, the transformations are

    .. math::

        h_1 = f(W_1 z + b_1)

    .. math::

        h_2 = f(W_2 h_1 + b_2)

    .. math::

        \vdots

    .. math::

        h_L = f(W_L h_{L-1} + b_L)

    The final layer produces logits of shape ``(batch_size, n_features, num_classes)``
    by reshaping a linear projection back to the (loci, genotype-state) grid.

    **Loss Function**

    Training minimizes ``SafeFocalCELoss``, which extends cross-entropy with class weighting, focal re-weighting, and masking so that only observed genotypes contribute to the objective.
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
            num_classes (int): Number of genotype states per locus (2 for haploid, 3 for diploid in practice). Defaults to 4 for backward compatibility.
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
        self.logger = configure_logger(
            logman.get_logger(), verbose=verbose, debug=debug
        )

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
        act: str = activation.lower()

        if act == "relu":
            return nn.ReLU()
        elif act == "elu":
            return nn.ELU()
        elif act == "leaky_relu":
            return nn.LeakyReLU()
        elif act == "selu":
            return nn.SELU()
        else:
            msg = f"Activation function {act} not supported."
            self.logger.error(msg)
            raise ValueError(msg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of the model.

        The input tensor is passed through the decoder network to produce logits,
        which are reshaped to align with the locus-by-class grid used by the loss.

        Args:
            x (torch.Tensor): The input tensor, which should represent the latent space vector.

        Returns:
            torch.Tensor: The reconstructed output tensor of shape `(batch_size, n_features, num_classes)`.
        """
        x = self.phase23_decoder(x)

        # Reshape to (batch, features, num_classes)
        return x.view(-1, *self.reshape)
