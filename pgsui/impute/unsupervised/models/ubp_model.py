from typing import Callable, List, Literal

import numpy as np
import torch
import torch.nn as nn
from snpio.utils.logging import LoggerManager

from pgsui.utils.logging_utils import configure_logger


class UBPModel(nn.Module):
    """An Unsupervised Backpropagation (UBP) decoder for genotype logits.

    The model reconstructs locus-level genotype probabilities (two states for haploid data or three for diploid data) from a latent vector. It exposes two decoding branches so the training schedule can follow the UBP recipe:

    1. **Phase 1 decoder** - a shallow linear layer that co-trains with latent codes.
    2. **Phase 2/3 decoder** - a deeper MLP with batch normalization and dropout that is first trained in isolation and later fine-tuned jointly with the latents.

    Both paths ultimately reshape their logits to ``(batch_size, n_features, num_classes)`` and training uses ``SafeFocalCELoss`` to focus on hard examples while masking missing entries.
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
            num_classes (int): Number of genotype states per locus (typically 2 or 3). Defaults to 3.
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
    ) -> Callable[[], nn.Module]:
        """Resolves an activation function factory from a string name.

        This method acts as a factory, returning a callable (lambda function) that produces the desired PyTorch activation function module when called.

        Args:
            activation (Literal["relu", "elu", "selu", "leaky_relu"]): The name of the activation function.

        Returns:
            Callable[[], nn.Module]: A factory function that, when called, returns an instance of the specified activation layer.

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

        This method routes the input tensor through the appropriate decoder based on
        the specified training ``phase`` and reshapes the logits to the
        `(batch_size, n_features, num_classes)` grid expected by the loss.

        Args:
            x (torch.Tensor): The input latent tensor of shape `(batch_size, latent_dim)`.
            phase (int): The training phase (1, 2, or 3), which determines which decoder path to use.

        Returns:
            torch.Tensor: Logits shaped as `(batch_size, n_features, num_classes)`.

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
