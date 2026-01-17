# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from snpio.utils.logging import LoggerManager

from pgsui.utils.logging_utils import configure_logger


class NLPCAModel(nn.Module):
    """Non-linear PCA (NLPCA) model implemented as UBP Phase-3-only.

    This model learns:
        - V: per-sample latent embeddings (nn.Embedding)
        - W: decoder network weights (MLP) jointly via backpropagation (i.e., the "non-linear refinement" phase of UBP).

    Forward maps embeddings -> logits over genotype classes for each locus.
    """

    def __init__(
        self,
        num_embeddings: int,
        n_features: int,
        prefix: str,
        *,
        embedding_init: torch.Tensor,
        num_classes: int = 3,
        hidden_layer_sizes: List[int] | np.ndarray = [64, 128],
        latent_dim: int = 2,
        dropout_rate: float = 0.2,
        activation: Literal["relu", "elu", "selu", "leaky_relu"] = "relu",
        device: torch.device | str = "cpu",
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize NLPCAModel.

        Args:
            num_embeddings (int): Total number of samples (rows).
            n_features (int): Number of loci/features (columns).
            prefix (str): Logging prefix.
            embedding_init (torch.Tensor): Tensor of shape (num_embeddings, latent_dim) used to initialize V (PCA warm-start).
            num_classes (int): Number of genotype classes (3 diploid, 2 haploid).
            hidden_layer_sizes (List[int] | np.ndarray): Hidden layer widths for the decoder MLP.
            latent_dim (int): Latent embedding dimension.
            dropout_rate (float): Dropout probability within the decoder.
            activation (Literal["relu", "elu", "selu", "leaky_relu"]): Activation function.
            device (torch.device | str): Torch device or device string.
            verbose (bool): Verbose logging.
            debug (bool): Debug logging.
        """
        super().__init__()
        self.num_classes = int(num_classes)
        self.n_features = int(n_features)
        self.latent_dim = int(latent_dim)
        self.device = device

        logman = LoggerManager(
            name=__name__, prefix=prefix, verbose=verbose, debug=debug
        )
        self.logger = configure_logger(
            logman.get_logger(), verbose=verbose, debug=debug
        )

        activation_module = self._resolve_activation(str(activation))

        hls = (
            hidden_layer_sizes.tolist()
            if isinstance(hidden_layer_sizes, np.ndarray)
            else list(hidden_layer_sizes)
        )

        # V: (n_samples, latent_dim)
        self.embedding = nn.Embedding(int(num_embeddings), self.latent_dim)

        if tuple(embedding_init.shape) != (int(num_embeddings), self.latent_dim):
            raise ValueError(
                f"Embedding init shape {tuple(embedding_init.shape)} mismatch. "
                f"Expected ({num_embeddings}, {self.latent_dim})."
            )

        embedding_init = embedding_init.to(
            dtype=self.embedding.weight.dtype, device=self.embedding.weight.device
        )
        with torch.no_grad():
            self.embedding.weight.copy_(embedding_init)

        # W: decoder MLP: latent_dim -> (n_features * num_classes)
        layers: list[nn.Module] = []
        input_dim = self.latent_dim
        for hidden_size in hls:
            layers.append(nn.Linear(input_dim, int(hidden_size)))
            layers.append(nn.LayerNorm(int(hidden_size)))
            layers.append(activation_module)
            layers.append(nn.Dropout(float(dropout_rate)))
            input_dim = int(hidden_size)

        self.hidden_layers = nn.Sequential(*layers)

        output_dim = self.n_features * self.num_classes
        self.dense_output = nn.Linear(input_dim, int(output_dim))
        self.reshape_dim = (self.n_features, self.num_classes)

    def forward(
        self,
        indices: Optional[torch.Tensor] = None,
        override_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass mapping latent embeddings -> genotype logits.

        Args:
            indices (Optional[torch.Tensor]): Tensor of sample indices, shape (B,).
            override_embeddings (Optional[torch.Tensor]): Direct embeddings, shape (B, latent_dim).

        Returns:
            Logits tensor of shape (B, n_features, num_classes).
        """
        if override_embeddings is not None:
            z = override_embeddings
            if z.dim() != 2 or z.shape[1] != self.latent_dim:
                raise ValueError(
                    f"override_embeddings must be (B, latent_dim={self.latent_dim}); got {tuple(z.shape)}."
                )
        elif indices is not None:
            if not torch.is_tensor(indices):
                raise TypeError(
                    f"indices must be a torch.Tensor, got {type(indices).__name__}."
                )
            if indices.dtype not in (torch.int32, torch.int64):
                indices = indices.long()
            if indices.dim() != 1:
                indices = indices.view(-1)
            z = self.embedding(indices)
        else:
            raise ValueError("Must provide either indices or override_embeddings.")

        x = self.hidden_layers(z)
        x = self.dense_output(x)
        return x.view(-1, *self.reshape_dim)

    @staticmethod
    def _resolve_activation(activation: str) -> nn.Module:
        """Resolve activation string to nn.Module.

        Args:
            activation: Activation function name.

        Returns:
            nn.Module: Corresponding activation module.
        """
        act = activation.lower()
        if act == "relu":
            return nn.ReLU()
        if act == "elu":
            return nn.ELU()
        if act in ("leaky_relu", "leakyrelu"):
            return nn.LeakyReLU()
        if act == "selu":
            return nn.SELU()
        raise ValueError(f"Activation {activation} not supported.")
