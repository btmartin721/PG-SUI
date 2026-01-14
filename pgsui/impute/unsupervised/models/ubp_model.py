# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from snpio.utils.logging import LoggerManager

from pgsui.utils.logging_utils import configure_logger


class UBPModel(nn.Module):
    """Unsupervised Backpropagation (UBP) Model.

    Unlike a standard Autoencoder, UBP does not have an Encoder network. Instead, it learns a latent vector (embedding) for every sample index directly via backpropagation. The 'forward' pass acts as a Decoder, mapping indices (via embeddings) to reconstructed outputs.
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
        device: Literal["cpu", "gpu", "mps"] = "cpu",
        verbose: bool = False,
        debug: bool = False,
    ):
        """Initialize UBP Model.

        Args:
            num_embeddings: Total number of samples (rows) in the dataset (n).
            n_features: Number of features/SNPs (d).
            num_classes: Genotype states (3 for diploid, 2 for haploid).
            hidden_layer_sizes: Sizes of hidden layers for the MLP (W).
            latent_dim: Size of the intrinsic/latent vector (t).
        """
        super(UBPModel, self).__init__()
        self.num_classes = num_classes
        self.n_features = n_features
        self.device = device
        self.latent_dim = latent_dim

        logman = LoggerManager(
            name=__name__, prefix=prefix, verbose=verbose, debug=debug
        )
        self.logger = configure_logger(
            logman.get_logger(), verbose=verbose, debug=debug
        )

        activation_module = self._resolve_activation(activation)

        if isinstance(hidden_layer_sizes, np.ndarray):
            hls = hidden_layer_sizes.tolist()
        else:
            hls = hidden_layer_sizes

        # Matrix V: n x t
        self.embedding = nn.Embedding(num_embeddings, latent_dim)

        if embedding_init.shape != (num_embeddings, latent_dim):
            raise ValueError(
                f"Embedding init shape {tuple(embedding_init.shape)} mismatch. "
                f"Expected ({num_embeddings}, {latent_dim})."
            )

        embedding_init = embedding_init.to(
            dtype=self.embedding.weight.dtype, device=self.embedding.weight.device
        )

        with torch.no_grad():
            self.embedding.weight.copy_(embedding_init)

        # Matrix W (The MLP): t -> d
        layers = []
        input_dim = latent_dim
        for hidden_size in hls:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(activation_module)
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_size

        self.hidden_layers = nn.Sequential(*layers)

        output_dim = n_features * num_classes
        self.dense_output = nn.Linear(input_dim, output_dim)
        self.reshape_dim = (n_features, num_classes)

    def forward(
        self,
        indices: Optional[torch.Tensor] = None,
        override_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass mapping latent V -> output logits.

        Args:
            indices: Sample indices (shape: [B]) used to lookup embeddings. Must be integer dtype.
            override_embeddings: Direct latent embeddings (shape: [B, latent_dim]) used instead of lookup.

        Returns:
            torch.Tensor: Logits of shape (B, n_features, num_classes).

        Raises:
            ValueError: If neither indices nor override_embeddings are provided, or shapes mismatch.
            TypeError: If indices is provided with non-integer dtype.
        """
        if override_embeddings is not None:
            z = override_embeddings
            if z.dim() != 2 or z.shape[1] != self.latent_dim:
                msg = f"override_embeddings must be (B, latent_dim={self.latent_dim}); got {tuple(z.shape)}."
                raise ValueError(msg)

        elif indices is not None:
            if not torch.is_tensor(indices):
                msg = f"indices must be a torch.Tensor, got {type(indices).__name__}."
                raise TypeError(msg)
            if indices.dtype not in (torch.int32, torch.int64):
                # Hard-cast, but be explicit for safety/debugging.
                indices = indices.long()
            if indices.dim() != 1:
                indices = indices.view(-1)
            z = self.embedding(indices)

        else:
            msg = "Must provide either indices or override_embeddings."
            raise ValueError(msg)

        x = self.hidden_layers(z)
        x = self.dense_output(x)
        return x.view(-1, *self.reshape_dim)

    def _resolve_activation(self, activation: str) -> torch.nn.Module:
        act = activation.lower()
        if act == "relu":
            return nn.ReLU()
        elif act == "elu":
            return nn.ELU()
        elif act in ("leaky_relu", "leakyrelu"):
            return nn.LeakyReLU()
        elif act == "selu":
            return nn.SELU()
        else:
            raise ValueError(f"Activation {activation} not supported.")
