from typing import List, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from snpio.utils.logging import LoggerManager


from typing import List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from snpio.utils.logging import LoggerManager


class WeightedMaskedCCELoss(nn.Module):
    def __init__(
        self,
        alpha: float | List[float] | torch.Tensor | None = None,
        reduction: Literal["mean", "sum"] = "mean",
    ):
        """
        A weighted, masked Categorical Cross-Entropy loss function.

        Args:
            alpha (float | List | Tensor, optional): A manual rescaling weight given to each class.
                If given, has to be a Tensor of size C (number of classes).
                Defaults to None.
            reduction (str, optional): Specifies the reduction to apply to the
                output: 'mean' or 'sum'. Defaults to "mean".
        """
        super(WeightedMaskedCCELoss, self).__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the masked categorical cross-entropy loss.

        Args:
            logits (torch.Tensor): Logits from the model of shape
                (batch_size, seq_len, num_classes).
            targets (torch.Tensor): Ground truth labels of shape (batch_size, seq_len).
            valid_mask (torch.Tensor, optional): Boolean mask of shape (batch_size, seq_len) where True indicates a valid (observed) value to include in the loss.
                Defaults to None, in which case all values are considered valid.

        Returns:
            torch.Tensor: The computed scalar loss value.
        """
        # Automatically detect the device from the input tensor
        device = logits.device
        num_classes = logits.shape[-1]

        # Ensure targets are on the correct device and are Long type
        targets = targets.to(device).long()

        # Prepare weights and pass them directly to the loss function
        class_weights = None
        if self.alpha is not None:
            if not isinstance(self.alpha, torch.Tensor):
                class_weights = torch.tensor(
                    self.alpha, dtype=torch.float, device=device
                )
            else:
                class_weights = self.alpha.to(device)

        loss = F.cross_entropy(
            logits.reshape(-1, num_classes),
            targets.reshape(-1),
            weight=class_weights,
            reduction="none",
            ignore_index=-1,  # Ignore all targets with the value -1
        )

        # If a mask is provided, filter the losses for the training set
        if valid_mask is not None:
            loss = loss[valid_mask.reshape(-1)]

        # If after masking no valid losses remain, return 0
        if loss.numel() == 0:
            return torch.tensor(0.0, device=device)

        # Apply the final reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Reduction mode '{self.reduction}' not supported.")


class MaskedFocalLoss(nn.Module):
    """Focal loss (gamma > 0) with optional class weights and a boolean valid mask."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        """Focal loss (gamma > 0) with optional class weights and a boolean valid mask.

        Args:
            gamma (float, optional): Focusing parameter. Defaults to 2.0.
            alpha (torch.Tensor | None, optional): Class weights. Defaults to None.
            reduction (str, optional): Reduction mode. Defaults to "mean".
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,  # (B, L, C)
        targets: torch.Tensor,  # (B, L) ints in [0..C-1] or -1
        valid_mask: torch.Tensor | None,  # (B, L) bool
    ) -> torch.Tensor:
        device = logits.device
        B, L, C = logits.shape
        t = targets.to(device).long().reshape(-1)
        logit2d = logits.reshape(-1, C)

        # standard CE per-token (no reduction)
        ce = F.cross_entropy(
            logit2d,
            t,
            weight=(self.alpha.to(device) if self.alpha is not None else None),
            reduction="none",
            ignore_index=-1,
        )

        # p_t from CE (no need to softmax explicitly)
        pt = torch.exp(-ce)  # p_t = exp(-CE)
        focal = ((1 - pt) ** self.gamma) * ce

        if valid_mask is not None:
            m = valid_mask.to(device).reshape(-1)
            focal = focal[m]

        if focal.numel() == 0:
            return torch.tensor(0.0, device=device)

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        else:
            raise ValueError(f"Reduction mode '{self.reduction}' not supported.")
