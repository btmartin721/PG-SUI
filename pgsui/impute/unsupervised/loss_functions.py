from typing import List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeFocalCELoss(nn.Module):
    """Focal cross-entropy with ignore_index and numeric guards.

    This class implements the focal loss function, which is designed to address class imbalance by down-weighting easy examples and focusing training on hard negatives. It also includes handling for ignored indices and numeric stability.
    """

    def __init__(
        self,
        gamma: float,
        weight: torch.Tensor | None = None,
        ignore_index: int = -1,
        eps: float = 1e-8,
    ):
        """Initialize the SafeFocalCELoss.

        This class sets up the focal loss with specified focusing parameter, class weights, ignore index, and a small epsilon for numerical stability.

        Args:
            gamma (float): Focusing parameter.
            weight (torch.Tensor | None): A manual rescaling weight given to each class. If given, has to be a Tensor of size C (number of classes). Defaults to None.
            ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient. Default is -1.
            eps (float): Small value to avoid numerical issues. Default is 1e-8.
        """
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates the focal loss on pre-flattened tensors.

        Args:
            logits (torch.Tensor): Logits from the model of shape (N, C) where N is the number of samples and C is the number of classes.
            targets (torch.Tensor): Ground truth labels of shape (N,).

        Returns:
            torch.Tensor: The computed scalar loss value.
        """
        # logits: (N, C), targets: (N,)
        valid = targets != self.ignore_index

        if not valid.any():
            return logits.new_tensor(0.0)

        logits_v = logits[valid]
        targets_v = targets[valid]

        logp = F.log_softmax(logits_v, dim=-1)  # stable
        ce = F.nll_loss(logp, targets_v, weight=self.weight, reduction="none")

        # p_t = exp(logp[range, targets])
        p_t = torch.exp(logp.gather(1, targets_v.unsqueeze(1)).squeeze(1))

        # focal factor with clamp to avoid 0**gamma and NaNs
        focal = (1.0 - p_t).clamp_min(self.eps).pow(self.gamma)

        loss_vec = focal * ce

        # guard remaining inf/nan if any slipped through
        loss_vec = torch.nan_to_num(loss_vec, nan=0.0, posinf=1e6, neginf=0.0)
        return loss_vec.mean()


class WeightedMaskedCCELoss(nn.Module):
    def __init__(
        self,
        alpha: float | List[float] | torch.Tensor | None = None,
        reduction: Literal["mean", "sum"] = "mean",
    ):
        """A weighted, masked Categorical Cross-Entropy loss function.

        This method computes the categorical cross-entropy loss while allowing for class weights and masking of invalid (missing) entries. It is particularly useful for sequence data where some positions may be missing or should not contribute to the loss calculation.

        Args:
            alpha (float | List | Tensor | None): A manual rescaling weight given to each class. If given, has to be a Tensor of size C (number of classes). Defaults to None.
            reduction (str, optional): Specifies the reduction to apply to the output: 'mean' or 'sum'. Defaults to "mean".
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
            msg = f"Reduction mode '{self.reduction}' not supported."
            raise ValueError(msg)


class MaskedFocalLoss(nn.Module):
    """Focal loss (gamma > 0) with optional class weights and a boolean valid mask.

    This method implements the focal loss function, which is designed to address class imbalance by down-weighting easy examples and focusing training on hard negatives. It also supports masking of invalid (missing) entries, making it suitable for sequence data with missing values.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: Literal["mean", "sum"] = "mean",
    ):
        """Initialize the MaskedFocalLoss.

        This class sets up the focal loss with specified focusing parameter, class weights, and reduction method. It is designed to handle missing data through a valid mask, ensuring that only relevant entries contribute to the loss calculation.

        Args:
            gamma (float): Focusing parameter.
            alpha (torch.Tensor | None): Class weights.
            reduction (Literal["mean", "sum"]): Reduction mode ('mean' or 'sum').
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,  # Expects (N, C) where N = batch*features
        targets: torch.Tensor,  # Expects (N,)
        valid_mask: torch.Tensor,  # Expects (N,)
    ) -> torch.Tensor:
        """Calculates the focal loss on pre-flattened tensors.

        Args:
            logits (torch.Tensor): Logits from the model of shape (N, C) where N is the number of samples (batch_size * seq_len) and C is the number of classes.
            targets (torch.Tensor): Ground truth labels of shape (N,).
            valid_mask (torch.Tensor): Boolean mask of shape (N,) where True indicates a valid (observed) value to include in the loss.

        Returns:
            torch.Tensor: The computed scalar loss value.
        """
        device = logits.device

        # Calculate standard cross-entropy loss per-token (no reduction)
        ce = F.cross_entropy(
            logits,
            targets,
            weight=(self.alpha.to(device) if self.alpha is not None else None),
            reduction="none",
            ignore_index=-1,
        )

        # Calculate p_t from the cross-entropy loss
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce

        # Apply the valid mask. We select only the elements that should contribute to the loss.
        focal = focal[valid_mask]

        # Return early if no valid elements exist to avoid NaN results
        if focal.numel() == 0:
            return torch.tensor(0.0, device=device)

        # Apply reduction
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        else:
            msg = f"Reduction mode '{self.reduction}' not supported."
            raise ValueError(msg)


def safe_kl_gauss_unit(
    mu: torch.Tensor, logvar: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """KL divergence between N(mu, exp(logvar)) and N(0, I) with guards."""
    logvar = logvar.clamp(min=-30.0, max=20.0)
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    if reduction == "sum":
        kl = kl.sum()
    elif reduction == "mean":
        kl = kl.mean()
    return torch.nan_to_num(kl, nan=0.0, posinf=1e6, neginf=0.0)


def compute_vae_loss(
    recon_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    class_weights: torch.Tensor | None,
    gamma: float,
    beta: float,
    ignore_index: int = -1,
) -> torch.Tensor:
    """Focal reconstruction + beta * KL with normalized class weights."""
    cw = None
    if class_weights is not None:
        cw = class_weights / class_weights.mean().clamp_min(1e-8)

    criterion = SafeFocalCELoss(
        gamma=gamma,
        weight=cw,
        ignore_index=ignore_index,
    )
    rec = criterion(recon_logits.view(-1, recon_logits.size(-1)), targets.view(-1))
    kl = safe_kl_gauss_unit(mu, logvar, reduction="mean")
    return rec + beta * kl
