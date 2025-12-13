from __future__ import annotations

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


def safe_kl_gauss_unit(
    mu: torch.Tensor, logvar: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """KL divergence between N(mu, exp(logvar)) and N(0, I) with guards.

    Args:
        mu (torch.Tensor): The mean of the latent Gaussian distribution.
        logvar (torch.Tensor): The log-variance of the latent Gaussian distribution.
        reduction (str): Specifies the reduction to apply to the output: 'mean' or 'sum'. Default is "mean".

    Returns:
        torch.Tensor: The computed KL divergence as a scalar tensor.
    """
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
    """Focal reconstruction + beta * KL with normalized class weights.

    Args:
        recon_logits (torch.Tensor): The reconstructed logits from the VAE decoder.
        targets (torch.Tensor): The ground truth target labels.
        mu (torch.Tensor): The mean of the latent Gaussian distribution.
        logvar (torch.Tensor): The log-variance of the latent Gaussian distribution.
        class_weights (torch.Tensor | None): Class weights for the focal loss. If None, no weighting is applied.
        gamma (float): Focusing parameter for the focal loss.
        beta (float): Weighting factor for the KL divergence term.
        ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient. Default is -1.

    Returns:
        torch.Tensor: The computed total loss as a scalar tensor.
    """
    cw = None
    if class_weights is not None:
        cw = class_weights / class_weights.mean().clamp_min(1e-8)

    criterion = SafeFocalCELoss(gamma=gamma, weight=cw, ignore_index=ignore_index)

    rec = criterion(recon_logits.view(-1, recon_logits.size(-1)), targets.view(-1))
    kl = safe_kl_gauss_unit(mu, logvar, reduction="mean")
    return rec + beta * kl
