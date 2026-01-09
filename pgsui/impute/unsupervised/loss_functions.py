from __future__ import annotations

from typing import Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalCELoss(nn.Module):
    """Focal cross-entropy with ignore_index and optional scaling.

    Supports logits of shape (N, C) or (N, C, d1, d2, ...). Targets must be shape-compatible: (N) or (N, d1, d2, ...).

    The optional `recon_scale` is useful in reconstruction settings (e.g., VAE) when your base reduction is "mean" over a sparse mask. Multiplying the final reduced loss by `recon_scale` makes the reconstruction term more "sum-like" per batch/sample, preventing KL from dominating.
    """

    def __init__(
        self,
        *,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        ignore_index: int = -1,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        """Initialize the focal cross-entropy loss.

        Args:
            alpha: Optional per-class weights of shape (C,).
            gamma: Focusing parameter.
            ignore_index: Target value to ignore.
            reduction: "mean", "sum", or "none".
        """
        super().__init__()
        self._gamma = float(gamma)
        self.ignore_index = int(ignore_index)
        self.reduction = reduction

        if alpha is not None:
            if alpha.dim() != 1:
                raise ValueError("alpha must be a 1D tensor of shape (C,).")
            # Register as buffer so it moves with the module across devices.
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        recon_scale: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Compute focal cross-entropy loss.

        Args:
            logits: Tensor of shape (N, C) or (N, C, d1, d2, ...).
            targets: Tensor of shape (N) or (N, d1, d2, ...).
            recon_scale: Optional scalar multiplier applied to the final loss.
                - If reduction is "mean" or "sum", multiplies the scalar loss.
                - If reduction is "none", multiplies elementwise.

        Returns:
            Loss tensor:
                - Scalar if reduction in {"mean","sum"}
                - Tensor shaped like `targets` if reduction == "none"
        """
        # Move C (dim 1) to the last position for flattening:
        # (N, C, d1, ...) -> (N, d1, ..., C)
        if logits.dim() > 2:
            logits = logits.permute(0, *range(2, logits.dim()), 1)

        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1).long()

        valid_mask = targets_flat != self.ignore_index

        # Early exit if everything is ignored
        if not bool(valid_mask.any()):
            out = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            # preserve grad path behavior if caller expects it
            out = out.requires_grad_(True)
            return out

        logits_v = logits_flat[valid_mask]
        targets_v = targets_flat[valid_mask]

        # Numerically stable log-softmax
        log_probs = F.log_softmax(logits_v, dim=-1)
        log_pt = log_probs.gather(1, targets_v.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        focal_term = (1.0 - pt).pow(self.gamma)
        loss_vec = -focal_term * log_pt

        if self.alpha is not None:
            loss_vec = loss_vec * self.alpha[targets_v]

        # Apply reduction
        if self.reduction == "mean":
            out = loss_vec.mean()
        elif self.reduction == "sum":
            out = loss_vec.sum()
        else:  # "none"
            out_flat = torch.zeros_like(targets_flat, dtype=loss_vec.dtype)
            out_flat[valid_mask] = loss_vec
            out = out_flat.view(targets.shape)

        # Optional scaling (useful for VAE recon term)
        if recon_scale is not None:
            if not isinstance(recon_scale, torch.Tensor):
                recon_scale = torch.tensor(
                    float(recon_scale), device=out.device, dtype=out.dtype
                )
            else:
                recon_scale = recon_scale.to(device=out.device, dtype=out.dtype)

            out = out * recon_scale

        return out

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, value: torch.Tensor | float) -> None:
        if isinstance(value, torch.Tensor):
            value = float(value.item())
        self._gamma = float(value)


def safe_kl_gauss_unit(
    mu: torch.Tensor, logvar: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """Compute KL divergence between N(mu, var) and N(0, I) with numeric guards.

    Args:
        mu (torch.Tensor): Latent mean (shape: [B, D]).
        logvar (torch.Tensor): Latent log-variance (shape: [B, D]).
        reduction (str): Reduction method ('mean' or 'sum').

    Returns:
        torch.Tensor: KL divergence (scalar).
    """
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())  # (B, D)
    kl = kl.sum(dim=-1)  # (B,)

    if reduction == "sum":
        kl = kl.sum()
    elif reduction == "mean":
        kl = kl.mean()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    return torch.nan_to_num(kl, nan=0.0, posinf=1e6, neginf=0.0)


def compute_vae_loss(
    criterion: nn.Module,
    recon_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_beta: torch.Tensor | float,
    reduction: str = "mean",
    recon_scale: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Compute VAE loss: reconstruction + KL divergence, with optional recon scaling.

    Args:
        criterion: Reconstruction loss module (e.g., FocalCELoss / CrossEntropyLoss).
            Must accept (logits_2d, targets_1d). If it supports `recon_scale`, it will
            be passed through; otherwise it will be called without it.
        recon_logits: Reconstruction logits from decoder. Shape: (N, L, C) or (N_eval, C).
        targets: Ground truth targets. Shape: (N, L) or (N_eval,).
        mu: Latent mean. Shape: (B, D) (or compatible with safe_kl_gauss_unit).
        logvar: Latent log-variance. Shape: (B, D).
        kl_beta: Scalar KL weight.
        reduction: KL reduction: "mean" or "sum".
        recon_scale: Optional scalar multiplier applied to reconstruction term.
            Use this to make reconstruction more "sum-like" for high-dimensional data.

    Returns:
        Scalar loss tensor.
    """
    # Flatten logits/targets to (N_total, C) and (N_total,)
    if recon_logits.dim() == 3:
        logits_2d = recon_logits.reshape(-1, recon_logits.size(-1))
    elif recon_logits.dim() == 2:
        logits_2d = recon_logits
    else:
        msg = f"recon_logits must be 2D or 3D; got shape {tuple(recon_logits.shape)}"
        raise ValueError(msg)

    tgt_1d = targets.reshape(-1) if targets.dim() > 1 else targets

    # Reconstruction loss (criterion may ignore_index internally)
    try:
        rec = criterion(logits_2d, tgt_1d, recon_scale=recon_scale)
    except TypeError:
        # Criterion doesn't accept recon_scale (e.g., torch.nn.CrossEntropyLoss)
        rec = criterion(logits_2d, tgt_1d)
        if recon_scale is not None:
            if isinstance(recon_scale, torch.Tensor):
                rec = rec * recon_scale.to(device=rec.device, dtype=rec.dtype)
            else:
                rec = rec * float(recon_scale)

    # KL term
    kl = safe_kl_gauss_unit(mu, logvar, reduction=reduction)
    loss = rec + (kl_beta * kl)

    return torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)
