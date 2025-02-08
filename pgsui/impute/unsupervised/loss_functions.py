from typing import Dict, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from snpio.utils.logging import LoggerManager


class MaskedFocalLoss(nn.Module):
    def __init__(
        self,
        class_weights: Dict[str, float] | None = None,
        alpha: float | List[float] = None,
        gamma: float = 2.0,
        reduction: Literal["mean", "sum"] = "mean",
        verbose: int = 0,
        debug: bool = False,
    ):
        """Compute the masked focal loss between predictions and targets.

        This class is used to compute the masked focal loss between model predictions and ground truth labels. The focal loss is a variant of the cross-entropy loss that focuses on hard-to-classify examples. The loss is computed as:

        .. math::
            L(p_t) = -\\alpha_t (1 - p_t)^\\gamma \\log(p_t)

        where :math:`p_t` is the predicted probability, :math:`\\alpha_t` is the class weight, and :math:`\\gamma` is the focusing parameter. The loss can be reduced by taking the mean or sum of the sample losses.

        Args:
            class_weights (list, optional): Class weights for imbalanced data. Default is None.
            alpha (float, list, optional): Weighting factor for the focal loss. Default is None.
            gamma (float, optional): Focusing parameter for the focal loss. Default is 2.0.
            reduction (str, optional): Reduction method for the loss. Must be one of {"mean", "sum"}. Default is "mean".
            verbose (int, optional): Verbosity level for logging messages. Default is 0.
            debug (bool, optional): Debug mode for logging messages. Default is False.

        Example:
            >>> import torch
            >>> from pgsui.impute.unsupervised.loss_functions import MaskedFocalLoss
            >>> predictions = torch.randn(2, 3, 5)
            >>> targets = torch.randint(0, 5, (2, 3))
            >>> criterion = MaskedFocalLoss(alpha=0.25, gamma=2.0)
            >>> loss = criterion(predictions, targets)
            >>> print(loss)
            tensor(1.1539)
        """
        super(MaskedFocalLoss, self).__init__()

        if isinstance(class_weights, list):
            class_weights = torch.tensor(class_weights, dtype=torch.float)

        self.class_weights = class_weights
        self.alpha = alpha

        if isinstance(self.alpha, list):
            self.alpha = torch.tensor(self.alpha, dtype=torch.float)

        self.gamma = gamma
        self.reduction = reduction

        logman = LoggerManager(
            name=__name__, prefix="pgsui", debug=debug, verbose=verbose >= 1
        )
        self.logger = logman.get_logger()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ):
        """Compute the masked focal loss between predictions and targets.

        This method computes the masked focal loss between the model predictions and the ground truth labels. The mask tensor is used to ignore certain values (< 0), and class weights can be provided to balance the loss.

        Args:
            predictions (torch.Tensor): Model predictions of shape (batch, seq, num_classes).
            targets (torch.Tensor): Ground truth labels of shape (batch, seq) or (batch, seq, num_classes).
            valid_mask (torch.Tensor, optional): Mask tensor to ignore certain values. Default is None.

        Returns:
            torch.Tensor: Computed focal loss value.

        Raises:
            ValueError: If the targets shape is invalid.
            ValueError: If the alpha type is invalid.
            ValueError: If the class weights size does not match the number of classes.
            ValueError: If the reduction type is invalid. Must be "mean" or "sum".
        """
        num_classes = predictions.shape[-1]

        # Validate targets shape
        if targets.dim() == 3 and targets.size(-1) == num_classes:
            targets = torch.argmax(targets, dim=-1)
        elif targets.dim() != 2:
            msg = "Targets must have shape (batch, seq) or (batch, seq, num_classes)."
            self.logger.error(msg)
            raise ValueError(msg)

        # Flatten predictions and targets
        predictions_2d = predictions.view(-1, num_classes)
        targets_1d = targets.view(-1)
        valid_mask = valid_mask.view(-1)

        # Mask invalid values
        if valid_mask.sum() == 0:
            self.logger.warning("No valid targets found. Returning zero loss.")
            return predictions_2d.new_tensor(0.0)

        valid_preds = predictions_2d[valid_mask]
        valid_tgts = targets_1d[valid_mask]
        valid_tgts = valid_tgts.long()

        # Compute probabilities and log probabilities.
        gathered_probs = F.softmax(valid_preds, dim=-1)[
            torch.arange(valid_tgts.size(0)), valid_tgts
        ]

        # Clamp probabilities to avoid log(0) for numerical stability.
        eps = 1e-6
        gathered_probs = gathered_probs.clamp(min=eps, max=1 - eps)
        gathered_log_probs = torch.log(gathered_probs)

        # Compute focal loss factor
        focal_factor = (1.0 - gathered_probs).pow(self.gamma)

        # Compute alpha factor
        if self.alpha is None:
            alpha_factor = 1.0

        elif isinstance(self.alpha, float):
            alpha_factor = self.alpha

        elif isinstance(self.alpha, torch.Tensor):
            alpha_factor = self.alpha[valid_tgts]

        else:
            pstr = type(self.alpha)
            msg = f"Invalid alpha type: {pstr}. Expected float or torch.Tensor."
            self.logger.error(msg)
            raise ValueError(msg)

        # Compute sample losses
        sample_losses = -gathered_log_probs * focal_factor * alpha_factor

        # Apply class weights if provided.
        if self.class_weights is not None:
            if self.class_weights.size(0) != num_classes:
                msg = "Mismatch in class_weights size."
                self.logger.error(msg)
                raise ValueError(msg)
            sample_losses *= self.class_weights[valid_tgts]

        # Reduce loss based on reduction type (mean or sum).
        if self.reduction == "sum":
            loss = sample_losses.sum()
        elif self.reduction == "mean":
            loss = sample_losses.sum() / valid_mask.sum()
        else:
            msg = f"Unsupported reduction type: {self.reduction}. Use 'mean' or 'sum'."
            self.logger.error(msg)
            raise ValueError(msg)

        return loss
