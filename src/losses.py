"""
losses.py - Custom loss functions for FiNER-ORD.

Currently exposes:

  - DiceLoss: Self-adjusting Dice Loss per Li et al. 2020,
    "Dice Loss for Data-imbalanced NLP Tasks."

The dice path is exploratory and runs only on the vanilla (non-CRF) training
path. Phase 5d evaluates whether dice on a single seed (88) lifts test
entity F1 by >= +0.005 over `efficient_training_seed88`. CRF + Dice
integration is explicitly out of scope for the spike; the training entry
point raises if both `use_crf` and `loss_type=dice` are set.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Self-adjusting multi-class Dice Loss for token classification.

    Implements equation (5) from Li et al. 2020. For each class c the
    self-adjusting dice score is:

        DSC^c = (2 * sum_i [(1 - p_ic)^alpha * p_ic * y_ic] + smooth)
              / (    sum_i [(1 - p_ic)^alpha * p_ic + y_ic] + smooth)

    where `p_ic` is the predicted probability of class c at position i and
    `y_ic` is the one-hot indicator. The loss is `1 - mean_c DSC^c`.

    The `(1 - p)^alpha` term down-weights easy positives so the gradient
    focuses on hard examples; small alpha (e.g. 0.01) yields a near-vanilla
    dice while still pushing the optimizer away from the trivial all-O
    solution that dominates token-classification CE on FiNER.

    Positions with label `-100` (HuggingFace's sentinel for special tokens
    and continuation subwords) are masked out before the dice computation,
    matching the masking convention used everywhere else in the repo.
    """

    def __init__(
        self,
        num_labels: int,
        smooth: float = 1.0,
        alpha: float = 0.01,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if num_labels <= 0:
            raise ValueError("`num_labels` must be > 0.")
        if smooth <= 0:
            raise ValueError("`smooth` must be > 0 to keep the denominator finite.")
        if alpha < 0:
            raise ValueError("`alpha` must be >= 0.")
        self.num_labels = int(num_labels)
        self.smooth = float(smooth)
        self.alpha = float(alpha)
        self.ignore_index = int(ignore_index)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the dice loss.

        Parameters
        ----------
        logits : torch.Tensor
            Shape (batch, seq_len, num_labels) or (N, num_labels).
        labels : torch.Tensor
            Shape (batch, seq_len) or (N,). Positions equal to
            `self.ignore_index` are excluded from the dice computation.

        Returns
        -------
        torch.Tensor
            Scalar loss in (0, 1].
        """

        if logits.shape[-1] != self.num_labels:
            raise ValueError(
                f"Last logits dim is {logits.shape[-1]}, expected {self.num_labels}."
            )

        flat_logits = logits.reshape(-1, self.num_labels)
        flat_labels = labels.reshape(-1)

        # Mask -100 BEFORE the softmax/one-hot to match the masking convention
        # used by KL distillation later. Doing it after produces nonsense
        # gradients on the ignored positions.
        valid_mask = flat_labels != self.ignore_index
        if not torch.any(valid_mask):
            return logits.sum() * 0.0

        valid_logits = flat_logits[valid_mask]
        valid_labels = flat_labels[valid_mask]

        # Clamp to a sane range so out-of-vocabulary or accidentally placed
        # label IDs don't quietly silently produce one-hot off-by-one bugs.
        if valid_labels.min() < 0 or valid_labels.max() >= self.num_labels:
            raise ValueError(
                "DiceLoss received label IDs outside [0, num_labels). "
                f"min={int(valid_labels.min())}, max={int(valid_labels.max())}."
            )

        probs = F.softmax(valid_logits.float(), dim=-1)
        one_hot = F.one_hot(valid_labels.long(), num_classes=self.num_labels).float()

        weights = (1.0 - probs).pow(self.alpha)
        weighted = weights * probs

        # Per-class numerator/denominator summed over valid positions.
        numerator = 2.0 * (weighted * one_hot).sum(dim=0) + self.smooth
        denominator = (weighted + one_hot).sum(dim=0) + self.smooth

        dice_per_class = numerator / denominator
        loss = 1.0 - dice_per_class.mean()

        # Cast back to the input dtype so the autocast context (fp16 on CUDA)
        # can keep the loss in mixed precision if it wants to.
        return loss.to(logits.dtype)
