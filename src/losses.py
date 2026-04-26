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

    The `(1 - p)^alpha` factor is the whole point: it down-weights easy
    positives so the gradient focuses on the hard examples that dominate
    the remaining error. The paper's NER experiments on OntoNotes 5.0
    sweep alpha in {0, 0.6, 1.0, 2.0, 3.0} and report best F1 around
    alpha=0.6. Setting alpha very close to 0 disables the self-adjusting
    factor; the loss collapses to plain soft-dice on a softmax with the
    O majority, the gradient norm shrinks toward zero, and the model
    settles on predicting O everywhere.

    The `outside_label_id` knob excludes the O (or whatever majority/
    background) class from the dice mean, which is the standard NER
    practice. This keeps the dice signal focused on the entity classes
    that actually carry the F1 we care about.

    Positions with label `-100` (HuggingFace's sentinel for special tokens
    and continuation subwords) are masked out before the dice computation,
    matching the masking convention used everywhere else in the repo.
    """

    def __init__(
        self,
        num_labels: int,
        smooth: float = 1.0,
        alpha: float = 0.6,
        ignore_index: int = -100,
        outside_label_id: int | None = 0,
    ) -> None:
        super().__init__()
        if num_labels <= 0:
            raise ValueError("`num_labels` must be > 0.")
        if smooth <= 0:
            raise ValueError("`smooth` must be > 0 to keep the denominator finite.")
        if alpha < 0:
            raise ValueError("`alpha` must be >= 0.")
        if outside_label_id is not None and not 0 <= outside_label_id < num_labels:
            raise ValueError(
                f"`outside_label_id` must be in [0, {num_labels}) or None; "
                f"got {outside_label_id}."
            )
        self.num_labels = int(num_labels)
        self.smooth = float(smooth)
        self.alpha = float(alpha)
        self.ignore_index = int(ignore_index)
        self.outside_label_id = (
            int(outside_label_id) if outside_label_id is not None else None
        )

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

        # Mask -100 before softmax/one-hot so ignored subwords do not produce
        # nonsense gradients.
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

        # Clamp to avoid (1 - 1.0)^alpha = 0 when alpha is positive, which
        # otherwise zeros out the gradient on perfectly-confident easy
        # positives the moment the model gets one right.
        eps = 1.0e-6
        probs_clamped = probs.clamp(min=eps, max=1.0 - eps)
        weights = (1.0 - probs_clamped).pow(self.alpha)
        weighted = weights * probs

        # Per-class numerator/denominator summed over valid positions.
        numerator = 2.0 * (weighted * one_hot).sum(dim=0) + self.smooth
        denominator = (weighted + one_hot).sum(dim=0) + self.smooth

        dice_per_class = numerator / denominator

        # Standard NER practice: drop the `O` (background) class from the
        # mean so the dice signal stays focused on entity classes. With it
        # included, the high `O` dice score dilutes the loss and the
        # gradient on rare classes vanishes, which is why the original
        # `alpha=0.01` config collapsed to all-O at ~0.007 grad norm.
        if self.outside_label_id is not None:
            keep_mask = torch.ones(
                self.num_labels, dtype=torch.bool, device=dice_per_class.device
            )
            keep_mask[self.outside_label_id] = False
            dice_per_class = dice_per_class[keep_mask]

        loss = 1.0 - dice_per_class.mean()

        # Cast back to the input dtype so the autocast context (fp16 on CUDA)
        # can keep the loss in mixed precision if it wants to.
        return loss.to(logits.dtype)
