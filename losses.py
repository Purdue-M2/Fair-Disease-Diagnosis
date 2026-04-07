"""FairLACVaR Loss: Logit-Adjusted CE with CVaR fairness aggregation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FairLACVaRLoss(nn.Module):
    """Fairness-aware loss combining Logit-Adjusted CE and CVaR.

    Logit-adjusted cross-entropy corrects for class imbalance by shifting
    logits with log-prior offsets. CVaR (Conditional Value-at-Risk)
    aggregates per-group losses to penalise the worst-performing
    demographic subgroup.

    Args:
        class_counts: Per-class sample counts for log-prior computation.
        alpha: CVaR confidence level in (0, 1]. Lower alpha focuses more
               on the worst-off group. alpha=1.0 reduces to simple mean.
        tau: Temperature scaling for log-prior adjustment (default: 1.0).
    """

    def __init__(self, class_counts, alpha=0.5, tau=1.0):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32)
        freqs = counts / counts.sum()
        log_prior = tau * torch.log(freqs)
        self.register_buffer("log_prior", log_prior)
        self.alpha = alpha

    def la_cross_entropy(self, logits, targets):
        adjusted_logits = logits + self.log_prior
        return F.cross_entropy(adjusted_logits, targets, reduction="none")

    def _binary_search_lambda(self, group_losses, num_iters=32):
        lo = group_losses.min().item()
        hi = group_losses.max().item()
        for _ in range(num_iters):
            mid = (lo + hi) / 2.0
            frac_above = (group_losses > mid).float().mean().item()
            deriv = 1.0 - frac_above / self.alpha
            if deriv < 0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    def forward(self, logits, targets, group_labels):
        per_sample_loss = self.la_cross_entropy(logits, targets)

        unique_groups = torch.unique(group_labels)
        group_losses = []
        for g in unique_groups:
            mask = group_labels == g
            if mask.sum() > 0:
                group_losses.append(per_sample_loss[mask].mean())

        if len(group_losses) == 1:
            return group_losses[0]

        group_losses = torch.stack(group_losses)
        num_groups = group_losses.shape[0]

        if self.alpha >= 1.0:
            return group_losses.mean()

        lam = self._binary_search_lambda(group_losses)
        hinge = F.relu(group_losses - lam)
        loss = lam + hinge.sum() / (self.alpha * num_groups)
        return loss
