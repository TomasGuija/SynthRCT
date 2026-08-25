"""Loss helpers for SynthRCT VAE training."""

from __future__ import annotations

import torch


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor | None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return the mean of ``values`` inside an optional mask."""
    if mask is None:
        return values.mean()

    if mask.ndim == values.ndim - 1:
        mask = mask.unsqueeze(1)

    mask = mask.to(device=values.device, dtype=values.dtype)

    if mask.shape[1] == 1 and values.shape[1] > 1:
        mask = mask.expand(-1, values.shape[1], *mask.shape[2:])

    return (values * mask).sum() / mask.sum().clamp_min(eps)


def kl_divergence(
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """Compute KL[N(mu, var) || N(0, I)] for diagonal Gaussians."""
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return kl.sum(dim=1).mean()


def kl_divergence_gaussians(
    q_mu: torch.Tensor,
    q_logvar: torch.Tensor,
    p_mu: torch.Tensor,
    p_logvar: torch.Tensor,
) -> torch.Tensor:
    """Compute KL[q || p] for diagonal Gaussian distributions."""
    q_var = q_logvar.exp()
    p_var = p_logvar.exp()

    kl = 0.5 * (
        p_logvar
        - q_logvar
        + (q_var + (q_mu - p_mu).pow(2)) / p_var
        - 1.0
    )

    return kl.sum(dim=1).mean()