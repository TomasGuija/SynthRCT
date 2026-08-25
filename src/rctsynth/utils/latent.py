"""Latent encoding, sampling, and PCA helpers."""

from __future__ import annotations

import numpy as np
import torch

from rctsynth.utils.core import model_device, resolve_device, to_torch_5d
from rctsynth.utils.types import LatentPCA

def _sample_from_mu_logvar(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    n_samples: int = 1,
) -> torch.Tensor:
    """Sample z from Gaussian parameters."""
    if mu.ndim == 1:
        mu = mu.unsqueeze(0)
    if logvar.ndim == 1:
        logvar = logvar.unsqueeze(0)

    if mu.shape[0] != 1:
        raise ValueError(
            f"Expected a single batch element for latent parameters, got {mu.shape}."
        )

    eps = torch.randn(
        (int(n_samples), mu.shape[-1]),
        device=mu.device,
        dtype=mu.dtype,
    )
    return mu.expand(int(n_samples), -1) + eps * torch.exp(0.5 * logvar).expand(int(n_samples), -1)


@torch.no_grad()
def encode_pair_z(
    model,
    fixed_full_dhw: np.ndarray,
    moving_full_dhw: np.ndarray,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Encode a fixed/moving pair and return the posterior mean latent code."""
    device = model_device(model) if device is None else resolve_device(device)
    model.eval().to(device)

    fixed_t = to_torch_5d(fixed_full_dhw, device)
    moving_t = to_torch_5d(moving_full_dhw, device)

    q_mu, _ = model.encode_posterior(
        fixed_full=fixed_t,
        moving_full=moving_t,
    )

    return q_mu


@torch.no_grad()
def encode_identity_z(
    model,
    moving_full_dhw: np.ndarray,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Encode the identity deformation by using the same image as fixed and moving."""
    return encode_pair_z(
        model=model,
        fixed_full_dhw=moving_full_dhw,
        moving_full_dhw=moving_full_dhw,
        device=device,
    )


@torch.no_grad()
def sample_prior_z(
    model,
    moving_full_dhw: np.ndarray,
    device: torch.device | str | None = None,
    n_samples: int = 1,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample latent codes from the anatomy-conditioned prior."""
    device = model_device(model) if device is None else resolve_device(device)
    model.eval().to(device)

    if seed is not None:
        torch.manual_seed(int(seed))

    moving_t = to_torch_5d(moving_full_dhw, device)

    p_mu, p_logvar = model.encode_prior(moving_full=moving_t)
    return _sample_from_mu_logvar(
        p_mu,
        p_logvar,
        n_samples=n_samples,
    )


def pca_scores_for_latents(latents: np.ndarray) -> LatentPCA:
    """Fit a simple PCA model from latent vectors [N, latent_dim]."""
    latents = np.asarray(latents, dtype=np.float32)

    if latents.ndim != 2:
        raise ValueError(f"Expected latents [N,L], got {latents.shape}.")

    mean = np.mean(latents, axis=0, keepdims=True)
    centered = latents - mean

    # SVD-based PCA
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt.astype(np.float32, copy=False)
    scores = (centered @ components.T).astype(np.float32, copy=False)

    if latents.shape[0] > 1:
        explained_variance = (s**2 / (latents.shape[0] - 1)).astype(np.float32, copy=False)
    else:
        explained_variance = np.zeros_like(s, dtype=np.float32)

    return LatentPCA(
        mean=mean[0].astype(np.float32, copy=False),
        components=components,
        explained_variance=explained_variance,
        scores=scores,
    )


def pc_traversal_latents(
    pca: LatentPCA,
    pc_index: int = 0,
    num_steps: int = 7,
    n_std: float = 2.0,
    base_z: np.ndarray | None = None,
) -> np.ndarray:
    """Generate latent vectors along a chosen principal component."""
    if pc_index < 0 or pc_index >= pca.components.shape[0]:
        raise IndexError(f"pc_index={pc_index} out of range for {pca.components.shape[0]} PCs.")

    if num_steps < 2:
        raise ValueError(f"num_steps must be >= 2, got {num_steps}.")

    base = pca.mean if base_z is None else np.asarray(base_z, dtype=np.float32).reshape(-1)
    direction = pca.components[pc_index]
    std = float(np.sqrt(max(float(pca.explained_variance[pc_index]), 0.0)))

    alphas = np.linspace(-float(n_std), float(n_std), num_steps, dtype=np.float32)
    zs = np.stack([base + a * std * direction for a in alphas], axis=0)
    return zs.astype(np.float32, copy=False)
