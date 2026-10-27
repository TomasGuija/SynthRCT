"""Notebook-oriented inference convenience functions."""

from __future__ import annotations

import numpy as np
import torch

from rctsynth.utils.core import model_device
from rctsynth.utils.latent import encode_prior_z, sample_prior_z
from rctsynth.utils.spatial import decode_full_volume_with_refiner
from rctsynth.utils.types import DeformationSample


RESPIRATORY_PHASES: tuple[float, ...] = tuple(
    index / 10.0 for index in range(10)
)


@torch.no_grad()
def decode_latent(
    model: torch.nn.Module,
    moving_dhw: np.ndarray,
    z: np.ndarray | torch.Tensor,
    *,
    target_phase: float | None = None,
    batch_size: int = 1,
) -> DeformationSample:
    """Decode one latent code into a warped image, DVF, and SVF."""
    device = model_device(model)

    if isinstance(z, torch.Tensor):
        z_t = z.to(device=device, dtype=torch.float32)
        if z_t.ndim == 1:
            z_t = z_t.unsqueeze(0)
        z_np = z_t[0].detach().cpu().numpy()
    else:
        z_np = np.asarray(z, dtype=np.float32).reshape(-1)
        z_t = torch.from_numpy(z_np[None]).to(device)

    warped_dhw, flow_3dhw, svf_3dhw = decode_full_volume_with_refiner(
        model=model,
        z=z_t,
        moving_dhw=np.asarray(moving_dhw, dtype=np.float32),
        target_phase=target_phase,
        batch_size=max(1, int(batch_size)),
    )

    if warped_dhw is None:
        raise RuntimeError(
            "Full-volume decoding did not return a warped image."
        )

    return DeformationSample(
        z=z_np,
        warped_dhw=warped_dhw,
        flow_3dhw=flow_3dhw,
        svf_3dhw=svf_3dhw,
    )


@torch.no_grad()
def generate_breathing_cycle(
    model: torch.nn.Module,
    source_dhw: np.ndarray,
) -> list[DeformationSample]:
    """Generate ten deterministic respiratory states from one source CT.

    States are returned in target-phase order from ``s=0.0`` through ``s=0.9``.
    """
    if not getattr(model, "condition_on_phase", False):
        raise ValueError("A phase-conditioned model is required.")

    source = np.asarray(source_dhw, dtype=np.float32)
    if source.ndim != 3:
        raise ValueError(f"Expected source volume [D,H,W], got {source.shape}.")

    samples: list[DeformationSample] = []
    for phase in RESPIRATORY_PHASES:
        z = encode_prior_z(
            model=model,
            moving_full_dhw=source,
            target_phase=phase,
        )
        samples.append(
            decode_latent(
                model=model,
                moving_dhw=source,
                z=z,
                target_phase=phase,
            )
        )

    return samples


def run_random_samples(
    model: torch.nn.Module,
    moving_dhw: np.ndarray,
    *,
    n_samples: int = 4,
    seed: int | None = None,
    batch_size: int = 1,
) -> list[DeformationSample]:
    """Sample conditional-prior latents and decode them."""
    latents = sample_prior_z(
        model=model,
        moving_full_dhw=moving_dhw,
        n_samples=n_samples,
        seed=seed,
    )

    return [
        decode_latent(
            model=model,
            moving_dhw=moving_dhw,
            z=z,
            batch_size=batch_size,
        )
        for z in latents
    ]