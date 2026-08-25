"""Device, tensor, and padding helpers."""

from __future__ import annotations

import numpy as np
import torch

def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve a requested device, falling back to CPU if CUDA is unavailable."""
    device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")

    return device


def model_device(model: torch.nn.Module) -> torch.device:
    """Return the device holding a model's parameters."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def to_torch_5d(
    volume_dhw: np.ndarray,
    device: torch.device | str,
) -> torch.Tensor:
    """Convert a DHW NumPy volume to (1, 1, D, H, W)."""
    device = resolve_device(device)
    volume = np.asarray(volume_dhw, dtype=np.float32)

    if volume.ndim != 3:
        raise ValueError(f"Expected volume [D,H,W], got {volume.shape}.")

    return torch.from_numpy(volume[None, None]).to(device=device, dtype=torch.float32)


def _pad_depth_end(
    volume_dhw: np.ndarray,
    target_depth: int,
    constant_value: float = 0.0,
) -> np.ndarray:
    """Pad a DHW volume at the end up to target_depth."""
    volume_dhw = np.asarray(volume_dhw, dtype=np.float32)

    if volume_dhw.ndim != 3:
        raise ValueError(f"Expected volume [D,H,W], got {volume_dhw.shape}.")

    depth = volume_dhw.shape[0]
    pad_after = target_depth - depth

    if pad_after < 0:
        raise ValueError(f"target_depth={target_depth} is smaller than depth={depth}.")
    if pad_after == 0:
        return volume_dhw

    return np.pad(
        volume_dhw,
        ((0, pad_after), (0, 0), (0, 0)),
        mode="constant",
        constant_values=float(constant_value),
    )
