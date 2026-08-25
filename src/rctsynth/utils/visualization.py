"""2D visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def mid_depth_slice(volume_dhw: np.ndarray) -> np.ndarray:
    """Return the center axial slice from a [D,H,W] volume."""
    volume_dhw = np.asarray(volume_dhw)

    if volume_dhw.ndim != 3:
        raise ValueError(f"Expected volume [D,H,W], got {volume_dhw.shape}.")

    return volume_dhw[volume_dhw.shape[0] // 2]


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """Robust min-max normalization for visualization."""
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(image, [1, 99])

    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)

    image = np.clip(image, lo, hi)
    return (image - lo) / (hi - lo)


def save_preview(
    output_path: Path,
    moving_dhw: np.ndarray,
    warped_dhw: np.ndarray | None,
    fixed_dhw: np.ndarray | None = None,
) -> None:
    """Save a simple axial center-slice preview."""
    panels: list[tuple[str, np.ndarray]] = [("Moving", mid_depth_slice(moving_dhw))]

    if warped_dhw is not None:
        panels.append(("Generated", mid_depth_slice(warped_dhw)))

    if fixed_dhw is not None:
        panels.append(("Reference", mid_depth_slice(fixed_dhw)))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))

    if len(panels) == 1:
        axes = [axes]

    for ax, (title, image) in zip(axes, panels):
        ax.imshow(normalize_for_display(image), cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def field_magnitude(field_3dhw: np.ndarray) -> np.ndarray:
    """Return voxelwise vector magnitude for a [3,D,H,W] field."""
    field_3dhw = np.asarray(field_3dhw, dtype=np.float32)

    if field_3dhw.ndim != 4 or field_3dhw.shape[0] != 3:
        raise ValueError(f"Expected field [3,D,H,W], got {field_3dhw.shape}.")

    return np.sqrt(np.sum(field_3dhw**2, axis=0)).astype(np.float32, copy=False)
