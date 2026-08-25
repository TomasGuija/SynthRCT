"""Torch helpers for stitching grouped image slabs."""

from __future__ import annotations

import torch


def resolve_stride(slab_depth: int, stride: int | None) -> int:
    """Resolve the axial stride between two overlapping slabs."""
    slab_depth = int(slab_depth)

    if slab_depth <= 0:
        raise ValueError(f"slab_depth must be positive, got {slab_depth}.")

    if stride is None:
        return slab_depth // 2

    stride = int(stride)

    if stride <= 0 or stride >= slab_depth:
        raise ValueError(
            f"stride must satisfy 0 < stride < slab_depth; "
            f"got stride={stride}, slab_depth={slab_depth}."
        )

    return stride


def stitch_torch_slabs(
    slabs: torch.Tensor,
    *,
    stride: int | None = None,
) -> torch.Tensor:
    """Stitch two overlapping image slabs.

    Parameters
    ----------
    slabs:
        Tensor with shape ``(B, 2, C, D, H, W)``.
    stride:
        Axial offset between the two slabs. If ``None``, half-overlap is used.

    Returns
    -------
    torch.Tensor
        Stitched tensor with shape ``(B, C, D + stride, H, W)``.
    """
    if slabs.ndim != 6:
        raise ValueError(f"Expected slabs [B,2,C,D,H,W], got {tuple(slabs.shape)}.")

    if slabs.shape[1] != 2:
        raise ValueError(f"Expected exactly two slabs, got {slabs.shape[1]}.")

    depth = int(slabs.shape[3])
    stride = resolve_stride(depth, stride)
    overlap = depth - stride

    left = slabs[:, 0]
    right = slabs[:, 1]

    return torch.cat(
        [
            left,
            right[:, :, overlap:],
        ],
        dim=2,
    )