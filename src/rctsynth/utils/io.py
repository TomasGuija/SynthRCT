"""NIfTI I/O helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def load_nifti_dhw(
    path: str | Path,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a canonical NIfTI XYZ array as a model-order DHW volume."""
    img = nib.as_closest_canonical(nib.load(str(path)))
    data_xyz = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)

    if data_xyz.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {data_xyz.shape} for {path}.")

    data_dhw = np.transpose(data_xyz, (2, 1, 0))

    if normalize:
        lo = float(np.min(data_dhw))
        hi = float(np.max(data_dhw))
        if hi > lo:
            data_dhw = (data_dhw - lo) / (hi - lo)
        else:
            data_dhw = np.zeros_like(data_dhw, dtype=np.float32)

    return (
        data_dhw.astype(np.float32, copy=False),
        np.asarray(img.affine, dtype=np.float32),
    )


def save_nifti_dhw(
    volume_dhw: np.ndarray,
    output_path: str | Path,
    affine: np.ndarray,
) -> None:
    """Save a model-order DHW volume as a NIfTI XYZ array."""
    volume_dhw = np.asarray(volume_dhw, dtype=np.float32)

    if volume_dhw.ndim != 3:
        raise ValueError(f"Expected [D,H,W], got {volume_dhw.shape}.")

    volume_xyz = np.transpose(volume_dhw, (2, 1, 0))
    img = nib.Nifti1Image(volume_xyz, affine=np.asarray(affine, dtype=np.float32))
    nib.save(img, str(output_path))


def save_flow_nifti(
    flow_3dhw: np.ndarray,
    output_path: str | Path,
    affine: np.ndarray,
) -> None:
    """Save a model-order ZYX field as an XYZ NIfTI with XYZ channels last."""
    flow_3dhw = np.asarray(flow_3dhw, dtype=np.float32)

    if flow_3dhw.ndim != 4 or flow_3dhw.shape[0] != 3:
        raise ValueError(f"Expected [3,D,H,W], got {flow_3dhw.shape}.")

    flow_xyz3 = np.transpose(flow_3dhw, (3, 2, 1, 0))[..., ::-1].copy()
    img = nib.Nifti1Image(flow_xyz3, affine=np.asarray(affine, dtype=np.float32))
    nib.save(img, str(output_path))


def save_nifti_outputs(
    output_dir: str | Path,
    affine: np.ndarray,
    warped_dhw: np.ndarray | None = None,
    flow_3dhw: np.ndarray | None = None,
    svf_3dhw: np.ndarray | None = None,
    prefix: str = "sample",
) -> None:
    """Convenience helper to save warped image / DVF / SVF."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if warped_dhw is not None:
        save_nifti_dhw(warped_dhw, output_dir / f"{prefix}_warped.nii.gz", affine)

    if flow_3dhw is not None:
        save_flow_nifti(flow_3dhw, output_dir / f"{prefix}_dvf.nii.gz", affine)

    if svf_3dhw is not None:
        save_flow_nifti(svf_3dhw, output_dir / f"{prefix}_svf.nii.gz", affine)
