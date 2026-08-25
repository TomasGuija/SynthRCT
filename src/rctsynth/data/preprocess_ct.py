#!/usr/bin/env python3
"""Preprocess one CT NIfTI into SynthRCT-style demo NIfTI files.

This is a lightweight example utility for preparing a single CT volume for
model demos. The generated masks are threshold-based heuristics intended for
cropping and loss support, not clinical segmentations.

Outputs:
  - preprocessed_ct.nii.gz   normalized CT, shape (D, H, W)
  - body_mask.nii.gz         heuristic body/anatomy mask
  - lung_mask.nii.gz         heuristic lung-air mask

Example:
python src/rctsynth/data/preprocess_ct.py \
  --input-nii data/raw/example_ct.nii.gz \
  --output-dir data/demo \
  --target-shape 207 256 256 \
  --spacing-mm 1.5 \
  --norm-mode clip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_to_output
import numpy as np
from scipy import ndimage as ndi


DEFAULT_TARGET_SHAPE = (207, 256, 256)
DEFAULT_SPACING_MM = 1.5
DEFAULT_CLIP_MIN_HU = -1000.0
DEFAULT_CLIP_MAX_HU = 500.0
DEFAULT_BODY_THRESHOLD_HU = -500.0
DEFAULT_LUNG_THRESHOLD_HU = -450.0
DEFAULT_BODY_TABLE_DISCONNECT_ITERS = 2
DEFAULT_BODY_ERODE_ITERS = 2
DEFAULT_LUNG_MIN_COMPONENT_SIZE = 1000
DEFAULT_LUNG_MIN_COMPONENT_FRACTION = 0.05


def load_resampled_ct(
    input_nii: str | Path,
    spacing_mm: float,
) -> np.ndarray:
    """Load, canonicalize, and resample a CT NIfTI to isotropic spacing."""
    img = nib.load(str(input_nii))
    img = nib.as_closest_canonical(img)
    img = resample_to_output(
        img,
        voxel_sizes=(float(spacing_mm), float(spacing_mm), float(spacing_mm)),
        order=1,
    )

    ct_xyz = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    ct_xyz = np.nan_to_num(ct_xyz, nan=-1024.0, posinf=3000.0, neginf=-1024.0)

    return ct_xyz


def xyz_to_dhw(volume_xyz: np.ndarray) -> np.ndarray:
    """Convert NIfTI XYZ array order to model DHW order."""
    return np.transpose(volume_xyz, (2, 1, 0))


def dhw_to_xyz(volume_dhw: np.ndarray) -> np.ndarray:
    """Convert model DHW order back to NIfTI XYZ array order."""
    return np.transpose(volume_dhw, (2, 1, 0))


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected component."""
    mask = np.asarray(mask, dtype=bool)
    labels, n_labels = ndi.label(mask)

    if n_labels == 0:
        return np.zeros_like(mask, dtype=bool)

    counts = np.bincount(labels.reshape(-1))
    counts[0] = 0

    return labels == int(np.argmax(counts))


def fill_holes_by_slice(mask: np.ndarray) -> np.ndarray:
    """Fill in-plane holes independently for each axial slice."""
    mask = np.asarray(mask, dtype=bool).copy()

    for z in range(mask.shape[0]):
        mask[z] = ndi.binary_fill_holes(mask[z])

    return mask


def filter_connected_components_by_size(
    mask: np.ndarray,
    min_size: int,
    min_fraction_of_largest: float,
) -> np.ndarray:
    """Keep components large enough to be plausible anatomy."""
    mask = np.asarray(mask, dtype=bool)
    labels, n_labels = ndi.label(mask)

    if n_labels == 0:
        return np.zeros_like(mask, dtype=bool)

    counts = np.bincount(labels.reshape(-1))
    counts[0] = 0
    largest = int(counts.max())
    threshold = max(min_size, round(largest * min_fraction_of_largest))

    keep = counts >= threshold
    keep[0] = False

    return keep[labels]


def build_body_mask(
    ct_dhw: np.ndarray,
    threshold_hu: float,
    table_disconnect_iters: int,
) -> np.ndarray:
    """Build a heuristic body mask from HU thresholding."""
    threshold_mask = ct_dhw > threshold_hu

    body = threshold_mask
    if table_disconnect_iters > 0:
        disconnected = ndi.binary_erosion(
            threshold_mask,
            iterations=table_disconnect_iters,
        )
        if np.any(disconnected):
            body = largest_connected_component(disconnected)
            body = ndi.binary_dilation(body, iterations=table_disconnect_iters)
            body &= threshold_mask

    body = largest_connected_component(body)
    body = ndi.binary_closing(body, iterations=2)
    body = fill_holes_by_slice(body)
    body = ndi.binary_closing(body, iterations=1)

    return body.astype(bool)


def build_lung_mask(
    ct_dhw: np.ndarray,
    body_mask: np.ndarray,
    lung_threshold_hu: float,
    body_erode_iters: int,
    min_component_size: int,
    min_component_fraction: float,
) -> np.ndarray:
    """Build a heuristic lung-air mask inside the body."""
    body_inner = body_mask

    if int(body_erode_iters) > 0:
        eroded = ndi.binary_erosion(body_mask, iterations=body_erode_iters)
        if np.any(eroded):
            body_inner = eroded

    lung = (ct_dhw < lung_threshold_hu) & body_inner
    lung = filter_connected_components_by_size(
        lung,
        min_size=min_component_size,
        min_fraction_of_largest=min_component_fraction,
    )

    lung = ndi.binary_closing(lung, iterations=2)
    lung = fill_holes_by_slice(lung)
    lung &= body_mask

    return lung.astype(bool)


def center_of_mask(mask: np.ndarray) -> np.ndarray | None:
    """Return mask center of mass as z/y/x."""
    mask = np.asarray(mask, dtype=bool)

    if not np.any(mask):
        return None

    return np.asarray(ndi.center_of_mass(mask), dtype=np.float32)


def choose_crop_center(
    body_mask: np.ndarray,
    lung_mask: np.ndarray,
) -> np.ndarray:
    """Choose z/y/x crop center.

    The axial center comes from the lung mask when available.
    The in-plane center comes from the body mask for stability.
    """
    body_center = center_of_mask(body_mask)
    lung_center = center_of_mask(lung_mask)

    if body_center is None and lung_center is None:
        shape = np.asarray(body_mask.shape, dtype=np.float32)
        return (shape - 1.0) / 2.0

    if body_center is None:
        return lung_center

    if lung_center is None:
        return body_center

    return np.asarray(
        [lung_center[0], body_center[1], body_center[2]],
        dtype=np.float32,
    )


def crop_or_pad_around_center(
    volume: np.ndarray,
    target_shape: tuple[int, int, int],
    center: np.ndarray,
    fill_value: float | int = 0,
) -> np.ndarray:
    """Crop or pad a DHW volume around a z/y/x center."""
    target_shape = tuple(target_shape)
    center = np.asarray(center, dtype=np.float32)

    output = np.full(target_shape, fill_value, dtype=volume.dtype)

    src_slices = []
    dst_slices = []

    for axis, target_size in enumerate(target_shape):
        size = volume.shape[axis]
        center_axis = round(center[axis])

        start = center_axis - target_size // 2
        end = start + target_size

        src_start = max(0, start)
        src_end = min(size, end)

        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)

        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))

    output[tuple(dst_slices)] = volume[tuple(src_slices)]

    return output


def normalize_ct(
    ct_dhw: np.ndarray,
    mode: str,
    clip_min_hu: float,
    clip_max_hu: float,
) -> np.ndarray:
    """Normalize CT to [0, 1]."""
    ct_dhw = np.asarray(ct_dhw, dtype=np.float32)
    mode = mode.lower()

    if mode == "minmax":
        finite = np.isfinite(ct_dhw)
        if not np.any(finite):
            raise ValueError("Input CT contains no finite values.")

        min_val = ct_dhw[finite].min()
        max_val = ct_dhw[finite].max()

        if max_val <= min_val:
            return np.zeros_like(ct_dhw, dtype=np.float32)

        out = (ct_dhw - min_val) / (max_val - min_val)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    if mode == "clip":
        ct_dhw = np.clip(ct_dhw, clip_min_hu, clip_max_hu)
        out = (ct_dhw - clip_min_hu) / (
            clip_max_hu - clip_min_hu
        )
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    raise ValueError(f"Unknown normalization mode: {mode}")


def save_nifti(
    path: str | Path,
    volume_dhw: np.ndarray,
    spacing_mm: float,
    dtype: np.dtype,
) -> None:
    """Save a DHW volume as NIfTI with a simple isotropic affine.

    The saved files are intended as model inputs in canonical DHW order. They
    do not preserve the source image origin or full scanner-space affine.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    volume_xyz = dhw_to_xyz(volume_dhw).astype(dtype)

    affine = np.diag(
        [
            spacing_mm,
            spacing_mm,
            spacing_mm,
            1.0,
        ]
    )

    nib.save(nib.Nifti1Image(volume_xyz, affine), path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess one CT NIfTI into normalized demo inputs and "
            "heuristic body/lung masks."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-nii",
        required=True,
        help="Path to the input CT NIfTI file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where output NIfTI files will be written.",
    )

    parser.add_argument(
        "--spacing-mm",
        type=float,
        default=DEFAULT_SPACING_MM,
        help="Isotropic voxel spacing used for resampling.",
    )

    parser.add_argument(
        "--target-shape",
        nargs=3,
        type=int,
        default=DEFAULT_TARGET_SHAPE,
        metavar=("D", "H", "W"),
        help="Output volume shape in model DHW order.",
    )

    parser.add_argument(
        "--norm-mode",
        choices=["minmax", "clip"],
        default="clip",
        help="Intensity normalization method for the output CT.",
    )
    parser.add_argument(
        "--clip-min-hu",
        type=float,
        default=DEFAULT_CLIP_MIN_HU,
        help="Lower HU bound used when --norm-mode clip.",
    )
    parser.add_argument(
        "--clip-max-hu",
        type=float,
        default=DEFAULT_CLIP_MAX_HU,
        help="Upper HU bound used when --norm-mode clip.",
    )

    parser.add_argument(
        "--body-threshold-hu",
        type=float,
        default=DEFAULT_BODY_THRESHOLD_HU,
        help="HU threshold for the initial body/anatomy candidate mask.",
    )
    parser.add_argument(
        "--lung-threshold-hu",
        type=float,
        default=DEFAULT_LUNG_THRESHOLD_HU,
        help="HU threshold for low-density lung-air candidates.",
    )
    parser.add_argument(
        "--body-table-disconnect-iters",
        type=int,
        default=DEFAULT_BODY_TABLE_DISCONNECT_ITERS,
        help=(
            "Binary erosion iterations before selecting the largest body "
            "component. Increase if the table remains attached; set to 0 to "
            "disable."
        ),
    )
    parser.add_argument(
        "--body-erode-iters",
        type=int,
        default=DEFAULT_BODY_ERODE_ITERS,
        help="Body-mask erosion iterations used before lung-air extraction.",
    )
    parser.add_argument(
        "--lung-min-component-size",
        type=int,
        default=DEFAULT_LUNG_MIN_COMPONENT_SIZE,
        help="Minimum connected-component size retained in the lung mask.",
    )
    parser.add_argument(
        "--lung-min-component-fraction",
        type=float,
        default=DEFAULT_LUNG_MIN_COMPONENT_FRACTION,
        help=(
            "Minimum lung component size as a fraction of the largest lung "
            "candidate component."
        ),
    )

    args = parser.parse_args()

    if args.spacing_mm <= 0:
        raise ValueError("--spacing-mm must be positive.")
    if any(size <= 0 for size in args.target_shape):
        raise ValueError("--target-shape values must be positive.")
    if args.clip_max_hu <= args.clip_min_hu:
        raise ValueError("--clip-max-hu must be greater than --clip-min-hu.")
    if args.body_table_disconnect_iters < 0:
        raise ValueError("--body-table-disconnect-iters must be non-negative.")
    if args.body_erode_iters < 0:
        raise ValueError("--body-erode-iters must be non-negative.")
    if args.lung_min_component_size < 0:
        raise ValueError("--lung-min-component-size must be non-negative.")
    if args.lung_min_component_fraction < 0:
        raise ValueError("--lung-min-component-fraction must be non-negative.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_shape = tuple(args.target_shape)

    ct_xyz = load_resampled_ct(
        args.input_nii,
        spacing_mm=args.spacing_mm,
    )
    ct_dhw = xyz_to_dhw(ct_xyz)

    body_mask = build_body_mask(
        ct_dhw,
        threshold_hu=args.body_threshold_hu,
        table_disconnect_iters=args.body_table_disconnect_iters,
    )
    lung_mask = build_lung_mask(
        ct_dhw,
        body_mask=body_mask,
        lung_threshold_hu=args.lung_threshold_hu,
        body_erode_iters=args.body_erode_iters,
        min_component_size=args.lung_min_component_size,
        min_component_fraction=args.lung_min_component_fraction,
    )

    center = choose_crop_center(
        body_mask=body_mask,
        lung_mask=lung_mask,
    )

    ct_norm = normalize_ct(
        ct_dhw,
        mode=args.norm_mode,
        clip_min_hu=args.clip_min_hu,
        clip_max_hu=args.clip_max_hu,
    )

    ct_crop = crop_or_pad_around_center(
        ct_norm,
        target_shape=target_shape,
        center=center,
        fill_value=0.0,
    )

    body_crop = crop_or_pad_around_center(
        body_mask.astype(np.uint8),
        target_shape=target_shape,
        center=center,
        fill_value=0,
    )

    lung_crop = crop_or_pad_around_center(
        lung_mask.astype(np.uint8),
        target_shape=target_shape,
        center=center,
        fill_value=0,
    )

    save_nifti(
        output_dir / "preprocessed_ct.nii.gz",
        ct_crop,
        spacing_mm=args.spacing_mm,
        dtype=np.float32,
    )
    save_nifti(
        output_dir / "body_mask.nii.gz",
        body_crop,
        spacing_mm=args.spacing_mm,
        dtype=np.uint8,
    )
    save_nifti(
        output_dir / "lung_mask.nii.gz",
        lung_crop,
        spacing_mm=args.spacing_mm,
        dtype=np.uint8,
    )

    print(f"Saved: {output_dir / 'preprocessed_ct.nii.gz'}")
    print(f"Saved: {output_dir / 'body_mask.nii.gz'}")
    print(f"Saved: {output_dir / 'lung_mask.nii.gz'}")
    print(f"Resampled shape DHW: {tuple(ct_dhw.shape)}")
    print(f"Output shape DHW: {tuple(ct_crop.shape)}")
    print(f"Crop center DHW: {tuple(round(x, 2) for x in center)}")
    print(f"CT range: [{ct_crop.min():.4f}, {ct_crop.max():.4f}]")
    print(f"Body voxels: {body_crop.sum()}")
    print(f"Lung voxels: {lung_crop.sum()}")


if __name__ == "__main__":
    main()
