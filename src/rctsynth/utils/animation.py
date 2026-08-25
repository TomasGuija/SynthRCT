"""GIF, mask warping, and 3D Slicer rendering helpers."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps

from rctsynth.utils.io import save_nifti_dhw
from rctsynth.utils.spatial import warp_full_volume
from rctsynth.utils.types import DeformationSample


# -----------------------------------------------------------------------------
# Optional dependencies
# -----------------------------------------------------------------------------

def _require_pillow():
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Animation export requires Pillow. Install it with: pip install pillow"
        ) from exc
    return Image


# -----------------------------------------------------------------------------
# 2D volume animations
# -----------------------------------------------------------------------------

_PLANE_TO_AXIS = {
    "axial": 0,
    "coronal": 1,
    "sagittal": 2,
}


def get_volume_plane(
    volume_dhw: np.ndarray,
    plane: str = "axial",
    index: int | None = None,
) -> np.ndarray:
    """Extract one axial, coronal, or sagittal image from a DHW volume.
    """
    volume = np.asarray(volume_dhw)

    if volume.ndim != 3:
        raise ValueError(f"Expected volume [D,H,W], got {volume.shape}.")

    plane = str(plane).strip().lower()
    axis = _PLANE_TO_AXIS.get(plane)
    if axis is None:
        valid = ", ".join(repr(name) for name in _PLANE_TO_AXIS)
        raise ValueError(f"plane must be one of: {valid}.")

    size = volume.shape[axis]
    resolved_index = size // 2 if index is None or index < 0 else index

    if resolved_index >= size:
        raise IndexError(
            f"index={resolved_index} is outside the {plane} axis with size {size}."
        )

    if plane == "axial":
        image = volume[resolved_index, :, :]
    elif plane == "coronal":
        image = np.flipud(volume[:, resolved_index, :])
    else:
        image = volume[:, :, resolved_index]

    return np.asarray(image)


def _display_limits(
    images: Sequence[np.ndarray],
    percentiles: tuple[float, float] = (1.0, 99.0),
) -> tuple[float, float]:
    """Return shared percentile limits across all images."""
    values = np.concatenate([image.ravel() for image in images])
    vmin, vmax = np.percentile(values, percentiles)

    if vmax <= vmin:
        vmax = vmin + 1.0

    return float(vmin), float(vmax)


def _image_to_rgb(
    image: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    cmap: str,
) -> np.ndarray:
    """Convert a 2D scalar image directly to an RGB uint8 frame."""
    image = np.asarray(image, dtype=np.float32)

    if vmax <= vmin:
        raise ValueError(f"vmax must be greater than vmin, got {vmin=} and {vmax=}.")

    normalized = np.nan_to_num(
        (image - float(vmin)) / (float(vmax) - float(vmin)),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    normalized = np.clip(normalized, 0.0, 1.0)

    try:
        rgba = colormaps[str(cmap)](normalized, bytes=True)
    except KeyError as exc:
        raise ValueError(f"Unknown Matplotlib colormap: {cmap!r}.") from exc

    return np.asarray(rgba[..., :3], dtype=np.uint8)


def _ping_pong_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Return a forward-and-back sequence without repeating both endpoints."""
    if len(frames) <= 2:
        return frames
    return frames + frames[-2:0:-1]


def save_rgb_frames_gif(
    frames: Sequence[np.ndarray],
    output_path: str | Path,
    *,
    fps: float = 5.0,
) -> Path:
    """Save NumPy frames as a looping ping-pong GIF."""
    Image = _require_pillow()

    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")
    if not frames:
        raise ValueError("At least one frame is required.")

    arrays = _ping_pong_frames(
        [np.asarray(frame, dtype=np.uint8) for frame in frames]
    )

    pillow_frames = []
    for frame in arrays:
        if frame.ndim == 2:
            image = Image.fromarray(frame).convert("RGB")
        elif frame.ndim == 3 and frame.shape[-1] in (3, 4):
            image = Image.fromarray(frame).convert("RGB")
        else:
            raise ValueError(
                f"Expected frame [H,W], [H,W,3], or [H,W,4], got {frame.shape}."
            )

        pillow_frames.append(image)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first, *remaining = pillow_frames
    first.save(
        output_path,
        save_all=True,
        append_images=remaining,
        duration=round(1000 / fps),
        loop=0,
        disposal=1,
        optimize=False,
    )

    return output_path


def save_volume_gif(
    volumes_dhw: Sequence[np.ndarray],
    output_path: str | Path,
    *,
    plane: str = "axial",
    slice_index: int | None = None,
    fps: float = 5.0,
    percentiles: tuple[float, float] = (1.0, 99.0),
    cmap: str = "gray",
) -> Path:
    """Create a consistently windowed looping ping-pong GIF."""
    volumes = [
        np.asarray(volume, dtype=np.float32)
        for volume in volumes_dhw
    ]

    if not volumes:
        raise ValueError("At least one volume is required.")

    reference_shape = volumes[0].shape
    if len(reference_shape) != 3:
        raise ValueError(
            f"Expected volumes [D,H,W], got {reference_shape}."
        )
    if any(volume.shape != reference_shape for volume in volumes):
        raise ValueError("All volumes must have the same shape.")

    planes = [
        get_volume_plane(
            volume,
            plane=plane,
            index=slice_index,
        )
        for volume in volumes
    ]

    vmin, vmax = _display_limits(
        planes,
        percentiles=percentiles,
    )

    frames = [
        _image_to_rgb(
            image,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        for image in planes
    ]

    return save_rgb_frames_gif(
        frames,
        output_path,
        fps=fps,
    )


# -----------------------------------------------------------------------------
# Mask warping and export
# -----------------------------------------------------------------------------


def save_warped_mask_sequence(
    mask_dhw: np.ndarray,
    flows_3dhw: Sequence[np.ndarray],
    output_dir: str | Path,
    affine: np.ndarray,
    *,
    device: torch.device | str = "cuda",
    prefix: str = "lung",
) -> list[Path]:
    """Warp one source mask with each flow and save the resulting NIfTI sequence."""
    flows = list(flows_3dhw)
    if not flows:
        raise ValueError("At least one deformation field is required.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for step, flow in enumerate(flows):
        warped_mask = warp_full_volume(
            volume_dhw=mask_dhw.astype(np.float32),
            flow_3dhw=np.asarray(flow, dtype=np.float32),
            device=device,
            mode="bilinear",
        ) > 0.5
        path = output_dir / f"{prefix}_{step:03d}.nii.gz"
        save_nifti_dhw(
            warped_mask.astype(np.float32, copy=False),
            path,
            affine,
        )
        paths.append(path)

    return paths


# -----------------------------------------------------------------------------
# 3D Slicer rendering
# -----------------------------------------------------------------------------


SLICER_RENDER_SCRIPT = Path(__file__).with_name("slicer_render.py")


def _resolve_slicer_executable(
    slicer_executable: str | Path,
) -> str:
    """Resolve the 3D Slicer executable."""
    path = Path(slicer_executable)

    if path.is_file():
        return str(path)

    executable = shutil.which(str(slicer_executable))
    if executable is not None:
        return executable

    raise FileNotFoundError(
        f"Could not find 3D Slicer executable: {slicer_executable}"
    )


def _hex_to_rgb01(value: str) -> list[float]:
    """Convert '#RRGGBB' or 'RRGGBB' to three floats in [0, 1]."""
    text = value.strip().lstrip("#")

    if len(text) != 6:
        raise ValueError(f"Expected a six-digit hex color, got {value!r}.")

    try:
        return [
            int(text[index:index + 2], 16) / 255.0
            for index in (0, 2, 4)
        ]
    except ValueError as exc:
        raise ValueError(f"Invalid hex color: {value!r}.") from exc


def render_masks_with_slicer(
    mask_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    slicer_executable: str | Path,
) -> list[Path]:
    """Render binary masks as fixed-style 3D Slicer screenshots."""
    masks = [Path(path).resolve() for path in mask_paths]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    png_paths = [
        output_dir / f"frame_{index:03d}.png"
        for index in range(len(masks))
    ]

    payload = {
        "common": {
            "camera_view": "oblique",
            "color": _hex_to_rgb01("#4388D6"),
            "background": _hex_to_rgb01("#F4F9FD"),
            "opacity": 1.0,
            "width": 800,
            "height": 800,
            "scale": 1,
            "zoom": 1.2,
            "smooth_iterations": 25,
            "pass_band": 0.08,
            "ambient": 0.25,
            "diffuse": 0.75,
            "specular": 0.18,
            "specular_power": 20,
        },
        "jobs": [
            {
                "mask": str(mask),
                "png": str(png.resolve()),
            }
            for mask, png in zip(masks, png_paths)
        ],
    }

    jobs_path = output_dir / "slicer_jobs.json"
    jobs_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    executable = _resolve_slicer_executable(
        slicer_executable
    )

    subprocess.run(
        [
            executable,
            "--no-splash",
            "--python-script",
            str(SLICER_RENDER_SCRIPT),
            str(jobs_path.resolve()),
        ],
        check=True,
    )

    return png_paths


# -----------------------------------------------------------------------------
# Rendered PNG post-processing and 3D GIF assembly
# -----------------------------------------------------------------------------

def crop_png_sequence_to_common_bbox(
    png_paths: Sequence[str | Path],
) -> list[Path]:
    """Crop all PNG frames using one shared foreground bounding box."""
    Image = _require_pillow()

    paths = [Path(path) for path in png_paths]
    background_rgb = np.array([255, 255, 255], dtype=np.int16)

    boxes = []
    image_size = None

    for path in paths:
        with Image.open(path) as image:
            rgba = np.asarray(image.convert("RGBA"))
            image_size = image_size or image.size

        rgb = rgba[..., :3].astype(np.int16)
        foreground = np.max(
            np.abs(rgb - background_rgb),
            axis=-1,
        ) > 8

        if np.any(foreground):
            rows, columns = np.where(foreground)
            boxes.append(
                (
                    columns.min(),
                    rows.min(),
                    columns.max() + 1,
                    rows.max() + 1,
                )
            )

    if not boxes:
        return paths

    margin = 30
    width, height = image_size

    common_box = (
        max(0, min(box[0] for box in boxes) - margin),
        max(0, min(box[1] for box in boxes) - margin),
        min(width, max(box[2] for box in boxes) + margin),
        min(height, max(box[3] for box in boxes) + margin),
    )

    for path in paths:
        with Image.open(path) as image:
            image.crop(common_box).save(path)

    return paths


def save_png_sequence_gif(
    png_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    fps: float = 5.0,
) -> Path:
    """Assemble PNG files into a looping ping-pong GIF."""
    Image = _require_pillow()

    frames = []
    for path in png_paths:
        with Image.open(path) as image:
            frames.append(
                np.asarray(image.convert("RGB"), dtype=np.uint8)
            )

    return save_rgb_frames_gif(
        frames,
        output_path,
        fps=fps,
    )


def create_slicer_lung_gif(
    mask_dhw: np.ndarray,
    samples: Sequence[DeformationSample],
    output_dir: str | Path,
    affine: np.ndarray,
    *,
    device: torch.device | str = "cuda",
    slicer_executable: str | Path,
    gif_name: str = "lung_3d.gif",
    fps: float = 5.0,
) -> Path:
    """Warp a lung mask, render it in Slicer, and assemble a 3D GIF."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_paths = save_warped_mask_sequence(
        mask_dhw=mask_dhw,
        flows_3dhw=[sample.flow_3dhw for sample in samples],
        output_dir=output_dir / "masks",
        affine=affine,
        device=device,
        prefix="lung",
    )

    png_paths = render_masks_with_slicer(
        mask_paths=mask_paths,
        output_dir=output_dir / "frames",
        slicer_executable=slicer_executable,
    )

    crop_png_sequence_to_common_bbox(png_paths)

    return save_png_sequence_gif(
        png_paths,
        output_dir / gif_name,
        fps=fps,
    )