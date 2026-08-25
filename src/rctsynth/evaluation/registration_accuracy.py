"""
Evaluate SynthRCT registration accuracy on selected H5 cases.

Metrics:
  - LNCC
  - RMSE

Example:
python -m rctsynth.evaluation.registration_accuracy \
  --ckpt checkpoints/best.ckpt \
  --h5-path data/DIR4DCT.h5 \
  --case-ids 0 1 2 \
  --output-json outputs/eval_accuracy.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from rctsynth.utils.core import resolve_device, to_torch_5d
from rctsynth.utils.latent import encode_pair_z
from rctsynth.utils.model_loading import load_vae
from rctsynth.utils.spatial import decode_full_volume_with_refiner


# ---------------------------------------------------------------------
# H5 helpers
# ---------------------------------------------------------------------
def h5_key(idx: int) -> str:
    return f"{idx:07d}"


def load_image(fh: Any, idx: int) -> np.ndarray:
    """Load one normalized image as DHW float32."""
    return np.asarray(fh["images"][h5_key(idx)], dtype=np.float32)


def load_anatomy_mask(fh: Any, idx: int) -> np.ndarray | None:
    """Load one optional anatomy mask as DHW bool."""
    if "anatomy_masks" not in fh:
        return None

    return np.asarray(fh["anatomy_masks"][h5_key(idx)], dtype=bool)


def build_loss_mask(
    fh: Any,
    moving_idx: int,
    fixed_idx: int,
) -> np.ndarray | None:
    """Build anatomy-overlap mask used for image metrics."""
    moving_mask = load_anatomy_mask(fh, moving_idx)
    fixed_mask = load_anatomy_mask(fh, fixed_idx)

    if moving_mask is None or fixed_mask is None:
        return None

    return moving_mask & fixed_mask


def shared_valid_range(
    fh: Any,
    moving_idx: int,
    fixed_idx: int,
) -> tuple[int, int]:
    """Return shared valid axial range for one pair."""
    valid_start = max(
        fh["valid_axis_start"][moving_idx],
        fh["valid_axis_start"][fixed_idx],
    )
    valid_end = min(
        fh["valid_axis_end"][moving_idx],
        fh["valid_axis_end"][fixed_idx],
    )

    return valid_start, valid_end


def build_ordered_pairs(
    case_ids: np.ndarray,
    time_ids: np.ndarray,
    selected_case_ids: list[int],
    min_scan_distance: int,
) -> list[tuple[int, int, int]]:
    """Build ordered intra-patient moving/fixed pairs."""
    selected_case_ids = set(selected_case_ids)

    by_case: dict[int, list[int]] = {}
    for idx, case_id in enumerate(case_ids):
        if case_id not in selected_case_ids:
            continue
        by_case.setdefault(case_id, []).append(idx)

    pairs: list[tuple[int, int, int]] = []

    for case_id, indices in by_case.items():
        indices = sorted(indices, key=lambda idx: time_ids[idx])

        for a, moving_idx in enumerate(indices):
            for b, fixed_idx in enumerate(indices):
                if a == b:
                    continue
                if abs(b - a) < int(min_scan_distance):
                    continue

                pairs.append((moving_idx, fixed_idx, case_id))

    if not pairs:
        raise ValueError(
            f"No valid ordered pairs found for case_ids={sorted(selected_case_ids)}."
        )

    return pairs


# ---------------------------------------------------------------------
# Image metrics
# ---------------------------------------------------------------------
def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean over finite values inside an optional mask."""
    finite = torch.isfinite(values)

    if mask is None:
        return (
            values[finite].mean()
            if torch.any(finite)
            else values.new_tensor(float("nan"))
        )

    if mask.shape != values.shape:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} does not match values shape "
            f"{tuple(values.shape)}."
        )

    valid = (mask > 0.5) & finite

    return (
        values[valid].mean()
        if torch.any(valid)
        else values.new_tensor(float("nan"))
    )


def same_pad3d(x: torch.Tensor, window: int) -> torch.Tensor:
    """Pad a 3D tensor so conv3d keeps the original spatial size."""
    total = window - 1
    left = total // 2
    right = total - left

    return F.pad(x, (left, right, left, right, left, right))


def lncc_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor | None = None,
    window: int = 9,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute masked local normalized cross-correlation."""
    filt = torch.ones(
        1,
        1,
        window,
        window,
        window,
        device=x.device,
        dtype=x.dtype,
    )
    n = float(window**3)

    x_sum = F.conv3d(same_pad3d(x, window), filt)
    y_sum = F.conv3d(same_pad3d(y, window), filt)
    x2_sum = F.conv3d(same_pad3d(x * x, window), filt)
    y2_sum = F.conv3d(same_pad3d(y * y, window), filt)
    xy_sum = F.conv3d(same_pad3d(x * y, window), filt)

    cross = xy_sum - x_sum * y_sum / n
    x_var = x2_sum - x_sum * x_sum / n
    y_var = y2_sum - y_sum * y_sum / n

    score = cross / torch.sqrt(
        x_var.clamp_min(0.0) * y_var.clamp_min(0.0) + eps
    )

    return masked_mean(score, mask)


def rmse_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute masked RMSE."""
    return torch.sqrt(masked_mean((x - y).pow(2), mask).clamp_min(0.0))


@torch.no_grad()
def compute_image_metrics(
    moving: np.ndarray,
    fixed: np.ndarray,
    warped: np.ndarray,
    mask: np.ndarray | None,
    device: torch.device,
) -> dict[str, float]:
    """Compute identity and registered LNCC/RMSE."""
    moving_t = to_torch_5d(moving, device)
    fixed_t = to_torch_5d(fixed, device)
    warped_t = to_torch_5d(warped, device)
    mask_t = None if mask is None else to_torch_5d(mask.astype(np.float32), device)

    lncc_identity = lncc_torch(moving_t, fixed_t, mask=mask_t)
    lncc_registered = lncc_torch(warped_t, fixed_t, mask=mask_t)

    rmse_identity = rmse_torch(moving_t, fixed_t, mask=mask_t)
    rmse_registered = rmse_torch(warped_t, fixed_t, mask=mask_t)

    return {
        "lncc_identity": float(lncc_identity.detach().cpu()),
        "lncc_registered": float(lncc_registered.detach().cpu()),
        "rmse_identity": float(rmse_identity.detach().cpu()),
        "rmse_registered": float(rmse_registered.detach().cpu()),
    }


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
def summarize(rows: list[dict]) -> dict:
    """Summarize pair-level metrics."""
    metrics = ("lncc", "rmse")
    output = {}

    for metric in metrics:
        identity_key = f"{metric}_identity"
        registered_key = f"{metric}_registered"

        identity = np.asarray([row[identity_key] for row in rows], dtype=np.float64)
        registered = np.asarray([row[registered_key] for row in rows], dtype=np.float64)

        finite = np.isfinite(identity) & np.isfinite(registered)

        identity = identity[finite]
        registered = registered[finite]

        output[metric] = {
            "identity_mean": float(identity.mean()) if identity.size else float("nan"),
            "identity_std": float(identity.std(ddof=1)) if identity.size > 1 else 0.0,
            "registered_mean": float(registered.mean()) if registered.size else float("nan"),
            "registered_std": float(registered.std(ddof=1)) if registered.size > 1 else 0.0,
        }

    return output


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    import h5py

    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--case-ids", nargs="+", type=int, required=True)
    parser.add_argument("--output-json", default="eval_accuracy.json")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decode-batch-size", type=int, default=1)
    parser.add_argument("--min-scan-distance", type=int, default=2)

    args = parser.parse_args()

    device = resolve_device(args.device)
    case_ids_requested = sorted({case_id for case_id in args.case_ids})

    _, model = load_vae(
        ckpt_path=args.ckpt,
        device=device,
    )
    slab_depth = int(model.slab_depth)
    stitch_stride = int(model.resolved_stitch_stride)

    with h5py.File(args.h5_path, "r") as fh:
        h5_case_ids = np.asarray(fh["case_id"][:], dtype=np.int32)
        h5_time_ids = np.asarray(fh["time_id"][:], dtype=np.int32)
        has_anatomy_masks = "anatomy_masks" in fh

        available_case_ids = sorted(x for x in np.unique(h5_case_ids))
        unknown_case_ids = sorted(set(case_ids_requested) - set(available_case_ids))

        if unknown_case_ids:
            raise ValueError(
                f"Requested case IDs {unknown_case_ids} are not present in "
                f"{args.h5_path}. Available case IDs: {available_case_ids}."
            )

        pairs = build_ordered_pairs(
            case_ids=h5_case_ids,
            time_ids=h5_time_ids,
            selected_case_ids=case_ids_requested,
            min_scan_distance=args.min_scan_distance,
        )

    rows = []

    with h5py.File(args.h5_path, "r") as fh:
        for moving_idx, fixed_idx, case_id in tqdm(pairs, desc="Evaluating pairs"):
            moving_idx = moving_idx
            fixed_idx = fixed_idx
            case_id = case_id

            valid_start, valid_end = shared_valid_range(fh, moving_idx, fixed_idx)

            moving_full = load_image(fh, moving_idx)
            fixed_full = load_image(fh, fixed_idx)

            moving_valid = moving_full[valid_start:valid_end]
            fixed_valid = fixed_full[valid_start:valid_end]

            loss_mask_full = build_loss_mask(fh, moving_idx, fixed_idx)
            loss_mask_valid = (
                None
                if loss_mask_full is None
                else loss_mask_full[valid_start:valid_end]
            )

            z = encode_pair_z(
                model=model,
                fixed_full_dhw=fixed_full,
                moving_full_dhw=moving_full,
                device=device,
            )

            warped_valid, _, _ = decode_full_volume_with_refiner(
                model=model,
                z=z,
                moving_full_dhw=moving_valid,
                device=device,
                batch_size=max(1, int(args.decode_batch_size)),
            )

            warped_valid = warped_valid[: moving_valid.shape[0]]

            image_metrics = compute_image_metrics(
                moving=moving_valid,
                fixed=fixed_valid,
                warped=warped_valid,
                mask=loss_mask_valid,
                device=device,
            )

            rows.append(
                {
                    "case_id": case_id,
                    "moving_idx": moving_idx,
                    "fixed_idx": fixed_idx,
                    "valid_start": int(valid_start),
                    "valid_end": int(valid_end),
                    **image_metrics,
                }
            )

    report = {
        "config": {
            "ckpt": args.ckpt,
            "h5_path": args.h5_path,
            "case_ids": case_ids_requested,
            "min_scan_distance": int(args.min_scan_distance),
            "slab_size": int(slab_depth),
            "stitch_stride": int(stitch_stride),
            "metrics": ["lncc", "rmse"],
            "metric_mask": "anatomy_masks_overlap" if has_anatomy_masks else None,
            "n_pairs": len(rows),
        },
        "summary": summarize(rows),
        "pairs": rows,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report["summary"], indent=2))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
