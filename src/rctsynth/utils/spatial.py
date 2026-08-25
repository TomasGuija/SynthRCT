"""Spatial warping and full-volume decoding utilities."""

from __future__ import annotations

import numpy as np
import torch

from rctsynth.spatial.transform import SpatialTransform, integrate_svf
from rctsynth.utils.core import _pad_depth_end, model_device, to_torch_5d


@torch.no_grad()
def integrate_full_svf(
    svf_3dhw: np.ndarray,
    device: torch.device | str,
    steps: int = 7,
) -> np.ndarray:
    """Integrate an SVF [3,D,H,W] into a dense displacement field."""
    svf = np.asarray(svf_3dhw, dtype=np.float32)
    device = torch.device(device)

    stn = SpatialTransform(tuple(svf.shape[1:])).to(device)
    svf_t = torch.from_numpy(svf[None]).to(device)

    flow_t = integrate_svf(
        svf_t,
        stn,
        steps=steps,
    )

    return flow_t[0].cpu().numpy()


@torch.no_grad()
def warp_full_volume(
    volume_dhw: np.ndarray,
    flow_3dhw: np.ndarray,
    device: torch.device | str,
    mode: str = "bilinear",
) -> np.ndarray:
    """Warp a volume [D,H,W] using a displacement field [3,D,H,W]."""
    volume = np.asarray(volume_dhw, dtype=np.float32)
    flow = np.asarray(flow_3dhw, dtype=np.float32)
    device = torch.device(device)

    stn = SpatialTransform(volume.shape).to(device)

    volume_t = to_torch_5d(volume, device)
    flow_t = torch.from_numpy(flow[None]).to(device)

    warped_t = stn(
        volume_t,
        flow_t,
        mode=mode,
    )

    return warped_t[0, 0].cpu().numpy()


def _sliding_slab_starts(
    depth: int,
    slab_depth: int,
    stride: int,
) -> tuple[list[int], int]:
    """Return slab starting positions and required padded depth."""
    if depth <= slab_depth:
        return [0], slab_depth

    last_start = int(
        np.ceil((depth - slab_depth) / stride) * stride
    )
    starts = list(range(0, last_start + 1, stride))

    return starts, last_start + slab_depth


@torch.no_grad()
def decode_full_volume_with_refiner(
    model: torch.nn.Module,
    z: np.ndarray | torch.Tensor,
    moving_dhw: np.ndarray,
    *,
    batch_size: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode one latent code into a warped volume, DVF, and SVF."""
    device = model_device(model)
    model.eval()

    moving = np.asarray(moving_dhw, dtype=np.float32)

    if isinstance(z, torch.Tensor):
        z_t = z.to(device=device, dtype=torch.float32)
    else:
        z_t = torch.as_tensor(
            z,
            dtype=torch.float32,
            device=device,
        )

    if z_t.ndim == 1:
        z_t = z_t.unsqueeze(0)

    slab_depth = int(model.slab_depth)
    stride = int(model.resolved_stitch_stride)
    overlap = slab_depth - stride

    original_depth, height, width = moving.shape

    starts, padded_depth = _sliding_slab_starts(
        original_depth,
        slab_depth,
        stride,
    )

    moving_padded = _pad_depth_end(
        moving,
        target_depth=padded_depth,
    )

    raw_slabs = []

    for batch_start in range(0, len(starts), batch_size):
        batch_starts = starts[
            batch_start : batch_start + batch_size
        ]

        slabs = np.stack(
            [
                moving_padded[
                    start : start + slab_depth
                ]
                for start in batch_starts
            ]
        )

        moving_t = torch.from_numpy(
            slabs[:, None]
        ).to(device)

        z_batch = z_t.expand(len(batch_starts), -1)

        raw_batch = model.decode_raw_slab(
            z_batch,
            moving_t,
        )

        raw_slabs.extend(
            raw_batch[index : index + 1]
            for index in range(raw_batch.shape[0])
        )

    svf_full = np.empty(
        (3, padded_depth, height, width),
        dtype=np.float32,
    )

    if len(raw_slabs) == 1:
        svf_full[:, :slab_depth] = (
            raw_slabs[0][0].cpu().numpy()
        )
    else:
        svf_full[:, :stride] = (
            raw_slabs[0][0, :, :stride]
            .cpu()
            .numpy()
        )

        for index in range(len(raw_slabs) - 1):
            grouped = torch.stack(
                [
                    raw_slabs[index],
                    raw_slabs[index + 1],
                ],
                dim=1,
            )

            bridge = moving_padded[
                starts[index] :
                starts[index] + slab_depth + stride
            ]

            bridge_t = torch.from_numpy(
                bridge[None, None]
            ).to(device)

            refined = model._refine_grouped_svf(
                grouped,
                bridge_t,
                model.svf_refiner,
                stride=stride,
            )[0].cpu().numpy()

            overlap_start = starts[index + 1]
            overlap_end = starts[index] + slab_depth

            svf_full[
                :,
                overlap_start:overlap_end,
            ] = refined[
                :,
                stride:slab_depth,
            ]

        last_start = starts[-1]

        svf_full[
            :,
            last_start + overlap :
            last_start + slab_depth,
        ] = (
            raw_slabs[-1][
                0,
                :,
                overlap:slab_depth,
            ]
            .cpu()
            .numpy()
        )

    svf = svf_full[:, :original_depth]

    stn = SpatialTransform(
        (original_depth, height, width)
    ).to(device)

    svf_t = torch.from_numpy(
        svf[None]
    ).to(device)

    flow_t = integrate_svf(
        svf_t,
        stn,
        steps=int(model.svf_steps),
    )

    moving_t = to_torch_5d(
        moving,
        device,
    )

    warped_t = stn(
        moving_t,
        flow_t,
    )

    warped = warped_t[0, 0].cpu().numpy()
    flow = flow_t[0].cpu().numpy()

    return warped, flow, svf