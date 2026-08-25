import torch
from torch import nn
import torch.nn.functional as F


class SpatialTransform(nn.Module):
    """
        This implementation was taken from:
        https://github.com/voxelmorph/voxelmorph/blob/master/voxelmorph/torch/layers.py
    """

    def __init__(self, size):
        super(SpatialTransform, self).__init__()
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow, mode="bilinear"):
        new_locs = self.grid + flow

        shape = flow.shape[2:]

        for i in range(len(shape)):
            if shape[i] > 1:
                new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
            else:
                new_locs[:, i, ...] = 0.0

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=mode, align_corners=True, padding_mode="border")  # nearest is slower



def integrate_svf(v, stn, steps=7):
    """
    Scaling & squaring integration of a stationary velocity field (SVF).

    v:   [B,3,D,H,W] velocity
    stn: SpatialTransform instance (used to warp vector fields)
    returns: [B,3,D,H,W] DVF displacement field corresponding to exp(v)
    """
    if steps <= 0:
        return v

    flow = v / (2 ** steps)
    for _ in range(steps):
        flow = flow + stn(flow, flow, mode="bilinear")
    return flow