"""Shared dataclasses for SynthRCT inference helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DeformationSample:
    """Decoded deformation sample and derived outputs."""
    z: np.ndarray
    warped_dhw: np.ndarray
    flow_3dhw: np.ndarray
    svf_3dhw: np.ndarray


@dataclass
class LatentPCA:
    """Simple PCA representation for latent traversal."""
    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    scores: np.ndarray
