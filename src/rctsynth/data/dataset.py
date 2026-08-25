"""H5 dataset for SynthRCT registration VAE training."""

from __future__ import annotations

import h5py
import numpy as np
from torch.utils.data import Dataset


class H5RegistrationDataset(Dataset):
    """Dataset returning full axial images, two-slab centers, and optional loss masks."""

    def __init__(
        self,
        h5_path: str,
        *,
        slab_size: int = 48,
        stitch_stride: int | None = None,
        num_sampled_regions: int = 1,
        min_scan_distance: int = 3,
        sampling_mode: str = "random",
        case_ids: list[int] | tuple[int, ...] | np.ndarray | None = None,
    ) -> None:
        super().__init__()

        self.h5_path = h5_path
        self.slab_size = slab_size
        self.stitch_stride = (self.slab_size // 2 if stitch_stride is None else stitch_stride)
        self.num_sampled_regions = num_sampled_regions
        self.min_scan_distance = min_scan_distance
        self.sampling_mode = sampling_mode.lower()
        self.included_case_ids = None if case_ids is None else set(case_ids)

        if self.slab_size <= 0 or self.slab_size % 2 != 0:
            raise ValueError("slab_size must be even and > 0.")
        if self.stitch_stride <= 0 or self.stitch_stride >= self.slab_size:
            raise ValueError(
                f"stitch_stride must satisfy 0 < stitch_stride < slab_size. "
                f"Got stitch_stride={self.stitch_stride}, slab_size={self.slab_size}."
            )
        if self.num_sampled_regions < 1:
            raise ValueError("num_sampled_regions must be >= 1.")
        if self.min_scan_distance < 1:
            raise ValueError("min_scan_distance must be >= 1.")
        if self.sampling_mode not in ("random", "deterministic"):
            raise ValueError("sampling_mode must be 'random' or 'deterministic'.")

        self.rng = np.random.default_rng()
        self._fh: h5py.File | None = None
        self._image_cache: dict[int, np.ndarray] = {}
        self._anatomy_cache: dict[int, np.ndarray] = {}

        with self._open_h5() as fh:
            self.case_ids = np.asarray(fh["case_id"][:], dtype=np.int32)
            self.time_ids = np.asarray(fh["time_id"][:], dtype=np.int32)

            valid_axis_start, valid_axis_end = self._read_valid_axis_bounds(fh)
            self.valid_axis_start = valid_axis_start
            self.valid_axis_end = valid_axis_end

            self.pairs = self._build_pair_index(
                self.case_ids,
                self.time_ids,
                valid_axis_start,
                valid_axis_end,
            )

        if self.included_case_ids is None:
            self.dataset_case_ids = sorted(int(x) for x in np.unique(self.case_ids))
        else:
            self.dataset_case_ids = sorted(self.included_case_ids)

    def _open_h5(self) -> h5py.File:
        """Open the H5 file."""
        return h5py.File(self.h5_path, "r")

    def _ensure_open(self) -> None:
        """Open the H5 file lazily inside each worker."""
        if self._fh is None:
            self._fh = self._open_h5()

    @staticmethod
    def _read_valid_axis_bounds(fh: h5py.File) -> tuple[np.ndarray, np.ndarray]:
        """Read valid axial bounds."""
        if "valid_axis_start" not in fh or "valid_axis_end" not in fh:
            raise KeyError(
                f"{fh.filename} is missing 'valid_axis_start'/'valid_axis_end'. "
                "Rebuild or crop the H5 with valid axial bounds."
            )

        return (
            np.asarray(fh["valid_axis_start"][:], dtype=np.int32),
            np.asarray(fh["valid_axis_end"][:], dtype=np.int32),
        )

    def _first_center_bounds(
        self,
        valid_start: int,
        valid_end: int,
    ) -> tuple[int, int] | None:
        """Return valid bounds for the first center in a two-slab region."""
        half = self.slab_size // 2

        center_start = valid_start + half
        center_stop = valid_end - half - self.stitch_stride + 1

        if center_stop <= center_start:
            return None

        return center_start, center_stop

    def _build_pair_index(
        self,
        case_ids: np.ndarray,
        time_ids: np.ndarray,
        valid_axis_start: np.ndarray,
        valid_axis_end: np.ndarray,
    ) -> list[tuple[int, int, int, int]]:
        """Build valid intra-patient moving/fixed pairs."""
        by_case: dict[int, list[int]] = {}
        for idx, case_id in enumerate(case_ids):
            if self.included_case_ids is not None and case_id not in self.included_case_ids:
                continue
            by_case.setdefault(case_id, []).append(idx)

        pairs: list[tuple[int, int, int, int]] = []

        for indices in by_case.values():
            if len(indices) < 2:
                continue

            indices = sorted(indices, key=lambda idx: int(time_ids[idx]))

            for a, moving_idx in enumerate(indices):
                for b in range(a + self.min_scan_distance, len(indices)):
                    fixed_idx = indices[b]

                    shared_start = max(
                        valid_axis_start[moving_idx],
                        valid_axis_start[fixed_idx],
                    )
                    shared_end = min(
                        valid_axis_end[moving_idx],
                        valid_axis_end[fixed_idx],
                    )

                    center_bounds = self._first_center_bounds(shared_start, shared_end)
                    if center_bounds is None:
                        continue

                    center_start, center_stop = center_bounds
                    pairs.append((moving_idx, fixed_idx, center_start, center_stop))

        if not pairs:
            raise ValueError(f"No valid intra-patient CT pairs were found.")

        return pairs

    def __len__(self) -> int:
        """Return number of image pairs."""
        return len(self.pairs)

    def _select_first_centers(
        self,
        center_start: int,
        center_stop: int,
    ) -> np.ndarray:
        """Select first centers for sampled two-slab regions."""
        possible = np.arange(center_start, center_stop, dtype=np.int32)
        bins = np.array_split(possible, self.num_sampled_regions)

        centers = []
        for bin_centers in bins:
            candidates = possible if len(bin_centers) == 0 else bin_centers
            if self.sampling_mode == "random":
                centers.append(candidates[self.rng.integers(len(candidates))])
            else:
                centers.append(candidates[len(candidates) // 2])

        return np.asarray(centers, dtype=np.int32)

    def _load_h5_array(
        self,
        group_name: str,
        idx: int,
        *,
        dtype: np.dtype,
    ) -> np.ndarray:
        """Load one array from an H5 group."""
        self._ensure_open()
        key = f"{idx:07d}"
        return np.asarray(self._fh[group_name][key], dtype=dtype)

    @staticmethod
    def _add_channel(arr: np.ndarray) -> np.ndarray:
        """Convert a DHW array to DHW1."""
        return arr[..., None]

    def _load_image(self, idx: int) -> np.ndarray:
        """Load one full image as DHW1."""
        image = self._image_cache.get(idx)
        if image is None:
            image = self._add_channel(
                self._load_h5_array("images", idx, dtype=np.float32)
            )
            self._image_cache[idx] = image

        return image

    def _load_anatomy(self, idx: int) -> np.ndarray | None:
        """Load one full anatomy mask as DHW."""
        if "anatomy_masks" not in self._fh:
            return None

        anatomy = self._anatomy_cache.get(idx)
        if anatomy is None:
            anatomy = self._load_h5_array("anatomy_masks", idx, dtype=np.uint8)
            self._anatomy_cache[idx] = anatomy

        return anatomy

    def shared_valid_axis_range(
        self,
        moving_idx: int,
        fixed_idx: int,
    ) -> tuple[int, int]:
        """Return the shared valid axial range for one image pair."""
        valid_start = max(
            self.valid_axis_start[moving_idx],
            self.valid_axis_start[fixed_idx],
        )
        valid_end = min(
            self.valid_axis_end[moving_idx],
            self.valid_axis_end[fixed_idx],
        )
        return valid_start, valid_end

    def build_full_loss_mask_dhw(
        self,
        moving_idx: int,
        fixed_idx: int,
    ) -> np.ndarray | None:
        """Build full moving/fixed anatomy-overlap mask as DHW."""
        moving_anatomy = self._load_anatomy(moving_idx)
        fixed_anatomy = self._load_anatomy(fixed_idx)

        if moving_anatomy is None or fixed_anatomy is None:
            return None

        return ((moving_anatomy > 0) & (fixed_anatomy > 0)).astype(np.uint8)

    def __getitem__(self, index: int) -> dict:
        """Return one registration training sample."""
        self._ensure_open()

        idx_a, idx_b, center_start, center_stop = self.pairs[index]

        first_centers = self._select_first_centers(center_start, center_stop)
        centers = np.stack(
            [
                first_centers,
                first_centers + self.stitch_stride,
            ],
            axis=1,
        ).astype(np.int32, copy=False)

        if self.sampling_mode == "deterministic" or self.rng.integers(2) == 0:
            moving_idx, fixed_idx = idx_a, idx_b
        else:
            moving_idx, fixed_idx = idx_b, idx_a

        moving_image = self._load_image(moving_idx)
        fixed_image = self._load_image(fixed_idx)

        loss_mask = self.build_full_loss_mask_dhw(moving_idx, fixed_idx)
        if loss_mask is not None:
            loss_mask = loss_mask[..., None]

        return {
            "moving": moving_image,
            "fixed": fixed_image,
            "center": centers,
            "loss_mask": loss_mask,
        }

    def __getstate__(self) -> dict:
        """Drop open H5 handles and caches before worker pickling."""
        state = self.__dict__.copy()
        state["_fh"] = None
        state["_image_cache"] = {}
        state["_anatomy_cache"] = {}
        return state

    def __del__(self) -> None:
        """Close any open H5 handle."""
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass