"""Reference-to-target phase dataset for conditional SynthRCT training."""

from __future__ import annotations

import numpy as np

from rctsynth.data.dataset import H5RegistrationDataset


class H5ReferencePhaseDataset(H5RegistrationDataset):
    """Use one fixed source phase and sample only the target phase."""

    def __init__(self, *args, reference_time_id: int = 7, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reference_time_id = int(reference_time_id)

        with self._open_h5() as fh:
            if "phase_fraction" not in fh:
                raise KeyError(f"{fh.filename} is missing 'phase_fraction'.")
            self.phase_fractions = np.asarray(fh["phase_fraction"][:], dtype=np.float32)

        if self.phase_fractions.shape != self.case_ids.shape:
            raise ValueError(
                "phase_fraction must contain one value per image; "
                f"got {len(self.phase_fractions)} values for "
                f"{len(self.case_ids)} images."
            )
        if np.any((self.phase_fractions < 0.0) | (self.phase_fractions >= 1.0)):
            raise ValueError("phase_fraction values must be in [0, 1).")

        case_samples = {}
        deterministic_samples = []
        for case_id in self.dataset_case_ids:
            indices = np.flatnonzero(self.case_ids == case_id).tolist()
            references = [
                idx
                for idx in indices
                if int(self.time_ids[idx]) == self.reference_time_id
            ]
            if len(references) != 1:
                raise ValueError(
                    f"Case {case_id} must have exactly one "
                    f"time_id={self.reference_time_id}; "
                    f"found {len(references)}."
                )

            reference_idx = references[0]
            valid_targets = []
            for target_idx in indices:
                valid_start, valid_end = self.shared_valid_axis_range(
                    reference_idx,
                    target_idx,
                )
                if self._first_center_bounds(valid_start, valid_end) is not None:
                    valid_targets.append(target_idx)

            if not valid_targets:
                raise ValueError(f"Case {case_id} has no valid target phases.")

            target_indices = tuple(valid_targets)
            case_samples[case_id] = (reference_idx, target_indices)
            deterministic_samples.extend(
                (case_id, target_idx) for target_idx in target_indices
            )

        self.case_samples = case_samples
        self.deterministic_samples = deterministic_samples

    def __len__(self) -> int:
        return len(self.deterministic_samples)

    def __getitem__(self, index: int) -> dict:
        """Return a fixed-reference image and a sampled target phase."""
        self._ensure_open()
        case_id, deterministic_target = self.deterministic_samples[index]
        moving_idx, target_indices = self.case_samples[case_id]
        if self.sampling_mode == "random":
            fixed_idx = target_indices[self.rng.integers(len(target_indices))]
        else:
            fixed_idx = deterministic_target

        valid_start, valid_end = self.shared_valid_axis_range(moving_idx, fixed_idx)
        center_bounds = self._first_center_bounds(valid_start, valid_end)
        center_start, center_stop = center_bounds
        first_centers = self._select_first_centers(center_start, center_stop)
        centers = np.stack(
            [first_centers, first_centers + self.stitch_stride],
            axis=1,
        ).astype(np.int32, copy=False)

        loss_mask = self.build_full_loss_mask_dhw(moving_idx, fixed_idx)
        if loss_mask is not None:
            loss_mask = loss_mask[..., None]

        return {
            "moving": self._load_image(moving_idx),
            "fixed": self._load_image(fixed_idx),
            "center": centers,
            "loss_mask": loss_mask,
            "target_phase": self.phase_fractions[fixed_idx],
        }
