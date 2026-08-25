"""Lightning datamodule for SynthRCT registration VAE training."""

from __future__ import annotations

import logging

import h5py
import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from rctsynth.data.dataset import H5RegistrationDataset


def registration_collate(batch: list[dict]) -> dict:
    """Collate registration samples with optional None values."""
    if not batch:
        return {}

    output = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]

        if all(value is None for value in values):
            output[key] = None
            continue

        if any(value is None for value in values):
            raise ValueError(f"Mixed None and non-None values for batch key '{key}'.")

        first = values[0]
        if isinstance(first, np.ndarray):
            shapes = [value.shape for value in values]
            if len(set(shapes)) != 1:
                raise ValueError(f"Shape mismatch for batch key '{key}': {shapes}.")
            output[key] = torch.from_numpy(np.stack(values, axis=0))
        elif torch.is_tensor(first):
            output[key] = torch.stack(values, dim=0)
        elif isinstance(first, (int, np.integer)):
            output[key] = torch.as_tensor(values, dtype=torch.long)
        else:
            output[key] = values

    return output


class VAEDataModule(LightningDataModule):
    """DataModule for intra-patient axial registration VAE training."""

    def __init__(
        self,
        h5_path: str,
        val_case_ids: list[int] | tuple[int, ...],
        *,
        slab_size: int = 48,
        stitch_stride: int | None = None,
        num_sampled_regions: int = 1,
        min_scan_distance: int = 3,
        batch_size: int = 4,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = True,
        prefetch_factor: int | None = 2,
        persistent_workers: bool | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        stitch_stride = slab_size // 2 if stitch_stride is None else stitch_stride

        if slab_size <= 0 or slab_size % 2 != 0:
            raise ValueError(f"slab_size must be even and > 0, got {slab_size}.")
        if stitch_stride <= 0 or stitch_stride >= slab_size:
            raise ValueError(
                f"stitch_stride must satisfy 0 < stitch_stride < slab_size. "
                f"Got stitch_stride={stitch_stride}, slab_size={slab_size}."
            )
        if num_sampled_regions < 1:
            raise ValueError("num_sampled_regions must be >= 1.")

        self.vol_size: tuple[int, int, int] | None = None
        self.train_case_ids: list[int] = []
        self.val_case_ids: list[int] = []
        self.train_ds: H5RegistrationDataset | None = None
        self.val_ds: H5RegistrationDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Create train and validation datasets."""
        with h5py.File(self.hparams.h5_path, "r") as fh:
            self.vol_size = tuple(x for x in fh.attrs["target_shape"])
            all_case_ids = sorted(x for x in np.unique(fh["case_id"][:]))

        val_case_ids = sorted({x for x in self.hparams.val_case_ids})
        if not val_case_ids:
            raise ValueError("val_case_ids must contain at least one case ID.")

        unknown_val_ids = sorted(set(val_case_ids) - set(all_case_ids))
        if unknown_val_ids:
            raise ValueError(
                f"Validation case IDs {unknown_val_ids} are not present in "
                f"{self.hparams.h5_path}. Available case IDs: {all_case_ids}."
            )

        train_case_ids = sorted(set(all_case_ids) - set(val_case_ids))
        if not train_case_ids:
            raise ValueError("Validation split produced no training cases.")

        self.train_case_ids = train_case_ids
        self.val_case_ids = val_case_ids

        self.train_ds = self._dataset(train_case_ids, sampling_mode="random")
        self.val_ds = self._dataset(val_case_ids, sampling_mode="deterministic")

        logging.info(
            "DataModule: h5_path=%s | train_cases=%s (%d pairs) | val_cases=%s (%d pairs)",
            self.hparams.h5_path,
            self.train_case_ids,
            len(self.train_ds),
            self.val_case_ids,
            len(self.val_ds),
        )

    def _dataset(
        self,
        case_ids: list[int],
        *,
        sampling_mode: str,
    ) -> H5RegistrationDataset:
        """Build one dataset split."""
        return H5RegistrationDataset(
            h5_path=self.hparams.h5_path,
            slab_size=self.hparams.slab_size,
            stitch_stride=self.hparams.stitch_stride,
            num_sampled_regions=self.hparams.num_sampled_regions,
            min_scan_distance=self.hparams.min_scan_distance,
            sampling_mode=sampling_mode,
            case_ids=case_ids,
        )

    def _loader(
        self,
        dataset: H5RegistrationDataset,
        *,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:
        """Build a dataloader."""
        num_workers = self.hparams.num_workers

        persistent_workers = self.hparams.persistent_workers
        if persistent_workers is None:
            persistent_workers = num_workers > 0

        kwargs = {
            "dataset": dataset,
            "batch_size": self.hparams.batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "persistent_workers": persistent_workers and num_workers > 0,
            "pin_memory": self.hparams.pin_memory,
            "drop_last": drop_last,
            "collate_fn": registration_collate,
        }

        if num_workers > 0 and self.hparams.prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(self.hparams.prefetch_factor)

        return DataLoader(**kwargs)

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        if self.train_ds is None:
            raise RuntimeError("DataModule has not been set up yet.")

        return self._loader(
            self.train_ds,
            shuffle=self.hparams.shuffle,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        if self.val_ds is None:
            raise RuntimeError("DataModule has not been set up yet.")

        return self._loader(
            self.val_ds,
            shuffle=False,
            drop_last=False,
        )
