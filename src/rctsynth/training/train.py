"""Training entry point for the SynthRCT registration VAE."""

from __future__ import annotations

import logging

import torch
from lightning.pytorch.cli import LightningCLI

from rctsynth.data.datamodule import VAEDataModule
from rctsynth.training.lightning_module import VAELightning


class VAELightningCLI(LightningCLI):
    """Lightning CLI with config logging."""

    def before_fit(self) -> None:
        logging.info("Loaded configuration:\n%s", self.parser.dump(self.config))


def main() -> None:
    """Run VAE training from a Lightning CLI config."""
    torch.set_float32_matmul_precision("medium")

    VAELightningCLI(
        model_class=VAELightning,
        datamodule_class=VAEDataModule,
        save_config_kwargs={"overwrite": True},
        seed_everything_default=333,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()