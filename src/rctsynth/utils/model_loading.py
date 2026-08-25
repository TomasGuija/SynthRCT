"""SynthRCT checkpoint loading."""

from __future__ import annotations

from inspect import signature
from pathlib import Path

import torch

from rctsynth.training.lightning_module import VAELightning
from rctsynth.utils.core import resolve_device


def load_vae(
    ckpt_path: str | bytes | Path,
    device: torch.device | str = "cuda",
) -> tuple[VAELightning, torch.nn.Module]:
    """Load SynthRCT from a self-contained Lightning checkpoint."""
    device = resolve_device(device)

    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    if "state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain a 'state_dict'.")

    hparams = dict(checkpoint.get("hyper_parameters", {}))

    model_slab_size = tuple(hparams.pop("model_slab_size"))
    slab_depth = int(hparams["slab_size"])
    stitch_stride = int(
        hparams.get("stitch_stride") or slab_depth // 2
    )

    valid_parameters = set(
        signature(VAELightning.__init__).parameters
    ) - {"self"}

    module = VAELightning(
        **{
            key: value
            for key, value in hparams.items()
            if key in valid_parameters
        }
    )

    module.build_model(
        slab_size=model_slab_size,
        stitch_stride=stitch_stride,
    )

    state_dict = {
        key.replace("._orig_mod.", ".").removeprefix("_orig_mod."): value
        for key, value in checkpoint["state_dict"].items()
    }

    module.load_state_dict(state_dict, strict=True)
    module.eval().to(device)

    model = module.model
    if model is None:
        raise RuntimeError(
            "VAELightning.build_model() did not create RegistrationVAE."
        )

    model.eval()
    return module, model