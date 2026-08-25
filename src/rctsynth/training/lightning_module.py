"""Lightning module for training the SynthRCT registration VAE."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from monai.losses import LocalNormalizedCrossCorrelationLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from rctsynth.training import losses as loss
from rctsynth.models.registration_vae import RegistrationVAE
from rctsynth.models.slab_stitching import stitch_torch_slabs


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class VAELightning(LightningModule):
    """Lightning wrapper for SynthRCT registration VAE training.

    Expected batch keys
    -------------------
    moving:
        Moving full volume with shape ``(B, D, H, W, 1)``.
    fixed:
        Fixed full volume with shape ``(B, D, H, W, 1)``.
    center:
        Two slab centers with shape ``(B, 2)`` or ``(B, R, 2)``.
    loss_mask:
        Optional full-volume mask with shape ``(B, D, H, W, 1)``.
    """

    def __init__(
        self,
        *,
        # Architecture
        slab_size: int = 48,
        stitch_stride: int | None = None,
        base_ch: int = 16,
        n_levels: int = 4,
        out_channels: int = 3,
        convs_per_block: int = 2,
        latent_dim: int = 32,
        svf_steps: int = 7,
        deep_supervision_scales: list[int] | None = None,

        # Loss weights
        sim_weight: float = 20.0,
        kl_weight: float = 1.0,
        prior_kl_weight: float = 0.25,
        deep_supervision_weights: list[float] | None = None,

        # Optimization
        lr: float = 1e-4,
        weight_decay: float = 1e-4,

        # KL warmup
        kl_start_epoch: int = 25,
        kl_warmup_epochs: int = 75,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.image_loss = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=3,
            kernel_size=9,
            reduction="none",
        )

        self.model: RegistrationVAE | None = None
        self.deep_supervision_weights = deep_supervision_weights if deep_supervision_weights is not None else []
        self.deep_supervision_scales = deep_supervision_scales if deep_supervision_scales is not None else []

        if self.deep_supervision_scales != sorted(self.deep_supervision_scales, reverse=True):
            raise ValueError(
                f"deep_supervision_scales must be ordered from coarse to fine, "
                f"e.g. [8, 4, 2]. Got {self.deep_supervision_scales}."
            )
        if len(self.deep_supervision_weights) != len(self.deep_supervision_scales):
            raise ValueError(
                f"Expected one deep-supervision weight per scale. "
                f"Got scales={self.deep_supervision_scales}, "
                f"weights={self.deep_supervision_weights}."
            )
        
    # ---------------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------------
    def setup(self, stage: str | None = None) -> None:
        """Build the model once the datamodule is available."""
        if self.model is not None:
            return

        datamodule = self.trainer.datamodule
        full_size = tuple(int(s) for s in datamodule.vol_size)

        slab_depth = int(self.hparams.slab_size)
        stitch_stride = (
            slab_depth // 2
            if self.hparams.stitch_stride is None
            else int(self.hparams.stitch_stride)
        )

        data_slab_depth = int(datamodule.hparams.slab_size)
        data_stitch_stride = (
            data_slab_depth // 2
            if datamodule.hparams.stitch_stride is None
            else int(datamodule.hparams.stitch_stride)
        )

        if data_slab_depth != slab_depth:
            raise ValueError(
                f"Lightning slab_size={slab_depth} does not match "
                f"datamodule slab_size={data_slab_depth}."
            )

        if data_stitch_stride != stitch_stride:
            raise ValueError(
                f"Lightning stitch_stride={stitch_stride} does not match "
                f"datamodule stitch_stride={data_stitch_stride}."
            )

        model_slab_size = (slab_depth, full_size[1], full_size[2])
        self.hparams["model_slab_size"] = list(model_slab_size)

        self.build_model(
            slab_size=model_slab_size,
            stitch_stride=stitch_stride,
        )

    def build_model(
        self,
        *,
        slab_size: tuple[int, int, int],
        stitch_stride: int,
    ) -> None:
        """Build the registration VAE."""
        slab_size = tuple(int(s) for s in slab_size)
        stitch_stride = int(stitch_stride)

        if len(slab_size) != 3:
            raise ValueError(f"Expected slab_size as DHW tuple, got {slab_size}.")

        factor = 2 ** int(self.hparams.n_levels)
        if any(size % factor != 0 for size in slab_size):
            raise ValueError(
                f"slab_size={slab_size} must be divisible by {factor} in all dimensions."
            )

        slab_depth = int(slab_size[0])
        if slab_depth % 2 != 0:
            raise ValueError(f"slab depth must be even, got {slab_depth}.")

        if stitch_stride <= 0 or stitch_stride >= slab_depth:
            raise ValueError(
                f"stitch_stride must satisfy 0 < stitch_stride < slab_depth. "
                f"Got stitch_stride={stitch_stride}, slab_depth={slab_depth}."
            )

        self.slab_depth = slab_depth
        self.stitch_stride = stitch_stride

        self.model = RegistrationVAE(
            slab_size=slab_size,
            deep_supervision_scales=self.deep_supervision_scales,
            stitch_stride=self.stitch_stride,
            out_channels=int(self.hparams.out_channels),
            base_ch=int(self.hparams.base_ch),
            n_levels=int(self.hparams.n_levels),
            convs_per_block=int(self.hparams.convs_per_block),
            latent_dim=int(self.hparams.latent_dim),
            svf_steps=int(self.hparams.svf_steps),
        )

    # ---------------------------------------------------------------------
    # Tensor helpers
    # ---------------------------------------------------------------------
    def _to_ncdhw(
        self,
        x: torch.Tensor,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor | None:
        """Convert ``(B, D, H, W, C)`` to ``(B, C, D, H, W)``."""
        if x is None: 
            return None
        
        if x.ndim != 5:
            raise ValueError(f"Expected tensor [B,D,H,W,C], got {tuple(x.shape)}.")

        return x.permute(0, 4, 1, 2, 3).contiguous().to(
            self.device,
            dtype=dtype,
            non_blocking=True,
        )

    @staticmethod
    def _downsample_by_scale(
        x: torch.Tensor | None,
        scale: int,
        *,
        is_mask: bool = False,
    ) -> torch.Tensor | None:
        """Downsample an image or mask by an integer isotropic scale."""
        if x is None:
            return None

        if scale == 1:
            return x.bool() if is_mask else x

        kernel = (scale, scale, scale)

        if is_mask:
            return F.max_pool3d(
                x.float(),
                kernel_size=kernel,
                stride=kernel,
            ).bool()

        return F.avg_pool3d(
            x,
            kernel_size=kernel,
            stride=kernel,
        )

    # ---------------------------------------------------------------------
    # Slab extraction
    # ---------------------------------------------------------------------
    def _extract_grouped_slabs(
        self,
        volume: torch.Tensor,
        centers: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Extract grouped two-slab regions from an NCDHW volume.

        Parameters
        ----------
        volume:
            Tensor with shape ``(B, C, D, H, W)``.
        centers:
            Tensor with shape ``(B, 2)`` or ``(B, R, 2)``.

        Returns
        -------
        slabs:
            Tensor with shape ``(B * R, 2, C, D_slab, H, W)``.
        num_regions:
            Number of two-slab regions per batch element.
        """
        if self.slab_depth is None:
            raise RuntimeError("Model has not been built yet; slab_depth is unknown.")

        if volume.ndim != 5:
            raise ValueError(f"Expected volume [B,C,D,H,W], got {tuple(volume.shape)}.")

        if centers.ndim == 2:
            centers = centers[:, None, :]
        elif centers.ndim != 3:
            raise ValueError(
                f"Expected centers [B,2] or [B,R,2], got {tuple(centers.shape)}."
            )

        if centers.shape[-1] != 2:
            raise ValueError(
                f"Final model expects two slab centers per region, got {tuple(centers.shape)}."
            )

        batch, channels, depth, height, width = volume.shape
        center_batch, num_regions, num_slabs = centers.shape

        if center_batch != batch:
            raise ValueError(
                f"Expected centers for {batch} batch elements, got {center_batch}."
            )

        centers = centers.to(volume.device, dtype=torch.long)
        flat_centers = centers.reshape(batch, num_regions * num_slabs)

        half = self.slab_depth // 2
        offsets = torch.arange(
            -half,
            half,
            device=volume.device,
            dtype=torch.long,
        )

        indices = flat_centers[:, :, None] + offsets[None, None, :]

        if torch.any(indices < 0) or torch.any(indices >= depth):
            raise ValueError(
                f"Slab centers produce out-of-bounds indices for volume depth={depth}."
            )

        expanded_volume = volume[:, None].expand(
            -1,
            num_regions * num_slabs,
            -1,
            -1,
            -1,
            -1,
        )

        gather_indices = indices[:, :, None, :, None, None].expand(
            -1,
            -1,
            channels,
            -1,
            height,
            width,
        )

        slabs = torch.gather(
            expanded_volume,
            dim=3,
            index=gather_indices,
        )

        slabs = slabs.reshape(
            batch,
            num_regions,
            num_slabs,
            channels,
            self.slab_depth,
            height,
            width,
        )

        return slabs.flatten(0, 1), num_regions
    
    # ---------------------------------------------------------------------
    # Loss helpers
    # ---------------------------------------------------------------------
    def _kl_weight(self) -> float:
        """KL warmup schedule."""
        target = float(self.hparams.kl_weight)
        start_epoch = int(self.hparams.kl_start_epoch)
        warmup_epochs = int(self.hparams.kl_warmup_epochs)

        if self.current_epoch < start_epoch:
            return 0.0

        if warmup_epochs <= 0:
            return target

        alpha = min(
            1.0,
            float(self.current_epoch - start_epoch + 1) / float(warmup_epochs),
        )
        return target * alpha

    def _image_similarity_loss(
        self,
        warped: torch.Tensor,
        fixed: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Masked LNCC image loss."""
        with torch.autocast(device_type="cuda", enabled=False):
            sim_map = self.image_loss(
                warped.float(),
                fixed.float(),
            )
            return 1.0 + loss.masked_mean(sim_map, mask)

    def compute_loss(
        self,
        *,
        fixed_full: torch.Tensor,
        moving_full: torch.Tensor,
        fixed_slab: torch.Tensor,
        moving_slab: torch.Tensor,
        loss_mask: torch.Tensor | None,
        num_regions: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the full training loss for one batch."""
        q_mu, q_logvar = self.model.encode_posterior(
            fixed_full=fixed_full,
            moving_full=moving_full,
        )
        p_mu, p_logvar = self.model.encode_prior(moving_full)
        z = self.model._reparameterize(q_mu, q_logvar)
        z_decode = (
            z.repeat_interleave(int(num_regions), dim=0)
            if int(num_regions) != 1
            else z
        )

        outputs = self.model.decode(
            z_decode,
            moving_slab,
            return_deep_supervision=bool(self.deep_supervision_scales),
        )
        primary = outputs[0]

        fixed_stitched = stitch_torch_slabs(fixed_slab, stride=self.stitch_stride)

        sim_loss = self._image_similarity_loss(
            warped=primary["warped"],
            fixed=fixed_stitched,
            mask=loss_mask,
        )

        aux_outputs = outputs[1:]

        aux_sim_loss = torch.zeros((), device=fixed_full.device)
        aux_sim_losses: dict[int, torch.Tensor] = {}

        for aux, scale, weight in zip(
            aux_outputs,
            self.deep_supervision_scales,
            self.deep_supervision_weights,
            strict=True,
        ):
            fixed_aux = self._downsample_by_scale(fixed_stitched, scale)
            mask_aux = self._downsample_by_scale(loss_mask, scale, is_mask=True)

            scale_loss = self._image_similarity_loss(
                warped=aux["warped"],
                fixed=fixed_aux,
                mask=mask_aux,
            )

            aux_sim_losses[scale] = scale_loss
            aux_sim_loss = aux_sim_loss + float(weight) * scale_loss

        with torch.autocast(device_type="cuda", enabled=False):
            posterior_to_prior = loss.kl_divergence_gaussians(
                q_mu.float(),
                q_logvar.float(),
                p_mu.float(),
                p_logvar.float(),
            )
            prior_to_standard = loss.kl_divergence(
                p_mu.float(),
                p_logvar.float(),
            )
            kl_loss = (
                posterior_to_prior
                + self.hparams.prior_kl_weight * prior_to_standard
            )

        total = (
            self.hparams.sim_weight * sim_loss
            + self.hparams.sim_weight * aux_sim_loss
            + self._kl_weight() * kl_loss
        )

        metrics = {
            "loss_total": total.detach(),
            "loss_sim": sim_loss.detach(),
            "loss_aux_sim": aux_sim_loss.detach(),
            "loss_kl": kl_loss.detach(),
            "loss_posterior_to_prior": posterior_to_prior.detach(),
            "loss_prior_to_standard": prior_to_standard.detach(),
            "kl_weight": torch.as_tensor(
                self._kl_weight(),
                device=fixed_full.device,
            ),
        }

        for scale, scale_loss in aux_sim_losses.items():
            metrics[f"loss_aux_sim_s{scale}"] = scale_loss.detach()

        return total, metrics

    # ---------------------------------------------------------------------
    # Training / validation
    # ---------------------------------------------------------------------
    def _step(
        self,
        stage: str,
        batch: dict,
    ) -> torch.Tensor:
        """Run one training or validation step."""
        moving_full = self._to_ncdhw(batch["moving"], dtype=torch.float32)
        fixed_full = self._to_ncdhw(batch["fixed"], dtype=torch.float32)

        loss_mask_full = self._to_ncdhw(
            batch.get("loss_mask"),
            dtype=torch.float32,
        )
        if loss_mask_full is not None:
            loss_mask_full = loss_mask_full > 0.5

        centers = batch["center"].to(
            self.device,
            dtype=torch.long,
            non_blocking=True,
        )

        moving_slab, num_regions = self._extract_grouped_slabs(
            moving_full,
            centers,
        )
        fixed_slab, _ = self._extract_grouped_slabs(
            fixed_full,
            centers,
        )

        loss_mask = None
        if loss_mask_full is not None:
            mask_slab, _ = self._extract_grouped_slabs(
                loss_mask_full,
                centers,
            )
            loss_mask = stitch_torch_slabs(
                mask_slab,
                stride=self.stitch_stride,
            )

        total, metrics = self.compute_loss(
            fixed_full=fixed_full,
            moving_full=moving_full,
            fixed_slab=fixed_slab,
            moving_slab=moving_slab,
            loss_mask=loss_mask,
            num_regions=num_regions,
        )

        self.log_dict(
            {f"{stage}/{name}": value for name, value in metrics.items()},
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=moving_slab.shape[0],
        )

        return total

    def training_step(
        self,
        batch: dict,
        batch_idx: int,
    ) -> torch.Tensor:
        """Run one training step."""
        return self._step("train", batch)

    def validation_step(
        self,
        batch: dict,
        batch_idx: int,
    ) -> torch.Tensor:
        """Run one validation step."""
        return self._step("val", batch)

    # ---------------------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------------------
    def configure_optimizers(self):
        """Configure optimizer and LR scheduler."""
        optimizer = AdamW(
            self.model.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay),
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=20,
            threshold=1e-5,
            cooldown=1,
            min_lr=1e-7,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss_total",
                "interval": "epoch",
                "strict": True,
                "name": "plateau",
            },
        }