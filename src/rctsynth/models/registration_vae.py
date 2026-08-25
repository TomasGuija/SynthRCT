"""Registration VAE model for SynthRCT.

The model learns a latent distribution over anatomical deformations. During
training, a posterior encoder observes the fixed/moving image pair. At inference
time, an anatomy-conditioned prior samples deformations from the moving image
alone.

The decoder operates on axial slabs. It predicts stationary velocity fields
(SVFs), refines overlapping slab predictions, integrates the SVF into a dense
deformation field, and warps the moving image.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from monai.networks.nets import UNet
from torch import nn

from rctsynth.models.modules import ConvPosteriorEncoder, ConvPriorEncoder, conv_block
from rctsynth.models.slab_stitching import stitch_torch_slabs
from rctsynth.spatial.transform import SpatialTransform, integrate_svf


POOL_STRIDE = (2, 2, 2)


def down_shape(
    inshape: tuple[int, int, int],
    pool_schedule: list[tuple[int, int, int]],
) -> tuple[int, int, int]:
    """Return the spatial shape after applying a sequence of pooling strides."""
    d, h, w = (int(s) for s in inshape)

    for pd, ph, pw in pool_schedule:
        d //= int(pd)
        h //= int(ph)
        w //= int(pw)

    return d, h, w


class RegistrationVAE(nn.Module):
    """SynthRCT registration VAE.

    The model learns a latent distribution over anatomical deformations.

    During training, the posterior encoder observes the fixed and moving images
    and predicts ``q(z | fixed, moving)``. The prior encoder observes only the
    moving image and predicts ``p(z | moving)``.

    The decoder receives a latent code and two overlapping moving-image slabs.
    It predicts one SVF per slab, refines the overlap with a residual UNet,
    integrates the stitched SVF into a deformation field, and warps the moving
    image.
    """

    def __init__(
        self,
        *,
        slab_size: tuple[int, int, int],
        deep_supervision_scales: tuple[int, ...] = (),
        stitch_stride: int | None = None,
        out_channels: int = 3,
        base_ch: int = 16,
        n_levels: int = 4,
        convs_per_block: int = 2,
        latent_dim: int = 64,
        svf_steps: int = 7,
    ) -> None:
        """Initialize the final SynthRCT registration VAE."""
        super().__init__()

        self.slab_size = tuple(int(s) for s in slab_size)
        self.stitch_stride = None if stitch_stride is None else int(stitch_stride)
        self.out_channels = int(out_channels)
        self.base_ch = int(base_ch)
        self.n_levels = int(n_levels)
        self.convs_per_block = int(convs_per_block)
        self.latent_dim = int(latent_dim)
        self.svf_steps = int(svf_steps)

        self._validate_slab_size()
        self._validate_stitch_stride()

        self.deep_supervision_scales = tuple(
            sorted((int(scale) for scale in deep_supervision_scales), reverse=True)
        )
        self.use_deep_supervision = len(self.deep_supervision_scales) > 0

        self.reconstruction = SpatialTransform(self._stitched_shape(self.slab_size))

        self.posterior_encoder = ConvPosteriorEncoder(
            base_ch=self.base_ch,
            n_levels=self.n_levels,
            convs_per_block=self.convs_per_block,
            latent_dim=self.latent_dim,
            pool_stride=POOL_STRIDE,
        )
        self.prior_encoder = ConvPriorEncoder(
            base_ch=self.base_ch,
            n_levels=self.n_levels,
            convs_per_block=self.convs_per_block,
            latent_dim=self.latent_dim,
            pool_stride=POOL_STRIDE,
        )

        self.mov_channels: list[int] = []
        self.mov_encoders = nn.ModuleList()
        self._build_moving_slab_encoder()

        decoder_pool_schedule = [POOL_STRIDE for _ in range(self.n_levels)]
        decoder_hidden_shape = down_shape(self.slab_size, decoder_pool_schedule)
        if any(size <= 0 for size in decoder_hidden_shape):
            raise ValueError(
                f"slab_size={self.slab_size} is too small for n_levels={self.n_levels} "
                f"and pool_schedule={self.decoder_pool_schedule}. "
                f"Got decoder_hidden_shape={decoder_hidden_shape}."
            )

        self.z_to_film = nn.Linear(self.latent_dim, 2 * self.mov_channels[-1])
        self._init_zero_linear(self.z_to_film)
        
        self.deep_supervision_decoder_indices: list[int] = []
        self.deep_supervision_out_convs = nn.ModuleList()
        self.deep_supervision_reconstructions = nn.ModuleList()
        self.deep_supervision_svf_refiners = nn.ModuleList()

        self.decoders = nn.ModuleList()
        self.decoder_films = nn.ModuleList()
        self._build_slab_decoder()

        self.out_conv = nn.Conv3d(
            self.decoder_out_channels,
            self.out_channels,
            kernel_size=3,
            padding=1,
        )
        self.svf_refiner = self._make_svf_refiner()

    def _validate_slab_size(self) -> None:
        """Validate the decoder slab size."""
        if len(self.slab_size) != 3:
            raise ValueError(f"slab_size must have three values, got {self.slab_size}.")
        if any(size <= 0 for size in self.slab_size):
            raise ValueError(f"slab_size values must be positive, got {self.slab_size}.")

    def _validate_stitch_stride(self) -> None:
        """Validate the optional full-resolution slab stride."""
        if self.stitch_stride is None:
            return

        slab_depth = self.slab_size[0]
        if self.stitch_stride <= 0 or self.stitch_stride >= slab_depth:
            raise ValueError(
                f"stitch_stride must satisfy 0 < stitch_stride < slab_depth; "
                f"got stitch_stride={self.stitch_stride}, slab_depth={slab_depth}."
            )

    @property
    def slab_depth(self) -> int:
        """Axial depth decoded by one slab."""
        return int(self.slab_size[0])

    @property
    def resolved_stitch_stride(self) -> int:
        """Axial stride used for full-resolution slab stitching."""
        if self.stitch_stride is None:
            return self.slab_depth // 2
        return int(self.stitch_stride)

    @property
    def encoder_downsample_factor(self) -> int:
        """Required depth multiple for full-volume latent encoding."""
        return int(POOL_STRIDE[0]) ** self.n_levels

    def pad_encoder_input(self, x: torch.Tensor) -> torch.Tensor:
        """Center-pad an NCDHW volume to the encoder's depth requirement."""
        if x.ndim != 5:
            raise ValueError(f"Expected tensor [B,C,D,H,W], got {tuple(x.shape)}.")

        factor = self.encoder_downsample_factor
        depth = int(x.shape[2])
        target_depth = ((depth + factor - 1) // factor) * factor
        pad_total = target_depth - depth

        if pad_total == 0:
            return x

        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return F.pad(
            x,
            (0, 0, 0, 0, pad_before, pad_after),
            mode="constant",
            value=0.0,
        )

    def _stitch_stride_for_depth(self, field_depth: int) -> int:
        """Return the axial stitching stride for a field at the given depth."""
        if self.stitch_stride is None:
            return field_depth // 2

        slab_depth = self.slab_size[0]
        depth_scale = slab_depth // field_depth

        return self.stitch_stride // depth_scale

    def _stitched_shape(self, field_shape: tuple[int, int, int]) -> tuple[int, int, int]:
        """Return the stitched shape produced by two slabs of ``field_shape``."""
        depth, height, width = (int(s) for s in field_shape)
        stride = self._stitch_stride_for_depth(depth)
        return depth + stride, height, width

    def _scaled_stitched_shape(self, scale: int) -> tuple[int, int, int]:
        """Return the stitched shape for one deep-supervision scale."""
        scale = int(scale)
        field_shape = tuple(size // scale for size in self.slab_size)
        return self._stitched_shape(field_shape)

    def _block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Build one convolutional block."""
        return conv_block(in_ch, out_ch, self.convs_per_block)

    def _build_moving_slab_encoder(self) -> None:
        """Build the moving-slab encoder used by the decoder."""
        prev_ch = 1
        for level in range(self.n_levels):
            out_ch = self.base_ch * (2**level)
            self.mov_channels.append(out_ch)
            self.mov_encoders.append(self._block(prev_ch, out_ch))
            prev_ch = out_ch

    def _build_slab_decoder(self) -> None:
        """Build decoder blocks, FiLM layers, and deep-supervision heads."""
        curr_ch = self.mov_channels[-1]

        for level in reversed(range(self.n_levels)):
            decoder_index = len(self.decoders)
            dec_ch = self.base_ch * (2**level)

            self.decoders.append(self._block(curr_ch, dec_ch))

            film = nn.Linear(self.latent_dim, 2 * dec_ch)
            self._init_zero_linear(film)
            self.decoder_films.append(film)

            curr_ch = dec_ch + self.mov_channels[level]

            scale = 2**level
            if scale in self.deep_supervision_scales:
                self.deep_supervision_decoder_indices.append(decoder_index)
                self.deep_supervision_out_convs.append(
                    nn.Conv3d(
                        curr_ch,
                        self.out_channels,
                        kernel_size=3,
                        padding=1,
                    )
                )
                self.deep_supervision_reconstructions.append(
                    SpatialTransform(self._scaled_stitched_shape(scale))
                )
                self.deep_supervision_svf_refiners.append(self._make_svf_refiner())

        self.decoder_out_channels = curr_ch

    def _make_svf_refiner(self) -> nn.Module:
        """Build the residual UNet used to refine overlapping slab SVFs."""
        refiner_ch = max(8, min(32, self.base_ch // 2))

        refiner = UNet(
            spatial_dims=3,
            in_channels=2 * self.out_channels + 1,
            out_channels=self.out_channels,
            channels=(refiner_ch, refiner_ch * 2, refiner_ch * 4),
            strides=((1, 2, 2), (1, 2, 2)),
            num_res_units=1,
            norm=None,
        )

        output_conv = next(
            module for module in reversed(list(refiner.modules())) if isinstance(module, nn.Conv3d)
        )
        nn.init.zeros_(output_conv.weight)
        nn.init.zeros_(output_conv.bias)

        return refiner

    @staticmethod
    def _init_zero_linear(layer: nn.Linear) -> None:
        """Initialize a FiLM layer to output zero modulation."""
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)

    @staticmethod
    def _apply_film(x: torch.Tensor, film: nn.Linear, z: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning to a feature tensor."""
        gamma, beta = film(z).chunk(2, dim=1)
        gamma = gamma.view(x.shape[0], -1, 1, 1, 1)
        beta = beta.view(x.shape[0], -1, 1, 1, 1)
        return (1.0 + gamma) * x + beta

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample during training and return the mean during evaluation."""
        if not self.training:
            return mu

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_posterior(
        self,
        fixed_full: torch.Tensor,
        moving_full: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode full fixed/moving volumes into posterior parameters."""
        if fixed_full.shape != moving_full.shape:
            raise ValueError(
                "fixed_full and moving_full must have identical shapes; "
                f"got {tuple(fixed_full.shape)} and {tuple(moving_full.shape)}."
            )

        fixed_full = self.pad_encoder_input(fixed_full)
        moving_full = self.pad_encoder_input(moving_full)
        return self.posterior_encoder(fixed_full, moving_full)

    def encode_prior(self, moving_full: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a full moving volume into prior parameters."""
        moving_full = self.pad_encoder_input(moving_full)
        return self.prior_encoder(moving_full)

    def encode_moving_slab(
        self,
        moving_slab: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode one moving slab and return bottleneck and skip features."""
        x = moving_slab
        skips: list[torch.Tensor] = []

        for encoder in self.mov_encoders:
            x = encoder(x)
            skips.append(x)
            x = F.max_pool3d(
                x,
                kernel_size=POOL_STRIDE,
                stride=POOL_STRIDE,
            )

        return x, skips

    def _decode_slab_features(
        self,
        z: torch.Tensor,
        moving_slab: torch.Tensor,
        *,
        return_deep_supervision: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Decode independent slabs before stitching/refinement."""
        mov_bottleneck, mov_skips = self.encode_moving_slab(moving_slab)

        x = self._apply_film(mov_bottleneck, self.z_to_film, z)
        deep_supervision_raws: list[torch.Tensor] = []
        deep_head_index = 0

        for decoder_index, decoder in enumerate(self.decoders):
            x = decoder(x)
            x = self._apply_film(x, self.decoder_films[decoder_index], z)
            x = F.interpolate(
                x,
                scale_factor=POOL_STRIDE,
                mode="trilinear",
                align_corners=True,
            )
            x = torch.cat([x, mov_skips[-(decoder_index + 1)]], dim=1)

            if (
                return_deep_supervision
                and deep_head_index < len(self.deep_supervision_decoder_indices)
                and decoder_index == self.deep_supervision_decoder_indices[deep_head_index]
            ):
                deep_supervision_raws.append(
                    self.deep_supervision_out_convs[deep_head_index](x)
                )
                deep_head_index += 1

        raw = self.out_conv(x)
        return raw, deep_supervision_raws

    def decode_raw_slab(
        self,
        z: torch.Tensor,
        moving_slab: torch.Tensor,
    ) -> torch.Tensor:
        """Decode independent moving slabs into raw SVFs without stitching or warping."""
        raw, _ = self._decode_slab_features(
            z,
            moving_slab,
            return_deep_supervision=False,
        )
        return raw

    def _build_svf_canvas(
        self,
        raw_grouped: torch.Tensor,
        *,
        stride: int | None = None,
    ) -> torch.Tensor:
        """Place grouped slab SVFs into a common padded canvas."""
        slab_depth = raw_grouped.shape[3]

        stride = self._stitch_stride_for_depth(slab_depth) if stride is None else int(stride)
        stitched_depth = slab_depth + stride

        blocks = []
        for slab_index in range(2):
            start = slab_index * stride
            end = start + slab_depth
            pad_before = start
            pad_after = stitched_depth - end
            blocks.append(
                F.pad(
                    raw_grouped[:, slab_index],
                    (0, 0, 0, 0, pad_before, pad_after),
                )
            )

        return torch.cat(blocks, dim=1)

    def _refine_grouped_svf(
        self,
        raw_grouped: torch.Tensor,
        moving_context: torch.Tensor,
        refiner: nn.Module,
        *,
        stride: int | None = None,
    ) -> torch.Tensor:
        """Refine and merge two overlapping slab SVFs."""
        slab_depth = raw_grouped.shape[3]

        stride = self._stitch_stride_for_depth(slab_depth) if stride is None else int(stride)
        overlap = slab_depth - stride

        if overlap <= 0:
            raise ValueError(
                f"Overlap refinement requires stride < slab depth, got stride={stride}, "
                f"depth={slab_depth}."
            )

        svf_canvas = self._build_svf_canvas(raw_grouped, stride=stride)

        residual_full = refiner(torch.cat([svf_canvas, moving_context], dim=1))
        residual_overlap = residual_full[:, :, stride:slab_depth]

        left = raw_grouped[:, 0]
        right = raw_grouped[:, 1]
        left_prefix = left[:, :, :stride]
        baseline_overlap = 0.5 * (left[:, :, stride:slab_depth] + right[:, :, :overlap])
        right_suffix = right[:, :, overlap:]

        return torch.cat(
            [left_prefix, baseline_overlap + residual_overlap, right_suffix],
            dim=2,
        )

    @staticmethod
    def _downsample_by_scale(x: torch.Tensor, scale: int) -> torch.Tensor:
        """Average-pool a tensor by an integer isotropic scale."""
        if scale == 1:
            return x

        return F.avg_pool3d(
            x,
            kernel_size=(scale, scale, scale),
            stride=(scale, scale, scale),
        )

    def _make_output(
        self,
        raw: torch.Tensor,
        moving: torch.Tensor,
        stn: nn.Module,
    ) -> dict[str, Any]:
        """Integrate a raw SVF, warp the moving image, and package the output."""
        flow = integrate_svf(raw, stn, steps=self.svf_steps)
        warped = stn(moving, flow)

        return {
            "flow": flow,
            "warped": warped,
        }

    def _prepare_grouped_decode_inputs(
        self,
        z: torch.Tensor,
        moving_slab: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flatten grouped two-slab input before decoding."""
        if moving_slab.ndim != 6:
            raise ValueError(
                "RegistrationVAE.decode() expects grouped two-slab input with shape "
                "(B, 2, 1, D_slab, H, W). Use decode_raw_slab() for independent slabs."
            )

        batch_size, num_slabs = moving_slab.shape[:2]
        if num_slabs != 2:
            raise ValueError(
                f"Grouped decoding expects {2} slabs, got {num_slabs}."
            )

        moving_flat = moving_slab.reshape(batch_size * num_slabs, *moving_slab.shape[2:])
        z_flat = z[:, None].expand(-1, num_slabs, -1).reshape(batch_size * num_slabs, -1)

        return z_flat, moving_flat

    def decode(
        self,
        z: torch.Tensor,
        moving_slab: torch.Tensor,
        *,
        return_deep_supervision: bool = False,
    ) -> list[dict[str, Any]]:
        """Decode latent codes into deformation outputs.

        ``moving_slab`` must have shape ``(B, 2, 1, D_slab, H, W)``.
        """
        batch_size = moving_slab.shape[0]

        return_deep_supervision = return_deep_supervision and self.use_deep_supervision

        z_flat, moving_flat = (
            self._prepare_grouped_decode_inputs(z, moving_slab)
        )

        raw_flat, deep_raws_flat = self._decode_slab_features(
            z_flat,
            moving_flat,
            return_deep_supervision=return_deep_supervision,
        )

        moving_stitched = stitch_torch_slabs(
            moving_slab,
            stride=self.stitch_stride,
        )

        raw_grouped_pre_refiner = raw_flat.reshape(
            batch_size,
            2,
            *raw_flat.shape[1:],
        )

        refined_raw = self._refine_grouped_svf(
            raw_grouped_pre_refiner,
            moving_stitched,
            self.svf_refiner,
        )

        outputs = [
            self._make_output(
                refined_raw,
                moving_stitched,
                self.reconstruction,
            )
        ]
        outputs[0]["raw_grouped_pre_refiner"] = raw_grouped_pre_refiner
        outputs[0]["raw"] = refined_raw

        if return_deep_supervision:
            for head_index, scale in enumerate(self.deep_supervision_scales):
                deep_raw_flat = deep_raws_flat[head_index]

                deep_raw_grouped = deep_raw_flat.reshape(
                    batch_size,
                    2,
                    *deep_raw_flat.shape[1:],
                )

                moving_aux = self._downsample_by_scale(moving_stitched, scale)

                refined_deep_raw = self._refine_grouped_svf(
                    deep_raw_grouped,
                    moving_aux,
                    self.deep_supervision_svf_refiners[head_index],
                )

                deep_output = self._make_output(
                    refined_deep_raw,
                    moving_aux,
                    self.deep_supervision_reconstructions[head_index],
                )
                deep_output["raw"] = refined_deep_raw
                deep_output["supervision_scale"] = scale

                outputs.append(deep_output)

        return outputs

    def forward(
        self,
        fixed_full: torch.Tensor,
        moving_full: torch.Tensor,
        moving_slab: torch.Tensor,
        *,
        return_deep_supervision: bool = False,
    ) -> dict[str, Any]:
        """Run the training/evaluation forward pass."""
        q_mu, q_logvar = self.encode_posterior(
            fixed_full=fixed_full,
            moving_full=moving_full,
        )
        p_mu, p_logvar = self.encode_prior(moving_full)
        z = self._reparameterize(q_mu, q_logvar)

        outputs = self.decode(
            z,
            moving_slab,
            return_deep_supervision=return_deep_supervision,
        )
        primary = outputs[0]

        return {
            "outputs": outputs,
            "raw": primary["raw"],
            "flow": primary["flow"],
            "warped": primary["warped"],
            "z": z,
            "q_mu": q_mu,
            "q_logvar": q_logvar,
            "p_mu": p_mu,
            "p_logvar": p_logvar,
        }