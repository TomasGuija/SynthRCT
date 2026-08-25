import torch
from torch import nn
import torch.nn.functional as F


def conv_block(in_ch: int, out_ch: int, convs_per_block: int) -> nn.Sequential:
    """Build a simple 3D convolutional block.

    Each block applies ``convs_per_block`` repetitions of:

    ``Conv3d -> LeakyReLU``

    Parameters
    ----------
    in_ch:
        Number of input channels.
    out_ch:
        Number of output channels.
    convs_per_block:
        Number of convolutional layers in the block.

    Returns
    -------
    nn.Sequential
        Sequential 3D convolutional feature extractor.
    """
    layers: list[nn.Module] = []

    for i in range(int(convs_per_block)):
        curr_in = in_ch if i == 0 else out_ch
        layers.append(nn.Conv3d(curr_in, out_ch, kernel_size=3, padding=1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    return nn.Sequential(*layers)


class _ConvLatentEncoder(nn.Module):
    """Shared CNN encoder that predicts Gaussian latent parameters.

    The encoder progressively extracts 3D features, downsamples them with max
    pooling, applies a bottleneck block, globally pools the result, and predicts
    ``mu`` and ``logvar`` for a diagonal Gaussian latent distribution.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        base_ch: int,
        n_levels: int,
        convs_per_block: int,
        latent_dim: int,
        pool_stride: tuple[int, int, int] = (2, 2, 2),
    ):
        """Initialize the shared convolutional latent encoder.

        Parameters
        ----------
        in_channels:
            Number of input image channels.
        base_ch:
            Number of channels in the first encoder level.
        n_levels:
            Number of downsampling encoder levels.
        convs_per_block:
            Number of convolutions per encoder block.
        latent_dim:
            Size of the latent vector.
        pool_stride:
            Max-pooling stride used after each encoder level.
        """
        super().__init__()

        self.pool_stride = tuple(pool_stride)

        encoders: list[nn.Module] = []
        prev_ch = int(in_channels)

        for level in range(int(n_levels)):
            out_ch = int(base_ch) * (2**level)
            encoders.append(conv_block(prev_ch, out_ch, convs_per_block))
            prev_ch = out_ch

        bottleneck_ch = prev_ch * 2

        self.encoders = nn.ModuleList(encoders)
        self.bottleneck = conv_block(prev_ch, bottleneck_ch, convs_per_block)
        self.pool = nn.AdaptiveAvgPool3d(1)

        self.mu = nn.Linear(bottleneck_ch, latent_dim)
        self.logvar = nn.Linear(bottleneck_ch, latent_dim)

        # Start with approximately unit variance: logvar ~= 0.
        nn.init.zeros_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an input volume into ``mu`` and ``logvar``.

        Parameters
        ----------
        x:
            Input tensor of shape ``(B, C, D, H, W)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Mean and log-variance tensors, both of shape ``(B, latent_dim)``.
        """
        for encoder in self.encoders:
            x = encoder(x)
            x = F.max_pool3d(x, kernel_size=self.pool_stride, stride=self.pool_stride)

        x = self.bottleneck(x)
        x = self.pool(x).flatten(1)

        return self.mu(x), self.logvar(x)


class ConvPosteriorEncoder(_ConvLatentEncoder):
    """Posterior encoder used during training.

    The posterior receives both the fixed and moving images and estimates
    ``q(z | fixed, moving)``. This encoder is only available during training or
    target-aware evaluation, because the fixed image is not known at deployment.
    """

    def __init__(
        self,
        *,
        base_ch: int,
        n_levels: int,
        convs_per_block: int,
        latent_dim: int,
        pool_stride: tuple[int, int, int] = (2, 2, 2),
    ):
        """Initialize the posterior encoder.

        Parameters
        ----------
        base_ch:
            Number of channels in the first encoder level.
        n_levels:
            Number of downsampling encoder levels.
        convs_per_block:
            Number of convolutions per encoder block.
        latent_dim:
            Size of the latent vector.
        pool_stride:
            Max-pooling stride used after each encoder level.
        """
        super().__init__(
            in_channels=2,
            base_ch=base_ch,
            n_levels=n_levels,
            convs_per_block=convs_per_block,
            latent_dim=latent_dim,
            pool_stride=pool_stride,
        )

    def forward(
        self,
        fixed_full: torch.Tensor,
        moving_full: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a fixed/moving image pair into posterior parameters.

        Parameters
        ----------
        fixed_full:
            Fixed image tensor of shape ``(B, 1, D, H, W)``.
        moving_full:
            Moving image tensor of shape ``(B, 1, D, H, W)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Posterior mean and log-variance, both of shape ``(B, latent_dim)``.
        """
        x = torch.cat([fixed_full, moving_full], dim=1)
        return super().forward(x)


class ConvPriorEncoder(_ConvLatentEncoder):
    """Anatomy-conditioned prior encoder used at inference time.

    The prior receives only the moving image and estimates ``p(z | moving)``.
    Sampling from this distribution allows the model to generate plausible
    repeat anatomies without observing a target fixed image.
    """

    def __init__(
        self,
        *,
        base_ch: int,
        n_levels: int,
        convs_per_block: int,
        latent_dim: int,
        pool_stride: tuple[int, int, int] = (2, 2, 2),
    ):
        """Initialize the prior encoder.

        Parameters
        ----------
        base_ch:
            Number of channels in the first encoder level.
        n_levels:
            Number of downsampling encoder levels.
        convs_per_block:
            Number of convolutions per encoder block.
        latent_dim:
            Size of the latent vector.
        pool_stride:
            Max-pooling stride used after each encoder level.
        """
        super().__init__(
            in_channels=1,
            base_ch=base_ch,
            n_levels=n_levels,
            convs_per_block=convs_per_block,
            latent_dim=latent_dim,
            pool_stride=pool_stride,
        )

    def forward(self, moving_full: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a moving image into prior parameters.

        Parameters
        ----------
        moving_full:
            Moving image tensor of shape ``(B, 1, D, H, W)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Prior mean and log-variance, both of shape ``(B, latent_dim)``.
        """
        return super().forward(moving_full)