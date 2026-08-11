import torch
import torch.nn as nn
import torch.nn.functional as F


def make_norm(channels: int, norm_groups: int = 16):
    groups = min(norm_groups, channels)

    # GroupNorm requires channels % groups == 0.
    while channels % groups != 0 and groups > 1:
        groups -= 1

    return nn.GroupNorm(num_groups=groups, num_channels=channels)


class ResBlock1D(nn.Module):
    """
    Basic 1D residual block.

    If stride > 1 or channel count changes, the shortcut uses a 1x1 projection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        norm_groups: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd.")

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm1 = make_norm(out_channels, norm_groups)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.norm2 = make_norm(out_channels, norm_groups)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                make_norm(out_channels, norm_groups),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + residual
        x = self.relu(x)

        return x


class ResStage1D(nn.Module):
    """
    A stage of repeated residual blocks.
    First block may downsample with stride=2.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_blocks: int,
        first_stride: int,
        kernel_size: int = 7,
        norm_groups: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        blocks = [
            ResBlock1D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=first_stride,
                norm_groups=norm_groups,
                dropout=dropout,
            )
        ]

        for _ in range(n_blocks - 1):
            blocks.append(
                ResBlock1D(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    norm_groups=norm_groups,
                    dropout=dropout,
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class DenseResNetFPN1D(nn.Module):
    """
    Strong dense ResNet-FPN for per-timestep prediction.

    Encoder:
        stem at full resolution
        stage1 full/near-full resolution
        stage2 downsample x2
        stage3 downsample x4
        stage4 downsample x8

    Decoder:
        lateral 1x1 projections at each scale
        top-down upsampling + fusion
        dense output at original sequence length
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        fpn_channels: int = 128,
        layers=(3, 4, 6, 3),
        kernel_size: int = 7,
        norm_groups: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                base_channels,
                kernel_size=15,
                padding=7,
                bias=False,
            ),
            make_norm(base_channels, norm_groups),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                base_channels,
                base_channels,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            make_norm(base_channels, norm_groups),
            nn.ReLU(inplace=True),
        )

        # Keep the first stage at full resolution.
        # Downsample only from stage2 onward.
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.stage1 = ResStage1D(
            in_channels=base_channels,
            out_channels=c1,
            n_blocks=layers[0],
            first_stride=1,
            kernel_size=kernel_size,
            norm_groups=norm_groups,
            dropout=dropout,
        )

        self.stage2 = ResStage1D(
            in_channels=c1,
            out_channels=c2,
            n_blocks=layers[1],
            first_stride=2,
            kernel_size=kernel_size,
            norm_groups=norm_groups,
            dropout=dropout,
        )

        self.stage3 = ResStage1D(
            in_channels=c2,
            out_channels=c3,
            n_blocks=layers[2],
            first_stride=2,
            kernel_size=kernel_size,
            norm_groups=norm_groups,
            dropout=dropout,
        )

        self.stage4 = ResStage1D(
            in_channels=c3,
            out_channels=c4,
            n_blocks=layers[3],
            first_stride=2,
            kernel_size=kernel_size,
            norm_groups=norm_groups,
            dropout=dropout,
        )

        # Lateral projections for FPN.
        self.lat1 = nn.Conv1d(c1, fpn_channels, kernel_size=1)
        self.lat2 = nn.Conv1d(c2, fpn_channels, kernel_size=1)
        self.lat3 = nn.Conv1d(c3, fpn_channels, kernel_size=1)
        self.lat4 = nn.Conv1d(c4, fpn_channels, kernel_size=1)

        # Smooth after each top-down fusion.
        self.smooth3 = nn.Sequential(
            nn.Conv1d(fpn_channels, fpn_channels, kernel_size=7, padding=3, bias=False),
            make_norm(fpn_channels, norm_groups),
            nn.ReLU(inplace=True),
        )
        self.smooth2 = nn.Sequential(
            nn.Conv1d(fpn_channels, fpn_channels, kernel_size=7, padding=3, bias=False),
            make_norm(fpn_channels, norm_groups),
            nn.ReLU(inplace=True),
        )
        self.smooth1 = nn.Sequential(
            nn.Conv1d(fpn_channels, fpn_channels, kernel_size=7, padding=3, bias=False),
            make_norm(fpn_channels, norm_groups),
            nn.ReLU(inplace=True),
        )

        # Dense output head at full resolution.
        self.output_head = nn.Sequential(
            nn.Conv1d(fpn_channels, fpn_channels, kernel_size=7, padding=3, bias=False),
            make_norm(fpn_channels, norm_groups),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(fpn_channels, fpn_channels // 2, kernel_size=7, padding=3, bias=False),
            make_norm(fpn_channels // 2, norm_groups),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(fpn_channels // 2, 1, kernel_size=1),
        )

    def forward(self, x):
        # x: (B, 2, T)
        input_len = x.shape[-1]

        x = self.stem(x)

        c1 = self.stage1(x)  # full resolution
        c2 = self.stage2(c1) # T/2
        c3 = self.stage3(c2) # T/4
        c4 = self.stage4(c3) # T/8

        p4 = self.lat4(c4)

        p3 = self.lat3(c3) + F.interpolate(
            p4,
            size=c3.shape[-1],
            mode="linear",
            align_corners=False,
        )
        p3 = self.smooth3(p3)

        p2 = self.lat2(c2) + F.interpolate(
            p3,
            size=c2.shape[-1],
            mode="linear",
            align_corners=False,
        )
        p2 = self.smooth2(p2)

        p1 = self.lat1(c1) + F.interpolate(
            p2,
            size=c1.shape[-1],
            mode="linear",
            align_corners=False,
        )
        p1 = self.smooth1(p1)

        # In case padding/stride creates an off-by-one, force exact input length.
        if p1.shape[-1] != input_len:
            p1 = F.interpolate(
                p1,
                size=input_len,
                mode="linear",
                align_corners=False,
            )

        logits = self.output_head(p1)
        return logits


class full_module(nn.Module):
    """
    V: strong dense ResNet-FPN 1D detector.

    Expected input:
        x: (batch, time, 2)

    Output:
        out: (batch, time, 1)

    By default returns sigmoid probabilities for BCELoss compatibility.
    If using BCEWithLogitsLoss, instantiate with return_logits=True.
    """

    def __init__(
        self,
        *args,
        base_channels: int = 64,
        fpn_channels: int = 128,
        layers=(3, 4, 6, 3),
        dropout: float = 0.0,
        return_logits: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.return_logits = return_logits

        self.model = DenseResNetFPN1D(
            in_channels=2,
            base_channels=base_channels,
            fpn_channels=fpn_channels,
            layers=layers,
            kernel_size=7,
            norm_groups=16,
            dropout=dropout,
        )

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape (batch, time, channels), got {x.shape}"
            )

        if x.shape[-1] != 2:
            raise ValueError(
                f"Expected last dimension to be 2 for H1/L1 channels, got {x.shape}"
            )

        # Your convention: (B, T, 2)
        # Conv1d convention: (B, 2, T)
        x = x.permute(0, 2, 1)

        logits = self.model(x)

        # (B, 1, T) -> (B, T, 1)
        logits = logits.permute(0, 2, 1)

        if self.return_logits:
            return logits

        return torch.sigmoid(logits)