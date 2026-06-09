import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualTCNBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("Use an odd kernel_size to preserve sequence length cleanly.")

        padding = dilation * (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + residual
        x = F.relu(x)

        return x


class full_module(nn.Module):
    """
    Drop-in early-fusion TCN replacement for the original AttenGW full_module.

    Expected input:
        x.shape = (batch, time, channels)

    For your current 2-detector setup:
        channels = 2, corresponding to H1/L1 or L1/H1 depending on your loader.

    Returned output:
        out.shape = (batch, time, 1)

    By default, returns sigmoid probabilities so it remains compatible with
    nn.BCELoss in the existing training code.

    If you later switch to nn.BCEWithLogitsLoss, instantiate with:
        full_module(return_logits=True)
    """

    def __init__(
        self,
        *args,
        input_channels: int = 2,
        hidden_channels: int = 32,
        kernel_size: int = 7,
        dilations=None,
        dropout: float = 0.0,
        return_logits: bool = False,
        **kwargs,
    ):
        super().__init__()

        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32, 64, 128]

        self.return_logits = return_logits

        self.input_conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.input_norm = nn.BatchNorm1d(hidden_channels)

        self.blocks = nn.ModuleList(
            [
                ResidualTCNBlock(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )

        self.output_conv = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=1,
        )

    def forward(self, x):
        # Current AttenGW code uses x.shape = (batch, time, channels).
        # Conv1d expects x.shape = (batch, channels, time).
        if x.ndim != 3:
            raise ValueError(f"Expected input with 3 dimensions (batch, time, channels), got {x.shape}")

        x = x.permute(0, 2, 1)

        x = self.input_conv(x)
        x = self.input_norm(x)
        x = F.relu(x)

        for block in self.blocks:
            x = block(x)

        logits = self.output_conv(x)

        # Back to shape (batch, time, 1), matching the original model.
        logits = logits.permute(0, 2, 1)

        if self.return_logits:
            return logits

        return torch.sigmoid(logits)