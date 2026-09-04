import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1, dropout: float = 0.0):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd to preserve sequence length cleanly.")

        padding = dilation * (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(channels)

        self.dropout = nn.Dropout(dropout)

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


class TCNStem(nn.Module):
    """
    Per-detector TCN stem.

    Use two independent instances for separate H1/L1 weights.

    Input:
        (B, 1, T)

    Output:
        (B, C, T)
    """

    def __init__(self, hidden_channels: int = 32, input_kernel_size: int = 15, block_kernel_size: int = 7, dilations=None, dropout: float = 0.0):
        super().__init__()

        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32, 64, 128]

        if input_kernel_size % 2 == 0:
            raise ValueError("input_kernel_size should be odd to preserve sequence length cleanly.")

        self.input_conv = nn.Conv1d(in_channels=1, out_channels=hidden_channels, kernel_size=input_kernel_size, padding=input_kernel_size // 2)
        self.input_norm = nn.BatchNorm1d(hidden_channels)

        self.blocks = nn.ModuleList(
            [
                ResidualTCNBlock(channels=hidden_channels, kernel_size=block_kernel_size, dilation=d, dropout=dropout)
                for d in dilations
            ]
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.input_norm(x)
        x = F.relu(x)

        for block in self.blocks:
            x = block(x)

        return x


class full_module(nn.Module):
    """
    A_sep: separate-stem TCN + simple learned fusion.

    Architecture:
        H1 -> TCN stem A ----\
                              learned Conv1d fusion -> output
        L1 -> TCN stem B ----/

    Expected input:
        x: (batch, time, 2)

    Output:
        out: (batch, time, 1)

    By default, returns sigmoid probabilities for BCELoss compatibility.
    If using BCEWithLogitsLoss:
        full_module(return_logits=True)
    """

    def __init__(self,*args,hidden_channels: int = 32,stem_dilations=None,dropout: float = 0.0,return_logits: bool = False,**kwargs):
        super().__init__()

        if stem_dilations is None:
            stem_dilations = [1, 2, 4, 8, 16, 32, 64, 128]

        self.hidden_channels = hidden_channels
        self.return_logits = return_logits

        self.stem_A = TCNStem(hidden_channels=hidden_channels,input_kernel_size=15,block_kernel_size=7,dilations=stem_dilations,dropout=dropout)
        self.stem_B = TCNStem(hidden_channels=hidden_channels,input_kernel_size=15,block_kernel_size=7,dilations=stem_dilations,dropout=dropout)

        self.fusion_head = nn.Sequential(
            nn.Conv1d(2 * hidden_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(hidden_channels, 1, kernel_size=1),
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

        # Input: (B, T, 2)
        # A/B:   (B, 1, T)
        A = x[:, :, 0].unsqueeze(1)
        B = x[:, :, 1].unsqueeze(1)

        # Separate TCN stems.
        # A_feat/B_feat: (B, C, T)
        A_feat = self.stem_A(A)
        B_feat = self.stem_B(B)

        # Simple learned fusion only.
        # (B, 2C, T)
        features = torch.cat([A_feat, B_feat], dim=1)

        logits = self.fusion_head(features)

        # (B, 1, T) -> (B, T, 1)
        logits = logits.permute(0, 2, 1)

        if self.return_logits:
            return logits

        return torch.sigmoid(logits)
