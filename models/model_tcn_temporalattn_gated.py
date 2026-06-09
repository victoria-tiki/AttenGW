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
            raise ValueError("kernel_size should be odd to preserve sequence length.")

        padding = dilation * (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm1 = nn.BatchNorm1d(channels)

        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
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


class EarlyFusionTCNStem(nn.Module):
    """
    Early-fusion TCN stem.

    Input:
        (batch, 2, time)

    Output:
        (batch, channels, time)
    """

    def __init__(
        self,
        channels: int = 32,
        input_kernel_size: int = 15,
        block_kernel_size: int = 7,
        dilations=None,
        dropout: float = 0.0,
    ):
        super().__init__()

        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32, 64, 128]

        if input_kernel_size % 2 == 0:
            raise ValueError("input_kernel_size should be odd to preserve sequence length.")

        self.input_conv = nn.Conv1d(
            in_channels=2,
            out_channels=channels,
            kernel_size=input_kernel_size,
            padding=input_kernel_size // 2,
        )
        self.input_norm = nn.BatchNorm1d(channels)

        self.blocks = nn.ModuleList(
            [
                ResidualTCNBlock(
                    channels=channels,
                    kernel_size=block_kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
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


class GatedTemporalAttentionBlock(nn.Module):
    """
    Temporal self-attention used as a gated context/modulation mechanism.

    Input:
        x: (batch, channels, time)

    Output:
        x_out: (batch, channels, time)

    Mechanism:
        attention_context = SelfAttention(x)
        gate = sigmoid(Conv1d([x, attention_context]))
        x_out = LayerNorm(x + gate * attention_context)
    """

    def __init__(
        self,
        channels: int = 32,
        num_heads: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(
                f"channels={channels} must be divisible by num_heads={num_heads}"
            )

        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attn_out = nn.Linear(channels, channels)

        # Gate sees both original TCN features and attention context.
        self.gate = nn.Sequential(
            nn.Conv1d(2 * channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T)
        x_seq = x.permute(0, 2, 1)  # (B, T, C)

        attn_context, _ = self.attn(
            query=x_seq,
            key=x_seq,
            value=x_seq,
            need_weights=False,
        )

        attn_context = self.attn_out(attn_context)
        attn_context = self.dropout(attn_context)

        # Back to Conv format for gate.
        attn_context_conv = attn_context.permute(0, 2, 1)  # (B, C, T)

        gate_input = torch.cat([x, attn_context_conv], dim=1)  # (B, 2C, T)
        gate = self.gate(gate_input)  # (B, C, T)

        updated = x + gate * attn_context_conv  # (B, C, T)

        # LayerNorm over feature/channel dimension.
        updated = self.norm(updated.permute(0, 2, 1)).permute(0, 2, 1)

        return updated


class full_module(nn.Module):
    """
    L: early-fusion TCN + gated temporal self-attention.

    Architecture:
        input H1/L1 together
        -> early-fusion TCN stem
        -> temporal self-attention context
        -> learned gate controls attention injection
        -> Conv1d output head

    Expected input:
        x: (batch, time, 2)

    Output:
        out: (batch, time, 1)

    By default, returns sigmoid probabilities for compatibility with BCELoss.

    If switching to BCEWithLogitsLoss, instantiate with:
        full_module(return_logits=True)
    """

    def __init__(
        self,
        *args,
        hidden_channels: int = 32,
        num_heads: int = 1,
        stem_dilations=None,
        dropout: float = 0.0,
        return_logits: bool = False,
        **kwargs,
    ):
        super().__init__()

        if stem_dilations is None:
            stem_dilations = [1, 2, 4, 8, 16, 32, 64, 128]

        self.hidden_channels = hidden_channels
        self.return_logits = return_logits

        self.stem = EarlyFusionTCNStem(
            channels=hidden_channels,
            input_kernel_size=15,
            block_kernel_size=7,
            dilations=stem_dilations,
            dropout=dropout,
        )

        self.gated_attention = GatedTemporalAttentionBlock(
            channels=hidden_channels,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.output_head = nn.Sequential(
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

        # Input x: (B, T, 2)
        # Conv1d wants: (B, 2, T)
        x = x.permute(0, 2, 1)

        # TCN features:
        # (B, hidden_channels, T)
        x = self.stem(x)

        # Gated temporal-attention refinement:
        # (B, hidden_channels, T)
        x = self.gated_attention(x)

        logits = self.output_head(x)

        # (B, 1, T) -> (B, T, 1)
        logits = logits.permute(0, 2, 1)

        if self.return_logits:
            return logits

        return torch.sigmoid(logits)