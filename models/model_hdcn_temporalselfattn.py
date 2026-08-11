import torch
import torch.nn as nn
import torch.nn.functional as F


class SubModule(nn.Module):
    def __init__(self, inp_shape=(4096, 1)):
        super(SubModule, self).__init__()

        self.n_filters = 32
        self.filter_width = 2
        self.dilation_rates = [2**i for i in range(11)] * 3

        self.conv1_firstit = nn.Conv1d(
            1, 16, kernel_size=1, padding="same"
        )

        self.conv1_postfirstit = nn.ModuleList(
            [
                nn.Conv1d(16, 16, kernel_size=1, padding="same")
                for _ in self.dilation_rates
            ]
        )[:-1]

        self.convs_f = nn.ModuleList(
            [
                nn.Conv1d(
                    16,
                    self.n_filters,
                    kernel_size=self.filter_width,
                    padding="same",
                    dilation=dilation_rate,
                )
                for dilation_rate in self.dilation_rates
            ]
        )

        self.convs_g = nn.ModuleList(
            [
                nn.Conv1d(
                    16,
                    self.n_filters,
                    kernel_size=self.filter_width,
                    padding="same",
                    dilation=dilation_rate,
                )
                for dilation_rate in self.dilation_rates
            ]
        )

        self.conv2 = nn.ModuleList(
            [
                nn.Conv1d(
                    self.n_filters,
                    16,
                    kernel_size=1,
                    padding="same",
                )
                for _ in self.dilation_rates
            ]
        )

    def forward(self, x):
        skips = []

        for i, dilation_rate in enumerate(self.dilation_rates):
            conv1 = self.conv1_firstit if i == 0 else self.conv1_postfirstit[i - 1]

            x = F.relu(conv1(x))

            x_f = self.convs_f[i](x)
            x_g = self.convs_g[i](x)

            z = torch.tanh(x_f) * torch.sigmoid(x_g)
            z = F.relu(self.conv2[i](z))

            x = x + z
            skips.append(z)

        out = F.relu(torch.sum(torch.stack(skips), dim=0))

        return out


class TemporalSelfAttentionBlock(nn.Module):
    """
    Full-resolution temporal self-attention over fused HDCN features.

    Input:
        x: (batch, channels, time)

    Output:
        x: (batch, channels, time)
    """

    def __init__(
        self,
        channels: int = 32,
        num_heads: int = 1,
        dropout: float = 0.0,
        use_ffn: bool = False,
        ff_multiplier: int = 2,
    ):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(
                f"channels={channels} must be divisible by num_heads={num_heads}"
            )

        self.use_ffn = use_ffn

        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attn_out = nn.Linear(channels, channels)
        self.norm_attn = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

        if self.use_ffn:
            ff_dim = ff_multiplier * channels
            self.ffn = nn.Sequential(
                nn.Linear(channels, ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, channels),
            )
            self.norm_ffn = nn.LayerNorm(channels)

    def forward(self, x):
        # Conv format:      (B, C, T)
        # Attention format: (B, T, C)
        x_seq = x.permute(0, 2, 1)

        attn_out, _ = self.attn(
            query=x_seq,
            key=x_seq,
            value=x_seq,
            need_weights=False,
        )

        attn_out = self.attn_out(attn_out)
        x_seq = self.norm_attn(x_seq + self.dropout(attn_out))

        if self.use_ffn:
            ffn_out = self.ffn(x_seq)
            x_seq = self.norm_ffn(x_seq + self.dropout(ffn_out))

        # Back to Conv format: (B, C, T)
        x = x_seq.permute(0, 2, 1)

        return x


class full_module(nn.Module):
    """
    N: heavy HDCN + temporal self-attention.

    Architecture:
        H1 -> HDCN ----\
                        learned fusion -> temporal self-attention -> output
        L1 -> HDCN ----/

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
        dropout: float = 0.0,
        use_ffn: bool = False,
        ff_multiplier: int = 2,
        return_logits: bool = False,
        **kwargs,
    ):
        super(full_module, self).__init__()

        self.return_logits = return_logits

        self.sub_mod_A = SubModule()
        self.sub_mod_B = SubModule()

        # Each HDCN outputs 16 channels.
        # Concatenate H1/L1 -> 32 channels.
        # Then project to hidden_channels.
        self.detector_fusion = nn.Sequential(
            nn.Conv1d(32, hidden_channels, kernel_size=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.temporal_attention = TemporalSelfAttentionBlock(
            channels=hidden_channels,
            num_heads=num_heads,
            dropout=dropout,
            use_ffn=use_ffn,
            ff_multiplier=ff_multiplier,
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

        seq_len = x.shape[1]

        # Split H1/L1.
        # (B, T, 2) -> (B, 1, T)
        x_A = x[:, :, 0].unsqueeze(1)
        x_B = x[:, :, 1].unsqueeze(1)

        # Heavy per-detector HDCN encoders.
        # Outputs: (B, 16, T)
        x_A = self.sub_mod_A(x_A)
        x_B = self.sub_mod_B(x_B)

        # Learned detector fusion.
        # (B, 32, T) -> (B, hidden_channels, T)
        x = torch.cat([x_A, x_B], dim=1)
        x = self.detector_fusion(x)

        # Temporal self-attention over fused HDCN features.
        x = self.temporal_attention(x)

        logits = self.output_head(x)

        # (B, 1, T) -> (B, T, 1)
        logits = logits.permute(0, 2, 1)

        if self.return_logits:
            return logits

        return torch.sigmoid(logits)