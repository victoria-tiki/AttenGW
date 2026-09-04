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

    Each detector should get its own instance of this module if we want
    separate H1/L1 weights.

    Input:  (B, 1, T)
    Output: (B, C, T)
    """

    def __init__(self, hidden_channels: int = 32, kernel_size: int = 15, dilations=None, dropout: float = 0.0):
        super().__init__()

        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32]

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd to preserve sequence length cleanly.")

        self.input_conv = nn.Conv1d(in_channels=1, out_channels=hidden_channels, kernel_size=kernel_size,padding=kernel_size // 2)
        self.input_norm = nn.BatchNorm1d(hidden_channels)

        self.blocks = nn.ModuleList(
            [
                ResidualTCNBlock(
                    channels=hidden_channels,
                    kernel_size=7,
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


class CrossAttentionBlock(nn.Module):
    """
    Bidirectional cross-attention between detector feature streams.

    Inputs:
        A: (B, C, T_low)
        B: (B, C, T_low)

    Outputs:
        A_context: (B, C, T_low), A attending to B
        B_context: (B, C, T_low), B attending to A
    """

    def __init__(self, channels: int = 32, num_heads: int = 2, dropout: float = 0.0):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(
                f"channels={channels} must be divisible by num_heads={num_heads}"
            )

        self.attn_A_to_B = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn_B_to_A = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.norm_A = nn.LayerNorm(channels)
        self.norm_B = nn.LayerNorm(channels)

        self.out_A = nn.Linear(channels, channels)
        self.out_B = nn.Linear(channels, channels)

    def forward(self, A, B):
        # Conv format: (B, C, T)
        # Attention format: (B, T, C)
        A_seq = A.permute(0, 2, 1)
        B_seq = B.permute(0, 2, 1)

        A_attn, _ = self.attn_A_to_B(query=A_seq, key=B_seq, value=B_seq, need_weights=False)
        B_attn, _ = self.attn_B_to_A(query=B_seq, key=A_seq, value=A_seq, need_weights=False)

        # Residual + norm
        A_context = self.norm_A(A_seq + A_attn)
        B_context = self.norm_B(B_seq + B_attn)

        A_context = self.out_A(A_context)
        B_context = self.out_B(B_context)

        # Back to Conv format: (B, C, T)
        A_context = A_context.permute(0, 2, 1)
        B_context = B_context.permute(0, 2, 1)

        return A_context, B_context


class full_module(nn.Module):
    """
    Separate-stem version of the downsampled cross-attention model.

    Architecture:
        H1 -> TCN stem A ----\
                              downsample -> cross-attention -> upsample
        L1 -> TCN stem B ----/

        Then:
            concatenate full-res local features + attention context
            learned Conv1d fusion/output head

    Expected input:
        x: (batch, time, 2)

    Output:
        out: (batch, time, 1)

    By default, returns sigmoid probabilities for compatibility with BCELoss.
    If you switch to BCEWithLogitsLoss, use:
        full_module(return_logits=True)
    """

    def __init__(self, *args, hidden_channels: int = 32, stem_kernel_size: int = 15, stem_dilations=None, attention_heads: int = 2, downsample_factor: int = 4, dropout: float = 0.0, return_logits: bool = False, **kwargs):
        super().__init__()

        if stem_dilations is None:
            stem_dilations = [1, 2, 4, 8, 16, 32]

        self.hidden_channels = hidden_channels
        self.downsample_factor = downsample_factor
        self.return_logits = return_logits

        # Separate detector stems: independent weights for H1/L1.
        self.stem_A = TCNStem(hidden_channels=hidden_channels, kernel_size=stem_kernel_size, dilations=stem_dilations, dropout=dropout)
        self.stem_B = TCNStem(hidden_channels=hidden_channels, kernel_size=stem_kernel_size,dilations=stem_dilations,dropout=dropout)

        # Separate learned downsampling too, to keep detector branches independent
        # before cross-attention.
        self.downsample_A = nn.Conv1d(in_channels=hidden_channels,out_channels=hidden_channels,kernel_size=downsample_factor,stride=downsample_factor)
        self.downsample_B = nn.Conv1d(in_channels=hidden_channels,out_channels=hidden_channels,kernel_size=downsample_factor,stride=downsample_factor)
        self.cross_attention = CrossAttentionBlock(channels=hidden_channels,num_heads=attention_heads,dropout=dropout)

        # We concatenate:
        #   A full-res local features
        #   B full-res local features
        #   A attention context, upsampled
        #   B attention context, upsampled
        #
        # Total = 4 * hidden_channels.
        self.fusion = nn.Sequential(
            nn.Conv1d(4 * hidden_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
        )

        self.output_conv = nn.Conv1d(in_channels=hidden_channels,out_channels=1,kernel_size=1)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                f"Expected x with shape (batch, time, channels), got {x.shape}"
            )

        if x.shape[-1] != 2:
            raise ValueError(
                f"Expected last dimension to have size 2 for two detectors, got {x.shape}"
            )

        seq_len = x.shape[1]

        # Split detector channels.
        # Input x: (B, T, 2)
        # A/B:     (B, 1, T)
        A = x[:, :, 0].unsqueeze(1)
        B = x[:, :, 1].unsqueeze(1)

        # Separate local TCN feature extraction.
        # A_feat/B_feat: (B, C, T)
        A_feat = self.stem_A(A)
        B_feat = self.stem_B(B)

        # Separate downsampling before attention.
        # A_low/B_low: (B, C, T_low)
        A_low = self.downsample_A(A_feat)
        B_low = self.downsample_B(B_feat)

        # Bidirectional cross-detector attention.
        # A_ctx_low/B_ctx_low: (B, C, T_low)
        A_ctx_low, B_ctx_low = self.cross_attention(A_low, B_low)

        # Upsample attention context back to original sequence length.
        A_ctx = F.interpolate(A_ctx_low, size=seq_len, mode="linear", align_corners=False)
        B_ctx = F.interpolate(B_ctx_low, size=seq_len, mode="linear", align_corners=False)

        # Fuse full-resolution local features with lower-resolution attention context.
        fused = torch.cat([A_feat, B_feat, A_ctx, B_ctx], dim=1)

        fused = self.fusion(fused)
        logits = self.output_conv(fused)

        # Back to original output format: (B, T, 1)
        logits = logits.permute(0, 2, 1)

        if self.return_logits:
            return logits

        return torch.sigmoid(logits)
