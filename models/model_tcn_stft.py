import math
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
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm1 = nn.BatchNorm1d(channels)

        self.conv2 = nn.Conv1d(
            channels,
            channels,
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


class TimeBranch(nn.Module):
    """
    Lightweight time-domain branch.

    Input:
        x: (batch, time, 2)

    Output:
        features: (batch, channels, time)
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
        # (B, T, 2) -> (B, 2, T)
        x = x.permute(0, 2, 1)

        x = self.input_conv(x)
        x = self.input_norm(x)
        x = F.relu(x)

        for block in self.blocks:
            x = block(x)

        return x


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for token sequences.

    Input/output:
        (batch, tokens, dim)
    """

    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        token_len = x.shape[1]

        if token_len > self.pe.shape[1]:
            raise ValueError(
                f"Token length {token_len} exceeds max positional length {self.pe.shape[1]}"
            )

        return x + self.pe[:, :token_len, :].to(dtype=x.dtype, device=x.device)


class STFTTransformerBranch(nn.Module):
    """
    Time-frequency branch.

    Computes STFT internally, creates one token per STFT time frame using
    all frequency bins and both detectors, applies a Transformer over STFT
    time frames, then upsamples back to the original time length.

    Input:
        x: (batch, time, 2)

    Output:
        features: (batch, channels, time)
    """

    def __init__(
        self,
        channels: int = 32,
        n_fft: int = 128,
        hop_length: int = 16,
        win_length: int = 128,
        transformer_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 1,
        ff_multiplier: int = 2,
        dropout: float = 0.0,
        max_frames: int = 1024,
    ):
        super().__init__()

        if transformer_dim % num_heads != 0:
            raise ValueError(
                f"transformer_dim={transformer_dim} must be divisible by num_heads={num_heads}"
            )

        self.channels = channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        freq_bins = n_fft // 2 + 1
        self.freq_bins = freq_bins

        # One token per STFT frame.
        # Token vector includes both detector channels and all frequency bins.
        token_in_dim = 2 * freq_bins

        self.token_proj = nn.Sequential(
            nn.Linear(token_in_dim, transformer_dim),
            nn.LayerNorm(transformer_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.pos_enc = SinusoidalPositionalEncoding(
            dim=transformer_dim,
            max_len=max_frames,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=ff_multiplier * transformer_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Project each STFT-frame token back to channels, then interpolate
        # along frame-time to the original sample-time.
        self.frame_to_channels = nn.Sequential(
            nn.Linear(transformer_dim, channels),
            nn.LayerNorm(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        window = torch.hann_window(win_length)
        self.register_buffer("window", window, persistent=False)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected x as (batch, time, 2), got {x.shape}")

        batch_size, seq_len, n_channels = x.shape

        if n_channels != 2:
            raise ValueError(f"Expected two detector channels, got {n_channels}")

        # (B, T, 2) -> (B, 2, T) -> (B*2, T)
        x_det = x.permute(0, 2, 1).contiguous()
        x_flat = x_det.view(batch_size * 2, seq_len)

        # STFT: (B*2, F, frames), complex
        stft = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=x.device, dtype=x.dtype),
            center=True,
            return_complex=True,
        )

        # Log magnitude: (B*2, F, frames)
        mag = torch.log1p(stft.abs())

        # Normalize each spectrogram roughly per sample/detector.
        # This prevents absolute STFT scale from dominating the branch.
        mag_mean = mag.mean(dim=(1, 2), keepdim=True)
        mag_std = mag.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        mag = (mag - mag_mean) / mag_std

        _, freq_bins, n_frames = mag.shape

        # (B*2, F, frames) -> (B, 2, F, frames)
        mag = mag.view(batch_size, 2, freq_bins, n_frames)

        # One token per STFT frame: concatenate detectors and frequencies.
        # (B, 2, F, frames) -> (B, frames, 2*F)
        tokens = mag.permute(0, 3, 1, 2).contiguous()
        tokens = tokens.view(batch_size, n_frames, 2 * freq_bins)

        tokens = self.token_proj(tokens)
        tokens = self.pos_enc(tokens)
        tokens = self.transformer(tokens)

        # (B, frames, transformer_dim) -> (B, frames, channels)
        frame_features = self.frame_to_channels(tokens)

        # (B, frames, channels) -> (B, channels, frames)
        frame_features = frame_features.permute(0, 2, 1)

        # Upsample STFT-frame features back to original sample length.
        features = F.interpolate(
            frame_features,
            size=seq_len,
            mode="linear",
            align_corners=False,
        )

        return features


class full_module(nn.Module):
    """
    S1: time-domain TCN branch + STFT Transformer branch.

    Architecture:
        input H1/L1 time series
        -> time-domain TCN branch
        -> STFT/log-magnitude branch + Transformer over STFT frames
        -> concatenate branches
        -> Conv1d fusion/output head

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
        time_channels: int = 32,
        tf_channels: int = 32,
        n_fft: int = 128,
        hop_length: int = 16,
        win_length: int = 128,
        transformer_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 1,
        ff_multiplier: int = 2,
        dropout: float = 0.0,
        return_logits: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.return_logits = return_logits

        self.time_branch = TimeBranch(
            channels=time_channels,
            dropout=dropout,
        )

        self.tf_branch = STFTTransformerBranch(
            channels=tf_channels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            transformer_dim=transformer_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_multiplier=ff_multiplier,
            dropout=dropout,
        )

        fused_channels = time_channels + tf_channels

        self.fusion_head = nn.Sequential(
            nn.Conv1d(fused_channels, fused_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(fused_channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(fused_channels, time_channels, kernel_size=1),
            nn.BatchNorm1d(time_channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(time_channels, 1, kernel_size=1),
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

        time_features = self.time_branch(x)
        tf_features = self.tf_branch(x)

        features = torch.cat([time_features, tf_features], dim=1)

        logits = self.fusion_head(features)

        # (B, 1, T) -> (B, T, 1)
        logits = logits.permute(0, 2, 1)

        if self.return_logits:
            return logits

        return torch.sigmoid(logits)