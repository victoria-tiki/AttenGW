import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class LocalConvEmbedding(nn.Module):
    """
    Small local embedding before Mamba.

    Input:
        x: (batch, time, 2)

    Output:
        x: (batch, time, hidden_dim)
    """

    def __init__(self, input_channels: int = 2, hidden_dim: int = 64, kernel_size: int = 15, dropout: float = 0.0):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd to preserve sequence length.")

        self.net = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # (B, T, 2) -> (B, 2, T)
        x = x.permute(0, 2, 1)

        # (B, hidden_dim, T)
        x = self.net(x)

        # (B, T, hidden_dim)
        x = x.permute(0, 2, 1)

        return x


class BiMambaResidualBlock(nn.Module):
    """
    Bidirectional residual Mamba block.

    Input:
        x: (batch, time, dim)

    Output:
        x: (batch, time, dim)

    Uses:
        forward Mamba over x
        backward Mamba over reversed x
        concatenate/squash back to dim
        residual connection
    """

    def __init__(self, dim: int = 64, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.0):
        super().__init__()

        if Mamba is None:
            raise ImportError(
                "mamba_ssm is not installed. Install it with something like:\n"
                "    pip install mamba-ssm causal-conv1d\n"
                "or load/install the package in your cluster environment."
            )

        self.norm = nn.LayerNorm(dim)

        self.mamba_fwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

        self.mamba_bwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

        self.proj = nn.Linear(2 * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x_norm = self.norm(x)

        y_fwd = self.mamba_fwd(x_norm)

        x_rev = torch.flip(x_norm, dims=[1])
        y_bwd = self.mamba_bwd(x_rev)
        y_bwd = torch.flip(y_bwd, dims=[1])

        y = torch.cat([y_fwd, y_bwd], dim=-1)
        y = self.proj(y)
        y = self.dropout(y)

        return residual + y


class full_module(nn.Module):
    """
    Q_bi_local: local Conv embedding + bidirectional Mamba detector.

    Architecture:
        input H1/L1 together
        -> small local Conv1d embedding
        -> stack of bidirectional residual Mamba blocks
        -> Conv1d output head
        -> timestep probability

    Expected input:
        x: (batch, time, 2)

    Output:
        out: (batch, time, 1)

    By default, returns sigmoid probabilities for compatibility with BCELoss.

    If switching to BCEWithLogitsLoss, instantiate with:
        full_module(return_logits=True)
    """

    def __init__(self, *args, hidden_dim: int = 64, num_layers: int = 4, d_state: int = 16, d_conv: int = 4, expand: int = 2, local_kernel_size: int = 15, dropout: float = 0.0, return_logits: bool = False, **kwargs):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.return_logits = return_logits

        self.embedding = LocalConvEmbedding(input_channels=2, hidden_dim=hidden_dim, kernel_size=local_kernel_size, dropout=dropout)

        self.blocks = nn.ModuleList(
            [
                BiMambaResidualBlock(
                    dim=hidden_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(hidden_dim)

        self.output_head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
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

        # (B, T, 2) -> (B, T, hidden_dim)
        x = self.embedding(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        # Conv1d head wants (B, C, T)
        x = x.permute(0, 2, 1)

        logits = self.output_head(x)

        # (B, 1, T) -> (B, T, 1)
        logits = logits.permute(0, 2, 1)

        if self.return_logits:
            return logits

        return torch.sigmoid(logits)
