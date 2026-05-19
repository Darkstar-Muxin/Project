from __future__ import annotations

import math

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("PyTorch is required for the IVE model. Install with: pip install torch") from exc


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class IVEModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_stocks: int,
        num_groups: int,
        num_horizons: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 390,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(num_features, d_model)
        self.stock_embedding = nn.Embedding(max(num_stocks, 1), d_model)
        self.group_embedding = nn.Embedding(max(num_groups, 1), d_model)
        self.position_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_len + 8)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.volume_mu_head = nn.Linear(d_model, num_horizons)
        self.volume_log_sigma_head = nn.Linear(d_model, num_horizons)
        self.vwap_return_head = nn.Linear(d_model, num_horizons)

    def forward(
        self,
        x: torch.Tensor,
        stock_id: torch.Tensor,
        group_id: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.input_projection(x)
        hidden = hidden + self.stock_embedding(stock_id).unsqueeze(1)
        hidden = hidden + self.group_embedding(group_id).unsqueeze(1)
        hidden = self.position_encoding(hidden)
        encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)
        if padding_mask is None:
            pooled = encoded[:, -1, :]
        else:
            lengths = (~padding_mask).sum(dim=1).clamp(min=1)
            batch_index = torch.arange(encoded.size(0), device=encoded.device)
            pooled = encoded[batch_index, lengths - 1, :]
        pooled = self.norm(pooled)
        return {
            "volume_mu": self.volume_mu_head(pooled),
            "volume_log_sigma": self.volume_log_sigma_head(pooled).clamp(min=-6.0, max=4.0),
            "vwap_return": self.vwap_return_head(pooled),
        }
