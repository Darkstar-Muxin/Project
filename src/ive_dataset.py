from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("PyTorch is required for IVE datasets. Install with: pip install torch") from exc


ID_COLUMNS = {"stock_code", "datetime", "date", "minute", "liquidity_group"}
LEAKY_FEATURE_COLUMNS = {
    "daily_volume",
    "daily_amount",
    "volume_ratio",
    "accumulated_volume_ratio",
    "amount_ratio",
    "accumulated_amount_ratio",
}


@dataclass
class Normalizer:
    mean: dict[str, float]
    std: dict[str, float]


def ive_label_columns(horizons: list[int]) -> list[str]:
    cols: list[str] = []
    for h in horizons:
        cols.extend([f"future_vwap_{h}", f"future_volume_{h}", f"future_volume_ratio_{h}", f"log_future_volume_ratio_{h}"])
    return cols


def get_ive_feature_columns(df: pd.DataFrame, horizons: list[int]) -> list[str]:
    excluded = set(ID_COLUMNS)
    excluded.update(LEAKY_FEATURE_COLUMNS)
    excluded.update(ive_label_columns(horizons))
    excluded.update(col for col in df.columns if col.startswith("future_") or col.startswith("log_future_"))
    cols = [col for col in df.columns if col not in excluded and pd.api.types.is_numeric_dtype(df[col])]
    return sorted(cols)


def fit_normalizer(df: pd.DataFrame, feature_columns: list[str]) -> Normalizer:
    work = df[feature_columns].replace([np.inf, -np.inf], np.nan)
    mean = work.mean(numeric_only=True).fillna(0.0).to_dict()
    std = work.std(numeric_only=True).replace(0, np.nan).fillna(1.0).to_dict()
    return Normalizer(mean={k: float(v) for k, v in mean.items()}, std={k: float(v) for k, v in std.items()})


def transform_features(df: pd.DataFrame, feature_columns: list[str], normalizer: Normalizer) -> np.ndarray:
    work = df.reindex(columns=feature_columns).replace([np.inf, -np.inf], np.nan)
    for col in feature_columns:
        work[col] = (pd.to_numeric(work[col], errors="coerce").fillna(normalizer.mean.get(col, 0.0)) - normalizer.mean.get(col, 0.0)) / normalizer.std.get(col, 1.0)
    return work.to_numpy(dtype=np.float32)


def build_vocab(values: pd.Series) -> dict[str, int]:
    uniq = sorted(str(v) for v in values.dropna().astype(str).unique())
    return {value: idx for idx, value in enumerate(uniq)}


def _valid_target_mask(df: pd.DataFrame, horizons: list[int]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for h in horizons:
        mask &= df[f"log_future_volume_ratio_{h}"].notna()
        mask &= df[f"future_vwap_{h}"].notna()
    return mask


class IVEDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        horizons: list[int],
        stock_vocab: dict[str, int],
        group_vocab: dict[str, int],
        normalizer: Normalizer,
        context_length: int = 390,
    ) -> None:
        self.df = df.sort_values(["stock_code", "date", "datetime"]).reset_index(drop=True).copy()
        self.df["datetime"] = pd.to_datetime(self.df["datetime"])
        self.feature_columns = feature_columns
        self.horizons = horizons
        self.stock_vocab = stock_vocab
        self.group_vocab = group_vocab
        self.normalizer = normalizer
        self.context_length = int(context_length)
        self.features = transform_features(self.df, feature_columns, normalizer)
        self.stock_ids = self.df["stock_code"].astype(str).map(stock_vocab).fillna(0).to_numpy(dtype=np.int64)
        self.group_ids = self.df["liquidity_group"].astype(str).map(group_vocab).fillna(0).to_numpy(dtype=np.int64)
        self.volume_targets = self.df[[f"log_future_volume_ratio_{h}" for h in horizons]].to_numpy(dtype=np.float32)
        base = self.df["vwap"].replace(0, np.nan).fillna(self.df["close"].replace(0, np.nan))
        self.vwap_targets = []
        for h in horizons:
            target = self.df[f"future_vwap_{h}"] / base - 1
            self.vwap_targets.append(target.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32))
        self.vwap_targets = np.vstack(self.vwap_targets).T.astype(np.float32)
        valid_mask = _valid_target_mask(self.df, horizons).to_numpy()
        self.indices = np.flatnonzero(valid_mask)
        self.start_bounds = np.zeros(len(self.df), dtype=np.int64)
        for _, group in self.df.groupby(["stock_code", "date"], sort=False):
            idx = group.index.to_numpy()
            self.start_bounds[idx] = int(idx[0])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row_idx = int(self.indices[item])
        start_bound = int(self.start_bounds[row_idx])
        start = max(start_bound, row_idx - self.context_length + 1)
        seq = self.features[start : row_idx + 1]
        valid_len = seq.shape[0]
        padded = np.zeros((self.context_length, len(self.feature_columns)), dtype=np.float32)
        padded[-valid_len:] = seq
        padding_mask = np.ones(self.context_length, dtype=bool)
        padding_mask[-valid_len:] = False
        return {
            "x": torch.from_numpy(padded),
            "padding_mask": torch.from_numpy(padding_mask),
            "stock_id": torch.tensor(self.stock_ids[row_idx], dtype=torch.long),
            "group_id": torch.tensor(self.group_ids[row_idx], dtype=torch.long),
            "volume_target": torch.from_numpy(self.volume_targets[row_idx]),
            "vwap_target": torch.from_numpy(self.vwap_targets[row_idx]),
        }


def metadata_from_training(
    df: pd.DataFrame,
    feature_columns: list[str],
    horizons: list[int],
    stock_vocab: dict[str, int],
    group_vocab: dict[str, int],
    normalizer: Normalizer,
    context_length: int,
) -> dict[str, Any]:
    daily_volume_prior = (
        df.groupby("stock_code")["daily_volume"].mean().replace([np.inf, -np.inf], np.nan).dropna().to_dict()
        if "daily_volume" in df.columns
        else {}
    )
    group_daily_volume_prior = (
        df.groupby("liquidity_group")["daily_volume"].mean().replace([np.inf, -np.inf], np.nan).dropna().to_dict()
        if "daily_volume" in df.columns
        else {}
    )
    return {
        "feature_columns": feature_columns,
        "horizons": horizons,
        "stock_vocab": stock_vocab,
        "group_vocab": group_vocab,
        "normalizer": {"mean": normalizer.mean, "std": normalizer.std},
        "context_length": context_length,
        "daily_volume_prior": {str(k): float(v) for k, v in daily_volume_prior.items()},
        "group_daily_volume_prior": {str(k): float(v) for k, v in group_daily_volume_prior.items()},
    }
