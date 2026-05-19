from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _build_group_future_labels(group: pd.DataFrame, horizons: Iterable[int], ratio_scale: float) -> pd.DataFrame:
    group = group.sort_values("datetime").copy()
    dt_ns = group["datetime"].astype("int64").to_numpy()
    volume = group["volume"].fillna(0).to_numpy(dtype=float)
    amount = group["amount"].fillna(0).to_numpy(dtype=float)
    daily_volume = float(np.nansum(volume))
    cum_volume = np.concatenate([[0.0], np.cumsum(volume)])
    cum_amount = np.concatenate([[0.0], np.cumsum(amount)])

    for h in horizons:
        end_ns = dt_ns + int(h) * 60 * 1_000_000_000
        end_idx = np.searchsorted(dt_ns, end_ns, side="left")
        start_idx = np.arange(len(group))
        future_volume = cum_volume[end_idx] - cum_volume[start_idx]
        future_amount = cum_amount[end_idx] - cum_amount[start_idx]
        ratio = np.where(daily_volume > 0, future_volume / daily_volume, np.nan)
        group[f"future_volume_{h}"] = future_volume
        group[f"future_vwap_{h}"] = np.where(future_volume > 0, future_amount / future_volume, np.nan)
        group[f"future_volume_ratio_{h}"] = ratio
        group[f"log_future_volume_ratio_{h}"] = np.log1p(np.maximum(ratio, 0) * ratio_scale)
    return group


def build_future_labels(
    minute_df: pd.DataFrame,
    horizons: Iterable[int],
    participation_limit: float = 0.30,
    volume_ratio_scale: float = 10000.0,
) -> pd.DataFrame:
    del participation_limit
    df = minute_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    grouped = df.groupby(["stock_code", "date"], sort=False)
    labeled = [_build_group_future_labels(group, horizons, volume_ratio_scale) for _, group in grouped]
    return pd.concat(labeled, ignore_index=True) if labeled else df
