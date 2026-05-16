from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import ensure_dirs
from src.utils import resolve_path


def classify_stocks(minute_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    ensure_dirs(config)
    daily = minute_df.groupby(["stock_code", "date"], as_index=False).agg(
        daily_amount=("amount", "sum"),
        daily_volume=("volume", "sum"),
    )
    summary = daily.groupby("stock_code", as_index=False).agg(
        avg_daily_amount=("daily_amount", "mean"),
        avg_daily_volume=("daily_volume", "mean"),
    )
    summary = summary.sort_values("avg_daily_amount", ascending=False).reset_index(drop=True)
    n = len(summary)
    high_cut = int(n * 0.30)
    medium_cut = int(n * 0.70)
    summary["liquidity_group"] = "low"
    summary.loc[: max(high_cut - 1, -1), "liquidity_group"] = "high"
    summary.loc[high_cut: max(medium_cut - 1, high_cut - 1), "liquidity_group"] = "medium"

    output_path = resolve_path(config["processed_data_dir"]) / "stock_liquidity_group.parquet"
    summary.to_parquet(output_path, index=False)
    return summary
