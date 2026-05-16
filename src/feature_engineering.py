from __future__ import annotations

from typing import Any
from time import perf_counter

import numpy as np
import pandas as pd

from src.config import ensure_dirs
from src.label_builder import build_future_labels
from src.utils import resolve_path, safe_divide


def _log_step(message: str) -> float:
    print(f"[feature] {message}", flush=True)
    return perf_counter()


def _log_done(message: str, start: float) -> None:
    elapsed = perf_counter() - start
    print(f"[feature] {message} done in {elapsed:.1f}s", flush=True)


def _add_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["stock_code", "datetime"]).copy()
    grouped = df.groupby("stock_code", group_keys=False)
    df["return_1m"] = grouped["close"].pct_change()
    df["return_5m"] = grouped["close"].pct_change(5)
    df["volume_5m_sum"] = grouped["volume"].rolling(5, min_periods=1).sum().reset_index(level=0, drop=True)
    df["amount_5m_sum"] = grouped["amount"].rolling(5, min_periods=1).sum().reset_index(level=0, drop=True)
    df["vwap_5m"] = safe_divide(df["amount_5m_sum"], df["volume_5m_sum"])
    df["volatility_5m"] = grouped["return_1m"].rolling(5, min_periods=2).std().reset_index(level=0, drop=True)
    df["volume_10m_sum"] = grouped["volume"].rolling(10, min_periods=1).sum().reset_index(level=0, drop=True)
    df["amount_10m_sum"] = grouped["amount"].rolling(10, min_periods=1).sum().reset_index(level=0, drop=True)
    df["volatility_10m"] = grouped["return_1m"].rolling(10, min_periods=2).std().reset_index(level=0, drop=True)
    df["hour"] = df["datetime"].dt.hour
    df["minute_of_day"] = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    return df


def _add_historical_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby(["stock_code", "date"], as_index=False).agg(
        daily_volume=("volume", "sum"),
        daily_amount=("amount", "sum"),
        daily_vwap=("amount", lambda x: np.nan),
    )
    daily["daily_vwap"] = daily["daily_amount"] / daily["daily_volume"].replace(0, np.nan)
    daily = daily.sort_values(["stock_code", "date"])
    for src, dst in [
        ("daily_volume", "stock_rolling_volume_mean_5d"),
        ("daily_amount", "stock_rolling_amount_mean_5d"),
        ("daily_vwap", "stock_rolling_vwap_mean_5d"),
    ]:
        daily[dst] = (
            daily.groupby("stock_code")[src]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )
    return df.merge(
        daily[
            [
                "stock_code",
                "date",
                "stock_rolling_volume_mean_5d",
                "stock_rolling_amount_mean_5d",
                "stock_rolling_vwap_mean_5d",
            ]
        ],
        on=["stock_code", "date"],
        how="left",
    )


def _add_same_minute_features(df: pd.DataFrame) -> pd.DataFrame:
    minute_stats = df.groupby(["stock_code", "date", "minute"], as_index=False).agg(
        minute_volume=("volume", "sum"),
        minute_amount=("amount", "sum"),
        minute_vwap=("vwap", "mean"),
    )
    minute_stats = minute_stats.sort_values(["stock_code", "minute", "date"])
    for src, dst in [
        ("minute_volume", "same_minute_volume_mean_5d"),
        ("minute_amount", "same_minute_amount_mean_5d"),
        ("minute_vwap", "same_minute_vwap_mean_5d"),
    ]:
        minute_stats[dst] = (
            minute_stats.groupby(["stock_code", "minute"])[src]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )
    return df.merge(
        minute_stats[
            [
                "stock_code",
                "date",
                "minute",
                "same_minute_volume_mean_5d",
                "same_minute_amount_mean_5d",
                "same_minute_vwap_mean_5d",
            ]
        ],
        on=["stock_code", "date", "minute"],
        how="left",
    )


def build_features(config: dict[str, Any]) -> pd.DataFrame:
    ensure_dirs(config)
    processed_dir = resolve_path(config["processed_data_dir"])
    feature_dir = resolve_path(config["feature_data_dir"])

    start = _log_step("loading minute_data.parquet")
    minute_df = pd.read_parquet(processed_dir / "minute_data.parquet")
    _log_done(f"loaded minute rows={len(minute_df):,}", start)

    start = _log_step("loading stock_liquidity_group.parquet")
    liquidity = pd.read_parquet(processed_dir / "stock_liquidity_group.parquet")
    _log_done(f"loaded stock groups={len(liquidity):,}", start)

    start = _log_step("merging liquidity group")
    minute_df["datetime"] = pd.to_datetime(minute_df["datetime"])
    df = minute_df.merge(liquidity[["stock_code", "liquidity_group"]], on="stock_code", how="left")
    _log_done(f"merged rows={len(df):,}", start)

    start = _log_step("building intraday rolling features")
    df = _add_intraday_features(df)
    _log_done("intraday rolling features", start)

    start = _log_step("building historical 5-day stock features")
    df = _add_historical_daily_features(df)
    _log_done("historical 5-day stock features", start)

    start = _log_step("building same-minute 5-day features")
    df = _add_same_minute_features(df)
    _log_done("same-minute 5-day features", start)

    start = _log_step("building future labels")
    df = build_future_labels(df, config["horizons"], config.get("participation_limit", 0.30))
    _log_done("future labels", start)

    start = _log_step("sorting and saving model_dataset.parquet")
    df = df.sort_values(["stock_code", "datetime"]).reset_index(drop=True)

    output_path = feature_dir / "model_dataset.parquet"
    df.to_parquet(output_path, index=False)
    _log_done(f"saved {output_path} rows={len(df):,}", start)
    return df
