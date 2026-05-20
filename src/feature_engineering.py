from __future__ import annotations

from time import perf_counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import ensure_dirs
from src.label_builder import build_future_labels
from src.utils import resolve_path, safe_divide


def _log_step(message: str) -> float:
    print(f"[feature] {message}", flush=True)
    return perf_counter()


def _log_done(message: str, start: float) -> None:
    print(f"[feature] {message} done in {perf_counter() - start:.1f}s", flush=True)


def _add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["stock_code", "datetime"]).copy()
    grouped = df.groupby("stock_code", group_keys=False)
    for col in ["buy_volume", "sell_volume", "buy_amount", "sell_amount", "trade_count"]:
        if col not in df.columns:
            df[col] = 0.0
    df["price"] = df["close"]
    df["buy_sell_volume_imbalance"] = safe_divide(df["buy_volume"] - df["sell_volume"], df["volume"])
    df["buy_sell_amount_imbalance"] = safe_divide(df["buy_amount"] - df["sell_amount"], df["amount"])
    df["return_1m"] = grouped["close"].pct_change()
    df["return_5m"] = grouped["close"].pct_change(5)
    df["return_10m"] = grouped["close"].pct_change(10)
    for window in [5, 10]:
        df[f"volume_{window}m_sum"] = grouped["volume"].rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
        df[f"amount_{window}m_sum"] = grouped["amount"].rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
        df[f"vwap_{window}m"] = safe_divide(df[f"amount_{window}m_sum"], df[f"volume_{window}m_sum"])
        df[f"volatility_{window}m"] = grouped["return_1m"].rolling(window, min_periods=2).std().reset_index(level=0, drop=True)
        df[f"vwap_deviation_{window}m"] = safe_divide(df["vwap"], df[f"vwap_{window}m"]) - 1
    df["minute_of_day"] = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    df["minutes_from_open"] = df["minute_of_day"] - (9 * 60 + 30)
    df["minutes_to_close"] = (15 * 60) - df["minute_of_day"]
    df["is_morning_session"] = (df["datetime"].dt.hour < 12).astype(int)
    df["abs_time_sin"] = np.sin(2 * np.pi * df["minute_of_day"] / (24 * 60))
    df["abs_time_cos"] = np.cos(2 * np.pi * df["minute_of_day"] / (24 * 60))
    return df


def _add_intraday_curve_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["stock_code", "date", "datetime"]).copy()
    grouped = df.groupby(["stock_code", "date"], group_keys=False)
    df["daily_volume"] = grouped["volume"].transform("sum")
    df["daily_amount"] = grouped["amount"].transform("sum")
    df["cum_volume"] = grouped["volume"].cumsum()
    df["cum_amount"] = grouped["amount"].cumsum()
    df["volume_ratio"] = safe_divide(df["volume"], df["daily_volume"])
    df["accumulated_volume_ratio"] = safe_divide(df["cum_volume"], df["daily_volume"])
    df["amount_ratio"] = safe_divide(df["amount"], df["daily_amount"])
    df["accumulated_amount_ratio"] = safe_divide(df["cum_amount"], df["daily_amount"])
    return df


def _daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby(["stock_code", "date"], as_index=False).agg(
        daily_total_volume=("volume", "sum"),
        daily_total_amount=("amount", "sum"),
    )
    daily["daily_total_vwap"] = safe_divide(daily["daily_total_amount"], daily["daily_total_volume"])
    daily = daily.sort_values(["stock_code", "date"])
    for window in [5, 10]:
        for src, dst in [
            ("daily_total_volume", f"stock_rolling_volume_mean_{window}d"),
            ("daily_total_amount", f"stock_rolling_amount_mean_{window}d"),
            ("daily_total_vwap", f"stock_rolling_vwap_mean_{window}d"),
        ]:
            daily[dst] = daily.groupby("stock_code")[src].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )
    return daily


def _add_historical_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    daily = _daily_summary(df)
    keep_cols = ["stock_code", "date"]
    keep_cols += [col for col in daily.columns if col.startswith("stock_rolling_")]
    return df.merge(daily[keep_cols], on=["stock_code", "date"], how="left")


def _add_same_minute_features(df: pd.DataFrame) -> pd.DataFrame:
    minute_stats = df.groupby(["stock_code", "date", "minute"], as_index=False).agg(
        minute_volume=("volume", "sum"),
        minute_amount=("amount", "sum"),
        minute_vwap=("vwap", "mean"),
        minute_volume_ratio=("volume_ratio", "sum"),
        minute_accumulated_volume_ratio=("accumulated_volume_ratio", "max"),
        minute_amount_ratio=("amount_ratio", "sum"),
        minute_accumulated_amount_ratio=("accumulated_amount_ratio", "max"),
    )
    minute_stats = minute_stats.sort_values(["stock_code", "minute", "date"])
    for window in [5, 10]:
        for src, dst in [
            ("minute_volume", f"same_minute_volume_mean_{window}d"),
            ("minute_amount", f"same_minute_amount_mean_{window}d"),
            ("minute_vwap", f"same_minute_vwap_mean_{window}d"),
            ("minute_volume_ratio", f"same_minute_volume_ratio_mean_{window}d"),
            ("minute_accumulated_volume_ratio", f"same_minute_accumulated_volume_ratio_mean_{window}d"),
            ("minute_amount_ratio", f"same_minute_amount_ratio_mean_{window}d"),
            ("minute_accumulated_amount_ratio", f"same_minute_accumulated_amount_ratio_mean_{window}d"),
        ]:
            minute_stats[dst] = minute_stats.groupby(["stock_code", "minute"])[src].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )
    keep_cols = ["stock_code", "date", "minute"]
    keep_cols += [col for col in minute_stats.columns if col.startswith("same_minute_")]
    return df.merge(minute_stats[keep_cols], on=["stock_code", "date", "minute"], how="left")


def _add_group_minute_features(df: pd.DataFrame) -> pd.DataFrame:
    if "liquidity_group" not in df.columns:
        return df
    stats = df.groupby(["liquidity_group", "date", "minute"], as_index=False).agg(
        group_minute_volume=("volume", "mean"),
        group_minute_amount=("amount", "mean"),
        group_minute_vwap=("vwap", "mean"),
        group_minute_volume_ratio=("volume_ratio", "mean"),
    )
    stats = stats.sort_values(["liquidity_group", "minute", "date"])
    for window in [5, 10]:
        for src, dst in [
            ("group_minute_volume", f"group_same_minute_volume_mean_{window}d"),
            ("group_minute_amount", f"group_same_minute_amount_mean_{window}d"),
            ("group_minute_vwap", f"group_same_minute_vwap_mean_{window}d"),
            ("group_minute_volume_ratio", f"group_same_minute_volume_ratio_mean_{window}d"),
        ]:
            stats[dst] = stats.groupby(["liquidity_group", "minute"])[src].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )
    keep_cols = ["liquidity_group", "date", "minute"]
    keep_cols += [col for col in stats.columns if col.startswith("group_same_minute_")]
    return df.merge(stats[keep_cols], on=["liquidity_group", "date", "minute"], how="left")


def _add_static_code_features(df: pd.DataFrame) -> pd.DataFrame:
    code = df["stock_code"].astype(str).str.lower()
    df["is_sh"] = code.str.endswith(".sh").astype(int)
    df["is_sz"] = code.str.endswith(".sz").astype(int)
    return df


def _date_from_path(path: Path) -> str:
    return path.stem[:8]


def _months_filter(paths: list[Path], months: list[int | str] | None) -> list[Path]:
    if not months:
        return paths
    month_set = {str(month) for month in months}
    return [path for path in paths if _date_from_path(path)[:6] in month_set]


def _minute_part_paths(config: dict[str, Any]) -> list[Path]:
    minute_parts_dir = resolve_path(config.get("minute_parts_dir", "data/processed/minute_parts"))
    paths = sorted(path for path in minute_parts_dir.glob("*.parquet") if path.stem[:8].isdigit())
    months = sorted({str(m) for m in [*config.get("train_months", []), *config.get("test_months", [])]}) or None
    return _months_filter(paths, months)


def _build_one_day_features(
    current_path: Path,
    history_paths: list[Path],
    config: dict[str, Any],
) -> pd.DataFrame:
    paths = [*history_paths, current_path]
    frames = [pd.read_parquet(path) for path in paths]
    minute_df = pd.concat(frames, ignore_index=True)
    minute_df["datetime"] = pd.to_datetime(minute_df["datetime"])
    current_date = pd.to_datetime(current_path.stem[:8], format="%Y%m%d").date().isoformat()
    df = minute_df
    for func in [
        _add_market_features,
        _add_intraday_curve_features,
        _add_historical_daily_features,
        _add_same_minute_features,
        _add_static_code_features,
    ]:
        df = func(df)
    df = build_future_labels(
        df,
        config["horizons"],
        config.get("participation_limit", 0.30),
        float(config.get("volume_ratio_scale", 10000.0)),
    )
    return df[df["date"].astype(str).eq(str(current_date))].sort_values(["stock_code", "datetime"]).reset_index(drop=True)


def build_feature_parts(config: dict[str, Any]) -> list[Path]:
    ensure_dirs(config)
    feature_parts_dir = resolve_path(config.get("feature_parts_dir", "data/features/model_parts"))
    feature_parts_dir.mkdir(parents=True, exist_ok=True)
    part_paths = _minute_part_paths(config)
    if not part_paths:
        raise FileNotFoundError("No minute part files found. Run scripts/01_preprocess.py first.")
    max_history_days = int(config.get("feature_history_days", 20))
    out_paths: list[Path] = []
    for idx, current_path in enumerate(tqdm(part_paths, desc="build feature parts")):
        out_path = feature_parts_dir / current_path.name
        if out_path.exists() and not bool(config.get("feature_overwrite", False)):
            out_paths.append(out_path)
            continue
        history_paths = part_paths[max(0, idx - max_history_days) : idx]
        day_df = _build_one_day_features(current_path, history_paths, config)
        day_df.to_parquet(out_path, index=False)
        out_paths.append(out_path)
    return out_paths


def build_features(config: dict[str, Any]) -> pd.DataFrame:
    ensure_dirs(config)
    processed_dir = resolve_path(config["processed_data_dir"])
    feature_dir = resolve_path(config["feature_data_dir"])
    minute_data_path = processed_dir / "minute_data.parquet"

    if not minute_data_path.exists() or not bool(config.get("build_combined_feature_data", False)):
        out_paths = build_feature_parts(config)
        if bool(config.get("build_combined_feature_data", False)):
            parts = [pd.read_parquet(path) for path in out_paths]
            df = pd.concat(parts, ignore_index=True).sort_values(["stock_code", "datetime"]).reset_index(drop=True)
            output_path = feature_dir / "model_dataset.parquet"
            df.to_parquet(output_path, index=False)
            return df
        print(
            "[feature] feature parts are ready; skip combined model_dataset.parquet "
            "because build_combined_feature_data=false",
            flush=True,
        )
        return pd.DataFrame()

    start = _log_step("loading minute_data.parquet")
    minute_df = pd.read_parquet(minute_data_path)
    minute_df["datetime"] = pd.to_datetime(minute_df["datetime"])
    _log_done(f"loaded minute rows={len(minute_df):,}", start)

    start = _log_step("loading stock_liquidity_group.parquet")
    liquidity = pd.read_parquet(processed_dir / "stock_liquidity_group.parquet")
    _log_done(f"loaded stock groups={len(liquidity):,}", start)

    start = _log_step("merging liquidity group")
    df = minute_df.merge(liquidity[["stock_code", "liquidity_group"]], on="stock_code", how="left")
    _log_done(f"merged rows={len(df):,}", start)

    for message, func in [
        ("building market features", _add_market_features),
        ("building intraday volume curve features", _add_intraday_curve_features),
        ("building historical daily features", _add_historical_daily_features),
        ("building same-minute stock features", _add_same_minute_features),
        ("building same-minute liquidity group features", _add_group_minute_features),
        ("building stock code features", _add_static_code_features),
    ]:
        start = _log_step(message)
        df = func(df)
        _log_done(message, start)

    start = _log_step("building future labels")
    df = build_future_labels(
        df,
        config["horizons"],
        config.get("participation_limit", 0.30),
        float(config.get("volume_ratio_scale", 10000.0)),
    )
    _log_done("future labels", start)

    start = _log_step("sorting and saving model_dataset.parquet")
    df = df.sort_values(["stock_code", "datetime"]).reset_index(drop=True)
    output_path = feature_dir / "model_dataset.parquet"
    df.to_parquet(output_path, index=False)
    _log_done(f"saved {output_path} rows={len(df):,}", start)
    return df
