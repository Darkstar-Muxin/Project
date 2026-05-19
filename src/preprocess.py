from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from src.config import ensure_dirs
from src.utils import resolve_path, safe_divide


TICK_KEEP_COLS = ["StockCode", "Date", "Time", "Price", "Volume", "Turnover", "BSFlag"]


def _date_from_path(path: Path) -> str:
    return path.stem[:8]


def _months_filter(paths: list[Path], months: list[int | str] | None) -> list[Path]:
    if not months:
        return paths
    month_set = {str(month) for month in months}
    return [path for path in paths if _date_from_path(path)[:6] in month_set]


def find_tick_files(config: dict[str, Any], months: list[int | str] | None = None) -> list[Path]:
    tick_dir = resolve_path(config.get("tick_data_dir", "data/tick_data"))
    if not tick_dir.exists():
        raise FileNotFoundError(f"tick_data_dir not found: {tick_dir}")
    paths = sorted(path for path in tick_dir.glob("*.parquet") if path.stem[:8].isdigit())
    return _months_filter(paths, months)


def _parse_tick_datetime(df: pd.DataFrame) -> pd.Series:
    dates = pd.to_numeric(df["Date"], errors="coerce").astype("Int64").astype(str)
    raw_time = pd.to_numeric(df["Time"], errors="coerce").fillna(0).astype("Int64").astype(str).str.zfill(9)
    hhmmss = raw_time.str[:6]
    return pd.to_datetime(dates + hhmmss, format="%Y%m%d%H%M%S", errors="coerce").dt.floor("min")


def _standardize_tick_batch(batch_df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in TICK_KEEP_COLS if col not in batch_df.columns]
    if missing:
        raise ValueError(f"Tick parquet missing required columns: {missing}")

    out = pd.DataFrame()
    out["stock_code"] = batch_df["StockCode"].astype("string")
    out["datetime"] = _parse_tick_datetime(batch_df)
    out["price"] = pd.to_numeric(batch_df["Price"], errors="coerce")
    out["volume"] = pd.to_numeric(batch_df["Volume"], errors="coerce").fillna(0)
    out["amount"] = pd.to_numeric(batch_df["Turnover"], errors="coerce")
    out["bs_flag"] = batch_df["BSFlag"].astype("string").str.upper().fillna("U")
    out = out.dropna(subset=["stock_code", "datetime", "price"])
    out = out[out["volume"] > 0].copy()
    out["amount"] = out["amount"].fillna(out["price"] * out["volume"])
    return out


def _aggregate_minute(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "stock_code",
                "datetime",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "trade_count",
                "buy_volume",
                "sell_volume",
                "buy_amount",
                "sell_amount",
            ]
        )
    df = df.sort_values(["stock_code", "datetime"])
    df["buy_volume_raw"] = np.where(df["bs_flag"].eq("B"), df["volume"], 0.0)
    df["sell_volume_raw"] = np.where(df["bs_flag"].eq("S"), df["volume"], 0.0)
    df["buy_amount_raw"] = np.where(df["bs_flag"].eq("B"), df["amount"], 0.0)
    df["sell_amount_raw"] = np.where(df["bs_flag"].eq("S"), df["amount"], 0.0)
    grouped = df.groupby(["stock_code", "datetime"], sort=False)
    return grouped.agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
        trade_count=("price", "size"),
        buy_volume=("buy_volume_raw", "sum"),
        sell_volume=("sell_volume_raw", "sum"),
        buy_amount=("buy_amount_raw", "sum"),
        sell_amount=("sell_amount_raw", "sum"),
    ).reset_index()


def _finalize_minute_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.sort_values(["stock_code", "datetime"]).copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["date"] = out["datetime"].dt.date.astype(str)
    out["minute"] = out["datetime"].dt.strftime("%H:%M")
    out["vwap"] = safe_divide(out["amount"], out["volume"])
    columns = [
        "stock_code",
        "datetime",
        "date",
        "minute",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "vwap",
        "trade_count",
        "buy_volume",
        "sell_volume",
        "buy_amount",
        "sell_amount",
    ]
    return out[columns]


def _merge_minute_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    combined["datetime"] = pd.to_datetime(combined["datetime"])
    combined = combined.sort_values(["stock_code", "datetime"])
    merged = combined.groupby(["stock_code", "datetime"], sort=False).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
        trade_count=("trade_count", "sum"),
        buy_volume=("buy_volume", "sum"),
        sell_volume=("sell_volume", "sum"),
        buy_amount=("buy_amount", "sum"),
        sell_amount=("sell_amount", "sum"),
    ).reset_index()
    return _finalize_minute_df(merged)


def preprocess_tick_file(path: str | Path, config: dict[str, Any], overwrite: bool | None = None) -> Path:
    tick_path = resolve_path(path)
    minute_parts_dir = resolve_path(config.get("minute_parts_dir", "data/processed/minute_parts"))
    minute_parts_dir.mkdir(parents=True, exist_ok=True)
    out_path = minute_parts_dir / f"{tick_path.stem}.parquet"
    overwrite = bool(config.get("preprocess_overwrite", False)) if overwrite is None else overwrite
    if out_path.exists() and not overwrite:
        print(f"[preprocess] cached minute part exists, skip {out_path}", flush=True)
        return out_path

    parquet_file = pq.ParquetFile(tick_path)
    available = set(parquet_file.schema_arrow.names)
    missing = [col for col in TICK_KEEP_COLS if col not in available]
    if missing:
        raise ValueError(f"{tick_path} missing required columns: {missing}")
    batch_size = int(config.get("batch_size", 1_000_000))
    parts: list[pd.DataFrame] = []

    for batch in tqdm(
        parquet_file.iter_batches(batch_size=batch_size, columns=TICK_KEEP_COLS),
        desc=f"preprocess {tick_path.name}",
    ):
        standardized = _standardize_tick_batch(batch.to_pandas())
        minute_part = _aggregate_minute(standardized)
        if not minute_part.empty:
            parts.append(minute_part)

    day_df = _merge_minute_parts(parts)
    day_df.to_parquet(out_path, index=False)
    print(f"[preprocess] saved {out_path} rows={len(day_df):,}", flush=True)
    return out_path


def combine_minute_parts(config: dict[str, Any], months: list[int | str] | None = None) -> pd.DataFrame:
    processed_dir = resolve_path(config["processed_data_dir"])
    minute_parts_dir = resolve_path(config.get("minute_parts_dir", "data/processed/minute_parts"))
    output_path = processed_dir / "minute_data.parquet"
    part_paths = sorted(minute_parts_dir.glob("*.parquet"))
    part_paths = _months_filter(part_paths, months)
    if not part_paths:
        raise FileNotFoundError(f"No minute part parquet files found in {minute_parts_dir}")
    parts = [pd.read_parquet(path) for path in tqdm(part_paths, desc="combine minute parts")]
    final = _merge_minute_parts(parts)
    final = final.sort_values(["stock_code", "datetime"]).reset_index(drop=True)
    final.to_parquet(output_path, index=False)
    print(f"[preprocess] saved {output_path} rows={len(final):,}", flush=True)
    return final


def preprocess_raw_data(config: dict[str, Any]) -> pd.DataFrame:
    ensure_dirs(config)
    resolve_path(config.get("minute_parts_dir", "data/processed/minute_parts")).mkdir(parents=True, exist_ok=True)
    months = sorted({str(m) for m in [*config.get("train_months", []), *config.get("test_months", [])]}) or None
    tick_files = find_tick_files(config, months=months)
    if not tick_files:
        raise FileNotFoundError("No tick parquet files found for configured train/test months")
    for path in tick_files:
        preprocess_tick_file(path, config)
    if bool(config.get("build_combined_minute_data", False)):
        return combine_minute_parts(config, months=months)
    print(
        "[preprocess] minute parts are ready; skip combined minute_data.parquet "
        "because build_combined_minute_data=false",
        flush=True,
    )
    return pd.DataFrame()
