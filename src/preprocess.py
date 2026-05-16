from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from src.config import ensure_dirs
from src.data_loader import infer_column_mapping
from src.utils import resolve_path, safe_divide


RAW_MONTH_FILES = ["202602.parquet", "202603.parquet", "202604.parquet"]


def _parse_datetime(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.Series:
    datetime_col = mapping.get("datetime")
    date_col = mapping.get("date")
    if datetime_col and datetime_col in df.columns and datetime_col.lower() not in {"time"}:
        return pd.to_datetime(df[datetime_col], errors="coerce").dt.floor("min")
    if date_col and datetime_col and datetime_col in df.columns:
        dates = df[date_col].astype("Int64").astype(str)
        times = df[datetime_col].fillna(0).astype("Int64").astype(str).str.zfill(9).str[:6]
        return pd.to_datetime(dates + times, format="%Y%m%d%H%M%S", errors="coerce").dt.floor("min")
    raise ValueError("Cannot infer datetime. Need a datetime column or date + time columns.")


def _standardize_batch(batch_df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    required = ["stock_code", "price", "volume"]
    missing = [name for name in required if not mapping.get(name)]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = pd.DataFrame()
    out["stock_code"] = batch_df[mapping["stock_code"]].astype("string")
    out["datetime"] = _parse_datetime(batch_df, mapping)
    out["price"] = pd.to_numeric(batch_df[mapping["price"]], errors="coerce")
    out["volume"] = pd.to_numeric(batch_df[mapping["volume"]], errors="coerce").fillna(0)

    amount_col = mapping.get("amount")
    if amount_col and amount_col in batch_df.columns:
        out["amount"] = pd.to_numeric(batch_df[amount_col], errors="coerce")
    else:
        out["amount"] = out["price"] * out["volume"]

    out = out.dropna(subset=["stock_code", "datetime", "price"])
    out = out[out["volume"] > 0]
    out["amount"] = out["amount"].fillna(out["price"] * out["volume"])
    return out


def _aggregate_minute(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values(["stock_code", "datetime"])
    grouped = df.groupby(["stock_code", "datetime"], sort=False)
    minute_df = grouped.agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
    ).reset_index()
    minute_df["date"] = minute_df["datetime"].dt.date.astype(str)
    minute_df["minute"] = minute_df["datetime"].dt.strftime("%H:%M")
    minute_df["vwap"] = safe_divide(minute_df["amount"], minute_df["volume"])
    return minute_df


def preprocess_raw_data(config: dict[str, Any]) -> pd.DataFrame:
    ensure_dirs(config)
    raw_dir = resolve_path(config.get("raw_data_dir", "data"))
    processed_dir = resolve_path(config["processed_data_dir"])
    tmp_dir = processed_dir / "_minute_parts"
    output_path = processed_dir / "minute_data.parquet"
    batch_size = int(config.get("batch_size", 1_000_000))

    if tmp_dir.exists():
        import shutil

        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    part_paths: list[Path] = []
    for file_name in RAW_MONTH_FILES:
        path = raw_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Required raw parquet not found: {path}")
        parquet_file = pq.ParquetFile(path)
        mapping = infer_column_mapping(parquet_file.schema_arrow.names, config)

        for batch_idx, batch in enumerate(
            tqdm(parquet_file.iter_batches(batch_size=batch_size), desc=f"preprocess {file_name}"),
            start=1,
        ):
            batch_df = batch.to_pandas()
            standardized = _standardize_batch(batch_df, mapping)
            minute_part = _aggregate_minute(standardized)
            part_path = tmp_dir / f"{path.stem}_part_{batch_idx:06d}.parquet"
            minute_part.to_parquet(part_path, index=False)
            part_paths.append(part_path)

    parts = [pd.read_parquet(path) for path in part_paths]
    combined = pd.concat(parts, ignore_index=True)
    combined["datetime"] = pd.to_datetime(combined["datetime"])
    combined = combined.sort_values(["stock_code", "datetime"])
    final = combined.groupby(["stock_code", "datetime"], sort=False).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
    ).reset_index()
    final["date"] = final["datetime"].dt.date.astype(str)
    final["minute"] = final["datetime"].dt.strftime("%H:%M")
    final["vwap"] = np.where(final["volume"] > 0, final["amount"] / final["volume"], np.nan)
    final = final[
        ["stock_code", "datetime", "date", "minute", "open", "high", "low", "close", "volume", "amount", "vwap"]
    ]
    final.to_parquet(output_path, index=False)

    import shutil

    shutil.rmtree(tmp_dir)
    return final
