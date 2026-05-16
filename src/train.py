from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import ensure_dirs
from src.utils import resolve_path


ID_COLUMNS = {"stock_code", "datetime", "date", "minute", "liquidity_group"}
RUNTIME_FEATURE_COLUMNS = [
    "is_sh",
    "is_sz",
    "minutes_from_open",
    "is_morning_session",
    "log_volume",
    "log_amount",
]


def filter_by_months(df: pd.DataFrame, months: list[int | str] | None) -> pd.DataFrame:
    if not months:
        return df
    month_set = {str(month) for month in months}
    dt = pd.to_datetime(df["datetime"])
    mask = dt.dt.strftime("%Y%m").isin(month_set)
    return df.loc[mask].copy()


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(ID_COLUMNS)
    excluded.update(col for col in df.columns if col.startswith("future_volume_") or col.startswith("future_vwap_"))
    candidates = [col for col in df.columns if col not in excluded]
    return [col for col in candidates if pd.api.types.is_numeric_dtype(df[col])]


def add_runtime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    code = out["stock_code"].astype(str)
    dt = pd.to_datetime(out["datetime"])
    out["is_sh"] = code.str.lower().str.endswith(".sh").astype(int)
    out["is_sz"] = code.str.lower().str.endswith(".sz").astype(int)
    out["minutes_from_open"] = dt.dt.hour * 60 + dt.dt.minute - (9 * 60 + 30)
    out["is_morning_session"] = (dt.dt.hour < 12).astype(int)
    if "volume" in out.columns:
        out["log_volume"] = np.log1p(pd.to_numeric(out["volume"], errors="coerce").clip(lower=0))
    if "amount" in out.columns:
        out["log_amount"] = np.log1p(pd.to_numeric(out["amount"], errors="coerce").clip(lower=0))
    return out


def _with_runtime_columns(feature_columns: list[str]) -> list[str]:
    merged = list(feature_columns)
    for col in RUNTIME_FEATURE_COLUMNS:
        if col not in merged:
            merged.append(col)
    return merged


def vwap_base(df: pd.DataFrame) -> pd.Series:
    if "vwap" in df.columns:
        base = pd.to_numeric(df["vwap"], errors="coerce")
    elif "close" in df.columns:
        base = pd.to_numeric(df["close"], errors="coerce")
    else:
        base = pd.Series(np.nan, index=df.index)
    return base.replace(0, np.nan)


def _schema_feature_columns(schema: pa.Schema) -> list[str]:
    excluded = set(ID_COLUMNS)
    excluded.update(name for name in schema.names if name.startswith("future_volume_") or name.startswith("future_vwap_"))
    numeric_types = (
        pa.types.is_integer,
        pa.types.is_floating,
        pa.types.is_decimal,
    )
    feature_columns: list[str] = []
    for field in schema:
        if field.name in excluded:
            continue
        if any(check(field.type) for check in numeric_types):
            feature_columns.append(field.name)
    return feature_columns


def _sample_group(df: pd.DataFrame, max_rows: int | None, random_state: int) -> pd.DataFrame:
    if max_rows and len(df) > max_rows:
        return df.sample(max_rows, random_state=random_state)
    return df


def _load_training_sample(
    dataset_path,
    config: dict[str, Any],
    feature_columns: list[str],
) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(dataset_path)
    horizons = [int(h) for h in config["horizons"]]
    label_columns = [f"future_vwap_{h}" for h in horizons] + [f"future_volume_{h}" for h in horizons]
    columns = ["stock_code", "datetime", "liquidity_group", *feature_columns, *label_columns]
    columns = [col for col in columns if col in parquet_file.schema_arrow.names]
    batch_size = int(config.get("train_batch_size", 200_000))
    max_rows = int(config.get("train_sample_per_group", 200_000))
    min_batches = int(config.get("train_min_batches_before_stop", 1))
    max_batches = config.get("train_max_batches")
    max_batches = int(max_batches) if max_batches else None
    random_state = int(config.get("random_state", 42))
    samples: dict[str, pd.DataFrame] = {}
    expected_groups = {"high", "medium", "low"}

    print(
        f"[train] streaming {dataset_path} with batch_size={batch_size:,}, "
        f"sample_per_group={max_rows:,}",
        flush=True,
    )
    for batch_no, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size, columns=columns), start=1):
        if max_batches and batch_no > max_batches:
            print(f"[train] reached train_max_batches={max_batches}; stop scanning parquet", flush=True)
            break
        batch_df = batch.to_pandas()
        batch_df = filter_by_months(batch_df, config.get("train_months"))
        if batch_df.empty:
            continue

        for group_name, group_df in batch_df.groupby("liquidity_group"):
            if pd.isna(group_name):
                continue
            key = str(group_name)
            current = samples.get(key)
            combined = group_df if current is None else pd.concat([current, group_df], ignore_index=True)
            if len(combined) > max_rows:
                combined = combined.sample(max_rows, random_state=random_state + batch_no)
            samples[key] = combined.reset_index(drop=True)

        sizes = {key: len(value) for key, value in samples.items()}
        print(f"[train] batch {batch_no}, sampled rows={sizes}", flush=True)
        if (
            batch_no >= min_batches
            and expected_groups.issubset(samples)
            and all(len(samples[group]) >= max_rows for group in expected_groups)
        ):
            print("[train] sample target reached for high/medium/low; stop scanning parquet", flush=True)
            break

    if not samples:
        raise ValueError("No training rows after applying train_months filter")
    return pd.concat(samples.values(), ignore_index=True)


def train_models(config: dict[str, Any]) -> None:
    ensure_dirs(config)
    dataset_path = resolve_path(config["feature_data_dir"]) / "model_dataset.parquet"
    model_dir = resolve_path(config["model_dir"])
    parquet_file = pq.ParquetFile(dataset_path)
    feature_columns = _schema_feature_columns(parquet_file.schema_arrow)
    df = _load_training_sample(dataset_path, config, feature_columns)
    df = add_runtime_features(df)
    feature_columns = _with_runtime_columns(feature_columns)
    random_state = int(config.get("random_state", 42))
    model_max_iter = int(config.get("model_max_iter", 100))
    learning_rate = float(config.get("model_learning_rate", 0.1))
    l2_regularization = float(config.get("model_l2_regularization", 0.0))

    for group_name, group_df in df.groupby("liquidity_group"):
        if pd.isna(group_name):
            continue
        print(f"[train] training group={group_name}, rows={len(group_df):,}", flush=True)
        group_path = model_dir / str(group_name)
        group_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(feature_columns, group_path / "feature_columns.joblib")
        work_df = group_df
        x_all = work_df[feature_columns]

        for h in config["horizons"]:
            vwap_target = f"future_vwap_{h}"
            volume_target = f"future_volume_{h}"

            vwap_mask = work_df[vwap_target].notna()
            if vwap_mask.sum() > 10:
                print(f"[train] group={group_name} horizon={h} vwap rows={int(vwap_mask.sum()):,}", flush=True)
                base = vwap_base(work_df).loc[vwap_mask]
                target = work_df.loc[vwap_mask, vwap_target] / base - 1
                valid = target.replace([np.inf, -np.inf], np.nan).notna()
                vwap_model = HistGradientBoostingRegressor(
                    random_state=random_state,
                    max_iter=model_max_iter,
                    learning_rate=learning_rate,
                    l2_regularization=l2_regularization,
                )
                vwap_model.fit(x_all.loc[vwap_mask].loc[valid], target.loc[valid])
                joblib.dump(vwap_model, group_path / f"vwap_h{h}.joblib")

            volume_mask = work_df[volume_target].notna() & (work_df[volume_target] >= 0)
            if volume_mask.sum() > 10:
                print(f"[train] group={group_name} horizon={h} volume rows={int(volume_mask.sum()):,}", flush=True)
                volume_model = HistGradientBoostingRegressor(
                    random_state=random_state,
                    max_iter=model_max_iter,
                    learning_rate=learning_rate,
                    l2_regularization=l2_regularization,
                )
                volume_model.fit(x_all.loc[volume_mask], np.log1p(work_df.loc[volume_mask, volume_target]))
                joblib.dump(volume_model, group_path / f"volume_h{h}.joblib")
