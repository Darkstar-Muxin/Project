from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import load_config
from src.train import add_runtime_features, vwap_base
from src.utils import normalize_side, resolve_path


def _find_feature_row_in_parquet(path, stock_code: str, start_time: str, columns: list[str] | None = None) -> pd.Series:
    start_ts = pd.to_datetime(start_time)
    parquet_file = pq.ParquetFile(path)
    available = set(parquet_file.schema_arrow.names)
    if columns is not None:
        columns = [col for col in columns if col in available]
    best_row = None
    best_time = None
    seen_stock = False

    for batch in parquet_file.iter_batches(batch_size=100_000, columns=columns):
        df = batch.to_pandas()
        if "stock_code" not in df.columns or "datetime" not in df.columns:
            raise ValueError("model_dataset.parquet must contain stock_code and datetime")
        stock_df = df[df["stock_code"].astype(str) == str(stock_code)].copy()
        if stock_df.empty:
            continue
        seen_stock = True
        stock_df["datetime"] = pd.to_datetime(stock_df["datetime"])
        after = stock_df[stock_df["datetime"] >= start_ts].sort_values("datetime")
        if after.empty:
            continue
        candidate = after.iloc[0]
        candidate_time = pd.to_datetime(candidate["datetime"])
        if best_time is None or candidate_time < best_time:
            best_row = candidate
            best_time = candidate_time
        if candidate_time == start_ts:
            break

    if best_row is None:
        if seen_stock:
            raise ValueError(f"No feature row found at or after {start_time} for {stock_code}")
        raise ValueError(f"stock_code not found in feature dataset: {stock_code}")
    return best_row


def _pick_candidate(candidates: list[dict[str, Any]], side: str, value_key: str, feasible_key: str) -> dict[str, Any] | None:
    feasible = [item for item in candidates if item.get(feasible_key) is True and pd.notna(item.get(value_key))]
    if not feasible:
        return None
    return min(feasible, key=lambda item: float(item[value_key])) if side == "buy" else max(feasible, key=lambda item: float(item[value_key]))


def _nan_to_none(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def predict_recommendation(
    stock_code: str,
    side: str,
    order_qty: float,
    start_time: str,
    config_path: str = "config.yaml",
) -> dict[str, Any]:
    config = load_config(config_path)
    side_norm = normalize_side(side)
    order_qty_float = float(order_qty)
    if order_qty_float <= 0:
        raise ValueError("order_qty must be positive")

    dataset_path = resolve_path(config["feature_data_dir"]) / "model_dataset.parquet"
    horizons = [int(h) for h in config["horizons"]]
    participation_limit = float(config.get("participation_limit", 0.30))
    label_columns = [f"future_vwap_{h}" for h in horizons] + [f"future_volume_{h}" for h in horizons]
    needed_columns = ["stock_code", "datetime", "liquidity_group", *label_columns]
    row = _find_feature_row_in_parquet(dataset_path, stock_code, start_time, needed_columns)
    liquidity_group = str(row["liquidity_group"])
    group_path = resolve_path(config["model_dir"]) / liquidity_group
    feature_columns = joblib.load(group_path / "feature_columns.joblib")
    row = _find_feature_row_in_parquet(
        dataset_path,
        stock_code,
        start_time,
        ["stock_code", "datetime", "liquidity_group", *feature_columns, *label_columns],
    )
    row_df = add_runtime_features(pd.DataFrame([row.to_dict()]))
    x = row_df.reindex(columns=feature_columns)
    x = x.apply(pd.to_numeric, errors="coerce")

    candidates: list[dict[str, Any]] = []
    for h in horizons:
        vwap_model = joblib.load(group_path / f"vwap_h{h}.joblib")
        volume_model = joblib.load(group_path / f"volume_h{h}.joblib")
        base_vwap = float(vwap_base(row_df).iloc[0])
        pred_vwap = float(base_vwap * (1 + vwap_model.predict(x)[0]))
        pred_volume = max(float(np.expm1(volume_model.predict(x)[0])), 0.0)
        participation = float(order_qty_float / pred_volume) if pred_volume > 0 else float("inf")
        feasible = bool(participation <= participation_limit)
        actual_vwap = _nan_to_none(row.get(f"future_vwap_{h}"))
        actual_volume = _nan_to_none(row.get(f"future_volume_{h}"))
        actual_participation = None
        actual_feasible = None
        if actual_volume is not None and float(actual_volume) > 0:
            actual_participation = float(order_qty_float / float(actual_volume))
            actual_feasible = bool(actual_participation <= participation_limit)
        candidates.append(
            {
                "horizon": int(h),
                "predicted_vwap": pred_vwap,
                "predicted_market_volume": pred_volume,
                "predicted_participation": participation,
                "feasible": feasible,
                "actual_vwap": None if actual_vwap is None else float(actual_vwap),
                "actual_market_volume": None if actual_volume is None else float(actual_volume),
                "actual_participation": actual_participation,
                "actual_feasible": actual_feasible,
                "vwap_error": None if actual_vwap is None else pred_vwap - float(actual_vwap),
                "volume_error": None if actual_volume is None else pred_volume - float(actual_volume),
            }
        )

    selected = _pick_candidate(candidates, side_norm, "predicted_vwap", "feasible")
    actual_best = _pick_candidate(candidates, side_norm, "actual_vwap", "actual_feasible")
    selected_actual = None
    regret = None
    optimal_hit = None
    if selected is not None:
        selected_actual = next((item for item in candidates if item["horizon"] == selected["horizon"]), None)
    if selected_actual is not None and actual_best is not None and selected_actual.get("actual_vwap") is not None:
        if side_norm == "buy":
            regret = float(selected_actual["actual_vwap"]) - float(actual_best["actual_vwap"])
        else:
            regret = float(actual_best["actual_vwap"]) - float(selected_actual["actual_vwap"])
        optimal_hit = selected_actual["horizon"] == actual_best["horizon"]
    has_actual_comparison = any(item.get("actual_vwap") is not None and item.get("actual_market_volume") is not None for item in candidates)

    return {
        "stock_code": str(stock_code),
        "side": side_norm,
        "order_qty": order_qty_float,
        "start_time": str(pd.to_datetime(start_time)),
        "matched_feature_time": str(pd.to_datetime(row["datetime"])),
        "liquidity_group": liquidity_group,
        "recommended_horizon": None if selected is None else selected["horizon"],
        "predicted_vwap": None if selected is None else selected["predicted_vwap"],
        "predicted_market_volume": None if selected is None else selected["predicted_market_volume"],
        "predicted_participation": None if selected is None else selected["predicted_participation"],
        "feasible": selected is not None,
        "has_actual_comparison": has_actual_comparison,
        "actual_best_horizon": None if actual_best is None else actual_best["horizon"],
        "actual_best_vwap": None if actual_best is None else actual_best["actual_vwap"],
        "actual_best_market_volume": None if actual_best is None else actual_best["actual_market_volume"],
        "actual_best_participation": None if actual_best is None else actual_best["actual_participation"],
        "recommended_actual_vwap": None if selected_actual is None else selected_actual["actual_vwap"],
        "recommended_actual_market_volume": None if selected_actual is None else selected_actual["actual_market_volume"],
        "recommended_actual_participation": None if selected_actual is None else selected_actual["actual_participation"],
        "recommended_actual_feasible": None if selected_actual is None else selected_actual["actual_feasible"],
        "optimal_hit": optimal_hit,
        "regret": regret,
        "candidate_table": candidates,
    }
