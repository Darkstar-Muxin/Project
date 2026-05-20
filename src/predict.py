from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("PyTorch is required for prediction. Install with: pip install torch") from exc

from src.config import load_config
from src.ive_dataset import Normalizer, transform_features
from src.ive_model import IVEModel
from src.utils import normalize_side, resolve_path


def _load_dataset(path: Path, start_time: str | None = None, config: dict[str, Any] | None = None) -> pd.DataFrame:
    if path.exists():
        df = pd.read_parquet(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df
    if start_time is None or config is None:
        raise FileNotFoundError(f"Feature dataset not found: {path}")
    date_key = pd.to_datetime(start_time).strftime("%Y%m%d")
    part_path = resolve_path(config.get("feature_parts_dir", "data/features/model_parts")) / f"{date_key}.parquet"
    if not part_path.exists():
        raise FileNotFoundError(f"Feature part not found: {part_path}. Run scripts/02_build_features_labels.py first.")
    df = pd.read_parquet(part_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def _find_feature_row(df: pd.DataFrame, stock_code: str, start_time: str) -> tuple[pd.Series, pd.DataFrame]:
    start_ts = pd.to_datetime(start_time)
    stock_df = df[df["stock_code"].astype(str) == str(stock_code)].copy()
    if stock_df.empty:
        raise ValueError(f"stock_code not found in feature dataset: {stock_code}")
    after = stock_df[stock_df["datetime"] >= start_ts].sort_values("datetime")
    if after.empty:
        raise ValueError(f"No feature row found at or after {start_time} for {stock_code}")
    row = after.iloc[0]
    same_day = stock_df[(stock_df["date"].astype(str) == str(row["date"])) & (stock_df["datetime"] <= row["datetime"])].sort_values("datetime")
    return row, same_day


def _model_group_path(config: dict[str, Any], liquidity_group: str, matched_date: str, rolling_window: int | None) -> Path:
    if rolling_window is not None:
        rolling_root = resolve_path(config.get("rolling_model_dir", "data/models/rolling"))
        path = rolling_root / f"window_{int(rolling_window)}d" / matched_date / liquidity_group
        if path.exists():
            return path
    model_path = resolve_path(config["model_dir"]) / liquidity_group
    if model_path.exists():
        return model_path
    rolling_root = resolve_path(config.get("rolling_model_dir", "data/models/rolling"))
    candidates = sorted(rolling_root.glob(f"window_*d/{matched_date}/{liquidity_group}"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No model found for group={liquidity_group}, date={matched_date}")


def _rolling_liquidity_group(config: dict[str, Any], stock_code: str, matched_date: str, rolling_window: int | None) -> str | None:
    if rolling_window is None:
        return None
    path = (
        resolve_path(config.get("rolling_model_dir", "data/models/rolling"))
        / f"window_{int(rolling_window)}d"
        / matched_date
        / "stock_liquidity_group.parquet"
    )
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    hit = df[df["stock_code"].astype(str).eq(str(stock_code))]
    if hit.empty:
        return "low"
    return str(hit.iloc[0]["liquidity_group"])


def _load_model(group_path: Path) -> tuple[IVEModel, dict[str, Any]]:
    import json

    meta = json.loads((group_path / "model_meta.json").read_text(encoding="utf-8"))
    model = IVEModel(
        num_features=len(meta["feature_columns"]),
        num_stocks=max(len(meta["stock_vocab"]), 1),
        num_groups=max(len(meta["group_vocab"]), 1),
        num_horizons=len(meta["horizons"]),
        d_model=int(meta.get("d_model", 64)),
        nhead=int(meta.get("nhead", 4)),
        num_layers=int(meta.get("num_layers", 2)),
        dropout=float(meta.get("dropout", 0.1)),
        max_len=int(meta.get("context_length", 390)),
    )
    state = torch.load(group_path / "ive_model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, meta


def _build_sequence(context_df: pd.DataFrame, row: pd.Series, meta: dict[str, Any]) -> dict[str, torch.Tensor]:
    feature_columns = list(meta["feature_columns"])
    normalizer = Normalizer(mean=meta["normalizer"]["mean"], std=meta["normalizer"]["std"])
    context_length = int(meta.get("context_length", 390))
    features = transform_features(context_df, feature_columns, normalizer)
    valid_len = min(len(context_df), context_length)
    padded = np.zeros((context_length, len(feature_columns)), dtype=np.float32)
    if valid_len > 0:
        padded[-valid_len:] = features[-valid_len:]
    padding_mask = np.ones(context_length, dtype=bool)
    padding_mask[-valid_len:] = False
    stock_id = int(meta["stock_vocab"].get(str(row["stock_code"]), 0))
    group_id = int(meta["group_vocab"].get(str(row["liquidity_group"]), 0))
    return {
        "x": torch.from_numpy(padded).unsqueeze(0),
        "padding_mask": torch.from_numpy(padding_mask).unsqueeze(0),
        "stock_id": torch.tensor([stock_id], dtype=torch.long),
        "group_id": torch.tensor([group_id], dtype=torch.long),
    }


def _daily_volume_prior(row: pd.Series, meta: dict[str, Any]) -> float:
    stock_prior = meta.get("daily_volume_prior", {}).get(str(row["stock_code"]))
    if stock_prior and float(stock_prior) > 0:
        return float(stock_prior)
    group_prior = meta.get("group_daily_volume_prior", {}).get(str(row["liquidity_group"]))
    if group_prior and float(group_prior) > 0:
        return float(group_prior)
    for col in ["stock_rolling_volume_mean_10d", "stock_rolling_volume_mean_5d"]:
        value = row.get(col)
        if pd.notna(value) and float(value) > 0:
            return float(value)
    return float(row.get("volume", 0.0))


def _pick_candidate(candidates: list[dict[str, Any]], side: str, value_key: str, feasible_key: str) -> dict[str, Any] | None:
    feasible = [item for item in candidates if item.get(feasible_key) is True and pd.notna(item.get(value_key))]
    if not feasible:
        return None
    return min(feasible, key=lambda item: float(item[value_key])) if side == "buy" else max(feasible, key=lambda item: float(item[value_key]))


def _nan_to_none(value: Any) -> Any:
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
    rolling_window: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    side_norm = normalize_side(side)
    order_qty_float = float(order_qty)
    if order_qty_float <= 0:
        raise ValueError("order_qty must be positive")

    dataset_path = resolve_path(config["feature_data_dir"]) / "model_dataset.parquet"
    df = _load_dataset(dataset_path, start_time=start_time, config=config)
    row, context_df = _find_feature_row(df, stock_code, start_time)
    matched_date = pd.to_datetime(row["datetime"]).date().isoformat()
    rolling_group = _rolling_liquidity_group(config, stock_code, matched_date, rolling_window)
    if rolling_group is not None:
        liquidity_group = rolling_group
        row = row.copy()
        row["liquidity_group"] = liquidity_group
        context_df = context_df.copy()
        context_df["liquidity_group"] = liquidity_group
    elif "liquidity_group" in row.index:
        liquidity_group = str(row["liquidity_group"])
    else:
        raise ValueError("No rolling liquidity classification found. Run scripts/06_rolling_backtest.py for this date/window first.")
    group_path = _model_group_path(config, liquidity_group, matched_date, rolling_window)
    model, meta = _load_model(group_path)
    model_input = _build_sequence(context_df, row, meta)
    with torch.no_grad():
        outputs = model(**model_input)
    volume_mu = outputs["volume_mu"].squeeze(0).cpu().numpy()
    volume_sigma = np.log1p(np.exp(outputs["volume_log_sigma"].squeeze(0).cpu().numpy()))
    vwap_return = outputs["vwap_return"].squeeze(0).cpu().numpy()

    horizons = [int(h) for h in meta["horizons"]]
    ratio_scale = float(config.get("volume_ratio_scale", 10000.0))
    daily_prior = _daily_volume_prior(row, meta)
    base_vwap = float(row["vwap"] if pd.notna(row.get("vwap")) and float(row.get("vwap")) > 0 else row["close"])
    candidates: list[dict[str, Any]] = []
    for idx, h in enumerate(horizons):
        predicted_volume_ratio = max(float(np.expm1(volume_mu[idx]) / ratio_scale), 0.0)
        predicted_volume_sigma = max(float(np.expm1(volume_sigma[idx]) / ratio_scale), 0.0)
        predicted_volume = max(predicted_volume_ratio * daily_prior, 0.0)
        predicted_vwap = float(base_vwap * (1 + vwap_return[idx]))
        predicted_participation = float(order_qty_float / predicted_volume) if predicted_volume > 0 else float("inf")
        actual_vwap = _nan_to_none(row.get(f"future_vwap_{h}"))
        actual_volume = _nan_to_none(row.get(f"future_volume_{h}"))
        actual_ratio = _nan_to_none(row.get(f"future_volume_ratio_{h}"))
        actual_participation = None
        actual_feasible = None
        if actual_volume is not None and float(actual_volume) > 0:
            actual_participation = float(order_qty_float / float(actual_volume))
            actual_feasible = bool(actual_participation <= float(config.get("participation_limit", 0.30)))
        candidates.append(
            {
                "horizon": h,
                "predicted_vwap": predicted_vwap,
                "predicted_market_volume": predicted_volume,
                "predicted_volume_ratio": predicted_volume_ratio,
                "predicted_volume_sigma": predicted_volume_sigma,
                "predicted_participation": predicted_participation,
                "feasible": bool(predicted_participation <= float(config.get("participation_limit", 0.30))),
                "actual_vwap": None if actual_vwap is None else float(actual_vwap),
                "actual_market_volume": None if actual_volume is None else float(actual_volume),
                "actual_volume_ratio": None if actual_ratio is None else float(actual_ratio),
                "actual_participation": actual_participation,
                "actual_feasible": actual_feasible,
                "vwap_error": None if actual_vwap is None else predicted_vwap - float(actual_vwap),
                "volume_error": None if actual_volume is None else predicted_volume - float(actual_volume),
            }
        )

    selected = _pick_candidate(candidates, side_norm, "predicted_vwap", "feasible")
    actual_best = _pick_candidate(candidates, side_norm, "actual_vwap", "actual_feasible")
    selected_actual = None if selected is None else next((item for item in candidates if item["horizon"] == selected["horizon"]), None)
    regret = None
    optimal_hit = None
    if selected_actual is not None and actual_best is not None and selected_actual.get("actual_vwap") is not None:
        if side_norm == "buy":
            regret = float(selected_actual["actual_vwap"]) - float(actual_best["actual_vwap"])
        else:
            regret = float(actual_best["actual_vwap"]) - float(selected_actual["actual_vwap"])
        optimal_hit = selected_actual["horizon"] == actual_best["horizon"]

    return {
        "stock_code": str(stock_code),
        "side": side_norm,
        "order_qty": order_qty_float,
        "start_time": str(pd.to_datetime(start_time)),
        "matched_feature_time": str(pd.to_datetime(row["datetime"])),
        "liquidity_group": liquidity_group,
        "model_path": str(group_path),
        "rolling_window": rolling_window,
        "recommended_horizon": None if selected is None else selected["horizon"],
        "predicted_vwap": None if selected is None else selected["predicted_vwap"],
        "predicted_market_volume": None if selected is None else selected["predicted_market_volume"],
        "predicted_participation": None if selected is None else selected["predicted_participation"],
        "feasible": selected is not None,
        "has_actual_comparison": any(item.get("actual_vwap") is not None for item in candidates),
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
