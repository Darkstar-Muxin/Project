from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import ensure_dirs
from src.rolling_train import _evaluate_predictions, _predict_frame
from src.train import filter_by_months
from src.utils import resolve_path


def evaluate_models(config: dict[str, Any]) -> pd.DataFrame:
    ensure_dirs(config)
    dataset_path = resolve_path(config["feature_data_dir"]) / "model_dataset.parquet"
    output_dir = resolve_path(config["output_dir"])
    model_dir = resolve_path(config["model_dir"])
    df = pd.read_parquet(dataset_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    metrics_rows: list[dict[str, Any]] = []
    detail_parts: list[pd.DataFrame] = []
    backtest_parts: list[pd.DataFrame] = []

    for split, months_key in [("train", "train_months"), ("test", "test_months")]:
        split_df = filter_by_months(df, config.get(months_key))
        if split_df.empty:
            continue
        for group_name, group_df in split_df.groupby("liquidity_group", sort=True):
            group_path = model_dir / str(group_name)
            if not group_path.exists():
                continue
            pred_df, horizons = _predict_frame(group_df, group_path, config)
            if pred_df.empty:
                continue
            meta = {"model_type": "ive_static", "split": split, "liquidity_group": str(group_name)}
            rows, detail_df, backtest_df = _evaluate_predictions(pred_df, horizons, config, meta)
            metrics_rows.extend(rows)
            if not detail_df.empty:
                detail_parts.append(detail_df)
            if not backtest_df.empty:
                backtest_parts.append(backtest_df)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "evaluation_metrics.csv", index=False, encoding="utf-8-sig")
    if detail_parts:
        detail_df = pd.concat(detail_parts, ignore_index=True)
        detail_df.to_csv(output_dir / "prediction_error_detail.csv", index=False, encoding="utf-8-sig")
        for filename, cols in {
            "prediction_error_by_date.csv": ["split", "date", "horizon"],
            "prediction_error_by_stock.csv": ["split", "stock_code", "horizon"],
            "prediction_error_by_minute.csv": ["split", "minute", "minute_of_day", "horizon"],
        }.items():
            detail_df.groupby(cols, as_index=False).agg(
                sample_count=("horizon", "size"),
                vwap_mae=("abs_vwap_error", "mean"),
                volume_mae=("abs_volume_error", "mean"),
                volume_ratio_mae=("volume_ratio_error", lambda s: float(s.abs().mean())),
            ).to_csv(output_dir / filename, index=False, encoding="utf-8-sig")
    if backtest_parts:
        backtest_df = pd.concat(backtest_parts, ignore_index=True)
        backtest_df.to_csv(output_dir / "recommendation_backtest_detail.csv", index=False, encoding="utf-8-sig")
        backtest_df.groupby(["split", "liquidity_group", "side"], as_index=False).agg(
            sample_count=("side", "size"),
            pred_feasible_rate=("has_pred_feasible", "mean"),
            true_feasible_rate=("has_true_feasible", "mean"),
            horizon_match_rate=("horizon_match", "mean"),
            avg_regret=("regret", "mean"),
            max_absolute_regret=("absolute_regret", "max"),
        ).to_csv(output_dir / "recommendation_backtest_summary.csv", index=False, encoding="utf-8-sig")
        backtest_df.sort_values("absolute_regret", ascending=False, na_position="last").head(
            int(config.get("worst_case_top_n", 100))
        ).to_csv(output_dir / "recommendation_backtest_worst_cases.csv", index=False, encoding="utf-8-sig")
    return metrics_df
