from __future__ import annotations

import gc
import os
import shutil
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import ensure_dirs
from src.evaluate import _baseline_volume, _baseline_vwap, _build_backtest_rows
from src.train import _schema_feature_columns, _with_runtime_columns, add_runtime_features, vwap_base
from src.utils import resolve_path


def _rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _date_strings(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.date.astype(str)


def get_trading_dates(dataset_path: str | Path, batch_size: int = 500_000) -> list[str]:
    parquet_file = pq.ParquetFile(dataset_path)
    dates: set[str] = set()
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["datetime"]):
        df = batch.to_pandas()
        dates.update(_date_strings(df["datetime"]).unique())
    return sorted(dates)


def _months_filter(dates: list[str], months: list[int | str]) -> list[str]:
    month_set = {str(month) for month in months}
    return [date for date in dates if date.replace("-", "")[:6] in month_set]


def _load_filtered_rows(
    dataset_path: str | Path,
    dates: set[str],
    group_name: str,
    columns: list[str],
    batch_size: int,
) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(dataset_path)
    available = set(parquet_file.schema_arrow.names)
    read_columns = [col for col in columns if col in available]
    parts: list[pd.DataFrame] = []
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=read_columns):
        df = batch.to_pandas()
        mask = (_date_strings(df["datetime"]).isin(dates)) & (df["liquidity_group"].astype(str) == str(group_name))
        if mask.any():
            parts.append(df.loc[mask].copy())
    if not parts:
        return pd.DataFrame(columns=read_columns)
    return pd.concat(parts, ignore_index=True)


def _model_params(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "random_state": int(config.get("random_state", 42)),
        "max_iter": int(config.get("model_max_iter", 120)),
        "learning_rate": float(config.get("model_learning_rate", 0.06)),
        "l2_regularization": float(config.get("model_l2_regularization", 0.01)),
    }


def _train_one_model(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    horizon: int,
    target_kind: str,
    config: dict[str, Any],
):
    target_col = f"future_{'vwap' if target_kind == 'vwap' else 'volume'}_{horizon}"
    train_df = add_runtime_features(train_df)
    x_all = train_df[feature_columns]
    mask = train_df[target_col].notna()
    if target_kind == "volume":
        mask = mask & (train_df[target_col] >= 0)
        y = np.log1p(train_df.loc[mask, target_col])
    else:
        base = vwap_base(train_df).loc[mask]
        y = train_df.loc[mask, target_col] / base - 1
        valid = y.replace([np.inf, -np.inf], np.nan).notna()
        mask_index = y.index[valid]
        return_x = x_all.loc[mask_index]
        return_y = y.loc[mask_index]
        model = HistGradientBoostingRegressor(**_model_params(config))
        model.fit(return_x, return_y)
        return model, len(return_y)

    model = HistGradientBoostingRegressor(**_model_params(config))
    model.fit(x_all.loc[mask], y)
    return model, int(mask.sum())


def _load_train_for_model(
    train_path: str | Path,
    feature_columns: list[str],
    horizon: int,
    target_kind: str,
) -> pd.DataFrame:
    target_col = f"future_{'vwap' if target_kind == 'vwap' else 'volume'}_{horizon}"
    columns = ["stock_code", "datetime", "liquidity_group", *feature_columns, target_col]
    available = set(pq.ParquetFile(train_path).schema_arrow.names)
    columns = [col for col in columns if col in available]
    return pd.read_parquet(train_path, columns=columns)


def _prepare_group_train_file(
    dataset_path: str | Path,
    train_dates: list[str],
    group_name: str,
    base_feature_columns: list[str],
    horizons: list[int],
    batch_size: int,
    temp_dir: Path,
) -> Path | None:
    label_columns = [f"future_vwap_{h}" for h in horizons] + [f"future_volume_{h}" for h in horizons]
    columns = ["stock_code", "datetime", "liquidity_group", *base_feature_columns, *label_columns]
    train_df = _load_filtered_rows(dataset_path, set(train_dates), group_name, columns, batch_size)
    if train_df.empty:
        return None
    temp_dir.mkdir(parents=True, exist_ok=True)
    out = temp_dir / f"train_{group_name}.parquet"
    train_df.to_parquet(out, index=False)
    del train_df
    gc.collect()
    return out


def _load_test_for_group(
    dataset_path: str | Path,
    test_date: str,
    group_name: str,
    feature_columns: list[str],
    horizons: list[int],
    batch_size: int,
) -> pd.DataFrame:
    label_columns = [f"future_vwap_{h}" for h in horizons] + [f"future_volume_{h}" for h in horizons]
    columns = ["stock_code", "datetime", "liquidity_group", *feature_columns, *label_columns]
    return _load_filtered_rows(dataset_path, {test_date}, group_name, columns, batch_size)


def _predict_model(model, x_all: pd.DataFrame, work_df: pd.DataFrame, target_kind: str) -> np.ndarray:
    pred = model.predict(x_all)
    if target_kind == "vwap":
        return vwap_base(work_df).to_numpy(dtype=float) * (1 + pred)
    return np.maximum(np.expm1(pred), 0)


def _evaluate_group_models(
    test_df: pd.DataFrame,
    feature_columns: list[str],
    group_path: Path,
    config: dict[str, Any],
    meta: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]]]:
    horizons = [int(h) for h in config["horizons"]]
    work_df = add_runtime_features(test_df.reset_index(drop=True))
    x_all = work_df[feature_columns]
    metrics: list[dict[str, Any]] = []
    detail_parts: list[pd.DataFrame] = []

    for h in horizons:
        row = {**meta, "horizon": h}
        for target_kind in ["vwap", "volume"]:
            model_path = group_path / f"{target_kind}_h{h}.joblib"
            if not model_path.exists():
                continue
            model = joblib.load(model_path)
            target_col = f"future_{target_kind}_{h}"
            mask = work_df[target_col].notna()
            if target_kind == "volume":
                mask = mask & (work_df[target_col] >= 0)
            if not mask.any():
                continue
            pred = _predict_model(model, x_all.loc[mask], work_df.loc[mask], target_kind)
            true = work_df.loc[mask, target_col]
            row[f"{target_kind}_mae"] = float(mean_absolute_error(true, pred))
            row[f"{target_kind}_rmse"] = _rmse(true, pred)
        if "vwap_mae" in row:
            baseline = _baseline_vwap(work_df)
            mask = work_df[f"future_vwap_{h}"].notna() & baseline.notna()
            row["baseline_vwap_mae"] = float(mean_absolute_error(work_df.loc[mask, f"future_vwap_{h}"], baseline.loc[mask]))
            row["vwap_mae_improvement"] = row["baseline_vwap_mae"] - row["vwap_mae"]
        if "volume_mae" in row:
            baseline = _baseline_volume(work_df, h)
            mask = work_df[f"future_volume_{h}"].notna() & baseline.notna()
            row["baseline_volume_mae"] = float(mean_absolute_error(work_df.loc[mask, f"future_volume_{h}"], baseline.loc[mask]))
            row["volume_mae_improvement"] = row["baseline_volume_mae"] - row["volume_mae"]
        metrics.append(row)

        vwap_path = group_path / f"vwap_h{h}.joblib"
        volume_path = group_path / f"volume_h{h}.joblib"
        detail_mask = work_df[f"future_vwap_{h}"].notna() & work_df[f"future_volume_{h}"].notna()
        if vwap_path.exists() and volume_path.exists() and detail_mask.any():
            detail_x = x_all.loc[detail_mask]
            pred_vwap = _predict_model(joblib.load(vwap_path), detail_x, work_df.loc[detail_mask], "vwap")
            pred_volume = _predict_model(joblib.load(volume_path), detail_x, work_df.loc[detail_mask], "volume")
            actual_vwap = work_df.loc[detail_mask, f"future_vwap_{h}"].to_numpy(dtype=float)
            actual_volume = work_df.loc[detail_mask, f"future_volume_{h}"].to_numpy(dtype=float)
            detail = pd.DataFrame(
                {
                    **meta,
                    "stock_code": work_df.loc[detail_mask, "stock_code"].to_numpy(),
                    "datetime": pd.to_datetime(work_df.loc[detail_mask, "datetime"]).to_numpy(),
                    "horizon": h,
                    "actual_vwap": actual_vwap,
                    "predicted_vwap": pred_vwap,
                    "vwap_error": pred_vwap - actual_vwap,
                    "abs_vwap_error": np.abs(pred_vwap - actual_vwap),
                    "actual_volume": actual_volume,
                    "predicted_volume": pred_volume,
                    "volume_error": pred_volume - actual_volume,
                    "abs_volume_error": np.abs(pred_volume - actual_volume),
                }
            )
            detail["date"] = pd.to_datetime(detail["datetime"]).dt.date.astype(str)
            detail["minute"] = pd.to_datetime(detail["datetime"]).dt.strftime("%H:%M")
            detail_parts.append(detail)

    backtest_rows = _build_backtest_rows(
        work_df,
        x_all,
        group_path,
        horizons,
        float(config.get("backtest_order_qty", 100000)),
        float(config.get("participation_limit", 0.30)),
    )
    for item in backtest_rows:
        item.update(meta)
        item["order_qty"] = float(config.get("backtest_order_qty", 100000))
    return metrics, detail_parts, backtest_rows


def run_rolling_backtest(config: dict[str, Any]) -> None:
    ensure_dirs(config)
    dataset_path = resolve_path(config["feature_data_dir"]) / "model_dataset.parquet"
    model_root = resolve_path(config.get("rolling_model_dir", "data/models/rolling"))
    output_root = resolve_path(config.get("rolling_output_dir", "data/outputs/rolling"))
    model_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    parquet_file = pq.ParquetFile(dataset_path)
    base_feature_columns = _schema_feature_columns(parquet_file.schema_arrow)
    feature_columns = _with_runtime_columns(base_feature_columns)
    horizons = [int(h) for h in config["horizons"]]
    windows = [int(w) for w in config.get("rolling_windows", [5, 8])]
    batch_size = int(config.get("rolling_batch_size", 300_000))
    keep_intermediate = bool(config.get("rolling_keep_intermediate", False))
    all_dates = get_trading_dates(dataset_path, batch_size=batch_size)
    test_dates = _months_filter(all_dates, config.get("rolling_test_months", config.get("test_months", [])))
    groups = ["high", "medium", "low"]

    metrics_rows: list[dict[str, Any]] = []
    detail_parts: list[pd.DataFrame] = []
    backtest_rows: list[dict[str, Any]] = []

    for test_date in test_dates:
        prior_dates = [date for date in all_dates if date < test_date]
        for window in windows:
            train_dates = prior_dates[-window:]
            if not train_dates:
                continue
            window_name = f"window_{window}d"
            print(f"[rolling] test_date={test_date} {window_name} train_dates={train_dates}", flush=True)
            for group_name in groups:
                group_path = model_root / window_name / test_date / group_name
                group_path.mkdir(parents=True, exist_ok=True)
                joblib.dump(feature_columns, group_path / "feature_columns.joblib")
                temp_dir = output_root / "_tmp" / window_name / test_date
                train_path = _prepare_group_train_file(
                    dataset_path,
                    train_dates,
                    group_name,
                    base_feature_columns,
                    horizons,
                    batch_size,
                    temp_dir,
                )
                if train_path is None:
                    continue

                for h in horizons:
                    for target_kind in ["vwap", "volume"]:
                        train_df = _load_train_for_model(
                            train_path,
                            base_feature_columns,
                            h,
                            target_kind,
                        )
                        if train_df.empty:
                            continue
                        model, row_count = _train_one_model(train_df, feature_columns, h, target_kind, config)
                        joblib.dump(model, group_path / f"{target_kind}_h{h}.joblib")
                        print(
                            f"[rolling] saved {window_name}/{test_date}/{group_name}/{target_kind}_h{h} rows={row_count:,}",
                            flush=True,
                        )
                        del train_df, model
                        gc.collect()

                test_df = _load_test_for_group(dataset_path, test_date, group_name, base_feature_columns, horizons, batch_size)
                if test_df.empty:
                    continue
                meta = {
                    "model_type": "rolling",
                    "window": window,
                    "test_date": test_date,
                    "liquidity_group": group_name,
                    "train_start_date": train_dates[0],
                    "train_end_date": train_dates[-1],
                    "train_day_count": len(train_dates),
                }
                group_metrics, group_details, group_backtest = _evaluate_group_models(test_df, feature_columns, group_path, config, meta)
                metrics_rows.extend(group_metrics)
                detail_parts.extend(group_details)
                backtest_rows.extend(group_backtest)
                del test_df
                if train_path.exists() and not keep_intermediate:
                    train_path.unlink()
                gc.collect()

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_root / "rolling_evaluation_metrics.csv", index=False, encoding="utf-8-sig")
    if detail_parts:
        detail_df = pd.concat(detail_parts, ignore_index=True)
        detail_df.to_csv(output_root / "rolling_prediction_error_detail.csv", index=False, encoding="utf-8-sig")
        for filename, cols in {
            "rolling_prediction_error_by_date.csv": ["window", "test_date", "horizon"],
            "rolling_prediction_error_by_stock.csv": ["window", "stock_code", "horizon"],
            "rolling_prediction_error_by_minute.csv": ["window", "minute", "horizon"],
        }.items():
            report = detail_df.groupby(cols, as_index=False).agg(
                sample_count=("horizon", "size"),
                vwap_mae=("abs_vwap_error", "mean"),
                volume_mae=("abs_volume_error", "mean"),
            )
            report.to_csv(output_root / filename, index=False, encoding="utf-8-sig")
    if backtest_rows:
        backtest_df = pd.DataFrame(backtest_rows)
        backtest_df.to_csv(output_root / "rolling_recommendation_backtest_detail.csv", index=False, encoding="utf-8-sig")
        summary = backtest_df.groupby(["window", "test_date", "liquidity_group", "side"], as_index=False).agg(
            sample_count=("side", "size"),
            pred_feasible_rate=("has_pred_feasible", "mean"),
            true_feasible_rate=("has_true_feasible", "mean"),
            horizon_match_rate=("horizon_match", "mean"),
            avg_regret=("regret", "mean"),
            max_absolute_regret=("absolute_regret", "max"),
        )
        summary.to_csv(output_root / "rolling_recommendation_backtest_summary.csv", index=False, encoding="utf-8-sig")
        worst_cases = backtest_df.sort_values("absolute_regret", ascending=False, na_position="last").head(
            int(config.get("worst_case_top_n", 100))
        )
        worst_cases.to_csv(output_root / "rolling_recommendation_backtest_worst_cases.csv", index=False, encoding="utf-8-sig")
    if not metrics_df.empty:
        comparison = metrics_df.groupby(["window"], as_index=False).agg(
            vwap_mae=("vwap_mae", "mean"),
            volume_mae=("volume_mae", "mean"),
            baseline_vwap_mae=("baseline_vwap_mae", "mean"),
            baseline_volume_mae=("baseline_volume_mae", "mean"),
            vwap_mae_improvement=("vwap_mae_improvement", "mean"),
            volume_mae_improvement=("volume_mae_improvement", "mean"),
        )
        comparison.to_csv(output_root / "rolling_window_comparison.csv", index=False, encoding="utf-8-sig")
    tmp_root = output_root / "_tmp"
    if tmp_root.exists() and not keep_intermediate:
        shutil.rmtree(tmp_root)
