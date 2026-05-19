from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("PyTorch is required for rolling training. Install with: pip install torch") from exc

from src.config import ensure_dirs
from src.ive_dataset import IVEDataset, Normalizer, get_ive_feature_columns
from src.ive_model import IVEModel
from src.train import train_ive_models
from src.utils import resolve_path


def _feature_part_dir(config: dict[str, Any]) -> Path:
    return resolve_path(config.get("feature_parts_dir", "data/features/model_parts"))


def _feature_part_paths(config: dict[str, Any]) -> list[Path]:
    root = _feature_part_dir(config)
    return sorted(path for path in root.glob("*.parquet") if path.stem[:8].isdigit())


def _date_strings(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.date.astype(str)


def _rmse(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def get_trading_dates(dataset_path: str | Path, batch_size: int = 500_000) -> list[str]:
    dataset_path = Path(dataset_path)
    if dataset_path.is_dir():
        return sorted(pd.to_datetime(path.stem[:8], format="%Y%m%d").date().isoformat() for path in dataset_path.glob("*.parquet") if path.stem[:8].isdigit())
    parquet_file = pq.ParquetFile(dataset_path)
    dates: set[str] = set()
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["datetime"]):
        df = batch.to_pandas()
        dates.update(_date_strings(df["datetime"]).unique())
    return sorted(dates)


def _months_filter(dates: list[str], months: list[int | str]) -> list[str]:
    month_set = {str(month) for month in months}
    return [date for date in dates if date.replace("-", "")[:6] in month_set]


def _rolling_eval_dates(all_dates: list[str], config: dict[str, Any]) -> list[tuple[str, str]]:
    train_months = config.get("rolling_train_months", config.get("train_months", []))
    test_months = config.get("rolling_test_months", config.get("test_months", []))
    train_dates = _months_filter(all_dates, train_months)
    test_dates = _months_filter(all_dates, test_months)
    return [("train", date) for date in train_dates] + [("test", date) for date in test_dates]


def _load_filtered_rows(
    dataset_path: str | Path,
    dates: set[str],
    group_name: str | None,
    columns: list[str],
    batch_size: int,
) -> pd.DataFrame:
    dataset_path = Path(dataset_path)
    if dataset_path.is_dir():
        parts: list[pd.DataFrame] = []
        date_to_path = {
            pd.to_datetime(path.stem[:8], format="%Y%m%d").date().isoformat(): path
            for path in dataset_path.glob("*.parquet")
            if path.stem[:8].isdigit()
        }
        for date in sorted(dates):
            path = date_to_path.get(date)
            if path is None:
                continue
            available = set(pq.ParquetFile(path).schema_arrow.names)
            read_columns = [col for col in columns if col in available]
            df = pd.read_parquet(path, columns=read_columns)
            if group_name is not None:
                df = df[df["liquidity_group"].astype(str).eq(str(group_name))].copy()
            if not df.empty:
                parts.append(df)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    parquet_file = pq.ParquetFile(dataset_path)
    available = set(parquet_file.schema_arrow.names)
    read_columns = [col for col in columns if col in available]
    parts: list[pd.DataFrame] = []
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=read_columns):
        df = batch.to_pandas()
        mask = _date_strings(df["datetime"]).isin(dates)
        if group_name is not None:
            mask &= df["liquidity_group"].astype(str).eq(str(group_name))
        if mask.any():
            parts.append(df.loc[mask].copy())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=read_columns)


def _drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [col for col in df.columns if col.startswith("group_same_minute_")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    if "liquidity_group" in df.columns:
        df = df.drop(columns=["liquidity_group"])
    return df


def _classify_window_liquidity(train_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    daily = train_df.groupby(["stock_code", "date"], as_index=False).agg(
        daily_amount=("amount", "sum"),
        daily_volume=("volume", "sum"),
        daily_trade_count=("trade_count", "sum") if "trade_count" in train_df.columns else ("volume", "size"),
    )
    summary = daily.groupby("stock_code", as_index=False).agg(
        avg_daily_amount=("daily_amount", "mean"),
        avg_daily_volume=("daily_volume", "mean"),
        avg_daily_trade_count=("daily_trade_count", "mean"),
    )
    if summary.empty:
        return pd.DataFrame(columns=["stock_code", "avg_daily_amount", "avg_daily_volume", "avg_daily_trade_count", "liquidity_group"])
    low_q = float(config.get("rolling_liquidity_low_quantile", 0.30))
    high_q = float(config.get("rolling_liquidity_high_quantile", 0.70))
    low_cut = summary["avg_daily_amount"].quantile(low_q)
    high_cut = summary["avg_daily_amount"].quantile(high_q)
    summary["liquidity_group"] = "medium"
    summary.loc[summary["avg_daily_amount"] >= high_cut, "liquidity_group"] = "high"
    summary.loc[summary["avg_daily_amount"] <= low_cut, "liquidity_group"] = "low"
    return summary.sort_values("avg_daily_amount", ascending=False).reset_index(drop=True)


def _apply_window_liquidity(df: pd.DataFrame, liquidity: pd.DataFrame) -> pd.DataFrame:
    out = _drop_leaky_columns(df).merge(liquidity[["stock_code", "liquidity_group"]], on="stock_code", how="left")
    out["liquidity_group"] = out["liquidity_group"].fillna("low")
    return out


def _load_model(group_path: Path) -> tuple[IVEModel, dict[str, Any]]:
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
    model.load_state_dict(torch.load(group_path / "ive_model.pt", map_location="cpu"))
    model.eval()
    return model, meta


def _predict_frame(test_df: pd.DataFrame, group_path: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, list[int]]:
    model, meta = _load_model(group_path)
    normalizer = Normalizer(mean=meta["normalizer"]["mean"], std=meta["normalizer"]["std"])
    dataset = IVEDataset(
        test_df,
        list(meta["feature_columns"]),
        [int(h) for h in meta["horizons"]],
        meta["stock_vocab"],
        meta["group_vocab"],
        normalizer,
        context_length=int(meta.get("context_length", 390)),
    )
    loader = DataLoader(dataset, batch_size=int(config.get("ive_batch_size", 256)), shuffle=False, num_workers=0)
    rows = dataset.df.iloc[dataset.indices].reset_index(drop=True).copy()
    if rows.empty:
        return rows, [int(h) for h in meta["horizons"]]
    volume_mu_parts: list[np.ndarray] = []
    volume_sigma_parts: list[np.ndarray] = []
    vwap_return_parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["x"], batch["stock_id"], batch["group_id"], batch["padding_mask"])
            volume_mu_parts.append(out["volume_mu"].cpu().numpy())
            volume_sigma_parts.append(np.log1p(np.exp(out["volume_log_sigma"].cpu().numpy())))
            vwap_return_parts.append(out["vwap_return"].cpu().numpy())
    volume_mu = np.vstack(volume_mu_parts)
    volume_sigma = np.vstack(volume_sigma_parts)
    vwap_return = np.vstack(vwap_return_parts)
    ratio_scale = float(config.get("volume_ratio_scale", 10000.0))
    horizons = [int(h) for h in meta["horizons"]]
    daily_prior = rows["stock_code"].astype(str).map(meta.get("daily_volume_prior", {}))
    daily_prior = daily_prior.fillna(rows["liquidity_group"].astype(str).map(meta.get("group_daily_volume_prior", {})))
    for col in ["stock_rolling_volume_mean_20d", "stock_rolling_volume_mean_10d", "stock_rolling_volume_mean_5d", "volume"]:
        if daily_prior.isna().any() and col in rows.columns:
            daily_prior = daily_prior.fillna(rows[col])
    daily_prior = daily_prior.astype(float).clip(lower=0)
    base_vwap = rows["vwap"].replace(0, np.nan).fillna(rows["close"]).astype(float)
    for i, h in enumerate(horizons):
        rows[f"predicted_volume_ratio_{h}"] = np.maximum(np.expm1(volume_mu[:, i]) / ratio_scale, 0)
        rows[f"predicted_volume_sigma_{h}"] = np.maximum(np.expm1(volume_sigma[:, i]) / ratio_scale, 0)
        rows[f"predicted_volume_{h}"] = rows[f"predicted_volume_ratio_{h}"] * daily_prior.to_numpy()
        rows[f"predicted_vwap_{h}"] = base_vwap.to_numpy() * (1 + vwap_return[:, i])
    return rows, horizons


def _recommend(rows: list[dict[str, Any]], side: str) -> dict[str, Any] | None:
    feasible = [row for row in rows if row["feasible"]]
    if not feasible:
        return None
    return min(feasible, key=lambda r: float(r["vwap"])) if side == "buy" else max(feasible, key=lambda r: float(r["vwap"]))


def _evaluate_predictions(
    pred_df: pd.DataFrame,
    horizons: list[int],
    config: dict[str, Any],
    meta: dict[str, Any],
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, Any]] = []
    detail_parts: list[pd.DataFrame] = []
    participation_limit = float(config.get("participation_limit", 0.30))
    order_qty = float(config.get("backtest_order_qty", 100000))
    for h in horizons:
        mask = pred_df[f"future_vwap_{h}"].notna() & pred_df[f"future_volume_{h}"].notna()
        if not mask.any():
            continue
        true_vwap = pred_df.loc[mask, f"future_vwap_{h}"].astype(float)
        pred_vwap = pred_df.loc[mask, f"predicted_vwap_{h}"].astype(float)
        true_volume = pred_df.loc[mask, f"future_volume_{h}"].astype(float)
        pred_volume = pred_df.loc[mask, f"predicted_volume_{h}"].astype(float)
        true_ratio = pred_df.loc[mask, f"future_volume_ratio_{h}"].astype(float)
        pred_ratio = pred_df.loc[mask, f"predicted_volume_ratio_{h}"].astype(float)
        metrics.append(
            {
                **meta,
                "horizon": h,
                "sample_count": int(mask.sum()),
                "vwap_mae": float(mean_absolute_error(true_vwap, pred_vwap)),
                "vwap_rmse": _rmse(true_vwap, pred_vwap.to_numpy()),
                "volume_mae": float(mean_absolute_error(true_volume, pred_volume)),
                "volume_rmse": _rmse(true_volume, pred_volume.to_numpy()),
                "volume_ratio_mae": float(mean_absolute_error(true_ratio, pred_ratio)),
                "volume_ratio_rmse": _rmse(true_ratio, pred_ratio.to_numpy()),
                "feasibility_accuracy": float(((order_qty / pred_volume.replace(0, np.nan) <= participation_limit) == (order_qty / true_volume.replace(0, np.nan) <= participation_limit)).mean()),
            }
        )
        detail = pred_df.loc[mask, ["stock_code", "datetime", "date", "minute", "liquidity_group"]].copy()
        detail = detail.assign(
            **meta,
            horizon=h,
            actual_vwap=true_vwap.to_numpy(),
            predicted_vwap=pred_vwap.to_numpy(),
            vwap_error=(pred_vwap - true_vwap).to_numpy(),
            abs_vwap_error=np.abs(pred_vwap - true_vwap).to_numpy(),
            actual_volume=true_volume.to_numpy(),
            predicted_volume=pred_volume.to_numpy(),
            volume_error=(pred_volume - true_volume).to_numpy(),
            abs_volume_error=np.abs(pred_volume - true_volume).to_numpy(),
            actual_volume_ratio=true_ratio.to_numpy(),
            predicted_volume_ratio=pred_ratio.to_numpy(),
            volume_ratio_error=(pred_ratio - true_ratio).to_numpy(),
            predicted_volume_sigma=pred_df.loc[mask, f"predicted_volume_sigma_{h}"].to_numpy(),
        )
        detail["minute_of_day"] = pd.to_datetime(detail["datetime"]).dt.hour * 60 + pd.to_datetime(detail["datetime"]).dt.minute
        detail_parts.append(detail)

    backtest_rows: list[dict[str, Any]] = []
    for _, sample in pred_df.iterrows():
        pred_candidates: list[dict[str, Any]] = []
        true_candidates: list[dict[str, Any]] = []
        for h in horizons:
            pred_volume = float(sample.get(f"predicted_volume_{h}", np.nan))
            true_volume = float(sample.get(f"future_volume_{h}", np.nan))
            pred_vwap = float(sample.get(f"predicted_vwap_{h}", np.nan))
            true_vwap = float(sample.get(f"future_vwap_{h}", np.nan))
            pred_part = order_qty / pred_volume if pred_volume > 0 else np.inf
            true_part = order_qty / true_volume if true_volume > 0 else np.inf
            pred_candidates.append({"horizon": h, "vwap": pred_vwap, "volume": pred_volume, "participation": pred_part, "feasible": pred_part <= participation_limit})
            if np.isfinite(true_vwap):
                true_candidates.append({"horizon": h, "vwap": true_vwap, "volume": true_volume, "participation": true_part, "feasible": true_part <= participation_limit})
        for side in ["buy", "sell"]:
            pred_pick = _recommend(pred_candidates, side)
            true_pick = _recommend(true_candidates, side)
            chosen_true = None if pred_pick is None else next((c for c in true_candidates if c["horizon"] == pred_pick["horizon"]), None)
            regret = np.nan
            if chosen_true is not None and true_pick is not None:
                regret = chosen_true["vwap"] - true_pick["vwap"] if side == "buy" else true_pick["vwap"] - chosen_true["vwap"]
            backtest_rows.append(
                {
                    **meta,
                    "stock_code": sample["stock_code"],
                    "datetime": sample["datetime"],
                    "side": side,
                    "has_pred_feasible": pred_pick is not None,
                    "has_true_feasible": true_pick is not None,
                    "recommended_horizon": None if pred_pick is None else pred_pick["horizon"],
                    "true_best_horizon": None if true_pick is None else true_pick["horizon"],
                    "predicted_vwap": np.nan if pred_pick is None else pred_pick["vwap"],
                    "predicted_volume": np.nan if pred_pick is None else pred_pick["volume"],
                    "predicted_participation": np.nan if pred_pick is None else pred_pick["participation"],
                    "recommended_actual_vwap": np.nan if chosen_true is None else chosen_true["vwap"],
                    "recommended_actual_volume": np.nan if chosen_true is None else chosen_true["volume"],
                    "recommended_actual_participation": np.nan if chosen_true is None else chosen_true["participation"],
                    "recommended_actual_feasible": False if chosen_true is None else chosen_true["feasible"],
                    "horizon_match": False if pred_pick is None or true_pick is None else pred_pick["horizon"] == true_pick["horizon"],
                    "regret": regret,
                    "absolute_regret": abs(regret) if np.isfinite(regret) else np.nan,
                }
            )
    detail_df = pd.concat(detail_parts, ignore_index=True) if detail_parts else pd.DataFrame()
    return metrics, detail_df, pd.DataFrame(backtest_rows)


def _write_reports(output_root: Path, metrics_rows: list[dict[str, Any]], details: list[pd.DataFrame], backtests: list[pd.DataFrame]) -> None:
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_root / "rolling_evaluation_metrics.csv", index=False, encoding="utf-8-sig")
    if details:
        detail_df = pd.concat(details, ignore_index=True)
        detail_df.to_csv(output_root / "rolling_prediction_error_detail.csv", index=False, encoding="utf-8-sig")
        for filename, cols in {
            "rolling_prediction_error_by_date.csv": ["split", "window", "test_date", "date", "horizon"],
            "rolling_prediction_error_by_stock.csv": ["split", "window", "stock_code", "horizon"],
            "rolling_prediction_error_by_minute.csv": ["split", "window", "minute", "minute_of_day", "horizon"],
        }.items():
            detail_df.groupby(cols, as_index=False).agg(
                sample_count=("horizon", "size"),
                vwap_mae=("abs_vwap_error", "mean"),
                volume_mae=("abs_volume_error", "mean"),
                volume_ratio_mae=("volume_ratio_error", lambda s: float(np.mean(np.abs(s)))),
            ).to_csv(output_root / filename, index=False, encoding="utf-8-sig")
    if backtests:
        backtest_df = pd.concat(backtests, ignore_index=True)
        backtest_df.to_csv(output_root / "rolling_recommendation_backtest_detail.csv", index=False, encoding="utf-8-sig")
        backtest_df.groupby(["split", "window", "test_date", "liquidity_group", "side"], as_index=False).agg(
            sample_count=("side", "size"),
            pred_feasible_rate=("has_pred_feasible", "mean"),
            true_feasible_rate=("has_true_feasible", "mean"),
            horizon_match_rate=("horizon_match", "mean"),
            avg_regret=("regret", "mean"),
            max_absolute_regret=("absolute_regret", "max"),
        ).to_csv(output_root / "rolling_recommendation_backtest_summary.csv", index=False, encoding="utf-8-sig")
        worst = backtest_df.sort_values("absolute_regret", ascending=False, na_position="last").head(100)
        worst.to_csv(output_root / "rolling_recommendation_backtest_worst_cases.csv", index=False, encoding="utf-8-sig")
    if not metrics_df.empty:
        metrics_df.groupby(["split", "window"], as_index=False).agg(
            vwap_mae=("vwap_mae", "mean"),
            volume_mae=("volume_mae", "mean"),
            volume_ratio_mae=("volume_ratio_mae", "mean"),
            feasibility_accuracy=("feasibility_accuracy", "mean"),
        ).to_csv(output_root / "rolling_window_comparison.csv", index=False, encoding="utf-8-sig")


def run_rolling_backtest(config: dict[str, Any]) -> None:
    ensure_dirs(config)
    feature_parts = _feature_part_paths(config)
    dataset_path = _feature_part_dir(config) if feature_parts else resolve_path(config["feature_data_dir"]) / "model_dataset.parquet"
    model_root = resolve_path(config.get("rolling_model_dir", "data/models/rolling"))
    output_root = resolve_path(config.get("rolling_output_dir", "data/outputs/rolling"))
    model_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    batch_size = int(config.get("rolling_batch_size", 300_000))
    all_dates = get_trading_dates(dataset_path, batch_size=batch_size)
    eval_dates = _rolling_eval_dates(all_dates, config)
    windows = [int(w) for w in config.get("rolling_windows", [5, 8])]
    schema_source = feature_parts[0] if feature_parts else dataset_path
    schema = pq.ParquetFile(schema_source).schema_arrow
    all_columns = list(schema.names)
    feature_columns: list[str] | None = None
    groups = ["high", "medium", "low"]
    metrics_rows: list[dict[str, Any]] = []
    detail_dfs: list[pd.DataFrame] = []
    backtest_dfs: list[pd.DataFrame] = []

    for split, test_date in eval_dates:
        prior_dates = [date for date in all_dates if date < test_date]
        for window in windows:
            train_dates = prior_dates[-window:]
            if not train_dates:
                continue
            print(f"[rolling] split={split} test_date={test_date} window={window} train_dates={train_dates}", flush=True)
            group_model_root = model_root / f"window_{window}d" / test_date
            train_window_df = _load_filtered_rows(dataset_path, set(train_dates), None, all_columns, batch_size)
            if train_window_df.empty:
                continue
            train_window_df = _drop_leaky_columns(train_window_df)
            liquidity = _classify_window_liquidity(train_window_df, config)
            group_model_root.mkdir(parents=True, exist_ok=True)
            liquidity.to_parquet(group_model_root / "stock_liquidity_group.parquet", index=False)
            train_window_df = _apply_window_liquidity(train_window_df, liquidity)
            test_window_df = _load_filtered_rows(dataset_path, {test_date}, None, all_columns, batch_size)
            if test_window_df.empty:
                continue
            test_window_df = _apply_window_liquidity(test_window_df, liquidity)
            if feature_columns is None:
                feature_columns = get_ive_feature_columns(train_window_df, [int(h) for h in config["horizons"]])
            for group_name in groups:
                train_df = train_window_df[train_window_df["liquidity_group"].astype(str).eq(group_name)].copy()
                if train_df.empty:
                    continue
                train_ive_models(train_df, group_model_root, config, feature_columns=feature_columns)
                test_df = test_window_df[test_window_df["liquidity_group"].astype(str).eq(group_name)].copy()
                if test_df.empty:
                    continue
                group_path = group_model_root / group_name
                pred_df, horizons = _predict_frame(test_df, group_path, config)
                if pred_df.empty:
                    continue
                meta = {
                    "model_type": "ive_rolling",
                    "split": split,
                    "window": window,
                    "test_date": test_date,
                    "liquidity_group": group_name,
                    "train_start_date": train_dates[0],
                    "train_end_date": train_dates[-1],
                    "train_day_count": len(train_dates),
                }
                group_metrics, group_detail, group_backtest = _evaluate_predictions(pred_df, horizons, config, meta)
                metrics_rows.extend(group_metrics)
                if not group_detail.empty:
                    detail_dfs.append(group_detail)
                if not group_backtest.empty:
                    backtest_dfs.append(group_backtest)

    _write_reports(output_root, metrics_rows, detail_dfs, backtest_dfs)
    tmp_root = output_root / "_tmp"
    if tmp_root.exists() and not bool(config.get("rolling_keep_intermediate", False)):
        shutil.rmtree(tmp_root)
