from __future__ import annotations

import gc
import json
import multiprocessing as mp
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


GROUPS = ["high", "medium", "low"]
MODEL_ARTIFACTS = [
    "ive_model.pt",
    "model_meta.json",
    "feature_columns.joblib",
    "stock_vocab.joblib",
    "group_vocab.joblib",
    "normalizer.joblib",
]


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


def build_rolling_tasks(config: dict[str, Any]) -> tuple[list[dict[str, Any]], Path, list[str]]:
    feature_parts = _feature_part_paths(config)
    dataset_path = _feature_part_dir(config) if feature_parts else resolve_path(config["feature_data_dir"]) / "model_dataset.parquet"
    batch_size = int(config.get("rolling_batch_size", 300_000))
    all_dates = get_trading_dates(dataset_path, batch_size=batch_size)
    eval_dates = _rolling_eval_dates(all_dates, config)
    windows = [int(w) for w in config.get("rolling_windows", [5, 8])]
    tasks: list[dict[str, Any]] = []
    for split, test_date in eval_dates:
        prior_dates = [date for date in all_dates if date < test_date]
        for window in windows:
            train_dates = prior_dates[-window:]
            if not train_dates:
                continue
            tasks.append(
                {
                    "split": split,
                    "test_date": test_date,
                    "window": window,
                    "train_dates": train_dates,
                }
            )
    return tasks, Path(dataset_path), all_dates


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


def _model_artifacts_complete(group_path: Path) -> bool:
    return all((group_path / name).exists() for name in MODEL_ARTIFACTS)


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
    for col in ["stock_rolling_volume_mean_10d", "stock_rolling_volume_mean_5d", "volume"]:
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


def _schema_columns(dataset_path: Path) -> list[str]:
    if dataset_path.is_dir():
        parts = sorted(path for path in dataset_path.glob("*.parquet") if path.stem[:8].isdigit())
        if not parts:
            return []
        schema_source = parts[0]
    else:
        schema_source = dataset_path
    return list(pq.ParquetFile(schema_source).schema_arrow.names)


def _task_model_root(config: dict[str, Any], task: dict[str, Any]) -> Path:
    model_root = resolve_path(config.get("rolling_model_dir", "data/models/rolling"))
    return model_root / f"window_{int(task['window'])}d" / str(task["test_date"])


def train_rolling_task(
    config: dict[str, Any],
    task: dict[str, Any],
    dataset_path: str | Path | None = None,
    all_columns: list[str] | None = None,
    overwrite_models: bool | None = None,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path) if dataset_path is not None else build_rolling_tasks(config)[1]
    batch_size = int(config.get("rolling_batch_size", 300_000))
    all_columns = all_columns or _schema_columns(dataset_path)
    overwrite = bool(config.get("rolling_overwrite_models", False)) if overwrite_models is None else bool(overwrite_models)
    group_model_root = _task_model_root(config, task)
    train_dates = [str(date) for date in task["train_dates"]]
    print(
        f"[rolling-train] split={task['split']} test_date={task['test_date']} "
        f"window={task['window']} train_dates={train_dates}",
        flush=True,
    )
    train_window_df = _load_filtered_rows(dataset_path, set(train_dates), None, all_columns, batch_size)
    if train_window_df.empty:
        return {**task, "status": "empty_train"}
    train_window_df = _drop_leaky_columns(train_window_df)
    liquidity = _classify_window_liquidity(train_window_df, config)
    group_model_root.mkdir(parents=True, exist_ok=True)
    liquidity.to_parquet(group_model_root / "stock_liquidity_group.parquet", index=False)
    train_window_df = _apply_window_liquidity(train_window_df, liquidity)
    feature_columns = get_ive_feature_columns(train_window_df, [int(h) for h in config["horizons"]])

    trained: list[str] = []
    skipped: list[str] = []
    empty: list[str] = []
    for group_name in GROUPS:
        group_path = group_model_root / group_name
        train_df = train_window_df[train_window_df["liquidity_group"].astype(str).eq(group_name)].copy()
        if train_df.empty:
            empty.append(group_name)
            continue
        if not overwrite and _model_artifacts_complete(group_path):
            print(f"[rolling-train] skip existing group={group_name} path={group_path}", flush=True)
            skipped.append(group_name)
            del train_df
            gc.collect()
            continue
        print(f"[rolling-train] start group={group_name} rows={len(train_df):,}", flush=True)
        train_ive_models(train_df, group_model_root, config, feature_columns=feature_columns)
        print(f"[rolling-train] saved group={group_name} path={group_path}", flush=True)
        trained.append(group_name)
        del train_df
        gc.collect()

    del train_window_df
    gc.collect()
    return {**task, "status": "trained", "trained_groups": trained, "skipped_groups": skipped, "empty_groups": empty}


def _part_dir(output_root: Path, task: dict[str, Any], group_name: str) -> Path:
    return output_root / "parts" / f"window_{int(task['window'])}d" / str(task["test_date"]) / group_name


def _write_prediction_part(
    output_root: Path,
    task: dict[str, Any],
    group_name: str,
    metrics_rows: list[dict[str, Any]],
    detail_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
) -> dict[str, str]:
    part_dir = _part_dir(output_root, task, group_name)
    part_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = part_dir / "metrics.csv"
    detail_path = part_dir / "detail.parquet"
    backtest_path = part_dir / "backtest.parquet"
    for stale_path in [metrics_path, detail_path, backtest_path]:
        if stale_path.exists():
            stale_path.unlink()
    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    if not detail_df.empty:
        detail_df.to_parquet(detail_path, index=False)
    if not backtest_df.empty:
        backtest_df.to_parquet(backtest_path, index=False)
    return {
        "metrics": str(metrics_path) if metrics_path.exists() else "",
        "detail": str(detail_path) if detail_path.exists() else "",
        "backtest": str(backtest_path) if backtest_path.exists() else "",
    }


def predict_rolling_group_task(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload["config"]
    task = payload["task"]
    group_name = str(payload["group_name"])
    dataset_path = Path(payload["dataset_path"])
    all_columns = payload["all_columns"]
    batch_size = int(config.get("rolling_batch_size", 300_000))
    output_root = resolve_path(config.get("rolling_output_dir", "data/outputs/rolling"))
    group_model_root = _task_model_root(config, task)
    group_path = group_model_root / group_name
    if not _model_artifacts_complete(group_path):
        return {**task, "group": group_name, "status": "missing_model"}
    liquidity_path = group_model_root / "stock_liquidity_group.parquet"
    if not liquidity_path.exists():
        return {**task, "group": group_name, "status": "missing_liquidity"}
    print(f"[rolling-predict] start split={task['split']} test_date={task['test_date']} window={task['window']} group={group_name}", flush=True)
    liquidity = pd.read_parquet(liquidity_path)
    test_window_df = _load_filtered_rows(dataset_path, {str(task["test_date"])}, None, all_columns, batch_size)
    if test_window_df.empty:
        return {**task, "group": group_name, "status": "empty_test"}
    test_window_df = _apply_window_liquidity(test_window_df, liquidity)
    test_df = test_window_df[test_window_df["liquidity_group"].astype(str).eq(group_name)].copy()
    del test_window_df
    if test_df.empty:
        return {**task, "group": group_name, "status": "empty_group"}
    pred_df, horizons = _predict_frame(test_df, group_path, config)
    del test_df
    if pred_df.empty:
        return {**task, "group": group_name, "status": "empty_prediction"}
    meta = {
        "model_type": "ive_rolling",
        "split": task["split"],
        "window": int(task["window"]),
        "test_date": task["test_date"],
        "liquidity_group": group_name,
        "train_start_date": task["train_dates"][0],
        "train_end_date": task["train_dates"][-1],
        "train_day_count": len(task["train_dates"]),
    }
    group_metrics, group_detail, group_backtest = _evaluate_predictions(pred_df, horizons, config, meta)
    paths = _write_prediction_part(output_root, task, group_name, group_metrics, group_detail, group_backtest)
    print(f"[rolling-predict] done split={task['split']} test_date={task['test_date']} window={task['window']} group={group_name}", flush=True)
    del pred_df, group_detail, group_backtest
    gc.collect()
    return {**task, "group": group_name, "status": "predicted", "paths": paths}


def _prediction_payloads(
    config: dict[str, Any],
    tasks: list[dict[str, Any]],
    dataset_path: Path,
    all_columns: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            "config": config,
            "task": task,
            "group_name": group_name,
            "dataset_path": str(dataset_path),
            "all_columns": all_columns,
        }
        for task in tasks
        for group_name in GROUPS
    ]


def _collect_part_reports(part_results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame]]:
    metrics_rows: list[dict[str, Any]] = []
    detail_dfs: list[pd.DataFrame] = []
    backtest_dfs: list[pd.DataFrame] = []
    for result in part_results:
        if result.get("status") != "predicted":
            continue
        paths = result.get("paths", {})
        metrics_raw = paths.get("metrics", "")
        metrics_path = Path(metrics_raw) if metrics_raw else None
        detail_raw = paths.get("detail", "")
        backtest_raw = paths.get("backtest", "")
        detail_path = Path(detail_raw) if detail_raw else None
        backtest_path = Path(backtest_raw) if backtest_raw else None
        if metrics_path is not None and metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            if not metrics.empty:
                metrics_rows.extend(metrics.to_dict("records"))
        if detail_path is not None and detail_path.exists():
            detail = pd.read_parquet(detail_path)
            if not detail.empty:
                detail_dfs.append(detail)
        if backtest_path is not None and backtest_path.exists():
            backtest = pd.read_parquet(backtest_path)
            if not backtest.empty:
                backtest_dfs.append(backtest)
    return metrics_rows, detail_dfs, backtest_dfs


def run_rolling_predictions(
    config: dict[str, Any],
    tasks: list[dict[str, Any]],
    dataset_path: str | Path,
    all_columns: list[str],
    predict_workers: int | None = None,
) -> list[dict[str, Any]]:
    output_root = resolve_path(config.get("rolling_output_dir", "data/outputs/rolling"))
    output_root.mkdir(parents=True, exist_ok=True)
    workers = int(predict_workers if predict_workers is not None else config.get("rolling_predict_workers", 1))
    workers = max(workers, 1)
    payloads = _prediction_payloads(config, tasks, Path(dataset_path), all_columns)
    if not payloads:
        return []
    print(f"[rolling-predict] tasks={len(payloads)} workers={workers}", flush=True)
    if workers == 1:
        results = [predict_rolling_group_task(payload) for payload in payloads]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers, maxtasksperchild=1) as pool:
            results = list(pool.imap_unordered(predict_rolling_group_task, payloads))
    metrics_rows, detail_dfs, backtest_dfs = _collect_part_reports(results)
    _write_reports(output_root, metrics_rows, detail_dfs, backtest_dfs)
    return results


def run_rolling_backtest(
    config: dict[str, Any],
    overwrite_models: bool | None = None,
    predict_workers: int | None = None,
) -> None:
    ensure_dirs(config)
    model_root = resolve_path(config.get("rolling_model_dir", "data/models/rolling"))
    output_root = resolve_path(config.get("rolling_output_dir", "data/outputs/rolling"))
    model_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    tasks, dataset_path, _ = build_rolling_tasks(config)
    all_columns = _schema_columns(dataset_path)
    print(f"[rolling] train phase tasks={len(tasks)}", flush=True)
    completed_tasks: list[dict[str, Any]] = []
    for task in tasks:
        result = train_rolling_task(config, task, dataset_path, all_columns, overwrite_models=overwrite_models)
        if result.get("status") in {"trained", "empty_train"}:
            completed_tasks.append(task)
    print(f"[rolling] predict phase tasks={len(completed_tasks)}", flush=True)
    run_rolling_predictions(config, completed_tasks, dataset_path, all_columns, predict_workers=predict_workers)
    tmp_root = output_root / "_tmp"
    if tmp_root.exists() and not bool(config.get("rolling_keep_intermediate", False)):
        shutil.rmtree(tmp_root)
