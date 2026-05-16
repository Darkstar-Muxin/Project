from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import ensure_dirs
from src.train import _schema_feature_columns, add_runtime_features, filter_by_months, vwap_base
from src.utils import resolve_path


def _rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _recommend_from_rows(rows: list[dict[str, float | int | bool]], side: str) -> dict[str, float | int | bool] | None:
    feasible = [row for row in rows if bool(row["feasible"])]
    if not feasible:
        return None
    if side == "buy":
        return min(feasible, key=lambda row: float(row["vwap"]))
    return max(feasible, key=lambda row: float(row["vwap"]))


def _candidate_by_horizon(rows: list[dict[str, float | int | bool]], horizon: int) -> dict[str, float | int | bool] | None:
    for row in rows:
        if int(row["horizon"]) == int(horizon):
            return row
    return None


def _build_backtest_rows(
    group_df: pd.DataFrame,
    x_all: pd.DataFrame,
    group_path,
    horizons: list[int],
    order_qty: float,
    participation_limit: float,
) -> list[dict[str, object]]:
    predictions: dict[int, dict[str, np.ndarray]] = {}
    for h in horizons:
        vwap_path = group_path / f"vwap_h{h}.joblib"
        volume_path = group_path / f"volume_h{h}.joblib"
        if not vwap_path.exists() or not volume_path.exists():
            continue
        pred_vwap = joblib.load(vwap_path).predict(x_all)
        pred_volume = np.maximum(np.expm1(joblib.load(volume_path).predict(x_all)), 0)
        predictions[h] = {"vwap": pred_vwap, "volume": pred_volume}

    rows: list[dict[str, object]] = []
    if not predictions:
        return rows

    for i, (_, sample) in enumerate(group_df.iterrows()):
        pred_candidates = []
        true_candidates = []
        for h in predictions:
            true_volume = sample.get(f"future_volume_{h}")
            true_vwap = sample.get(f"future_vwap_{h}")
            pred_volume = float(predictions[h]["volume"][i])
            pred_vwap = float(predictions[h]["vwap"][i])
            pred_participation = order_qty / pred_volume if pred_volume > 0 else np.inf
            true_participation = order_qty / true_volume if pd.notna(true_volume) and true_volume > 0 else np.inf
            pred_candidates.append(
                {
                    "horizon": h,
                    "vwap": pred_vwap,
                    "volume": pred_volume,
                    "participation": pred_participation,
                    "feasible": pred_participation <= participation_limit,
                }
            )
            if pd.notna(true_vwap):
                true_candidates.append(
                    {
                        "horizon": h,
                        "vwap": float(true_vwap),
                        "volume": float(true_volume),
                        "participation": true_participation,
                        "feasible": true_participation <= participation_limit,
                    }
                )

        for side in ["buy", "sell"]:
            pred_pick = _recommend_from_rows(pred_candidates, side)
            true_pick = _recommend_from_rows(true_candidates, side)
            base = {
                "stock_code": sample.get("stock_code"),
                "datetime": sample.get("datetime"),
                "side": side,
                "has_pred_feasible": pred_pick is not None,
                "has_true_feasible": true_pick is not None,
                "recommended_horizon": None if pred_pick is None else pred_pick["horizon"],
                "true_best_horizon": None if true_pick is None else true_pick["horizon"],
                "true_best_vwap": None if true_pick is None else true_pick["vwap"],
                "true_best_volume": None if true_pick is None else true_pick["volume"],
                "true_best_participation": None if true_pick is None else true_pick["participation"],
            }
            if pred_pick is None or true_pick is None:
                if pred_pick is None:
                    rows.append(
                        {
                            **base,
                            "predicted_vwap": np.nan,
                            "predicted_volume": np.nan,
                            "predicted_participation": np.nan,
                            "recommended_actual_vwap": np.nan,
                            "recommended_actual_volume": np.nan,
                            "recommended_actual_participation": np.nan,
                            "recommended_actual_feasible": False,
                            "feasibility_correct": true_pick is None,
                            "regret": np.nan,
                            "absolute_regret": np.nan,
                            "horizon_match": False,
                            "comparison_status": "no_pred_feasible" if true_pick is not None else "both_no_feasible",
                        }
                    )
                    continue

                chosen_h = int(pred_pick["horizon"])
                chosen_true = _candidate_by_horizon(true_candidates, chosen_h)
                rows.append(
                    {
                        **base,
                        "predicted_vwap": pred_pick["vwap"],
                        "predicted_volume": pred_pick["volume"],
                        "predicted_participation": pred_pick["participation"],
                        "recommended_actual_vwap": None if chosen_true is None else chosen_true["vwap"],
                        "recommended_actual_volume": None if chosen_true is None else chosen_true["volume"],
                        "recommended_actual_participation": None if chosen_true is None else chosen_true["participation"],
                        "recommended_actual_feasible": False if chosen_true is None else chosen_true["feasible"],
                        "feasibility_correct": False if chosen_true is None else bool(chosen_true["feasible"]),
                        "regret": np.nan,
                        "absolute_regret": np.nan,
                        "horizon_match": False,
                        "comparison_status": "pred_feasible_but_true_no_feasible",
                    }
                )
                continue

            chosen_h = int(pred_pick["horizon"])
            chosen_true = _candidate_by_horizon(true_candidates, chosen_h)
            if chosen_true is None:
                continue
            if side == "buy":
                regret = float(chosen_true["vwap"]) - float(true_pick["vwap"])
            else:
                regret = float(true_pick["vwap"]) - float(chosen_true["vwap"])
            actual_feasible = bool(chosen_true["feasible"])
            if not actual_feasible:
                comparison_status = "pred_feasible_but_actual_infeasible"
            elif chosen_h == int(true_pick["horizon"]):
                comparison_status = "optimal"
            else:
                comparison_status = "suboptimal"
            rows.append(
                {
                    **base,
                    "predicted_vwap": pred_pick["vwap"],
                    "predicted_volume": pred_pick["volume"],
                    "predicted_participation": pred_pick["participation"],
                    "recommended_actual_vwap": chosen_true["vwap"],
                    "recommended_actual_volume": chosen_true["volume"],
                    "recommended_actual_participation": chosen_true["participation"],
                    "recommended_actual_feasible": actual_feasible,
                    "feasibility_correct": actual_feasible,
                    "regret": regret,
                    "absolute_regret": abs(regret),
                    "horizon_match": chosen_h == int(true_pick["horizon"]),
                    "comparison_status": comparison_status,
                }
            )
    return rows


def _baseline_vwap(work_df: pd.DataFrame) -> pd.Series:
    for col in ["same_minute_vwap_mean_5d", "stock_rolling_vwap_mean_5d", "vwap", "close"]:
        if col in work_df.columns:
            baseline = pd.to_numeric(work_df[col], errors="coerce")
            if baseline.notna().any():
                return baseline
    return pd.Series(np.nan, index=work_df.index)


def _baseline_volume(work_df: pd.DataFrame, horizon: int) -> pd.Series:
    if "same_minute_volume_mean_5d" in work_df.columns:
        baseline = pd.to_numeric(work_df["same_minute_volume_mean_5d"], errors="coerce") * horizon
    elif "volume_10m_sum" in work_df.columns:
        baseline = pd.to_numeric(work_df["volume_10m_sum"], errors="coerce") / 10 * horizon
    elif "volume_5m_sum" in work_df.columns:
        baseline = pd.to_numeric(work_df["volume_5m_sum"], errors="coerce") / 5 * horizon
    else:
        baseline = pd.to_numeric(work_df.get("volume", pd.Series(np.nan, index=work_df.index)), errors="coerce") * horizon
    return baseline.clip(lower=0)


def _load_split_sample(
    dataset_path,
    config: dict[str, Any],
    feature_columns: list[str],
    split: str,
    months_key: str,
) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(dataset_path)
    horizons = [int(h) for h in config["horizons"]]
    label_columns = [f"future_vwap_{h}" for h in horizons] + [f"future_volume_{h}" for h in horizons]
    id_columns = ["stock_code", "datetime", "liquidity_group"]
    columns = [*id_columns, *feature_columns, *label_columns]
    columns = [col for col in columns if col in parquet_file.schema_arrow.names]
    batch_size = int(config.get("evaluation_batch_size", config.get("train_batch_size", 100_000)))
    max_rows = int(config.get("evaluation_sample_per_group", 50_000))
    random_state = int(config.get("random_state", 42))
    samples: dict[str, pd.DataFrame] = {}
    expected_groups = {"high", "medium", "low"}

    print(
        f"[eval] streaming {split} from {dataset_path} with batch_size={batch_size:,}, "
        f"sample_per_group={max_rows:,}, months={config.get(months_key)}",
        flush=True,
    )
    for batch_no, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size, columns=columns), start=1):
        batch_df = batch.to_pandas()
        batch_df = filter_by_months(batch_df, config.get(months_key))
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
        print(f"[eval] batch {batch_no}, sampled rows={sizes}", flush=True)
        if expected_groups.issubset(samples) and all(len(samples[group]) >= max_rows for group in expected_groups):
            print("[eval] sample target reached for high/medium/low; stop scanning parquet", flush=True)
            break

    if not samples:
        raise ValueError(f"No evaluation rows after applying {months_key} filter")
    df = pd.concat(samples.values(), ignore_index=True)
    df["split"] = split
    return df


def _evaluate_split(
    df: pd.DataFrame,
    config: dict[str, Any],
    feature_columns: list[str],
) -> tuple[list[dict[str, object]], list[pd.DataFrame], list[dict[str, object]]]:
    model_dir = resolve_path(config["model_dir"])
    split = str(df["split"].iloc[0])
    rows: list[dict[str, object]] = []
    detail_parts: list[pd.DataFrame] = []
    backtest_rows: list[dict[str, object]] = []
    order_qty = float(config.get("backtest_order_qty", 100000))
    participation_limit = float(config.get("participation_limit", 0.30))

    for group_name, group_df in df.groupby("liquidity_group"):
        group_path = model_dir / str(group_name)
        if not group_path.exists():
            continue
        work_df = add_runtime_features(group_df.reset_index(drop=True))
        feature_columns = joblib.load(group_path / "feature_columns.joblib")
        x_all = work_df[feature_columns]

        for h in config["horizons"]:
            metrics = {"split": split, "liquidity_group": group_name, "horizon": h}
            vwap_path = group_path / f"vwap_h{h}.joblib"
            volume_path = group_path / f"volume_h{h}.joblib"
            vwap_pred = None
            volume_pred = None

            if vwap_path.exists():
                target = f"future_vwap_{h}"
                mask = work_df[target].notna()
                if mask.any():
                    base = vwap_base(work_df).loc[mask]
                    vwap_pred = base.to_numpy(dtype=float) * (1 + joblib.load(vwap_path).predict(x_all.loc[mask]))
                    metrics["vwap_mae"] = float(mean_absolute_error(work_df.loc[mask, target], vwap_pred))
                    metrics["vwap_rmse"] = _rmse(work_df.loc[mask, target], vwap_pred)
                    baseline = _baseline_vwap(work_df).loc[mask]
                    baseline_mask = baseline.notna()
                    if baseline_mask.any():
                        true_values = work_df.loc[mask, target].loc[baseline_mask]
                        baseline_values = baseline.loc[baseline_mask]
                        metrics["baseline_vwap_mae"] = float(mean_absolute_error(true_values, baseline_values))
                        metrics["vwap_mae_improvement"] = metrics["baseline_vwap_mae"] - metrics["vwap_mae"]

            if volume_path.exists():
                target = f"future_volume_{h}"
                mask = work_df[target].notna() & (work_df[target] >= 0)
                if mask.any():
                    volume_pred = np.expm1(joblib.load(volume_path).predict(x_all.loc[mask]))
                    metrics["volume_mae"] = float(mean_absolute_error(work_df.loc[mask, target], volume_pred))
                    metrics["volume_rmse"] = _rmse(work_df.loc[mask, target], volume_pred)
                    baseline = _baseline_volume(work_df, int(h)).loc[mask]
                    baseline_mask = baseline.notna()
                    if baseline_mask.any():
                        true_values = work_df.loc[mask, target].loc[baseline_mask]
                        baseline_values = baseline.loc[baseline_mask]
                        metrics["baseline_volume_mae"] = float(mean_absolute_error(true_values, baseline_values))
                        metrics["volume_mae_improvement"] = metrics["baseline_volume_mae"] - metrics["volume_mae"]

            rows.append(metrics)

            detail_mask = work_df[f"future_vwap_{h}"].notna() & work_df[f"future_volume_{h}"].notna()
            if detail_mask.any() and vwap_path.exists() and volume_path.exists():
                detail_x = x_all.loc[detail_mask]
                base = vwap_base(work_df).loc[detail_mask]
                pred_vwap = base.to_numpy(dtype=float) * (1 + joblib.load(vwap_path).predict(detail_x))
                pred_volume = np.maximum(np.expm1(joblib.load(volume_path).predict(detail_x)), 0)
                actual_vwap = work_df.loc[detail_mask, f"future_vwap_{h}"].to_numpy(dtype=float)
                actual_volume = work_df.loc[detail_mask, f"future_volume_{h}"].to_numpy(dtype=float)
                baseline_vwap = _baseline_vwap(work_df).loc[detail_mask].to_numpy(dtype=float)
                baseline_volume = _baseline_volume(work_df, int(h)).loc[detail_mask].to_numpy(dtype=float)
                detail = pd.DataFrame(
                    {
                        "split": split,
                        "liquidity_group": group_name,
                        "stock_code": work_df.loc[detail_mask, "stock_code"].to_numpy(),
                        "datetime": pd.to_datetime(work_df.loc[detail_mask, "datetime"]).to_numpy(),
                        "horizon": int(h),
                        "actual_vwap": actual_vwap,
                        "predicted_vwap": pred_vwap,
                        "baseline_vwap": baseline_vwap,
                        "vwap_error": pred_vwap - actual_vwap,
                        "abs_vwap_error": np.abs(pred_vwap - actual_vwap),
                        "baseline_abs_vwap_error": np.abs(baseline_vwap - actual_vwap),
                        "actual_volume": actual_volume,
                        "predicted_volume": pred_volume,
                        "baseline_volume": baseline_volume,
                        "volume_error": pred_volume - actual_volume,
                        "abs_volume_error": np.abs(pred_volume - actual_volume),
                        "baseline_abs_volume_error": np.abs(baseline_volume - actual_volume),
                    }
                )
                detail["date"] = pd.to_datetime(detail["datetime"]).dt.date.astype(str)
                detail["minute"] = pd.to_datetime(detail["datetime"]).dt.strftime("%H:%M")
                detail["minute_of_day"] = pd.to_datetime(detail["datetime"]).dt.hour * 60 + pd.to_datetime(detail["datetime"]).dt.minute
                detail_parts.append(detail)

        backtest_part = _build_backtest_rows(
            work_df,
            x_all,
            group_path,
            [int(h) for h in config["horizons"]],
            order_qty,
            participation_limit,
        )
        for item in backtest_part:
            item["split"] = split
            item["liquidity_group"] = group_name
            item["order_qty"] = order_qty
        backtest_rows.extend(backtest_part)
    return rows, detail_parts, backtest_rows


def _write_error_reports(detail_df: pd.DataFrame, output_dir) -> None:
    detail_df.to_csv(output_dir / "prediction_error_detail.csv", index=False, encoding="utf-8-sig")
    group_cols = {
        "prediction_error_by_date.csv": ["split", "date", "horizon"],
        "prediction_error_by_stock.csv": ["split", "stock_code", "horizon"],
        "prediction_error_by_minute.csv": ["split", "minute", "minute_of_day", "horizon"],
    }
    for filename, cols in group_cols.items():
        report = detail_df.groupby(cols, as_index=False).agg(
            sample_count=("horizon", "size"),
            vwap_mae=("abs_vwap_error", "mean"),
            baseline_vwap_mae=("baseline_abs_vwap_error", "mean"),
            vwap_bias=("vwap_error", "mean"),
            volume_mae=("abs_volume_error", "mean"),
            baseline_volume_mae=("baseline_abs_volume_error", "mean"),
            volume_bias=("volume_error", "mean"),
        )
        report["vwap_mae_improvement"] = report["baseline_vwap_mae"] - report["vwap_mae"]
        report["volume_mae_improvement"] = report["baseline_volume_mae"] - report["volume_mae"]
        report.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")


def evaluate_models(config: dict[str, Any]) -> pd.DataFrame:
    ensure_dirs(config)
    dataset_path = resolve_path(config["feature_data_dir"]) / "model_dataset.parquet"
    output_path = resolve_path(config["output_dir"]) / "evaluation_metrics.csv"
    parquet_file = pq.ParquetFile(dataset_path)
    feature_columns = _schema_feature_columns(parquet_file.schema_arrow)
    output_dir = resolve_path(config["output_dir"])
    split_specs = [
        ("train", "train_months"),
        ("test", "test_months"),
    ]
    rows: list[dict[str, object]] = []
    detail_parts: list[pd.DataFrame] = []
    backtest_rows: list[dict[str, object]] = []

    for split, months_key in split_specs:
        split_df = _load_split_sample(dataset_path, config, feature_columns, split, months_key)
        split_rows, split_details, split_backtest = _evaluate_split(split_df, config, feature_columns)
        rows.extend(split_rows)
        detail_parts.extend(split_details)
        backtest_rows.extend(split_backtest)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    if detail_parts:
        detail_df = pd.concat(detail_parts, ignore_index=True)
        _write_error_reports(detail_df, output_dir)
    if backtest_rows:
        backtest_df = pd.DataFrame(backtest_rows)
        summary = backtest_df.groupby(["split", "liquidity_group", "side"], as_index=False).agg(
            sample_count=("side", "size"),
            pred_feasible_rate=("has_pred_feasible", "mean"),
            true_feasible_rate=("has_true_feasible", "mean"),
            actual_feasible_rate=("recommended_actual_feasible", "mean"),
            feasibility_correct_rate=("feasibility_correct", "mean"),
            horizon_match_rate=("horizon_match", "mean"),
            avg_regret=("regret", "mean"),
            median_regret=("regret", "median"),
            max_regret=("regret", "max"),
            max_absolute_regret=("absolute_regret", "max"),
        )
        worst_n = int(config.get("worst_case_top_n", 100))
        worst_cases = backtest_df.sort_values("absolute_regret", ascending=False, na_position="last").head(worst_n)
        backtest_df.to_csv(output_dir / "recommendation_backtest_detail.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(output_dir / "recommendation_backtest_summary.csv", index=False, encoding="utf-8-sig")
        worst_cases.to_csv(output_dir / "recommendation_backtest_worst_cases.csv", index=False, encoding="utf-8-sig")
    return metrics_df
