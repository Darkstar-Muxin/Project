from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.predict import MODEL_ARTIFACTS, predict_recommendation
from src.utils import resolve_path


def _fmt_number(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.{digits}f}"


def _fmt_pct(value) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.2%}"


def _fmt_bool(value) -> str:
    if value is True:
        return "是"
    if value is False:
        return "否"
    return "-"


@st.cache_data(show_spinner=False)
def _read_csv_if_exists(path: str) -> pd.DataFrame | None:
    p = resolve_path(path)
    if not p.exists() and path.startswith("data/outputs/rolling/"):
        fallback = resolve_path("data/outputs/rolling/rolling") / p.name
        if fallback.exists():
            p = fallback
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return None


@st.cache_data(show_spinner=False)
def _available_rolling_models() -> dict[int, list[str]]:
    root = resolve_path("data/models/rolling")
    if not root.exists():
        return {}
    available: dict[int, list[str]] = {}
    for window_dir in sorted(root.glob("window_*d")):
        try:
            window = int(window_dir.name.removeprefix("window_").removesuffix("d"))
        except ValueError:
            continue
        dates: list[str] = []
        for date_dir in sorted(path for path in window_dir.iterdir() if path.is_dir()):
            has_complete_group = False
            for group in ["high", "medium", "low"]:
                group_dir = date_dir / group
                if group_dir.exists() and all((group_dir / name).exists() for name in MODEL_ARTIFACTS):
                    has_complete_group = True
                    break
            if has_complete_group and (date_dir / "stock_liquidity_group.parquet").exists():
                dates.append(date_dir.name)
        if dates:
            available[window] = dates
    return available


def _rolling_parts_roots() -> list[Path]:
    roots = [
        resolve_path("data/outputs/rolling/parts"),
        resolve_path("data/outputs/rolling/rolling/parts"),
    ]
    return [root for root in roots if root.exists()]


def _day_part_dirs(rolling_window: int, selected_date: str) -> list[Path]:
    dirs: list[Path] = []
    for root in _rolling_parts_roots():
        day_root = root / f"window_{int(rolling_window)}d" / selected_date
        if day_root.exists():
            dirs.extend(path for path in sorted(day_root.iterdir()) if path.is_dir())
    return dirs


@st.cache_data(show_spinner=False)
def _read_day_metrics(rolling_window: int, selected_date: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for part_dir in _day_part_dirs(rolling_window, selected_date):
        path = part_dir / "metrics.csv"
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _read_day_backtest_summary(rolling_window: int, selected_date: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    columns = ["liquidity_group", "side", "horizon_match", "absolute_regret", "has_pred_feasible", "has_true_feasible"]
    for part_dir in _day_part_dirs(rolling_window, selected_date):
        path = part_dir / "backtest.parquet"
        if not path.exists() or path.stat().st_size == 0:
            continue
        df = pd.read_parquet(path, columns=columns)
        if df.empty:
            continue
        frames.append(
            df.groupby(["liquidity_group", "side"], as_index=False).agg(
                sample_count=("horizon_match", "size"),
                horizon_match_rate=("horizon_match", "mean"),
                mean_absolute_regret=("absolute_regret", "mean"),
                pred_feasible_rate=("has_pred_feasible", "mean"),
                true_feasible_rate=("has_true_feasible", "mean"),
            )
        )
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined.groupby(["liquidity_group", "side"], as_index=False).agg(
        sample_count=("sample_count", "sum"),
        horizon_match_rate=("horizon_match_rate", "mean"),
        mean_absolute_regret=("mean_absolute_regret", "mean"),
        pred_feasible_rate=("pred_feasible_rate", "mean"),
        true_feasible_rate=("true_feasible_rate", "mean"),
    )


@st.cache_data(show_spinner=False)
def _read_day_stock_codes(rolling_window: int, selected_date: str) -> list[str]:
    path = (
        resolve_path("data/models/rolling")
        / f"window_{int(rolling_window)}d"
        / selected_date
        / "stock_liquidity_group.parquet"
    )
    if not path.exists() or path.stat().st_size == 0:
        return []
    df = pd.read_parquet(path, columns=["stock_code"])
    return sorted(df["stock_code"].astype(str).unique().tolist())


@st.cache_data(show_spinner=False)
def _read_day_stock_detail(rolling_window: int, selected_date: str, stock_code: str) -> pd.DataFrame:
    columns = [
        "stock_code",
        "datetime",
        "minute",
        "liquidity_group",
        "horizon",
        "actual_vwap",
        "predicted_vwap",
        "actual_volume",
        "predicted_volume",
        "actual_volume_ratio",
        "predicted_volume_ratio",
    ]
    frames: list[pd.DataFrame] = []
    for part_dir in _day_part_dirs(rolling_window, selected_date):
        path = part_dir / "detail.parquet"
        if not path.exists() or path.stat().st_size == 0:
            continue
        df = pd.read_parquet(path, columns=columns, filters=[("stock_code", "==", str(stock_code))])
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=columns)
    out = pd.concat(frames, ignore_index=True)
    out["datetime"] = pd.to_datetime(out["datetime"])
    return out.sort_values(["horizon", "datetime"])


@st.cache_data(show_spinner=False)
def _read_day_stock_curve(rolling_window: int, selected_date: str, stock_code: str, horizon: int) -> pd.DataFrame:
    columns = [
        "stock_code",
        "datetime",
        "minute",
        "liquidity_group",
        "horizon",
        "actual_vwap",
        "predicted_vwap",
        "actual_volume",
        "predicted_volume",
        "actual_volume_ratio",
        "predicted_volume_ratio",
    ]
    frames: list[pd.DataFrame] = []
    for part_dir in _day_part_dirs(rolling_window, selected_date):
        path = part_dir / "detail.parquet"
        if not path.exists() or path.stat().st_size == 0:
            continue
        df = pd.read_parquet(
            path,
            columns=columns,
            filters=[("stock_code", "==", str(stock_code)), ("horizon", "==", int(horizon))],
        )
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=columns)
    out = pd.concat(frames, ignore_index=True)
    out["datetime"] = pd.to_datetime(out["datetime"])
    return out.sort_values("datetime")


def _render_day_prediction_report() -> None:
    available_models = _available_rolling_models()
    st.subheader("当天整体预测分析")
    if not available_models:
        st.info("没有在 data/models/rolling 下找到 rolling 模型。")
        return

    windows = sorted(available_models)
    default_window_index = windows.index(5) if 5 in windows else len(windows) - 1
    c1, c2 = st.columns(2)
    with c1:
        rolling_window = st.selectbox("报告 Rolling 窗口", windows, index=default_window_index, format_func=lambda x: f"{x}d")
    date_options = available_models[int(rolling_window)]
    with c2:
        selected_date = st.selectbox("报告日期", date_options, index=len(date_options) - 1)

    day_metrics = _read_day_metrics(int(rolling_window), selected_date)
    day_backtest = _read_day_backtest_summary(int(rolling_window), selected_date)
    if day_metrics.empty:
        st.info("没有找到该日期的 rolling 预测分片 metrics.csv。")
    else:
        total_samples = int(day_metrics.groupby(["liquidity_group", "horizon"])["sample_count"].max().sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("预测日期", selected_date)
        m2.metric("Rolling 窗口", f"{int(rolling_window)}d")
        m3.metric("流动性组数", day_metrics["liquidity_group"].nunique())
        m4.metric("样本行数", f"{total_samples:,}")
        day_summary = day_metrics.groupby(["liquidity_group", "horizon"], as_index=False).agg(
            sample_count=("sample_count", "max"),
            vwap_mae=("vwap_mae", "mean"),
            volume_ratio_mae=("volume_ratio_mae", "mean"),
            feasibility_accuracy=("feasibility_accuracy", "mean"),
        )
        st.dataframe(day_summary, width="stretch")
        st.plotly_chart(
            px.line(day_summary, x="horizon", y="vwap_mae", color="liquidity_group", markers=True),
            use_container_width=True,
        )
    if not day_backtest.empty:
        st.subheader("当天推荐回测概览")
        st.dataframe(day_backtest, width="stretch")
        st.plotly_chart(
            px.bar(day_backtest, x="liquidity_group", y="horizon_match_rate", color="side", barmode="group"),
            use_container_width=True,
        )

    stock_options = _read_day_stock_codes(int(rolling_window), selected_date)
    if not stock_options:
        st.info("没有找到该日期的股票列表。")
        return
    curve_left, curve_right = st.columns(2)
    with curve_left:
        selected_stock = st.selectbox("查看某只股票的全天预测曲线", stock_options, index=0)
    horizons = sorted(day_metrics["horizon"].dropna().astype(int).unique().tolist()) if not day_metrics.empty else [5, 10, 15, 30, 60]
    with curve_right:
        selected_horizon = st.selectbox("曲线 Horizon", horizons, index=0, format_func=lambda x: f"{x} 分钟")
    curve_df = _read_day_stock_curve(int(rolling_window), selected_date, selected_stock, int(selected_horizon))
    if curve_df.empty:
        st.info("没有找到该股票和 horizon 的 detail.parquet 预测明细。")
        return

    curve_title = f"{selected_stock} - {selected_date} - {selected_horizon}分钟"
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader(f"{curve_title} VWAP")
        vwap_curve = curve_df[["datetime", "actual_vwap", "predicted_vwap"]].melt(
            "datetime", var_name="series", value_name="vwap"
        )
        st.plotly_chart(px.line(vwap_curve, x="datetime", y="vwap", color="series"), use_container_width=True)
    with c_right:
        st.subheader(f"{curve_title} Volume")
        volume_curve = curve_df[["datetime", "actual_volume", "predicted_volume"]].melt(
            "datetime", var_name="series", value_name="volume"
        )
        st.plotly_chart(px.line(volume_curve, x="datetime", y="volume", color="series"), use_container_width=True)


@st.cache_data(show_spinner=False)
def _read_feature_slice(stock_code: str, matched_time: str) -> pd.DataFrame:
    p = resolve_path("data/features/model_dataset.parquet")
    ts = pd.to_datetime(matched_time)
    if not p.exists():
        p = resolve_path("data/features/model_parts") / f"{ts.strftime('%Y%m%d')}.parquet"
        if not p.exists():
            return pd.DataFrame()
    df = pd.read_parquet(p)
    df["datetime"] = pd.to_datetime(df["datetime"])
    mask = (df["stock_code"].astype(str) == str(stock_code)) & (df["datetime"].dt.date == ts.date())
    day = df.loc[mask].sort_values("datetime")
    if day.empty:
        return day
    start = ts - pd.Timedelta(minutes=30)
    end = ts + pd.Timedelta(minutes=90)
    return day[(day["datetime"] >= start) & (day["datetime"] <= end)].copy()


def _render_recommendation_tab() -> None:
    available_models = _available_rolling_models()
    with st.sidebar:
        st.header("交易要素输入")
        stock_code = st.text_input("股票代码", value="000001.sz")
        side = st.selectbox("交易方向", ["buy", "sell"], format_func=lambda x: "买入" if x == "buy" else "卖出")
        order_qty = st.number_input("交易数量", min_value=1.0, value=100000.0, step=1000.0)
        if available_models:
            windows = sorted(available_models)
            default_window_index = windows.index(5) if 5 in windows else len(windows) - 1
            rolling_window = st.selectbox("Rolling 窗口", windows, index=default_window_index, format_func=lambda x: f"{x}d")
            date_options = available_models[int(rolling_window)]
            selected_date = st.selectbox("可用模型日期", date_options, index=len(date_options) - 1)
            start_clock = st.text_input("开始时刻", value="09:30:00")
            start_time = f"{selected_date} {start_clock}"
        else:
            st.warning("没有在 data/models/rolling 下找到 rolling 模型，请先运行 rolling 训练。")
            rolling_window = st.number_input("Rolling 窗口", min_value=1, value=5, step=1)
            start_time = st.text_input("开始时间", value="2026-02-03 09:30:00")
        run = st.button("生成推荐", type="primary", disabled=not bool(available_models))

    if not run:
        st.info("选择已有 rolling 日期和窗口，输入交易要素后点击生成推荐。")
        return

    try:
        result = predict_recommendation(
            stock_code,
            side,
            order_qty,
            start_time,
            rolling_window=int(rolling_window),
        )
    except Exception as exc:
        st.error(str(exc))
        return

    candidate_df = pd.DataFrame(result["candidate_table"])
    st.caption(f"匹配特征时间：{result['matched_feature_time']} | 模型路径：{result['model_path']}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("流动性组", result["liquidity_group"])
    c2.metric("推荐区间", "-" if result["recommended_horizon"] is None else f"{result['recommended_horizon']} 分钟")
    c3.metric("预测 VWAP", _fmt_number(result["predicted_vwap"], 4))
    c4.metric("预测成交量", _fmt_number(result["predicted_market_volume"], 0))
    c5.metric("预测成交占比", _fmt_pct(result["predicted_participation"]))

    if result["has_actual_comparison"]:
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("真实最优区间", "-" if result["actual_best_horizon"] is None else f"{result['actual_best_horizon']} 分钟")
        a2.metric("推荐区间真实可行", _fmt_bool(result["recommended_actual_feasible"]))
        a3.metric("命中真实最优", _fmt_bool(result["optimal_hit"]))
        a4.metric("Regret", _fmt_number(result["regret"], 6))

    st.subheader("候选区间预测与真实对比")
    display_columns = [
        "horizon",
        "predicted_vwap",
        "predicted_market_volume",
        "predicted_volume_ratio",
        "predicted_volume_sigma",
        "predicted_participation",
        "feasible",
        "actual_vwap",
        "actual_market_volume",
        "actual_volume_ratio",
        "actual_participation",
        "actual_feasible",
        "vwap_error",
        "volume_error",
    ]
    st.dataframe(candidate_df[[c for c in display_columns if c in candidate_df.columns]], width="stretch")

    st.subheader("价格 / VWAP 分钟曲线")
    price_df = _read_feature_slice(result["stock_code"], result["matched_feature_time"])
    if not price_df.empty:
        plot_df = price_df[["datetime", "close", "vwap"]].melt("datetime", var_name="series", value_name="price")
        fig = px.line(plot_df, x="datetime", y="price", color="series", markers=False)
        fig.add_vline(x=pd.to_datetime(result["matched_feature_time"]), line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("没有找到可用于画图的分钟特征数据。")

    left, right = st.columns(2)
    with left:
        st.subheader("候选区间 VWAP")
        vwap_cols = ["predicted_vwap"]
        if candidate_df["actual_vwap"].notna().any():
            vwap_cols.append("actual_vwap")
        fig_vwap = px.line(
            candidate_df.melt("horizon", value_vars=vwap_cols, var_name="series", value_name="vwap"),
            x="horizon",
            y="vwap",
            color="series",
            markers=True,
        )
        st.plotly_chart(fig_vwap, use_container_width=True)
    with right:
        st.subheader("成交占比与 30% 约束")
        part_cols = ["predicted_participation"]
        if candidate_df["actual_participation"].notna().any():
            part_cols.append("actual_participation")
        fig_part = px.bar(
            candidate_df.melt("horizon", value_vars=part_cols, var_name="series", value_name="participation"),
            x="horizon",
            y="participation",
            color="series",
            barmode="group",
        )
        fig_part.add_hline(y=0.30, line_dash="dash", line_color="red")
        fig_part.update_layout(yaxis_tickformat=".1%")
        st.plotly_chart(fig_part, use_container_width=True)

    st.subheader("Volume Ratio 不确定性")
    fig_sigma = px.bar(candidate_df, x="horizon", y="predicted_volume_sigma", text="predicted_volume_sigma")
    fig_sigma.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    st.plotly_chart(fig_sigma, use_container_width=True)


def _render_report_tab() -> None:
    metrics = _read_csv_if_exists("data/outputs/evaluation_metrics.csv")
    detail = _read_csv_if_exists("data/outputs/prediction_error_detail.csv")
    rec_summary = _read_csv_if_exists("data/outputs/recommendation_backtest_summary.csv")
    rolling_compare = _read_csv_if_exists("data/outputs/rolling/rolling_window_comparison.csv")
    rolling_metrics = _read_csv_if_exists("data/outputs/rolling/rolling_evaluation_metrics.csv")
    rolling_detail = _read_csv_if_exists("data/outputs/rolling/rolling_prediction_error_detail.csv")
    rolling_rec = _read_csv_if_exists("data/outputs/rolling/rolling_recommendation_backtest_summary.csv")
    worst_cases = _read_csv_if_exists("data/outputs/rolling/rolling_recommendation_backtest_worst_cases.csv")

    _render_day_prediction_report()

    st.subheader("Rolling 预测准确性")
    if rolling_metrics is not None and not rolling_metrics.empty:
        st.dataframe(rolling_metrics, width="stretch")
        fig = px.line(
            rolling_metrics,
            x="horizon",
            y="vwap_mae",
            color="liquidity_group",
            line_dash="split",
            facet_col="window",
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("尚未找到 rolling 评估报告。运行：python scripts/06_rolling_backtest.py --config config.yaml")

    with st.expander("静态 baseline 报告（可选）", expanded=False):
        if metrics is not None and not metrics.empty:
            st.dataframe(metrics, width="stretch")
            metric_cols = [col for col in ["vwap_mae", "volume_ratio_mae"] if col in metrics.columns]
            if metric_cols:
                plot_df = metrics.melt(
                    id_vars=[col for col in ["horizon", "split", "liquidity_group"] if col in metrics.columns],
                    value_vars=metric_cols,
                    var_name="metric",
                    value_name="value",
                )
                st.plotly_chart(
                    px.bar(plot_df, x="horizon", y="value", color="metric", barmode="group", facet_col="liquidity_group"),
                    use_container_width=True,
                )
        else:
            st.info("尚未找到静态 baseline 报告。")

    if detail is not None and not detail.empty:
        by_date = detail.groupby(["split", "date"], as_index=False).agg(vwap_mae=("abs_vwap_error", "mean"), volume_mae=("abs_volume_error", "mean"))
        st.subheader("静态 baseline 误差随日期变化")
        st.plotly_chart(px.line(by_date, x="date", y="vwap_mae", color="split", markers=True), use_container_width=True)

    st.subheader("Rolling Window 对比")
    if rolling_compare is not None and not rolling_compare.empty:
        st.dataframe(rolling_compare, width="stretch")
        st.plotly_chart(px.bar(rolling_compare, x="window", y="vwap_mae", color="split", barmode="group"), use_container_width=True)
    else:
        st.info("尚未找到 rolling window 对比报告。")

    if rolling_metrics is not None and not rolling_metrics.empty:
        st.subheader("Rolling 按流动性组 / Horizon 分析")
        st.plotly_chart(px.line(rolling_metrics, x="horizon", y="volume_ratio_mae", color="liquidity_group", line_dash="split", facet_col="window", markers=True), use_container_width=True)
        st.dataframe(rolling_metrics, width="stretch")

    if rolling_detail is not None and not rolling_detail.empty:
        st.subheader("Rolling 按股票和分钟时段误差")
        stock_rank = rolling_detail.groupby("stock_code", as_index=False).agg(vwap_mae=("abs_vwap_error", "mean"), sample_count=("horizon", "size")).sort_values("vwap_mae")
        minute_rank = rolling_detail.groupby(["minute", "minute_of_day"], as_index=False).agg(vwap_mae=("abs_vwap_error", "mean"), sample_count=("horizon", "size")).sort_values("minute_of_day")
        c1, c2 = st.columns(2)
        c1.dataframe(stock_rank.head(30), width="stretch")
        c2.plotly_chart(px.line(minute_rank, x="minute", y="vwap_mae"), use_container_width=True)

    st.subheader("推荐回测")
    if rolling_rec is not None and not rolling_rec.empty:
        st.dataframe(rolling_rec, width="stretch")
        st.plotly_chart(px.bar(rolling_rec, x="liquidity_group", y="horizon_match_rate", color="side", facet_col="window", barmode="group"), use_container_width=True)
    elif rec_summary is not None and not rec_summary.empty:
        st.dataframe(rec_summary, width="stretch")
    else:
        st.info("尚未找到推荐回测报告。")

    if worst_cases is not None and not worst_cases.empty:
        st.subheader("Worst Cases")
        st.dataframe(worst_cases.head(100), width="stretch")


def main() -> None:
    st.set_page_config(page_title="A股最优交易完成时间推荐系统", layout="wide")
    st.title("A股最优交易完成时间推荐系统")

    tab_recommend, tab_report = st.tabs(["单笔推荐", "评估报告"])
    with tab_recommend:
        _render_recommendation_tab()
    with tab_report:
        _render_report_tab()


if __name__ == "__main__":
    main()
