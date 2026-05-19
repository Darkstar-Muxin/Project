from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.predict import predict_recommendation
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
    if not p.exists():
        return None
    return pd.read_csv(p)


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
    with st.sidebar:
        st.header("交易要素输入")
        stock_code = st.text_input("股票代码", value="000001.sz")
        side = st.selectbox("交易方向", ["buy", "sell"], format_func=lambda x: "买入" if x == "buy" else "卖出")
        order_qty = st.number_input("交易数量", min_value=1.0, value=100000.0, step=1000.0)
        start_time = st.text_input("交易开始时间", value="2026-04-01 09:30:00")
        rolling_window = st.number_input("Rolling 窗口(交易日)", min_value=1, value=5, step=1)
        run = st.button("生成推荐", type="primary")

    if not run:
        st.info("输入交易要素后点击生成推荐。历史样本日期会展示预测值、真实值和误差。")
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
    c1.metric("股票类别", result["liquidity_group"])
    c2.metric("推荐区间", "无可行区间" if result["recommended_horizon"] is None else f"{result['recommended_horizon']} 分钟")
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
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("没有找到可用于画图的分钟特征数据。")

    left, right = st.columns(2)
    with left:
        st.subheader("候选区间 VWAP")
        vwap_cols = ["predicted_vwap"]
        if candidate_df["actual_vwap"].notna().any():
            vwap_cols.append("actual_vwap")
        fig_vwap = px.line(candidate_df.melt("horizon", value_vars=vwap_cols, var_name="series", value_name="vwap"), x="horizon", y="vwap", color="series", markers=True)
        st.plotly_chart(fig_vwap, width="stretch")
    with right:
        st.subheader("成交占比与 30% 约束")
        part_cols = ["predicted_participation"]
        if candidate_df["actual_participation"].notna().any():
            part_cols.append("actual_participation")
        fig_part = px.bar(candidate_df.melt("horizon", value_vars=part_cols, var_name="series", value_name="participation"), x="horizon", y="participation", color="series", barmode="group")
        fig_part.add_hline(y=0.30, line_dash="dash", line_color="red")
        fig_part.update_layout(yaxis_tickformat=".1%")
        st.plotly_chart(fig_part, width="stretch")

    st.subheader("Volume Ratio 不确定性")
    fig_sigma = px.bar(candidate_df, x="horizon", y="predicted_volume_sigma", text="predicted_volume_sigma")
    fig_sigma.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    st.plotly_chart(fig_sigma, width="stretch")


def _render_report_tab() -> None:
    metrics = _read_csv_if_exists("data/outputs/evaluation_metrics.csv")
    detail = _read_csv_if_exists("data/outputs/prediction_error_detail.csv")
    rec_summary = _read_csv_if_exists("data/outputs/recommendation_backtest_summary.csv")
    rolling_compare = _read_csv_if_exists("data/outputs/rolling/rolling_window_comparison.csv")
    rolling_metrics = _read_csv_if_exists("data/outputs/rolling/rolling_evaluation_metrics.csv")
    rolling_detail = _read_csv_if_exists("data/outputs/rolling/rolling_prediction_error_detail.csv")
    rolling_rec = _read_csv_if_exists("data/outputs/rolling/rolling_recommendation_backtest_summary.csv")
    worst_cases = _read_csv_if_exists("data/outputs/rolling/rolling_recommendation_backtest_worst_cases.csv")

    st.subheader("样本内 / 样本外 Rolling 预测准确性")
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
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("尚未找到 rolling 评估报告。运行：python scripts/06_rolling_backtest.py --config config.yaml")

    with st.expander("静态模型评估（可选 baseline）", expanded=False):
        if metrics is not None and not metrics.empty:
            st.dataframe(metrics, width="stretch")
            fig = px.bar(metrics, x="horizon", y=["vwap_mae", "volume_ratio_mae"], color="split", barmode="group", facet_col="liquidity_group")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("尚未找到静态评估报告。运行：python scripts/04_evaluate.py --config config.yaml")

    if detail is not None and not detail.empty:
        by_date = detail.groupby(["split", "date"], as_index=False).agg(vwap_mae=("abs_vwap_error", "mean"), volume_mae=("abs_volume_error", "mean"))
        st.subheader("误差随日期变化")
        st.plotly_chart(px.line(by_date, x="date", y="vwap_mae", color="split", markers=True), width="stretch")

    st.subheader("Rolling Window 对比")
    if rolling_compare is not None and not rolling_compare.empty:
        st.dataframe(rolling_compare, width="stretch")
        st.plotly_chart(px.bar(rolling_compare, x="window", y="vwap_mae", color="split", barmode="group"), width="stretch")
    else:
        st.info("尚未找到 rolling 报告。运行：python scripts/06_rolling_backtest.py --config config.yaml")

    if rolling_metrics is not None and not rolling_metrics.empty:
        st.subheader("Rolling 按流动性组 / Horizon 分析")
        st.plotly_chart(px.line(rolling_metrics, x="horizon", y="volume_ratio_mae", color="liquidity_group", line_dash="split", facet_col="window", markers=True), width="stretch")
        st.dataframe(rolling_metrics, width="stretch")

    if rolling_detail is not None and not rolling_detail.empty:
        st.subheader("按股票和分钟时段误差")
        stock_rank = rolling_detail.groupby("stock_code", as_index=False).agg(vwap_mae=("abs_vwap_error", "mean"), sample_count=("horizon", "size")).sort_values("vwap_mae")
        minute_rank = rolling_detail.groupby(["minute", "minute_of_day"], as_index=False).agg(vwap_mae=("abs_vwap_error", "mean"), sample_count=("horizon", "size")).sort_values("minute_of_day")
        c1, c2 = st.columns(2)
        c1.dataframe(stock_rank.head(30), width="stretch")
        c2.plotly_chart(px.line(minute_rank, x="minute", y="vwap_mae"), width="stretch")

    st.subheader("推荐回测")
    if rolling_rec is not None and not rolling_rec.empty:
        st.dataframe(rolling_rec, width="stretch")
        st.plotly_chart(px.bar(rolling_rec, x="liquidity_group", y="horizon_match_rate", color="side", facet_col="window", barmode="group"), width="stretch")
    elif rec_summary is not None and not rec_summary.empty:
        st.dataframe(rec_summary, width="stretch")
    else:
        st.info("尚未找到推荐回测报告。")

    if worst_cases is not None and not worst_cases.empty:
        st.subheader("Worst Cases")
        st.dataframe(worst_cases.head(100), width="stretch")


st.set_page_config(page_title="A股最优交易完成时间推荐系统", layout="wide")
st.title("A股分类建模的最优交易完成时间推荐系统")

tab_recommend, tab_report = st.tabs(["单笔推荐", "评估报告"])
with tab_recommend:
    _render_recommendation_tab()
with tab_report:
    _render_report_tab()
