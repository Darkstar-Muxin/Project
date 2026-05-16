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


def _render_recommendation_tab() -> None:
    with st.sidebar:
        st.header("订单输入")
        stock_code = st.text_input("股票代码", value="000001.sz")
        side = st.selectbox("交易方向", ["buy", "sell"])
        order_qty = st.number_input("交易数量", min_value=1.0, value=100000.0, step=1000.0)
        start_time = st.text_input("交易开始时间", value="2026-04-01 09:30:00")
        run = st.button("生成推荐", type="primary")

    if not run:
        dataset_path = resolve_path("data/features/model_dataset.parquet")
        if dataset_path.exists():
            st.info("填写左侧订单信息后点击生成推荐。历史样本会自动展示预测与真实结果对比。")
        else:
            st.warning("尚未找到特征数据，请先完成 01 预处理、02 特征构造和 03 模型训练。")
        return

    try:
        result = predict_recommendation(stock_code, side, order_qty, start_time)
        candidate_df = pd.DataFrame(result["candidate_table"])

        st.caption(f"匹配特征时间：{result['matched_feature_time']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("股票类别", result["liquidity_group"])
        c2.metric("模型推荐区间", "无可行区间" if result["recommended_horizon"] is None else f"{result['recommended_horizon']} 分钟")
        c3.metric("预测 VWAP", _fmt_number(result["predicted_vwap"], 4))
        c4.metric("预测成交占比", _fmt_pct(result["predicted_participation"]))

        if result["has_actual_comparison"]:
            st.subheader("历史真实结果对比")
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("真实最优区间", "-" if result["actual_best_horizon"] is None else f"{result['actual_best_horizon']} 分钟")
            a2.metric("推荐区间真实可行", _fmt_bool(result["recommended_actual_feasible"]))
            a3.metric("是否命中真实最优", _fmt_bool(result["optimal_hit"]))
            a4.metric("Regret", _fmt_number(result["regret"], 6))

            b1, b2, b3 = st.columns(3)
            b1.metric("推荐区间真实 VWAP", _fmt_number(result["recommended_actual_vwap"], 4))
            b2.metric("推荐区间真实成交量", _fmt_number(result["recommended_actual_market_volume"], 0))
            b3.metric("推荐区间真实成交占比", _fmt_pct(result["recommended_actual_participation"]))
        else:
            st.info("当前样本没有可用于对比的未来真实数据，仅展示模型预测结果。")

        display_columns = [
            "horizon",
            "predicted_vwap",
            "predicted_market_volume",
            "predicted_participation",
            "feasible",
            "actual_vwap",
            "actual_market_volume",
            "actual_participation",
            "actual_feasible",
            "vwap_error",
            "volume_error",
        ]
        st.subheader("候选区间预测与真实对比")
        st.dataframe(candidate_df[display_columns], width="stretch")

        st.subheader("VWAP 相对变化")
        vwap_plot = candidate_df[["horizon", "predicted_vwap", "actual_vwap", "vwap_error"]].copy()
        base = vwap_plot["actual_vwap"].dropna().iloc[0] if vwap_plot["actual_vwap"].notna().any() else vwap_plot["predicted_vwap"].iloc[0]
        vwap_plot["predicted_vwap_bp"] = (vwap_plot["predicted_vwap"] / base - 1) * 10000
        vwap_plot["actual_vwap_bp"] = (vwap_plot["actual_vwap"] / base - 1) * 10000
        long_vwap = vwap_plot.melt(
            id_vars="horizon",
            value_vars=["predicted_vwap_bp", "actual_vwap_bp"],
            var_name="series",
            value_name="bp_vs_base",
        )
        fig = px.line(long_vwap, x="horizon", y="bp_vs_base", color="series", markers=True)
        fig.update_layout(yaxis_title="相对首个真实/预测 VWAP 的变化(bp)", xaxis_title="完成时间(分钟)")
        st.plotly_chart(fig)

        st.subheader("VWAP 预测误差")
        err_df = candidate_df[["horizon", "vwap_error"]].dropna()
        if not err_df.empty:
            fig_err = px.bar(err_df, x="horizon", y="vwap_error", text="vwap_error")
            fig_err.update_traces(texttemplate="%{text:.5f}", textposition="outside")
            fig_err.update_layout(yaxis_title="预测 VWAP - 真实 VWAP", xaxis_title="完成时间(分钟)")
            st.plotly_chart(fig_err)
        else:
            st.info("没有真实 VWAP，暂不能绘制误差图。")

        st.subheader("成交占比：预测 vs 真实")
        part_cols = ["predicted_participation"]
        if "actual_participation" in candidate_df.columns and candidate_df["actual_participation"].notna().any():
            part_cols.append("actual_participation")
        part_plot = candidate_df[["horizon", *part_cols]].melt("horizon", var_name="series", value_name="participation")
        fig_part = px.bar(part_plot, x="horizon", y="participation", color="series", barmode="group")
        fig_part.update_layout(yaxis_tickformat=".1%", yaxis_title="成交占比", xaxis_title="完成时间(分钟)")
        st.plotly_chart(fig_part)
    except Exception as exc:
        st.error(str(exc))


def _render_report_tab() -> None:
    metrics = _read_csv_if_exists("data/outputs/evaluation_metrics.csv")
    by_date = _read_csv_if_exists("data/outputs/prediction_error_by_date.csv")
    by_stock = _read_csv_if_exists("data/outputs/prediction_error_by_stock.csv")
    by_minute = _read_csv_if_exists("data/outputs/prediction_error_by_minute.csv")
    rec_summary = _read_csv_if_exists("data/outputs/recommendation_backtest_summary.csv")
    worst_cases = _read_csv_if_exists("data/outputs/recommendation_backtest_worst_cases.csv")
    rolling_compare = _read_csv_if_exists("data/outputs/rolling/rolling_window_comparison.csv")
    rolling_metrics = _read_csv_if_exists("data/outputs/rolling/rolling_evaluation_metrics.csv")

    if metrics is None or by_date is None or by_stock is None or by_minute is None:
        st.warning("尚未找到完整评估报告文件。请先运行：python scripts/04_evaluate.py --config config.yaml")
        return

    st.subheader("样本内 vs 样本外整体准确性")
    overall = metrics.groupby("split", as_index=False).agg(
        vwap_mae=("vwap_mae", "mean"),
        baseline_vwap_mae=("baseline_vwap_mae", "mean"),
        vwap_mae_improvement=("vwap_mae_improvement", "mean"),
        vwap_rmse=("vwap_rmse", "mean"),
        volume_mae=("volume_mae", "mean"),
        baseline_volume_mae=("baseline_volume_mae", "mean"),
        volume_mae_improvement=("volume_mae_improvement", "mean"),
        volume_rmse=("volume_rmse", "mean"),
    )
    c1, c2, c3, c4 = st.columns(4)
    train_row = overall[overall["split"] == "train"]
    test_row = overall[overall["split"] == "test"]
    c1.metric("样本内 VWAP MAE", "-" if train_row.empty else _fmt_number(train_row.iloc[0]["vwap_mae"], 6))
    c2.metric("样本外 VWAP MAE", "-" if test_row.empty else _fmt_number(test_row.iloc[0]["vwap_mae"], 6))
    c3.metric("样本外 VWAP 相对基准改善", "-" if test_row.empty else _fmt_number(test_row.iloc[0]["vwap_mae_improvement"], 6))
    c4.metric("样本外 Volume 相对基准改善", "-" if test_row.empty else _fmt_number(test_row.iloc[0]["volume_mae_improvement"], 0))

    st.write(
        "这份报告衡量两个核心预测目标：未来区间 VWAP 和未来区间市场成交量。"
        "样本内对应 2、3 月，样本外对应 4 月。样本外误差更能代表真实预测能力。"
        "相对基准改善 = baseline MAE - model MAE，数值为正说明模型超过简单基准。"
    )

    metric_compare = metrics.melt(
        id_vars=["split", "liquidity_group", "horizon"],
        value_vars=["vwap_mae", "baseline_vwap_mae"],
        var_name="method",
        value_name="mae",
    )
    fig_metrics = px.bar(metric_compare, x="horizon", y="mae", color="method", facet_row="split", facet_col="liquidity_group", barmode="group")
    fig_metrics.update_layout(yaxis_title="VWAP MAE", xaxis_title="完成时间(分钟)")
    st.plotly_chart(fig_metrics)

    st.subheader("预测准确性随日期变化")
    date_plot = by_date.groupby(["split", "date"], as_index=False).agg(
        vwap_mae=("vwap_mae", "mean"),
        baseline_vwap_mae=("baseline_vwap_mae", "mean"),
        vwap_mae_improvement=("vwap_mae_improvement", "mean"),
        volume_mae=("volume_mae", "mean"),
    )
    fig_date = px.line(date_plot, x="date", y="vwap_mae", color="split", markers=True)
    fig_date.update_layout(yaxis_title="VWAP MAE", xaxis_title="日期")
    st.plotly_chart(fig_date)

    st.subheader("什么股票预测最准 / 最差")
    stock_rank = by_stock.groupby(["split", "stock_code"], as_index=False).agg(
        vwap_mae=("vwap_mae", "mean"),
        baseline_vwap_mae=("baseline_vwap_mae", "mean"),
        vwap_mae_improvement=("vwap_mae_improvement", "mean"),
        volume_mae=("volume_mae", "mean"),
        sample_count=("sample_count", "sum"),
    )
    min_count = st.slider("股票排名最小样本数", min_value=10, max_value=1000, value=100, step=10)
    stock_rank = stock_rank[stock_rank["sample_count"] >= min_count]
    for split in ["train", "test"]:
        split_rank = stock_rank[stock_rank["split"] == split].sort_values("vwap_mae")
        left, right = st.columns(2)
        left.markdown(f"**{split} VWAP 最准股票**")
        left.dataframe(split_rank.head(10), width="stretch")
        right.markdown(f"**{split} 相对基准改善最差股票**")
        right.dataframe(split_rank.sort_values("vwap_mae_improvement").head(10), width="stretch")

    st.subheader("什么时间预测最准 / 最差")
    minute_rank = by_minute.groupby(["split", "minute", "minute_of_day"], as_index=False).agg(
        vwap_mae=("vwap_mae", "mean"),
        baseline_vwap_mae=("baseline_vwap_mae", "mean"),
        vwap_mae_improvement=("vwap_mae_improvement", "mean"),
        volume_mae=("volume_mae", "mean"),
        sample_count=("sample_count", "sum"),
    )
    fig_minute = px.line(minute_rank.sort_values("minute_of_day"), x="minute", y="vwap_mae", color="split")
    fig_minute.update_layout(yaxis_title="VWAP MAE", xaxis_title="分钟")
    st.plotly_chart(fig_minute)

    st.subheader("交易推荐回测")
    if rec_summary is not None:
        st.dataframe(rec_summary, width="stretch")
        if "horizon_match_rate" in rec_summary.columns:
            fig_rec = px.bar(rec_summary, x="liquidity_group", y="horizon_match_rate", color="split", facet_col="side", barmode="group")
            fig_rec.update_layout(yaxis_title="真实最优区间命中率", xaxis_title="流动性分组")
            st.plotly_chart(fig_rec)
    if worst_cases is not None:
        st.markdown("**偏离真实最优最大的样本**")
        st.dataframe(worst_cases.head(50), width="stretch")

    st.subheader("Rolling 全量窗口对比")
    if rolling_compare is None:
        st.info("尚未找到 rolling 报告。运行 python scripts/06_rolling_backtest.py --config config.yaml 后，这里会显示 5日/8日窗口对比。")
        return
    st.dataframe(rolling_compare, width="stretch")
    fig_roll = px.bar(
        rolling_compare,
        x="window",
        y=["vwap_mae", "baseline_vwap_mae"],
        barmode="group",
        title="Rolling VWAP MAE vs Baseline",
    )
    st.plotly_chart(fig_roll)
    if rolling_metrics is not None:
        fig_roll_h = px.line(
            rolling_metrics,
            x="horizon",
            y="vwap_mae",
            color="liquidity_group",
            facet_col="window",
            markers=True,
            title="Rolling 不同 horizon 的 VWAP MAE",
        )
        st.plotly_chart(fig_roll_h)


st.set_page_config(page_title="A股最优交易完成时间推荐", layout="wide")
st.title("A股分类建模的最优交易完成时间推荐系统")

tab_recommend, tab_report = st.tabs(["单笔推荐", "评估报告"])
with tab_recommend:
    _render_recommendation_tab()
with tab_report:
    _render_report_tab()
