# A股分类建模的最优交易完成时间推荐系统

本项目基于 A 股历史成交数据，按股票流动性分类建模，推荐订单在候选完成时间 `[5, 10, 15, 30, 60]` 分钟中的最优完成区间。

输入：

- 股票代码
- 交易方向：`buy` 或 `sell`
- 交易数量：`order_qty`
- 交易开始时间：`start_time`

输出：

- 推荐完成时间
- 预测未来区间 VWAP
- 预测未来市场成交量
- 预测成交占比
- 所有候选区间预测表

## 数据目录说明

项目只读取：

```text
data/202602.parquet
data/202603.parquet
data/202604.parquet
```

`raw_data/` 是原始备份目录，本项目不读取、不修改、不删除。

运行过程中会生成：

```text
data/processed/
data/features/
data/models/
data/outputs/
```

## 30% 成交占比约束

对每个候选完成时间 `h`，模型会预测未来 `h` 分钟市场成交量：

```text
participation = order_qty / predicted_market_volume
```

只有 `participation <= 0.30` 的候选区间才算可行。

- `buy`：在可行区间中选择预测 VWAP 最低的区间。
- `sell`：在可行区间中选择预测 VWAP 最高的区间。
- 如果没有可行区间，则返回“无可行区间”，并展示所有候选预测结果。

## 脚本说明

### `scripts/00_check_schema.py`

检查 `data/` 下 parquet 文件的字段结构和字段映射，输出：

```text
data/outputs/schema_summary.json
```

运行：

```bash
python scripts/00_check_schema.py --data_dir data --config config.yaml --out data/outputs/schema_summary.json
```

### `scripts/01_preprocess.py`

把原始成交数据处理成分钟级行情数据，并生成股票流动性分组。

输出：

```text
data/processed/minute_data.parquet
data/processed/stock_liquidity_group.parquet
```

运行：

```bash
python scripts/01_preprocess.py --config config.yaml
```

### `scripts/02_build_features_labels.py`

构造模型训练数据，包括行情特征、历史 rolling 特征、同分钟历史特征，以及未来 VWAP / 成交量标签。

输出：

```text
data/features/model_dataset.parquet
```

运行：

```bash
python scripts/02_build_features_labels.py --config config.yaml
```

### `scripts/03_train_models.py`

训练静态模型。默认使用 `202602`、`202603` 作为样本内训练月份，`202604` 留作样本外测试。

当前 VWAP 模型预测的是：

```text
future_vwap_h / current_vwap - 1
```

成交量模型预测的是：

```text
log1p(future_volume_h)
```

输出：

```text
data/models/{liquidity_group}/vwap_h{h}.joblib
data/models/{liquidity_group}/volume_h{h}.joblib
data/models/{liquidity_group}/feature_columns.joblib
```

运行：

```bash
python scripts/03_train_models.py --config config.yaml
```

### `scripts/04_evaluate.py`

评估静态模型，输出样本内 / 样本外预测准确性、baseline 对比、推荐回测结果和 worst cases。

输出：

```text
data/outputs/evaluation_metrics.csv
data/outputs/prediction_error_detail.csv
data/outputs/prediction_error_by_date.csv
data/outputs/prediction_error_by_stock.csv
data/outputs/prediction_error_by_minute.csv
data/outputs/recommendation_backtest_summary.csv
data/outputs/recommendation_backtest_worst_cases.csv
```

运行：

```bash
python scripts/04_evaluate.py --config config.yaml
```

### `scripts/05_run_demo.py`

启动 Streamlit 网页 Demo。

运行：

```bash
python scripts/05_run_demo.py
```

等价于：

```bash
streamlit run app/streamlit_app.py
```

网页 Demo 只要求已经跑完 `03_train_models.py`，不强制要求跑 `04_evaluate.py`。如果已经跑过 `04` 或 `06`，网页“评估报告”页会自动读取相应报告。

### `scripts/06_rolling_backtest.py`

运行 rolling 全量训练实验。

该流程不抽样训练。对每个 4 月预测日，分别使用该日前最近 5 个交易日、最近 8 个交易日的全量历史样本训练模型，然后预测该日。

输出：

```text
data/models/rolling/
data/outputs/rolling/
```

主要结果：

```text
data/outputs/rolling/rolling_evaluation_metrics.csv
data/outputs/rolling/rolling_window_comparison.csv
data/outputs/rolling/rolling_recommendation_backtest_summary.csv
data/outputs/rolling/rolling_recommendation_backtest_worst_cases.csv
```

运行：

```bash
python scripts/06_rolling_backtest.py --config config.yaml
```

## 完整执行流程

### 1. 基础静态模型流程

第一次完整运行：

```bash
pip install -r requirements.txt

python scripts/00_check_schema.py --data_dir data --config config.yaml --out data/outputs/schema_summary.json

python scripts/01_preprocess.py --config config.yaml

python scripts/02_build_features_labels.py --config config.yaml

python scripts/03_train_models.py --config config.yaml

python scripts/04_evaluate.py --config config.yaml

streamlit run app/streamlit_app.py
```

如果只是重新打开网页：

```bash
streamlit run app/streamlit_app.py
```

### 2. Rolling 全量训练流程

在基础流程完成 `02_build_features_labels.py` 之后，可以运行 rolling 实验：

```bash
python scripts/06_rolling_backtest.py --config config.yaml
```

如果想在网页评估报告中看到 rolling 对比，推荐运行：

```bash
python scripts/04_evaluate.py --config config.yaml
python scripts/06_rolling_backtest.py --config config.yaml
streamlit run app/streamlit_app.py
```

注意：`06_rolling_backtest.py` 会训练很多模型，耗时明显长于 `03_train_models.py`。

## 训练和预测阶段区别

- 训练阶段：可以使用历史真实未来数据构造 `future_vwap_h` 和 `future_volume_h` 标签。
- 预测阶段：不能使用未来真实数据，只能使用模型预测的 VWAP 和成交量判断 30% 约束。
- 网页里的真实对比只用于历史样本复盘，不参与真实预测决策。
