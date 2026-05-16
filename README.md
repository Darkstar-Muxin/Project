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

作用：检查 `data/` 下 parquet 文件的字段结构和字段映射。

它会：

- 递归扫描 `data/` 下的 parquet 文件。
- 排除 `raw_data/`、`data/processed/`、`data/features/`、`data/models/`、`data/outputs/`。
- 自动识别股票代码、时间、价格、成交量、成交额字段。
- 输出 schema 检查结果到 JSON。

输出：

```text
data/outputs/schema_summary.json
```

运行：

```bash
python scripts/00_check_schema.py --data_dir data --config config.yaml --out data/outputs/schema_summary.json
```

### `scripts/01_preprocess.py`

作用：把原始成交数据处理成分钟级行情数据，并生成股票流动性分组。

它会：

- 读取 `data/202602.parquet`、`data/202603.parquet`、`data/202604.parquet`。
- 不读取 `raw_data/`。
- 自动识别并统一字段名：
  - `stock_code`
  - `datetime`
  - `price`
  - `volume`
  - `amount`
- 按 `stock_code + minute` 聚合为分钟级 OHLC、成交量、成交额、VWAP。
- 按平均日成交额把股票分成 `high`、`medium`、`low` 三类。

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

作用：构造模型训练数据，包括特征和未来标签。

它会：

- 读取 `data/processed/minute_data.parquet`。
- 读取 `data/processed/stock_liquidity_group.parquet`。
- 构造行情特征、时间特征、历史 rolling 特征、同时间段历史特征。
- 对每个候选区间 `[5, 10, 15, 30, 60]` 构造：
  - `future_volume_h`
  - `future_vwap_h`

注意：未来标签窗口采用 `[t, t+h)`，即从当前分钟开始到未来 `h` 分钟内。

输出：

```text
data/features/model_dataset.parquet
```

运行：

```bash
python scripts/02_build_features_labels.py --config config.yaml
```

### `scripts/03_train_models.py`

作用：训练分组、分 horizon 的预测模型。

它会：

- 读取 `data/features/model_dataset.parquet`。
- 只使用 `config.yaml` 中 `train_months` 指定的样本内月份，默认是 `202602`、`202603`。
- 对每个流动性分组 `high / medium / low` 单独训练模型。
- 对每个候选区间单独训练两个模型：
  - VWAP 模型：预测 `future_vwap_h`
  - Volume 模型：预测 `log1p(future_volume_h)`
- 模型使用 `HistGradientBoostingRegressor`。

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

作用：用样本外月份评估已训练模型的预测误差和推荐效果。

它会：

- 读取训练好的模型。
- 只使用 `config.yaml` 中 `test_months` 指定的样本外月份，默认是 `202604`。
- 在特征数据上计算：
  - `vwap_mae`
  - `vwap_rmse`
  - `volume_mae`
  - `volume_rmse`
- 用 `backtest_order_qty` 模拟订单，比较模型推荐区间和历史真实最优可行区间。

输出：

```text
data/outputs/evaluation_metrics.csv
data/outputs/recommendation_backtest_detail.csv
data/outputs/recommendation_backtest_summary.csv
data/outputs/recommendation_backtest_worst_cases.csv
```

其中：

- `recommendation_backtest_detail.csv`：逐笔样本外推荐对比，包含模型推荐区间、预测 VWAP、预测成交量、预测成交占比、推荐区间的真实 VWAP、真实成交量、真实成交占比、真实最优区间和 regret。
- `recommendation_backtest_summary.csv`：按流动性组和买卖方向汇总，包括真实可行率、推荐区间实际可行率、最优区间命中率、平均 regret、最大 regret。
- `recommendation_backtest_worst_cases.csv`：按偏离真实最优的绝对 regret 排序，列出最差的样本，用于定位模型推荐失败案例。

运行：

```bash
python scripts/04_evaluate.py --config config.yaml
```

### `scripts/05_run_demo.py`

作用：启动 Streamlit 网页 Demo。

它等价于运行：

```bash
streamlit run app/streamlit_app.py
```

运行：

```bash
python scripts/05_run_demo.py
```

网页 Demo 只要求已经跑完 `03_train_models.py`，不强制要求跑 `04_evaluate.py`。如果输入的是历史样本中存在未来标签的时间点，页面会自动展示预测结果和真实未来 VWAP、真实成交量、真实成交占比、真实最优区间、是否命中真实最优和 regret。这个单笔真实对比直接来自 `data/features/model_dataset.parquet`，不依赖批量评估输出。

如果提示 `No module named streamlit`，说明当前 Python 环境还没安装 Streamlit，需要先运行：

```bash
pip install -r requirements.txt
```

## 完整执行流程

```bash
pip install -r requirements.txt

python scripts/00_check_schema.py --data_dir data --config config.yaml --out data/outputs/schema_summary.json

python scripts/01_preprocess.py --config config.yaml

python scripts/02_build_features_labels.py --config config.yaml

python scripts/03_train_models.py --config config.yaml

python scripts/04_evaluate.py --config config.yaml

streamlit run app/streamlit_app.py
```

## 训练和预测阶段区别

- 训练阶段：可以使用历史真实未来数据构造 `future_vwap_h` 和 `future_volume_h` 标签。
- 预测阶段：不能使用未来真实数据，只能使用模型预测的 VWAP 和成交量判断 30% 约束。
