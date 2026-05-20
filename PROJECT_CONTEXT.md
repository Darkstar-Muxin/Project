# Project Context: A股最优交易完成时间推荐系统

本文档用于让新的 Codex 对话快速理解当前项目状态。后续凡是修改数据口径、特征、模型、评估、网页或运行方式，都必须同步更新 `README.md` 和本文件。

## 1. 项目目标

课题名称：A股分类建模的最优交易完成时间推荐系统。

系统输入：

- 股票代码
- 交易方向：buy / sell
- 交易数量
- 交易开始时间，精确到分钟
- rolling window，单位为交易日

系统输出：

- 推荐完成时间长度：5/10/15/30/60 分钟
- 推荐区间预测 VWAP
- 推荐区间预测市场成交量
- 推荐区间预测成交占比
- 是否满足成交占比不超过 30%
- 股票在当前 rolling 窗口下所属流动性类别
- 候选区间预测表和历史真实对比

业务规则：

- 买入选择满足 30% 约束下预测 VWAP 最低的区间。
- 卖出选择满足 30% 约束下预测 VWAP 最高的区间。
- 不满足 30% 约束的候选区间不可选。

## 2. 参考文献和模型

主要参考文献：`paper/01.pdf`，IVE: Enhanced Probabilistic Forecasting of Intraday Volume Ratio with Transformers。

论文关键点：

- 数据频率是一分钟级，不是 tick 级。
- 预测 intraday volume ratio。
- context length 为 390 个一分钟步，代表一整天分钟上下文。
- 使用 Transformer 架构、时间编码、股票特征和概率式输出。

本项目工程化适配：

- 原始数据是 tick 成交数据，先流式聚合为分钟数据。
- 使用 IVE 风格 Transformer encoder。
- 输出 volume ratio 的均值/不确定性，以及 VWAP return。
- 通过 volume ratio 和历史日均量先验反推市场成交量。
- 保留课题要求的 VWAP 最优区间推荐逻辑。

## 3. 数据和缓存

原始 tick 数据：

```text
data/tick_data/YYYYMMDD.parquet
```

列：

```text
StockCode, Date, Time, Price, Volume, Turnover, BSFlag
```

分钟缓存：

```text
data/processed/minute_parts/YYYYMMDD.parquet
```

特征缓存：

```text
data/features/model_parts/YYYYMMDD.parquet
```

默认不生成全量大表：

- `data/processed/minute_data.parquet`
- `data/features/model_dataset.parquet`

相关配置：

```yaml
build_combined_minute_data: false
build_combined_feature_data: false
feature_overwrite: true
```

这样做是为了避免一次性 concat 全部分钟数据导致内存爆掉。

## 4. 样本期和 Rolling 口径

当前配置：

```yaml
train_months: [202602, 202603]
test_months: [202604]
rolling_train_months: [202602, 202603]
rolling_test_months: [202604]
rolling_windows: [5, 8]
```

主流程是 `scripts/06_rolling_backtest.py`：

- 对每个样本内日期和样本外日期都执行 rolling。
- 对每个预测日，只使用该日前最近 N 个交易日训练。
- 训练后只预测该日。
- `split=train` 用来调窗口和模型超参数。
- `split=test` 用来做最终样本外评估。

## 5. 流动性分类

重要：流动性分类不能使用全样本。

当前 rolling 主流程在每个 `test_date + window` 上重新分类：

1. 读取预测日前最近 N 个交易日的 feature parts。
2. 对每只股票计算窗口内：
   - 平均日成交额
   - 平均日成交量
   - 平均日成交笔数
3. 按平均日成交额分位数划分：
   - 高流动性 high
   - 中流动性 medium
   - 低流动性 low
4. 将这个窗口分类应用于训练窗口和预测日。
5. 保存：

```text
data/models/rolling/window_{N}d/{test_date}/stock_liquidity_group.parquet
```

全局 `data/processed/stock_liquidity_group.parquet` 只作为 baseline 参考，主流程不依赖它。

## 6. 特征体系

分钟聚合字段：

- `open, high, low, close, volume, amount, vwap`
- `trade_count`
- `buy_volume, sell_volume, buy_amount, sell_amount`

行情和统计特征：

- `price`
- `buy_sell_volume_imbalance`
- `buy_sell_amount_imbalance`
- `return_1m, return_5m, return_10m`
- `volume_5m_sum, volume_10m_sum, volume_20m_sum`
- `amount_5m_sum, amount_10m_sum, amount_20m_sum`
- `vwap_5m, vwap_10m, vwap_20m`
- `volatility_5m, volatility_10m, volatility_20m`
- `vwap_deviation_5m, vwap_deviation_10m, vwap_deviation_20m`
- `stock_rolling_volume_mean_5d/10d`
- `stock_rolling_amount_mean_5d/10d`
- `stock_rolling_vwap_mean_5d/10d`
- `same_minute_volume_mean_5d/10d`
- `same_minute_amount_mean_5d/10d`
- `same_minute_vwap_mean_5d/10d`
- `same_minute_volume_ratio_mean_5d/10d`
- `same_minute_accumulated_volume_ratio_mean_5d/10d`
- `same_minute_amount_ratio_mean_5d/10d`
- `same_minute_accumulated_amount_ratio_mean_5d/10d`

时间特征：

- `minute_of_day`
- `minutes_from_open`
- `minutes_to_close`
- `is_morning_session`
- `abs_time_sin`
- `abs_time_cos`

代码特征：

- `is_sh`
- `is_sz`

交易要素特征：

- 方向、数量、数量/预测成交量、数量/历史成交量主要在推理和推荐阶段使用。

当前无盘口字段：

- tick 数据没有 bid/ask/order book。
- 不伪造盘口特征。

## 7. 标签

标签由 `src/label_builder.py` 生成：

- `future_vwap_{h}`
- `future_volume_{h}`
- `future_volume_ratio_{h}`
- `log_future_volume_ratio_{h}`

规则：

- 只在同股票、同交易日内计算。
- 不跨日、不跨股票。
- 标签用于训练和历史评估，不进入模型输入特征。

## 8. 防泄露规则

已经修正的泄露点：

- 不再使用全样本流动性分类做 rolling 训练。
- 特征日文件不再生成全局流动性组历史特征。
- rolling 训练会丢弃旧特征文件里可能存在的 `liquidity_group` 和 `group_same_minute_*` 列。
- 模型输入排除当天全天才知道的字段：
  - `daily_volume`
  - `daily_amount`
  - `volume_ratio`
  - `accumulated_volume_ratio`
  - `amount_ratio`
  - `accumulated_amount_ratio`
- 历史日的 ratio/profile 特征可以使用，例如同分钟历史 volume ratio、同分钟历史累计成交进度、同分钟历史 amount ratio。
- 成交量预测从 volume ratio 反推成交量时，使用训练窗口历史日均量先验，不使用预测日真实全天成交量。
- 默认不合并全量大表，rolling 按窗口日期读取所需日文件。

仍需注意：

- 如果旧的 `data/features/model_parts/*.parquet` 是修复前生成的，需要重新运行 `scripts/02_build_features_labels.py`，因为配置中 `feature_overwrite: true` 会覆盖旧文件。
- 如果将来新增盘口数据，必须保证盘口特征在预测时可见，不能使用未来订单簿状态。

## 9. 核心文件职责

`config.yaml`

- 全局配置、样本月份、rolling windows、缓存目录、模型超参数、防合并开关。

`src/preprocess.py`

- 逐日读取 tick parquet。
- 分 batch 聚合到分钟级。
- 保存每日分钟缓存。
- 默认不 combine。

`src/stock_classification.py`

- baseline 股票分类。
- `classify_stocks_from_minute_parts` 只用 `train_months`，避免 baseline 文件包含 4 月信息。
- rolling 主流程另行做窗口分类。

`src/feature_engineering.py`

- 按日生成 feature part。
- 每天读取当前日和最多 10 个历史日来计算历史特征。
- 不注入全局流动性组。
- 不生成 group-level 历史特征，避免依赖全样本分类。

`src/label_builder.py`

- 构造未来 VWAP、成交量、volume ratio 标签。

`src/ive_dataset.py`

- 构造 390 分钟上下文序列。
- 做标准化、padding mask、股票 ID、流动性组 ID。
- 排除泄露特征。

`src/ive_model.py`

- IVE 风格 Transformer。
- 输出 volume ratio 分布参数和 VWAP return。

`src/train.py`

- 静态 baseline 训练，不是主流程。

`src/rolling_train.py`

- 主流程。
- 读取窗口日期数据。
- 计算窗口流动性分类。
- 按流动性组训练模型。
- 预测当天。
- 输出 rolling 评估和推荐回测。

`src/predict.py`

- 单笔推理。
- 读取对应日期 feature part。
- 读取对应 rolling window/date 的流动性分类。
- 加载对应流动性组模型。
- 输出推荐区间和候选表。

`src/evaluate.py`

- 静态 baseline 评估。

`app/streamlit_app.py`

- 单笔推荐页面。
- rolling 报告页面。

## 10. 推荐运行顺序

```bash
pip install -r requirements.txt
python scripts/00_check_schema.py --data_dir data/tick_data --config config.yaml --out data/outputs/schema_summary.json
python scripts/01_preprocess.py --config config.yaml
python scripts/02_build_features_labels.py --config config.yaml
python scripts/02_build_features_labels_parallel.py --config config.yaml --workers 2
python scripts/06_rolling_backtest.py --config config.yaml
streamlit run app/streamlit_app.py
```

`02_build_features_labels_parallel.py` 是可选加速版，不替代原 `02_build_features_labels.py`。它使用 Python 标准库 `multiprocessing.Pool`，按交易日并行生成 `data/features/model_parts/YYYYMMDD.parquet`，默认跳过已存在文件；如需重算加 `--overwrite`。每个 worker 会读取当前日和最多 10 个历史日，内存紧张时不要把 `--workers` 开太大。

并行版默认 `--maxtasks-per-child 1`，每个 worker 完成一个交易日后退出重建，避免 pandas/numpy 在长生命周期进程里持续持有峰值内存。大数据机器上建议 `--workers 2` 起步，不建议直接开 6。

并行版支持按月份分批运行：

```bash
python scripts/02_build_features_labels_parallel.py --config config.yaml --workers 2 --months 202602
```

`--months` 只限制输出目标日期，历史滞后特征仍会使用目标日期之前的 minute parts。

静态 baseline 可选：

```bash
python scripts/03_train_models.py --config config.yaml
python scripts/04_evaluate.py --config config.yaml
```

## 11. 当前限制

- 当前环境如果未安装 PyTorch，模型训练和推理会失败，需要 `pip install torch`。
- rolling 训练会训练大量模型，耗时明显；这是为了避免抽样和未来信息泄露。
- 网页单笔预测需要对应日期/window 的 rolling 模型已经生成。

## 12. 维护约定

- 修改数据处理、特征、模型、rolling、评估、网页或运行命令时，必须同步更新 `README.md` 和本文件。
- 任何使用全样本统计的新增逻辑都要先检查是否会泄露未来信息。
- 默认优先按日期文件读取，不恢复大范围 combine。
