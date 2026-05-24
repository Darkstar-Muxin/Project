# Project Context: A 股最优交易完成时间推荐系统

本文档用于让新的 Codex 对话快速理解当前项目状态。后续凡是修改数据口径、特征、模型、评估、网页或运行方式，都必须同步更新 `README.md` 和本文件。

## 1. 项目目标

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
- 训练、测试/评估、live trading forecast 是分开的；推理预测阶段不反向传播，不更新模型参数。

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

这样做是为了避免一次性 concat 全部分钟数据导致内存爆掉。

## 4. Rolling 口径

当前配置：

```yaml
rolling_train_months: [202602, 202603]
rolling_test_months: [202604]
rolling_windows: [5, 8]
```

主流程：

- 对每个样本内日期和样本外日期都执行 rolling。
- 对每个预测日，只使用该日前最近 N 个交易日训练。
- 样本内 `split=train` 用于调窗口和模型超参数。
- 样本外 `split=test` 用于最终评估。
- 不使用未来日期的数据做训练、分类或标准化。

## 5. 流动性分类

流动性分类不能使用全样本。

当前 rolling 主流程在每个 `test_date + window` 上重新分类：

1. 读取预测日前最近 N 个交易日的 feature parts。
2. 对每只股票计算窗口内平均日成交额、平均日成交量、平均日成交笔数。
3. 主要按平均日成交额分位数划分 `high / medium / low`。
4. 将窗口分类应用于训练窗口和预测日。
5. 保存到：

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

当前 tick 数据没有 bid/ask/order book，因此盘口特征只作为预留能力，不伪造。

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
- 模型输入排除当天全天才知道的字段：`daily_volume, daily_amount, volume_ratio, accumulated_volume_ratio, amount_ratio, accumulated_amount_ratio`。
- 历史日的 ratio/profile 特征可以使用，例如同分钟历史 volume ratio、同分钟历史累计成交进度、同分钟历史 amount ratio。
- 成交量预测从 volume ratio 反推成交量时，使用训练窗口历史日均量先验，不使用预测日真实全天成交量。
- 默认不合并全量大表，rolling 按窗口日期读取所需日文件。

仍需注意：

- 如果旧的 `data/features/model_parts/*.parquet` 是修复前生成的，需要重新运行 `02`。
- 如果将来新增盘口数据，必须保证盘口特征在预测时可见，不能使用未来订单簿状态。

## 9. 训练和预测流程

`scripts/06_rolling_backtest.py` 是稳定串行入口，但内部已拆成两个阶段：

1. 训练阶段：遍历所有 `split + test_date + window`，每个任务先完成 `high / medium / low` 三组模型训练。
2. 预测评估阶段：按 `date + window + liquidity_group` 形成预测任务，可用 CPU 多进程并行，写分片后汇总总报告。

预测阶段使用 `model.eval()` 和 `torch.no_grad()`，不会反向传播，也不会更新模型参数。训练阶段才会调用 `loss.backward()` 和 `optimizer.step()`。

模型默认不覆盖：

```yaml
rolling_overwrite_models: false
```

如果 `ive_model.pt, model_meta.json, feature_columns.joblib, stock_vocab.joblib, group_vocab.joblib, normalizer.joblib` 都已存在，则跳过该日期/window/group 的训练。需要重训时使用：

```bash
python scripts/06_rolling_backtest.py --config config.yaml --overwrite-models
```

CPU 预测并行：

```bash
python scripts/06_rolling_backtest.py --config config.yaml --predict-workers 4
```

只跑指定 rolling window 时使用 `--windows`，例如：

```bash
python scripts/06_rolling_backtest.py --config config.yaml --windows 5
python scripts/06_rolling_backtest_parallel.py --config config.yaml --windows 5 --train-workers 2 --predict-workers 4
```

只跑指定目标月份时使用 `--months`，例如 `--months 202604`。这个参数只过滤预测/回测目标日期，训练窗口仍然从目标日前已有交易日中取最近 N 天，不改变防泄露口径。

```bash
python scripts/06_rolling_backtest_parallel.py --config config.yaml --months 202604 --windows 5 --train-workers 2 --predict-workers 4
```

只训练模型、暂不做预测评估时使用 `--train-only`：

```bash
python scripts/06_rolling_backtest_parallel.py --config config.yaml --months 202604 --windows 5 --train-workers 2 --train-only
```

只做预测评估时使用 `--predict-only`。该模式直接读取已有模型和已有 `stock_liquidity_group.parquet`，不会读取训练窗口、不重算流动性分类，也不会覆盖分类文件。

```bash
python scripts/06_rolling_backtest_parallel.py --config config.yaml --months 202604 --windows 5 --predict-only --predict-workers 4
```

预测评估性能优化状态：

- 预测任务默认按 `date + window` 拆分，每个 worker 对同一天只读取一次 feature part，然后顺序处理 `high / medium / low`。
- 预测可用 `--predict-unit group` 改为按 `date + window + liquidity_group` 拆分，适合早期大日期的 `medium` 组长时间占住 worker 时做负载均衡。
- 预测可用 `--skip-existing-predictions` 跳过已完成分片；完成条件是对应 `window/date/group` 下 `metrics.csv`、`detail.parquet`、`backtest.parquet` 都存在。
- 分月补跑时可加 `--skip-final-reports` 只写 `data/outputs/rolling/parts/...`，避免多个预测批次抢写最终 CSV。
- 所有分片完成后可用 `--aggregate-only` 只扫描已有 parts 并生成最终报告。
- `IVEDataset` 使用数组保存每行所属交易日序列起点，替代逐行 Python dict。
- `_predict_frame` 不再使用 PyTorch DataLoader 逐样本调用 `__getitem__`，而是手写 batch 构造上下文张量，减少 Python/collate 开销。
- 推荐回测使用 numpy 向量化计算推荐 horizon、真实最优 horizon 和 regret。
- `ive_predict_batch_size` 控制预测 batch，`predict_log_every_batches` 控制 forward 进度日志。
- 预测日里没有出现在 rolling 窗口流动性分类表中的股票会被跳过，并打印 `skip unclassified`。不要再把未分类股票默认填成 `low`，否则早期窗口不满时会把大量窗口外股票塞进 low 组，导致预测极慢且分组含义不对。
- 如果预测仍然卡顿，优先降低 `--predict-workers` 到 2 或 4，避免多个进程同时持有过大的日级 DataFrame。

多 GPU 训练入口：

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/06_rolling_backtest_parallel.py --config config.yaml --train-workers 2 --predict-workers 4
```

并行训练按 `date + window` 分发任务到可见 GPU。每个任务内部仍按 `high / medium / low` 顺序训练，避免同一日期内部抢显存。也可以用 `--devices 0,1` 显式指定卡。

预测分片目录：

```text
data/outputs/rolling/parts/window_{N}d/{test_date}/{liquidity_group}/
```

推荐分月补跑命令：

```bash
python scripts/06_rolling_backtest_parallel.py --config config.yaml --months 202602 --windows 5 --predict-only --predict-workers 6 --predict-unit group --skip-existing-predictions --skip-final-reports
python scripts/06_rolling_backtest_parallel.py --config config.yaml --months 202603 --windows 5 --predict-only --predict-workers 6 --predict-unit group --skip-existing-predictions --skip-final-reports
python scripts/06_rolling_backtest_parallel.py --config config.yaml --months 202604 --windows 5 --predict-only --predict-workers 6 --predict-unit group --skip-existing-predictions --skip-final-reports
python scripts/06_rolling_backtest_parallel.py --config config.yaml --windows 5 --aggregate-only
```

最终报告仍写入：

```text
data/outputs/rolling/
```

## 10. 核心文件职责

`config.yaml`

- 全局配置、样本月份、rolling windows、缓存目录、模型超参数、防合并开关、训练覆盖和 worker 数。

`src/preprocess.py`

- 逐日读取 tick parquet。
- 分 batch 聚合到分钟级。
- 保存每日分钟缓存。
- 默认不 combine。

`src/feature_engineering.py`

- 按日生成 feature part。
- 每天读取当前日和最多 10 个历史日来计算历史特征。
- 不注入全局流动性组。
- 不生成 group-level 历史特征，避免依赖全样本分类。

`src/ive_dataset.py`

- 构造 390 分钟上下文序列。
- 做标准化、padding mask、股票 ID、流动性组 ID。
- 排除泄露特征。

`src/ive_model.py`

- IVE 风格 Transformer。
- 输出 volume ratio 分布参数和 VWAP return。

`src/train.py`

- 模型训练、损失函数、模型保存。
- 训练时根据 `ive_device` 使用 CPU 或 CUDA。

`src/rolling_train.py`

- 主 rolling 逻辑。
- 构造 rolling 任务。
- 读取窗口数据。
- 计算窗口流动性分类。
- 训练/预测分离。
- CPU 多进程预测评估。
- 汇总 rolling 评估和推荐回测。

`scripts/06_rolling_backtest_parallel.py`

- 多 GPU date/window 级训练调度。
- 训练结束后调用 CPU 并行预测评估。

`src/predict.py`

- 单笔推理。
- 读取对应日期 feature part。
- 读取对应 rolling window/date 的流动性分类。
- 加载对应流动性组模型。
- 输出推荐区间和候选表。

`app/streamlit_app.py`

- 单笔推荐页面。
- rolling 报告页面。

## 11. 推荐运行顺序

```bash
pip install -r requirements.txt
python scripts/00_check_schema.py --data_dir data/tick_data --config config.yaml --out data/outputs/schema_summary.json
python scripts/01_preprocess.py --config config.yaml
python scripts/02_build_features_labels_parallel.py --config config.yaml --workers 2
python scripts/06_rolling_backtest.py --config config.yaml --predict-workers 4
streamlit run app/streamlit_app.py
```

多 GPU 训练时：

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/06_rolling_backtest_parallel.py --config config.yaml --months 202604 --windows 5 --train-workers 2 --predict-workers 4
```

## 12. 当前限制

- 当前环境如果未安装 PyTorch，模型训练和推理会失败，需要安装 torch。
- rolling 训练会训练大量模型，耗时明显；这是为了避免抽样和未来信息泄露。
- 多 GPU 并行会增加磁盘 IO 和内存压力，建议 `train-workers <= 可用 GPU 数`。
- 网页单笔预测需要对应日期/window 的 rolling 模型已经生成。

## 13. 维护约定

- 修改数据处理、特征、模型、rolling、评估、网页或运行命令时，必须同步更新 `README.md` 和本文件。
- 任何使用全样本统计的新增逻辑都要先检查是否会泄露未来信息。
- 默认优先按日期文件读取，不恢复大范围 combine。
## 2026-05-23 update: rolling-only web prediction

- `app/streamlit_app.py` and `src/predict.py` now use rolling IVE models only.
- Single-order prediction requires `rolling_window` and loads models from `data/models/rolling/window_{N}d/{test_date}/{liquidity_group}/`.
- Static model fallback under `data/models/high`, `data/models/medium`, and `data/models/low` is disabled for web prediction.
- Missing rolling model artifacts now raise an explicit rolling-model error instead of trying to load static-model paths.
- The web app discovers available rolling windows and dates from `data/models/rolling` and defaults to an existing rolling date.
- Rolling report CSVs are read from `data/outputs/rolling/*.csv`; the app temporarily also checks `data/outputs/rolling/rolling/*.csv` for existing older outputs.
- The single-order page also shows same-day rolling metrics plus selectable per-stock VWAP and volume prediction curves from rolling `detail.parquet`.
- Same-day rolling metrics and per-stock VWAP/volume curves now live on the Evaluation Reports tab; curve reads are filtered by stock and horizon to keep horizon switching responsive.
