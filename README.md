# A 股最优交易完成时间推荐系统

本项目用于在给定股票代码、买卖方向、交易数量和开始时间后，推荐 5/10/15/30/60 分钟中的最优交易完成区间。推荐规则是：买入选择满足 30% 成交占比约束下预测 VWAP 最低的区间；卖出选择满足约束下预测 VWAP 最高的区间。

模型参考 `paper/01.pdf` 的 IVE 思路：一分钟级 intraday volume ratio 预测、390 个一分钟步上下文、股票 embedding、时间/位置编码、Transformer encoder 和概率式 volume ratio 输出。本项目原始数据是 tick 成交数据，因此先流式聚合为分钟数据，再训练 rolling Transformer 模型。

## 数据与缓存

原始数据统一读取：

```text
data/tick_data/YYYYMMDD.parquet
```

每个 tick parquet 应包含：

```text
StockCode, Date, Time, Price, Volume, Turnover, BSFlag
```

预处理默认不合并大表，只生成每日分钟缓存：

```text
data/processed/minute_parts/YYYYMMDD.parquet
```

特征也默认按日保存：

```text
data/features/model_parts/YYYYMMDD.parquet
```

项目不通过抽样减少数据量。性能通过逐日处理、parquet batch、按日期读取、懒加载、batch 训练和多进程任务拆分解决。

## Rolling 口径

当前样本期配置在 `config.yaml`：

```yaml
rolling_train_months: [202602, 202603]
rolling_test_months: [202604]
rolling_windows: [5, 8]
```

样本内和样本外都使用同一 rolling 逻辑：对每个预测日，只使用该日前最近 N 个交易日训练，并只预测该日。`split=train` 的结果用于调窗口和模型超参数；`split=test` 用于最终样本外报告。

流动性分类也在每个 `test_date + window` 上重新计算，只使用预测日前窗口内的数据。分类依据是股票在窗口内的平均日成交额、日成交量和成交笔数，主要按平均日成交额分位数分为 `high / medium / low`。分类结果保存到：

```text
data/models/rolling/window_{N}d/{test_date}/stock_liquidity_group.parquet
```

## 特征与标签

分钟聚合字段包括 `open, high, low, close, volume, amount, vwap, trade_count, buy_volume, sell_volume, buy_amount, sell_amount`。

模型候选特征包括行情特征、历史统计特征、时间特征、股票代码特征和 IVE 风格特征。历史特征默认最多使用前 10 个历史交易日，包括股票日均量/额/VWAP、同股票同分钟历史均值、历史同分钟 volume ratio 和累计成交进度等。

当前 tick 数据没有 bid/ask/order book 字段，因此盘口特征仅作为预留能力，不伪造。

每个 horizon 构造：

```text
future_vwap_{h}
future_volume_{h}
future_volume_ratio_{h}
log_future_volume_ratio_{h}
```

标签严格在同股票、同交易日内构造，不跨日、不跨股票。标签只用于训练和历史评估，不进入预测输入。

## 防未来信息泄露

主流程遵守以下规则：

- 不使用全样本流动性分类，rolling 分类只用预测日前窗口数据。
- 不默认生成或读取全量 `minute_data.parquet` / `model_dataset.parquet`。
- 模型输入排除当天全天才知道的字段：`daily_volume, daily_amount, volume_ratio, accumulated_volume_ratio, amount_ratio, accumulated_amount_ratio`。
- 历史日的 ratio/profile 特征可以使用，例如 `same_minute_volume_ratio_mean_*d` 和 `same_minute_accumulated_volume_ratio_mean_*d`。
- 历史同分钟、历史日均等特征使用滞后窗口，当前日特征只引用历史日。
- future labels 只用于训练和评估，不作为预测输入。
- volume ratio 反推成交量时，使用 rolling 训练窗口中的历史日均量先验，不使用预测日真实全天成交量。

## 模型

核心模型为 IVE 风格 Transformer：

- 数值特征 linear projection
- 股票 embedding
- 流动性组 embedding
- sinusoidal positional encoding
- Transformer encoder
- 多 horizon 输出头

训练损失为：

```text
成交量比例 Gaussian NLL + VWAP return SmoothL1Loss
```

预测阶段使用 `model.eval()` 和 `torch.no_grad()`，不会反向传播，也不会更新模型参数。因此 `06` 主流程已经拆成“先训练、后预测评估”。

rolling 模型保存到：

```text
data/models/rolling/window_{N}d/{test_date}/{liquidity_group}/
```

每个组目录包含：

```text
ive_model.pt
model_meta.json
feature_columns.joblib
stock_vocab.joblib
group_vocab.joblib
normalizer.joblib
```

## 运行方式

安装依赖：

```bash
pip install -r requirements.txt
```

检查 schema：

```bash
python scripts/00_check_schema.py --data_dir data/tick_data --config config.yaml --out data/outputs/schema_summary.json
```

预处理 tick 到每日分钟缓存：

```bash
python scripts/01_preprocess.py --config config.yaml
```

构建每日特征和标签：

```bash
python scripts/02_build_features_labels.py --config config.yaml
```

并行构建每日特征和标签：

```bash
python scripts/02_build_features_labels_parallel.py --config config.yaml --workers 2
python scripts/02_build_features_labels_parallel.py --config config.yaml --workers 2 --months 202602
```

并行版按交易日生成 `data/features/model_parts/YYYYMMDD.parquet`，默认跳过已存在文件；如需重算加 `--overwrite`。每个 worker 读取当前日和最多前 10 个历史日，默认 `--maxtasks-per-child 1`，用来释放 pandas/numpy 峰值内存。

串行 rolling 训练和回测：

```bash
python scripts/06_rolling_backtest.py --config config.yaml
```

只跑指定 rolling window 时使用 `--windows`，不会改动 `config.yaml`：

```bash
python scripts/06_rolling_backtest.py --config config.yaml --windows 5
python scripts/06_rolling_backtest.py --config config.yaml --windows 5 8
```

只跑指定目标月份时使用 `--months`。它只限制预测/回测目标日期，训练窗口仍然从目标日前的历史交易日中取最近 N 天：

```bash
python scripts/06_rolling_backtest.py --config config.yaml --months 202604 --windows 5
```

默认先训练所有 rolling 任务，再用 CPU 预测评估并汇总报告。默认不覆盖已有模型；如需重训：

```bash
python scripts/06_rolling_backtest.py --config config.yaml --overwrite-models
```

只训练模型、暂不做预测评估：

```bash
python scripts/06_rolling_backtest.py --config config.yaml --months 202604 --windows 5 --train-only
```

只做预测评估、不进入训练阶段：

```bash
python scripts/06_rolling_backtest.py --config config.yaml --months 202604 --windows 5 --predict-only --predict-workers 4
```

`--predict-only` 会直接读取已有模型和已有 `stock_liquidity_group.parquet`，不会重算或覆盖窗口流动性分类。

预测评估已按“日期”为任务读取数据：每个 worker 对同一天只读取一次 feature part，再顺序处理 `high / medium / low`，避免同一天被三个流动性组重复读取。推荐回测也使用向量化计算，避免逐行 Python 循环。若机器仍然卡顿，先从 `--predict-workers 2` 或 `--predict-workers 4` 开始。

预测阶段还会输出 dataset 构造和 forward 进度日志。`ive_predict_batch_size` 控制预测 batch 大小，`predict_log_every_batches` 控制每隔多少个 batch 打印一次进度；显存不参与 CPU 预测，内存足够时可以尝试把 `ive_predict_batch_size` 调到 512。

预测日中没有出现在 rolling 训练窗口流动性分类表里的股票会被跳过，并打印 `skip unclassified` 行数和股票数。这样避免早期窗口不满时把大量未分类股票强行塞进 `low` 组，造成预测异常变慢和分组含义失真。

CPU 多进程预测评估：

```bash
python scripts/06_rolling_backtest.py --config config.yaml --predict-workers 4
```

多 GPU rolling 训练入口：

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/06_rolling_backtest_parallel.py --config config.yaml --train-workers 2 --predict-workers 4
```

多 GPU 且只跑 window=5：

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/06_rolling_backtest_parallel.py --config config.yaml --windows 5 --train-workers 2 --predict-workers 4
```

多 GPU 且只跑 202604 的 window=5：

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/06_rolling_backtest_parallel.py --config config.yaml --months 202604 --windows 5 --train-workers 2 --predict-workers 4
```

多 GPU 只训练、不预测：

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/06_rolling_backtest_parallel.py --config config.yaml --months 202604 --windows 5 --train-workers 2 --train-only
```

已有模型后只跑 CPU 预测评估：

```bash
python scripts/06_rolling_backtest_parallel.py --config config.yaml --months 202604 --windows 5 --predict-only --predict-workers 4
```

预测默认按 `date + window` 并行。若某些日期的 `medium` 组特别大，后续日期已跑完而早期大组仍占住 worker，可改用 `--predict-unit group`，把 `date + window + liquidity_group` 作为并行单位：

```bash
python scripts/06_rolling_backtest_parallel.py --config config.yaml --months 202602 --windows 5 --predict-only --predict-workers 6 --predict-unit group
```

预测分片已存在时可用 `--skip-existing-predictions` 跳过已完成的 `window/date/group`。只有该分片下 `metrics.csv`、`detail.parquet`、`backtest.parquet` 都存在时才认为完成：

```bash
python scripts/06_rolling_backtest_parallel.py --config config.yaml --months 202602 --windows 5 --predict-only --predict-workers 6 --predict-unit group --skip-existing-predictions
```

分月补跑时建议先只写分片、不生成最终总表，避免多个批次抢写 `data/outputs/rolling/*.csv`：

```bash
python scripts/06_rolling_backtest_parallel.py --config config.yaml --months 202602 --windows 5 --predict-only --predict-workers 6 --predict-unit group --skip-existing-predictions --skip-final-reports
python scripts/06_rolling_backtest_parallel.py --config config.yaml --months 202603 --windows 5 --predict-only --predict-workers 6 --predict-unit group --skip-existing-predictions --skip-final-reports
python scripts/06_rolling_backtest_parallel.py --config config.yaml --months 202604 --windows 5 --predict-only --predict-workers 6 --predict-unit group --skip-existing-predictions --skip-final-reports
```

所有月份分片都完成后，再单独汇总最终报告：

```bash
python scripts/06_rolling_backtest_parallel.py --config config.yaml --windows 5 --aggregate-only
```

并行训练版按 `date + window` 分发任务到可见 GPU。每个任务内部仍按 `high / medium / low` 顺序训练，避免同一日期内部抢显存。也可以显式指定卡：

```bash
python scripts/06_rolling_backtest_parallel.py --config config.yaml --devices 0,1 --train-workers 2 --predict-workers 4
```

预测评估分片写入：

```text
data/outputs/rolling/parts/window_{N}d/{test_date}/{liquidity_group}/
```

最终仍汇总为原有报告文件：

```text
rolling_evaluation_metrics.csv
rolling_prediction_error_detail.csv
rolling_prediction_error_by_date.csv
rolling_prediction_error_by_stock.csv
rolling_prediction_error_by_minute.csv
rolling_recommendation_backtest_detail.csv
rolling_recommendation_backtest_summary.csv
rolling_recommendation_backtest_worst_cases.csv
rolling_window_comparison.csv
```

启动网页：

```bash
streamlit run app/streamlit_app.py
```

## 核心文件

- `src/preprocess.py`：逐日、分 batch 读取 tick，聚合为分钟 OHLCV/VWAP，保存每日分钟缓存。
- `src/feature_engineering.py`：按日构建行情、历史统计、时间、IVE 特征和 future labels。
- `scripts/02_build_features_labels_parallel.py`：按交易日多进程构建 feature parts。
- `src/ive_dataset.py`：构造 390 分钟上下文序列，做标准化、padding mask、股票 ID、流动性组 ID，并排除泄露字段。
- `src/ive_model.py`：IVE 风格 Transformer 模型。
- `src/train.py`：模型训练、损失函数、模型保存。
- `src/rolling_train.py`：主 rolling 流程；窗口分类、训练/预测分离、CPU 预测并行、报告汇总。
- `scripts/06_rolling_backtest_parallel.py`：多 GPU date/window 级训练调度入口。
- `src/predict.py`：单笔预测推荐，读取对应 rolling 模型和窗口分类。
- `app/streamlit_app.py`：网页 Demo 和报告展示。

## 网页功能

网页包含：

- 单笔推荐：输入股票代码、买/卖、交易数量、开始时间、rolling window。
- 推荐展示：股票类别、推荐区间、预测 VWAP、预测成交量、预测成交占比、是否满足 30%。
- 候选区间表：5/10/15/30/60 分钟预测值、真实值、误差、可行性。
- 可视化：价格/VWAP 曲线、候选区间 VWAP 曲线、成交占比与 30% 约束线、volume ratio 不确定性。
- 报告：样本内/样本外 rolling 准确性、window 对比、流动性组和 horizon 对比、日期/股票/分钟误差、推荐命中率、regret、worst cases。

## 维护约定

后续每次修改核心流程、数据口径、特征、模型、评估或网页展示时，都要同步更新本 README 和 `PROJECT_CONTEXT.md`。
## 2026-05-23 update: rolling-only web prediction

- `app/streamlit_app.py` and `src/predict.py` now use rolling IVE models only.
- Single-order prediction requires `rolling_window` and loads models from `data/models/rolling/window_{N}d/{test_date}/{liquidity_group}/`.
- Static model fallback under `data/models/high`, `data/models/medium`, and `data/models/low` is disabled for web prediction.
- Missing rolling model artifacts now raise an explicit rolling-model error instead of trying to load static-model paths.
- The web app discovers available rolling windows and dates from `data/models/rolling` and defaults to an existing rolling date.
- Rolling report CSVs are read from `data/outputs/rolling/*.csv`; the app temporarily also checks `data/outputs/rolling/rolling/*.csv` for existing older outputs.
- The single-order page also shows same-day rolling metrics plus selectable per-stock VWAP and volume prediction curves from rolling `detail.parquet`.
- Same-day rolling metrics and per-stock VWAP/volume curves now live on the Evaluation Reports tab; curve reads are filtered by stock and horizon to keep horizon switching responsive.
