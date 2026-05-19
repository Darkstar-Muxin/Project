# A股分类建模的最优交易完成时间推荐系统

本项目用于在给定股票代码、买卖方向、交易数量和开始时间后，推荐 5/10/15/30/60 分钟中的最优交易完成区间。推荐规则是：买入选择满足 30% 成交占比约束下预测 VWAP 最低的区间；卖出选择满足约束下预测 VWAP 最高的区间。

模型参考 `paper/01.pdf` 的 IVE 思路：一分钟级 intraday volume ratio 预测、390 个一分钟步上下文、股票 embedding、时间/位置编码、Transformer encoder 和概率式 volume ratio 输出。本项目原始数据是 tick 成交数据，因此先流式聚合为分钟数据，再训练 rolling Transformer 模型。

## 数据与缓存

原始数据：

```text
data/tick_data/YYYYMMDD.parquet
```

每个 tick parquet 应包含：

```text
StockCode, Date, Time, Price, Volume, Turnover, BSFlag
```

预处理默认不会合并大表，只生成每日分钟缓存：

```text
data/processed/minute_parts/YYYYMMDD.parquet
```

`data/processed/minute_data.parquet` 是可选产物，默认不生成。项目不通过抽样减少数据量，所有有效 tick 都参与分钟聚合；性能问题通过逐日处理、parquet batch、按日期读取、懒加载和 batch 训练解决。

特征也默认按日保存：

```text
data/features/model_parts/YYYYMMDD.parquet
```

`data/features/model_dataset.parquet` 是可选合并产物，默认不生成，避免内存占满。

## 样本划分与 Rolling 口径

配置在 `config.yaml`：

```yaml
train_months: [202602, 202603]
test_months: [202604]
rolling_train_months: [202602, 202603]
rolling_test_months: [202604]
rolling_windows: [5, 8]
```

rolling 是主流程。样本内和样本外都用同一逻辑：对每个预测日，只使用该日前最近 N 个交易日训练，并只预测该日。`split=train` 的 rolling 结果用于调窗口和模型超参数；`split=test` 的 rolling 结果用于最终样本外报告。

## 流动性分类

流动性分类不能使用全样本。rolling 主流程会在每个 `test_date + window` 上重新分类：

- 只使用预测日前最近 N 个交易日。
- 对每只股票计算窗口内平均日成交额、平均日成交量、平均日成交笔数。
- 主要按平均日成交额分位数划分：高流动性、中流动性、低流动性。
- 分类结果保存到：

```text
data/models/rolling/window_{N}d/{test_date}/stock_liquidity_group.parquet
```

全局 `data/processed/stock_liquidity_group.parquet` 只作为可选 baseline 参考，rolling 主模型不依赖它。

## 特征

分钟聚合字段：

- 行情：`open, high, low, close, vwap`
- 成交：`volume, amount, trade_count`
- 买卖方向成交：`buy_volume, sell_volume, buy_amount, sell_amount`

模型候选特征包括：

- 行情特征：价格、VWAP、成交量、成交额、成交笔数、买卖成交不平衡、1/5/10 分钟收益率、5/10/20 分钟滚动成交量、滚动成交额、滚动 VWAP、滚动波动率、VWAP 偏离。
- 历史统计特征：过去 5/10/20 日股票日均量、日均额、日均 VWAP；同股票同分钟历史均值；历史同分钟 volume ratio 均值。
- 时间特征：分钟序号、距开盘分钟数、距收盘分钟数、上午/下午标记、绝对时间 sin/cos。
- IVE 特征：分钟成交量、累计成交量、成交额、累计成交额、股票 embedding、流动性组 embedding、位置编码、绝对时间编码。
- 交易要素：买卖方向、交易数量、交易数量相对预测成交量/历史成交量的占比主要在推理和推荐阶段使用。

当前 tick 数据没有 bid/ask/order book 字段，因此盘口特征仅作为预留能力，不伪造。

## 标签

每个 horizon 构造：

```text
future_vwap_{h}
future_volume_{h}
future_volume_ratio_{h}
log_future_volume_ratio_{h}
```

标签严格在同一股票、同一交易日内构造，不跨日、不跨股票。

## 防止未来信息泄露

当前主流程遵守以下规则：

- 不使用全样本流动性分类；rolling 分类只用预测日前窗口数据。
- 不默认生成或读取全量 `minute_data.parquet` / `model_dataset.parquet`。
- 模型输入排除当天全天才知道的字段：`daily_volume, daily_amount, volume_ratio, accumulated_volume_ratio, amount_ratio, accumulated_amount_ratio`。
- 历史同分钟、历史日均等特征使用 `shift(1)`，当前日特征只引用历史日。
- feature part 不生成全局流动性组历史特征；rolling 阶段才注入窗口分类。
- future labels 只用于训练和历史评估，不作为预测输入。
- volume ratio 反推成交量时，使用 rolling 训练窗口中的历史日均量先验，不使用预测日真实全天成交量。

## 模型

核心模型为 IVE 风格 Transformer：

- 数值特征 linear projection
- 股票 embedding
- 流动性组 embedding
- sinusoidal positional encoding
- absolute time encoding
- Transformer encoder
- 多 horizon 输出头

模型输出：

- `predicted_vwap`
- `predicted_market_volume`
- `predicted_volume_ratio`
- `predicted_volume_sigma`
- `predicted_participation`

rolling 模型保存到：

```text
data/models/rolling/window_{N}d/{test_date}/{liquidity_group}/
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

运行 rolling 训练和回测，这是主流程：

```bash
python scripts/06_rolling_backtest.py --config config.yaml
```

静态训练和评估只作为 optional baseline：

```bash
python scripts/03_train_models.py --config config.yaml
python scripts/04_evaluate.py --config config.yaml
```

启动网页：

```bash
streamlit run app/streamlit_app.py
```

## 核心文件

- `src/preprocess.py`：逐日、分 batch 读取 tick；聚合为分钟 OHLCV/VWAP；保存每日分钟缓存。
- `src/stock_classification.py`：生成 baseline 流动性分类；rolling 主流程不依赖全样本分类。
- `src/feature_engineering.py`：按日构建行情、历史统计、时间、IVE 特征和 future labels；默认输出每日 feature part。
- `src/label_builder.py`：同股票、同交易日内构造未来 VWAP、成交量和 volume ratio 标签。
- `src/ive_dataset.py`：构造 390 分钟上下文序列，做标准化、padding mask、股票 ID、流动性组 ID；排除泄露字段。
- `src/ive_model.py`：IVE 风格 Transformer 模型。
- `src/rolling_train.py`：主流程；按 split/date/window 读取所需日期、做窗口流动性分类、训练模型、预测当天、输出评估和回测。
- `src/predict.py`：单笔预测推荐；读取对应日期 feature part 和 rolling 窗口分类。
- `app/streamlit_app.py`：网页 Demo 和报告展示。

## 网页功能

网页包含：

- 单笔推荐：输入股票代码、买/卖、交易数量、开始时间、rolling window。
- 推荐展示：股票类别、推荐区间、预测 VWAP、预测成交量、预测成交占比、是否满足 30%。
- 候选区间表：5/10/15/30/60 分钟预测值、真实值、误差、可行性。
- 可视化：价格/VWAP 曲线、候选区间 VWAP 曲线、成交占比与 30% 约束线、volume ratio 不确定性。
- 报告：样本内/样本外 rolling 准确性、window 对比、流动性组和 horizon 对比、日期/股票/分钟误差、推荐命中率、regret、worst cases。

## 后续维护约定

后续每次修改核心流程、数据口径、特征、模型、评估或网页展示时，都要同步更新本 README 和 `PROJECT_CONTEXT.md`。
