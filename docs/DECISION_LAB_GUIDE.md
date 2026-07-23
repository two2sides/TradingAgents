# Decision Lab 使用与集成指南

Decision Lab 是职责一提供的本地交易实验工作台。它把历史行情、B 的记忆、C
的目标仓位、模拟成交和账户账本保存在同一份可回放档案中。

## 1. 安装

在仓库根目录执行：

```powershell
uv sync --extra dev --extra webui
```

项目跟踪 `uv.lock`，因此团队成员使用相同 Python 平台时会得到同一套已解析依赖。

若要使用真实 `TradingAgents + RAG` 模式，还需安装 B 的本地向量记忆依赖：

```powershell
uv sync --extra dev --extra webui --extra memory
```

## 2. 启动 WebUI

```powershell
uv run --frozen streamlit run webui/app.py
```

默认地址通常是 <http://localhost:8501>。主题配置位于
`.streamlit/config.toml`。

页面有两种决策引擎：

- `Fast demo`：不需要 API Key，不访问 LLM，适合快速展示全部审计与执行能力；
- `TradingAgents + RAG`：使用 `.env` 中配置的模型、C 的完整 Agent 图和 B 的
  `EnhancedMemoryProvider`。

真实行情默认使用 `Yahoo Chart · cached`：它复用项目统一的 Chart JSON
数据路径和磁盘缓存，不经过 yfinance 的 crumb/cookie 流程，也不需要行情 Key。
真实模式还提供 `Built-in execution sandbox`，用于外部服务不可用时继续验证
Agent、RAG、Broker 和 WebUI。Sandbox 不使用真实执行价格，因此不能把其收益指标
解释为真实历史回测。

若升级前的环境出现 `No module named 'torchvision'`，重新同步 memory extra
并重启 Streamlit。若 Yahoo Chart 持续返回 HTTP 429，页面会在指数退避重试后
明确提示限流；可以稍后重试，或临时选择 execution sandbox。

第一次体验建议使用 `Fast demo`：

1. 打开左侧 **Run Experiment**；
2. 保持 `Built-in deterministic demo` 数据源；
3. 点击 **Launch historical replay**；
4. 运行结束后进入 **Decision Replay**；
5. 点击净值图上的 BUY/SELL 标记，切换到 **Decision audit** 查看证据；
6. 在 **Execution What-if** 中提高费用或滑点并保存场景；
7. 进入 **Compare Runs** 比较原始运行和 What-if。

内置演示行情是确定性生成的，同样的标的和时间窗口每次产生相同 OHLCV 数据，适合自动测试和现场演示。它不是实际市场数据。

## 3. 使用 Yahoo Chart

在运行表单中把 Market source 切换为 `Yahoo Chart · cached`。该适配器先查询
共享磁盘缓存，只在需要时访问 Yahoo Chart JSON，再把请求窗口内的日线一次性
加载到不可变的内存数据源；回测循环不会在每个决策日重复发起网络请求。

`HistoricalMarketDataProvider.from_yfinance()` 仍作为兼容接口保留，但不再是
WebUI 默认路径。当前 Yahoo Chart 回放不单独模拟现金分红和拆股后的持股数量变化。

终端默认输出 INFO 级别的初始化、行情、决策、成交、完成、限流和异常日志。
需要逐日估值细节时可以在启动前设置：

```powershell
$env:TRADINGAGENTS_LOG_LEVEL = "DEBUG"
```

## 4. 四个页面

### Overview

- 展示本地运行数量、完成状态和最好收益；
- 浏览最近档案；
- 选择一个运行加入 Decision Replay。

### Run Experiment

- 在 `Fast demo` 与 `TradingAgents + RAG` 间切换；
- 配置标的、时间窗口、初始资金和决策间隔；
- 显式配置手续费、最低费用和滑点；
- 真实模式可选择参与图运行的分析师与单标的建仓上限；
- 提交前估算完整 Agent 图调用次数，一周版本单次最多 12 次；
- 实时查看 `RunEvent` 进度流水线；
- 运行成功或失败都会写入 SQLite。

### Decision Replay

- 策略净值与 Buy & Hold 基准；
- 回撤和逐日资金分配；
- 价格与 Agent 目标仓位联动图；
- 点击交易标记定位决策；
- 查看当时行情、记忆、理由、诊断、成交和账本；
- 真实模式在 **Agent dossier** 中按阶段浏览分析师报告、研究计划、交易提案、
  Portfolio Manager 决策及审计对象数量；
- 下载完整运行 JSON；
- 发起不调用 Agent 的执行层 What-if。

### Compare Runs

- 把不同初始资金的运行归一化到同一起点；
- 对比收益、风险、换手和费用；
- 按 `(symbol, as_of)` 对齐目标仓位；
- 判断差异来自 Agent 决策还是执行条件。

## 5. 数据保存位置

默认数据库：

```text
~/.tradingagents/paper_trading/runs.sqlite3
```

可以用独立环境变量改到其他位置：

```powershell
$env:TRADINGAGENTS_RUN_STORE = "D:\data\tradingagents-runs.sqlite3"
uv run --frozen streamlit run webui/app.py
```

数据库包含：

- `BacktestRequest`；
- `BacktestResult`；
- 账户账本和回测用行情；
- Agent 决策上下文；
- `RunEvent`；
- 失败原因。

## 6. B/C 真实集成

默认图输出 `Buy / Overweight / Hold / Underweight / Sell` 五级评级，而 Broker
只执行连续的 `target_weight`。职责一提供了两层显式适配：

- `TradingAgentsGraphDecisionProvider`：运行默认图、严格读取 Portfolio Manager
  评级并返回公共 `DecisionEnvelope`；
- `RatingAllocationPolicy`：把评级转换为 long-only 目标仓位。

代码入口如下：

```python
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.extensions.contracts import BacktestRequest
from tradingagents.extensions.memory import EnhancedMemoryProvider
from tradingagents.extensions.paper_trading import (
    BacktestApplicationService,
    HistoricalMarketDataProvider,
    RatingAllocationPolicy,
    SQLiteRunStore,
    TradingAgentsGraphDecisionProvider,
)
from tradingagents.graph.trading_graph import TradingAgentsGraph

graph = TradingAgentsGraph(config=dict(DEFAULT_CONFIG))
memory = EnhancedMemoryProvider(
    dict(DEFAULT_CONFIG),
    llm_client=graph.quick_thinking_llm,
)
decision = TradingAgentsGraphDecisionProvider(
    graph,
    RatingAllocationPolicy(max_position_weight=0.35),
)
request = BacktestRequest(...)
market_data = HistoricalMarketDataProvider.from_yahoo_chart(
    request.symbols, request.start, request.end
)
service = BacktestApplicationService(market_data, SQLiteRunStore())

stored = service.run_and_store(
    request=request,
    decision_provider=decision,
    memory_provider=memory,
    label="B/C integration run",
)
```

### 记忆生命周期为什么只由 A 管理

回测器先通过公共 `MemoryProvider.retrieve()` 获取 `as_of <= T` 的记忆。适配器
在本次图调用期间把这份上下文作为只读 Provider 注入各 Agent，并临时关闭图
内部的 Markdown/RAG 写入。图结束后：

1. A 执行评级映射后的目标仓位；
2. A 用真实 `ExecutionReport` 调用 B 的 `record_decision()`；
3. 结果窗口到达后，A 再调用 B 的 `record_outcome()`。

因此同一决策只写入一次，历史回测也不会被默认图的“当前时间反思”路径污染。

### 默认评级仓位政策

默认单标的建仓上限为 35%，多标的还会使用 `min(35%, 1/N)`：

| Portfolio Manager 评级 | 目标仓位规则 |
|---|---|
| Buy | 只增不减，向分散上限靠拢 |
| Overweight | 只增不减，向上限的 75% 靠拢 |
| Hold | 保持当前仓位 |
| Underweight | 只减不增，降到不高于上限的 25% |
| Sell | 清零 |

显式评级缺失时不会默认解释为 `Hold`，而是返回 `FAILED_SAFE` 并保持当前仓位。

## 7. 时间与成交语义

```text
T 日收盘前已经可见的数据
→ B 检索 available_at <= T 的记忆
→ C 产生 T 日目标仓位
→ T+1 共同可交易 K 线的开盘价报价
→ Broker 应用滑点、费用、整数股和现金约束
→ T+1 收盘重新估值
```

多标的使用共同交易日历，使同一次组合再平衡拥有同步执行时点。当前账户为 long-only，仓位权重范围是 0–100%。

## 8. What-if 的正确解释

执行层 What-if 会复用父运行的 `target_weight`，然后改变：

- 初始资金；
- 手续费；
- 最低费用；
- 滑点。

它不会重新调用 B 或 C，也不会声称 Agent 在新账户状态下仍会做同样决策。结果包含：

```text
run_kind = EXECUTION_WHAT_IF
parent_run_id = ...
agent_calls_reused = ...
```

若要比较记忆开关、Agent 模型或风险逻辑，应执行两次完整运行，而不是使用执行层 What-if。

## 9. 验证命令

职责一和公共契约：

```powershell
uv run --frozen pytest tests/paper_trading tests/test_extension_contracts.py -q
```

代码规范：

```powershell
uv run --frozen ruff check tradingagents/extensions webui tests/paper_trading
uv run --frozen ruff format --check tradingagents/extensions webui tests/paper_trading
```

启动 smoke test：

```powershell
uv run --frozen streamlit run webui/app.py --server.headless true
```

## 10. 当前限制

- 只支持日线、long-only 和整数股；
- 只实现 `NEXT_OPEN` 执行策略；
- 不模拟盘口深度、停牌、涨跌停、融资和做空；
- 复权行情不单独处理分红现金流；
- Streamlit 适合本地演示与研究，不提供多用户任务隔离；
- 真实模式每个决策点都会运行完整多 Agent 图，不适合用很密的日频决策间隔；
- 首次加载本地 embedding 可能需要下载模型；一次完整图调用通常以分钟计，而不是
  演示策略的秒级执行。页面默认只选 Market Analyst、约 21 天窗口和 30 bars
  间隔，以形成一次完整图调用；需要时再增加分析师或决策点；
- Agent 图内部的数据工具仍有各自的历史可得性限制，Decision Lab 保存并展示其
  降级状态与证据链，不把缺失数据伪装成精确回测输入。
