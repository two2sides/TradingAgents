# Decision Lab 使用与集成指南

Decision Lab 是职责一提供的本地交易实验工作台。它把历史行情、B 的记忆、C
的目标仓位、模拟成交和账户账本保存在同一份可回放档案中。

## 1. 安装

在仓库根目录执行：

```powershell
uv sync --extra dev --extra webui
```

项目跟踪 `uv.lock`，因此团队成员使用相同 Python 平台时会得到同一套已解析依赖。

## 2. 启动 WebUI

```powershell
uv run --frozen streamlit run webui/app.py
```

默认地址通常是 <http://localhost:8501>。主题配置位于
`.streamlit/config.toml`。

第一次体验不需要配置 API Key：

1. 打开左侧 **Run Experiment**；
2. 保持 `Built-in deterministic demo` 数据源；
3. 点击 **Launch historical replay**；
4. 运行结束后进入 **Decision Replay**；
5. 点击净值图上的 BUY/SELL 标记，切换到 **Decision audit** 查看证据；
6. 在 **Execution What-if** 中提高费用或滑点并保存场景；
7. 进入 **Compare Runs** 比较原始运行和 What-if。

内置演示行情是确定性生成的，同样的标的和时间窗口每次产生相同 OHLCV 数据，适合自动测试和现场演示。它不是实际市场数据。

## 3. 使用 yfinance

在运行表单中把 Market source 切换为 `yfinance daily`。该适配器下载复权后的日线并一次性加载到不可变的内存数据源；回测循环不会在每个决策日重复发起网络请求。

当前版本使用复权 OHLCV 规避拆股产生的虚假价格跳变，但不单独模拟现金分红和拆股后的持股数量变化。

## 4. 四个页面

### Overview

- 展示本地运行数量、完成状态和最好收益；
- 浏览最近档案；
- 选择一个运行加入 Decision Replay。

### Run Experiment

- 配置标的、时间窗口、初始资金和决策间隔；
- 显式配置手续费、最低费用和滑点；
- 实时查看 `RunEvent` 进度流水线；
- 运行成功或失败都会写入 SQLite。

### Decision Replay

- 策略净值与 Buy & Hold 基准；
- 回撤和逐日资金分配；
- 价格与 Agent 目标仓位联动图；
- 点击交易标记定位决策；
- 查看当时行情、记忆、理由、诊断、成交和账本；
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

## 6. 接入 B 和 C 的真实实现

WebUI 不导入 B、C 的内部模块。集成层只需要把满足协议的对象交给应用服务：

```python
from tradingagents.extensions.contracts import BacktestRequest
from tradingagents.extensions.paper_trading import (
    BacktestApplicationService,
    HistoricalMarketDataProvider,
    SQLiteRunStore,
)

request = BacktestRequest(...)
market_data = HistoricalMarketDataProvider.from_yfinance(
    request.symbols, request.start, request.end
)
service = BacktestApplicationService(market_data, SQLiteRunStore())

stored = service.run_and_store(
    request=request,
    decision_provider=real_c_provider,
    memory_provider=real_b_provider,
    label="B/C integration run",
)
```

真实对象分别实现：

- B：`MemoryProvider`；
- C：`DecisionProvider`。

演示页面当前使用 `DemoMemoryProvider` 和 `MovingAverageDecisionProvider`。后者只用于让 A 的引擎和 UI 独立可运行，不代表 C 的最终策略。

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
- 接入真实 B/C Provider 后才代表完整三人项目策略。
