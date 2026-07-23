# TradingAgents 小组公共接口约定

## 1. 文档目的

本文档规定三名成员所负责模块之间的公共接口、数据语义和职责边界。

本文档只约定模块之间交换什么，不规定各成员内部如何实现。以下内容均不属于公共约束：

- 使用哪一种数据库、文件格式或缓存；
- RAG 使用关键词、向量、规则还是混合检索；
- 决策模块包含多少 Agent、LangGraph 节点或 Tool；
- 量化指标、融合算法、Prompt 和模型的具体选择；
- 模拟 Broker 内部使用事件驱动还是逐日循环；
- WebUI 的组件结构和绘图库。

只要实现满足本文档的输入、输出、时间语义和错误语义，各模块就可以独立替换。

公共契约已经落地为可直接导入的 Python 代码：

```text
tradingagents/extensions/contracts.py   # 共享数据模型
tradingagents/extensions/protocols.py   # A/B/C 对外接口
```

实现者应从这两个模块导入契约，不能在自己的实现目录中重新定义同名模型或复制一份接口。

## 2. 成员职责边界

| 成员 | 负责能力 | 对外提供 | 不负责 |
|---|---|---|---|
| A：模拟交易、回测与 WebUI | 市场时间推进、账户、成交、回测、结果展示 | 市场快照、账户快照、执行报告、回测结果 | 如何分析股票、如何检索记忆、如何决定目标仓位 |
| B：记忆、上下文、RAG 与状态感知 | 检索历史经验，记录决策及事后结果 | 记忆上下文、记忆引用 | 修改账户、执行交易、生成最终目标仓位 |
| C：Agent、仓位决策、验证与 Tools | 综合原项目分析、市场、账户和记忆形成决策 | 最终交易意图、决策状态、可选诊断信息 | 计算实际成交股数、扣除现金、保存账户 |

唯一允许改变账户现金和持仓的模块是 A 的 Broker。C 输出目标仓位，但不能直接修改账户；B 只能读取账户快照。

## 3. 模块调用关系

一次决策周期的标准调用顺序如下：

```text
A 获取 MarketSnapshot 和 PortfolioState
                    ↓
B.retrieve(...) 返回 MemoryContext
                    ↓
C.decide(...) 返回 DecisionEnvelope
                    ↓
A.rebalance(...) 返回 ExecutionReport
                    ↓
B.record_decision(...) 记录当时可知的信息
                    ↓
结果可知后 B.record_outcome(...) 补充事后结果
```

调用方只能依赖公共接口，不能读取其他成员模块的内部数据库、LangGraph State 或私有字段。

## 4. 共同数据契约

下面的 Python 类型表示公共接口语义。仓库中的权威实现位于
`tradingagents.extensions.contracts`；本文档中的片段用于说明，不应复制成另一套模型。

### 4.1 市场快照

```python
class MarketBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketSnapshot:
    symbol: str
    as_of: datetime
    bars: list[MarketBar]
    metadata: dict[str, Any]
```

约束：

- `bars` 中不得出现晚于 `as_of` 的数据；
- 所有价格使用标的报价货币；
- `bars` 按时间升序排列；
- `metadata` 仅用于扩展，任何模块都不能依赖未写入本文档的私有键完成核心流程。

### 4.2 账户快照

```python
class Position:
    symbol: str
    quantity: int
    average_cost: float
    market_price: float
    market_value: float
    weight: float


class PortfolioState:
    as_of: datetime
    cash: float
    total_equity: float
    positions: dict[str, Position]
```

约束：

- `PortfolioState` 是只读快照；
- `weight` 和后续所有目标仓位均使用 `[0, 1]` 区间，而不是百分数；
- 不持有某标的时，可以省略对应 `Position`，调用方应将其当前仓位视为 `0`；
- 只有 Broker 执行成功后才能产生新的账户快照。

### 4.3 交易意图

`target_weight` 是执行层唯一可信的交易方向来源。公共接口不同时维护独立的 `BUY/SELL/HOLD` 字段，以避免动作和目标仓位互相矛盾。

```python
class TradeIntent:
    decision_id: str
    symbol: str
    as_of: datetime
    target_weight: float
    confidence: float
    rationale: str
    warnings: list[str]
    metadata: dict[str, Any]
```

语义：

- `target_weight > current_weight`：执行层计算增持订单；
- `target_weight < current_weight`：执行层计算减持订单；
- `target_weight == current_weight`：不交易；
- `target_weight = 0`：目标是退出该标的；
- `confidence` 使用 `[0, 1]`；
- `metadata` 可携带原始评级、验证报告等诊断信息，但 A 不得通过解析私有键决定是否执行。

### 4.4 Trace

```python
class TraceEvent:
    timestamp: datetime
    source: str
    event_type: str
    summary: str
    payload: dict[str, Any]
```

Trace 只用于调试、实验和 WebUI 展示。核心交易流程不得依赖自然语言 `summary`。

## 5. A 提供的公共接口

### 5.1 市场数据

```python
class MarketDataProvider(Protocol):
    def get_snapshot(
        self,
        symbol: str,
        as_of: datetime,
        lookback: int,
    ) -> MarketSnapshot:
        ...

    def get_execution_quote(
        self,
        symbol: str,
        after: datetime,
    ) -> ExecutionQuote:
        ...
```

`get_snapshot()` 返回决策时已经可见的信息；`get_execution_quote()` 返回决策时间之后允许成交的第一个报价。历史回测和当前模拟盘可以使用不同实现，但必须遵守相同时间语义。

### 5.2 Broker

```python
class Broker(Protocol):
    def get_portfolio(self, as_of: datetime) -> PortfolioState:
        ...

    def rebalance(
        self,
        intent: TradeIntent,
        quote: ExecutionQuote,
    ) -> ExecutionReport:
        ...
```

```python
class ExecutionReport:
    decision_id: str
    status: Literal["FILLED", "PARTIAL", "REJECTED", "NO_ACTION"]
    requested_target_weight: float
    achieved_weight: float
    fills: list[Fill]
    fees: float
    rejection_reason: str | None
```

Broker 必须保证账户硬约束。非法意图应返回 `REJECTED`，不能以“成功”为名静默执行完全不同的交易。由于整数股、费用或流动性约束，`achieved_weight` 可以和请求仓位不同。

### 5.3 回测

```python
class BacktestRunner(Protocol):
    def run(
        self,
        request: BacktestRequest,
        decision_provider: DecisionProvider,
        memory_provider: MemoryProvider,
        observer: RunObserver | None = None,
    ) -> BacktestResult:
        ...
```

```python
class ExecutionConfig:
    commission_rate: float = 0.0005
    slippage_rate: float = 0.001
    minimum_fee: float = 0
    execution_policy: Literal["NEXT_OPEN"] = "NEXT_OPEN"

class BacktestRequest:
    symbols: list[str]
    start: datetime
    end: datetime
    initial_cash: float
    lookback: int
    decision_interval_bars: int
    outcome_horizon_bars: int
    execution: ExecutionConfig

class BacktestResult:
    decisions: list[DecisionEnvelope]
    executions: list[ExecutionReport]
    equity_curve: list[EquityPoint]
    portfolio_history: list[PortfolioState]
    benchmark_curves: dict[str, list[EquityPoint]]
    metrics: dict[str, float]
    warnings: list[str]
```

`commission_rate` 和 `slippage_rate` 均为小数比例，例如 `0.001` 表示
0.1%。第一版只支持 `NEXT_OPEN`，即决策产生后的下一个可交易开盘价成交；以后若增加成交策略，应扩展枚举而不是在实现中静默改变语义。

`decision_interval_bars` 表示两次 Agent 决策之间间隔多少根共同可交易 K 线，默认 5；`outcome_horizon_bars` 表示经过多少根 K 线后将结果反馈给 B，默认也是 5。使用 K 线数量而不是“自然日/周”可以避免节假日语义不一致。

`benchmark_curves` 的键是稳定、可展示的基准名称，例如
`buy_and_hold`。WebUI 只消费公共结果对象，不直接访问 Broker、记忆库或
Agent 的内部状态。`portfolio_history` 用于回放每个估值时点的现金、持仓、成本和权重，不要求 WebUI 重新推导账户状态。

长回测可选传入进度观察者：

```python
class RunEvent:
    timestamp: datetime
    stage: str
    message: str
    progress: float | None  # 0 到 1
    payload: dict[str, Any]

class RunObserver(Protocol):
    def on_event(self, event: RunEvent) -> None:
        ...
```

观察者只接收事件，不控制回测流程；CLI、测试和 WebUI 可以分别提供自己的实现，B、C 不需要依赖 Streamlit。

## 6. B 提供的公共接口

### 6.1 检索记忆

```python
class MemoryQuery:
    symbol: str
    as_of: datetime
    market: MarketSnapshot
    portfolio: PortfolioState
    limit: int


class MemoryContext:
    as_of: datetime
    items: list[MemoryItem]
    summary: str
    warnings: list[str]


class MemoryProvider(Protocol):
    def retrieve(self, query: MemoryQuery) -> MemoryContext:
        ...
```

B 自行决定存储和检索方式。公共约束只有：

- 不得返回在 `query.as_of` 时尚不可知的经验；
- 每个 `MemoryItem` 必须带有信息可用时间 `available_at`；
- 没有匹配结果时返回合法的空 `MemoryContext`；
- C 不能依赖 B 的相似度算法、数据库 ID 或内部字段。

### 6.2 记录决策

```python
class DecisionRecord:
    intent: TradeIntent
    portfolio_before: PortfolioState
    market_at_decision: MarketSnapshot
    execution: ExecutionReport | None


class MemoryProvider(Protocol):
    def record_decision(self, record: DecisionRecord) -> MemoryReference:
        ...
```

### 6.3 补充事后结果

```python
class DecisionOutcome:
    observed_at: datetime
    holding_period_return: float | None
    max_adverse_move: float | None
    portfolio_impact: float | None
    metadata: dict[str, Any]


class MemoryProvider(Protocol):
    def record_outcome(
        self,
        reference: MemoryReference,
        outcome: DecisionOutcome,
    ) -> None:
        ...
```

决策和结果必须分开记录，因为决策发生时未来收益尚不可知。`record_outcome()` 的内容只有在 `observed_at` 到达后才能被后续检索使用。

## 7. C 提供的公共接口

### 7.1 决策请求

```python
class DecisionRequest:
    symbol: str
    as_of: datetime
    market: MarketSnapshot
    portfolio: PortfolioState
    memory: MemoryContext
    mode: str
    metadata: dict[str, Any]
```

`metadata` 可以携带原 TradingAgents 报告、评级或实验配置，但核心调用方不得要求某个未标准化的私有键一定存在。

### 7.2 决策输出

```python
class DecisionEnvelope:
    intent: TradeIntent
    status: Literal["SUCCESS", "DEGRADED", "FAILED_SAFE"]
    trace: list[TraceEvent]
    diagnostics: dict[str, Any]


class DecisionProvider(Protocol):
    def decide(self, request: DecisionRequest) -> DecisionEnvelope:
        ...
```

约束：

- `intent` 是 A 唯一允许执行的对象；
- `trace` 和 `diagnostics` 只用于解释、实验和展示；
- `SUCCESS` 表示正常产生决策；
- `DEGRADED` 表示部分信息不可用但仍产生了可执行决策；
- `FAILED_SAFE` 表示无法可靠判断，此时目标仓位必须等于请求中的当前仓位；
- 不能用固定 `target_weight = 0` 表示普通失败，否则模型或网络故障会被误解释为清仓。

C 内部是否使用原始 TradingAgents、额外 Agent、验证节点、量化 Tool 或确定性规则，均不属于公共接口。

## 8. 错误与时间语义

所有模块共同遵守：

1. 预期内的数据缺失、无检索结果、LLM 失败和订单拒绝应通过返回对象表达，不应直接终止整个回测。
2. 编程错误、违反公共契约或状态损坏可以抛出异常，不应伪装成正常结果。
3. 所有时间字段必须带有明确时区或由项目统一约定为同一时区。
4. 决策只能读取 `as_of` 时已经可知的信息。
5. 成交报价时间必须晚于决策信息截止时间。
6. 记忆中的事后结果必须等到 `observed_at` 后才能参与检索。

## 9. 接口变更规则

公共契约合并后，任何成员都不能在自己的功能分支中单方面改变字段语义。

接口变更应满足：

1. 先说明调用方为什么无法通过现有扩展字段完成需求；
2. 三人确认字段名称、单位、默认值和兼容策略；
3. 单独提交接口 PR；
4. 同时更新契约测试和本文档；
5. 接口 PR 合并后，各功能分支再同步修改。

建议至少建立以下契约测试：

- 公共模型序列化和反序列化；
- `target_weight`、`confidence` 的范围校验；
- `FAILED_SAFE` 保持当前仓位；
- 市场快照不存在未来数据；
- 记忆检索不存在 `available_at > as_of` 的条目；
- Broker 对非法意图返回明确拒绝；
- `BacktestResult` 能被 WebUI 在不导入 B、C 内部模块的情况下读取。

## 10. 最小集成验收

在三个人的真实实现完成前，应分别提供符合公共接口的 Mock。最小集成链必须能够完成：

```text
MarketDataProvider
→ PortfolioState
→ MemoryProvider
→ DecisionProvider
→ Broker
→ BacktestResult
```

只要 Mock 和真实实现可以通过依赖注入相互替换，且不要求修改调用方代码，就说明模块边界有效。

## 11. 开发入口与目录归属

三个实现目录已经创建：

```text
tradingagents/extensions/
├── contracts.py          # 公共，只通过单独的接口 PR 修改
├── protocols.py          # 公共，只通过单独的接口 PR 修改
├── paper_trading/        # A 独立实现
├── memory/               # B 独立实现
└── decision/             # C：分析师工具与特征库（原 HybridDecisionProvider 闸门已移除）
```

推荐直接从公共模块导入：

```python
from tradingagents.extensions.contracts import (
    BacktestRequest,
    DecisionEnvelope,
    DecisionRequest,
    ExecutionReport,
    MemoryContext,
    PortfolioState,
    TradeIntent,
)
from tradingagents.extensions.protocols import (
    BacktestRunner,
    Broker,
    DecisionProvider,
    MemoryProvider,
    RunObserver,
)
```

各成员的最低开发起点是：

- A 在 `paper_trading/` 中提供满足 `MarketDataProvider`、`Broker` 和
  `BacktestRunner` 的对象；
- B 在 `memory/` 中提供满足 `MemoryProvider` 的对象；
- C 在 `decision/tools/` 中维护 Market/Sentiment 等 **Analyst Tools**；
  `DecisionProvider` 协议仍保留供日后薄 Policy / Broker 接入，**当前默认图不再挂接 Hybrid 闸门**；
- A 在 `paper_trading/integrations.py` 提供
  `TradingAgentsGraphDecisionProvider`，把默认图的最终五级评级通过独立的
  `RatingAllocationPolicy` 转成公共 `TradeIntent`；这层是集成政策，不属于
  C 的内部决策实现；
- WebUI 和集成代码只导入公共契约与协议，不导入 B、C 的内部类；
- 公共层的契约测试位于 `tests/test_extension_contracts.py`。
