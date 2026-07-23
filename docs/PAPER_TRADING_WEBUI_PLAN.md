# Paper Trading 与 Decision Lab 实施计划

> 负责人：职责一（市场回放、模拟 Broker、回测、WebUI 与集成）  
> 开发分支：`feature/paper-trading-webui`  
> 目标周期：约一周

## 1. 产品目标

本工作的目标不是给原项目增加几张结果图，而是把单次文本分析扩展成一个
**可执行、可审计、可回放、可比较**的交易实验环境。

最终演示应能回答四个问题：

1. Agent 在当时看到了什么信息，是否存在未来数据？
2. Agent 给出的目标仓位如何变成真实成交？
3. 账户净值为何发生变化，每一笔现金和持仓变化来自哪里？
4. 改变费用、滑点或策略实现后，结果发生了什么变化？

## 2. 不可破坏的边界

- A 只依赖 B 的 `MemoryProvider` 和 C 的 `DecisionProvider`，不导入其内部类。
- B、C 不依赖 Broker、SQLite、Streamlit 或 Plotly。
- 历史时点 `T` 的决策只能读取 `as_of <= T` 的市场和记忆数据。
- 默认在决策后的下一根可交易 K 线开盘价成交。
- Agent 只给目标仓位；现金、整数股、费用和成交约束由 Broker 决定。
- WebUI 只通过公共契约、运行存储和 A 的应用服务读取数据。

公共接口以以下文件为准：

- `tradingagents/extensions/contracts.py`
- `tradingagents/extensions/protocols.py`
- `docs/TEAM_PUBLIC_INTERFACES.md`

## 3. 目标架构

```text
WebUI / CLI
    │
    ▼
Backtest Application Service ─────── RunStore (SQLite)
    │                         └────── RunObserver
    ├── MarketDataProvider
    ├── MemoryProvider (B)
    ├── TradingAgentsGraphDecisionProvider
    │       ├── TradingAgentsGraph (C 的五级评级)
    │       └── RatingAllocationPolicy (评级 → 目标仓位)
    └── LedgerBroker
            └── immutable LedgerEntry events
```

建议文件布局：

```text
tradingagents/extensions/paper_trading/
├── ledger.py          # 不可变账户事件与状态重建
├── broker.py          # 目标仓位、费用、滑点、拒单和部分成交
├── market_data.py     # 内存历史数据与 yfinance 适配
├── backtest.py        # 时间推进和 B/C 编排
├── metrics.py         # 收益、风险、费用和基准
├── storage.py         # SQLite 运行档案
├── replay.py          # 执行层 What-if
├── demo.py            # 不依赖 LLM 的离线/演示 Provider
├── integrations.py    # 真实 Agent 图适配与显式评级仓位策略
└── observers.py       # 组合、记录进度事件

webui/
├── app.py             # Streamlit 入口与导航
├── pages/             # 运行、回放、对比页面
├── components/        # 图表、卡片、时间线
└── assets/styles.css  # Decision Lab 视觉系统
```

## 4. 分阶段交付

### M0：公共接口基线（已完成）

- [x] `ExecutionConfig` 明确费用、滑点和成交规则。
- [x] `benchmark_curves` 提供命名基准曲线。
- [x] `RunEvent` / `RunObserver` 解耦回测进度与界面。
- [x] 公共契约测试通过。

### M1：可审计 Broker

- [x] 用不可变 `LedgerEntry` 保存入金、成交、费用和估值事件。
- [x] 从账本重建现金、数量、平均成本和净值。
- [x] 将目标仓位转换为整数股买卖。
- [x] 支持手续费、最低费用、滑点、现金不足和明确拒单。
- [x] 测试账本可重放以及所有账户硬约束。

验收：给定同一组事件可恢复完全相同的账户；任何成交后现金和数量均不会为负。

### M2：历史回放与评价

- [x] 提供离线 DataFrame/MarketBar 数据源。
- [x] 提供 yfinance 历史数据适配器并缓存一次下载结果。
- [x] 按决策时钟依次调用 B、C 和 Broker。
- [x] 生成账户曲线、持仓历史、成交和警告。
- [x] 计算 Buy & Hold 基准、收益、回撤、Sharpe、换手率和费用侵蚀。
- [x] 通过时间边界测试，禁止未来 K 线进入 `MarketSnapshot`。

验收：固定输入与固定 Provider 得到确定性结果，并能解释每一笔成交价格来自哪根 K 线。

### M3：运行档案与实验能力

- [x] 使用 SQLite 保存请求、结果、账本和进度事件。
- [x] 支持运行列表、详情、删除和导出。
- [x] 支持复用已有 Agent 决策的“仅执行层 What-if”。
- [x] 明确区分完整 Agent 重跑与近似的执行层反事实实验。

验收：应用重启后仍可打开历史运行；修改费用后无需重新调用 LLM 即可生成对照结果。

### M4：Decision Lab WebUI

- [x] 建立深石墨、青蓝和琥珀色视觉系统。
- [x] 回测表单展示费用、滑点、时间窗口和 Provider 信息。
- [x] 实时展示阶段、进度、当前日期和降级警告。
- [x] 在同一时间轴展示 K 线、交易、仓位、净值和回撤。
- [x] 点击决策点查看 Agent 结论、记忆、成交和后续表现。
- [x] 支持两次运行并排比较以及执行层 What-if。
- [x] 提供无需 API Key 的确定性演示模式。

验收：首次使用者不阅读代码即可完成“发起回测 → 定位一笔交易 → 解释结果 → 比较实验”流程。

### M5：集成与演示加固

- [x] 用 B/C 的 Mock 完成全链路契约测试。
- [x] 接入 B 的 `EnhancedMemoryProvider` 和 C 所在的默认 `TradingAgentsGraph`。
- [x] 以请求级只读桥接把 B 的时点安全记忆注入 C，避免重复检索和重复写入。
- [x] 以显式、可配置的 `RatingAllocationPolicy` 把五级评级转换为目标仓位。
- [x] WebUI 可在确定性演示与真实 `TradingAgents + RAG` 两种引擎间切换。
- [x] 准备一个短周期快速演示数据集和一个完整结果样例。
- [x] 补充启动命令、故障提示和演示说明。

真实路径没有修改 Broker、账本、回测循环和运行存储。适配层位于 A 的
`paper_trading/integrations.py`，因此 C 仍然只负责最终评级，B 仍然只负责
记忆；评级如何变成可执行仓位是可单测、可替换的集成政策。

默认仓位政策使用 35% 单标的建仓上限，并同时应用 `1/N` 分散上限：

- `Buy`：只增不减，向建仓上限靠拢；
- `Overweight`：只增不减，目标为上限的 75%；
- `Hold`：保持当前仓位；
- `Underweight`：只减不增，目标不高于上限的 25%；
- `Sell`：目标仓位为 0。

这不是 C 的内部实现假设，而是 C 的五级评级与 A 的连续目标仓位契约之间必须
存在的明确执行政策。参数可以替换，默认值不会藏在 Prompt 或 WebUI 中。

## 5. WebUI 交互重点

WebUI 的核心页面命名为 **Decision Replay**，而不是普通“结果详情”。用户选择某个决策时间点后，页面联动展示：

```text
当时可见行情
→ 检索到的记忆
→ Agent 最终目标仓位与置信度
→ 下一交易时点的执行报价
→ 实际成交与费用
→ 账户状态和随后表现
```

图表本身也是输入控件。点击净值或交易标记后，详情面板切换到相应决策；筛选和详情更新不应重新运行回测。

## 6. 明确暂不实现

一周版本暂不承诺以下能力：

- 实盘券商连接和真实下单；
- 高频或分钟级撮合；
- 做空、融资、期权和复杂保证金；
- 盘口深度及市场冲击模型；
- 多用户权限与远程任务队列；
- 用自定义 JavaScript 前端替代 Streamlit。

这些内容以后可以通过新增 Broker 或 MarketDataProvider 实现，不改变 B/C 的接口。

## 7. 测试与提交策略

每个里程碑单独提交，至少执行：

```powershell
uv run --frozen pytest tests/test_extension_contracts.py -q
uv run --frozen pytest tests/paper_trading -q
uv run --frozen ruff check tradingagents/extensions webui tests/paper_trading
uv run --frozen ruff format --check tradingagents/extensions webui tests/paper_trading
```

涉及 UI 时另外执行一次本地 Streamlit smoke test。未经明确要求，功能分支只本地提交，不推送远端。
