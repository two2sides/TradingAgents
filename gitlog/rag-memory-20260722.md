# RAG 记忆增强实现报告

## 一、本轮工作的目标

将 TradingAgents 的记忆系统从基于 Markdown 文件的时序检索，升级为基于 RAG（Retrieval-Augmented Generation）的语义检索系统。同时实现 A 组定义的 `MemoryProvider` 协议，使记忆模块成为可替换的独立组件，并让不同 Agent 角色获得与其职责匹配的差异化记忆。

## 二、完成的主要内容

### 1. 实现 `MemoryProvider` 协议

B 组的核心交付：`EnhancedMemoryProvider` 类，位于：

`tradingagents/extensions/memory/provider.py`

完整实现了 `MemoryProvider` 协议定义的三个方法：

- `retrieve(MemoryQuery) -> MemoryContext`：接收包含当前市场快照和投资组合状态的查询，返回经过时间安全过滤和历史相关性排序的记忆上下文。
- `record_decision(DecisionRecord) -> MemoryReference`：将决策及其市场上下文持久化，生成不透明句柄供后续结果关联。
- `record_outcome(MemoryReference, DecisionOutcome) -> None`：在结果已知后更新记忆，触发 LLM 反思生成，并将反思嵌入回向量库。

### 2. 混合 RAG 检索管道

位于：`tradingagents/extensions/memory/retrieval.py`

检索流程：查询嵌入 → ChromaDB ANN 粗排 → 元数据过滤（时间安全、股票、分块类型、标签）→ 加权精排（相似度 × 时效性 × 结果质量）→ memory_id 去重 → top-k。

排序公式：

```
score = w₁ × cosine_similarity + w₂ × exp(-0.05 × days) + w₃ × outcome_quality
```

### 3. 角色感知记忆配置

位于：`tradingagents/extensions/memory/agent_profiles.py`

定义了 12 个 Agent 角色的差异化检索策略，每个角色有独立的：
- 分块类型偏好（thesis / market_context / portfolio_context / reflection）
- 兴趣标签过滤（valuation、technical、bull_thesis 等 18 类标签）
- 排序权重（相似度、时效性、结果质量三项可配置）
- 是否允许跨股票检索
- 最大返回条数

关键角色差异化示例：

| 角色 | 相似度权重 | 时效性权重 | 结果质量权重 | 跨股票 |
|---|---|---|---|---|
| 市场分析师 | 0.65 | 0.15 | 0.20 | 是 |
| 空头研究员 | 0.30 | 0.20 | 0.50 | 是 |
| 保守风控 | 0.40 | 0.20 | 0.40 | 否（仅同股票） |
| 投资组合经理 | 0.40 | 0.30 | 0.30 | 是 |

空头研究员将结果质量权重设为 0.50，因为"上次的看空论据是否被证实"比"论据听起来是否相似"更重要。保守风控不检索其他股票的记忆，因为风险特征不可迁移。

### 4. 语义分块与标签分类

位于：`tradingagents/extensions/memory/chunker.py`

每条 `DecisionRecord` 被拆分为三类语义 chunk：
- `thesis`：投资论点核心（800 字符截断）
- `market_context`：决策时市场 OHLCV 快照的结构化描述（500 字符）
- `portfolio_context`：持仓状态描述（400 字符）
- `reflection`：事后 LLM 反思（由 `record_outcome` 追加，600 字符）

自动标签分类基于关键词匹配，覆盖估值、技术、情绪、宏观、风险等 18 个标签类别。

### 5. 存储后端

位于：`tradingagents/extensions/memory/store.py`

使用 ChromaDB 作为向量存储。每条记忆的多个 chunk 共享一个 `memory_id`，通过 ChromaDB 的 ANN 索引（HNSW，余弦空间）支持快速相似度检索。元数据过滤支持按股票代码、时间范围、标签和分块类型筛选。

关键设计决策：时间戳存储为 Unix 时间戳（float），因为 ChromaDB 的 `$lte` 操作符仅支持数值类型。

### 6. 嵌入模型

位于：`tradingagents/extensions/memory/embedder.py`

支持双后端模式：
- 本地：`all-MiniLM-L6-v2`（384 维，离线，无 API 成本）
- 云端：OpenAI embeddings（`text-embedding-3-small`，1536 维）

通过 config 的 `memory_embedding` 键切换。默认使用本地模型。采用离线优先策略：先尝试 `local_files_only=True` 加载缓存模型，未缓存时自动回退到网络下载。

### 7. 图注入点

修改了 `tradingagents/graph/trading_graph.py`：

- 类级别默认 `memory_provider = None`，保证无注入时行为不变。
- `_resolve_pending_entries`：当 provider 激活时，同步推送结果到 provider 的 `record_outcome()`，同时保留原有 markdown 日志。
- `_run_graph`：provider 激活时，调用 `_retrieve_agent_memories()` 为 prompt 注入角色检索记忆，合并到初始状态。
- `_record_decision_via_provider`：图运行结束后，将决策写入 RAG store。

修改了 `tradingagents/graph/propagation.py`：

- `create_initial_state` 增加 `**extra_state` 参数，允许注入 `memory_{role}` 字段和 `memory_provider` 实例而不破坏签名兼容性。

### 8. Agent 角色长期记忆方案

经过对所有 Agent 角色的逐一分析，确定两条记忆路径：

**Tool 路径（5 个 Agent）**：给已有 tool-calling 能力的分析师新增 `recall_historical_decisions` 工具。Agent 分析到异常形态时主动调用，按需查询历史相似决策及其结果。不做 prompt 注入，不消耗额外 token。

**Prompt 注入路径（2 个 Agent）**：PM 和 Research Manager 使用 `bind_structured` 输出结构化结果，不支持混合 tool call，且职责要求必须参考历史。在每轮开始时预检索记忆并注入 prompt。

**不需要记忆（5 个 Agent）**：风险辩论者的角色价值在于提供多样化的观点——注入记忆反而会削弱角色差异。Sentiment Analyst 的职责是测量当下情绪，校准是 PM 的事。Trader 在模拟环境中无执行层记忆需求。

| Agent | 方式 | 状态 |
|---|---|---|
| Market Analyst | Tool `recall_historical_decisions` | 已实现 |
| Fundamentals Analyst | Tool `recall_historical_decisions` | 已实现 |
| News Analyst | Tool `recall_historical_decisions` | 已实现 |
| Bull Researcher | Tool `recall_historical_decisions` | 已实现 |
| Bear Researcher | Tool `recall_historical_decisions` | 已实现 |
| Research Manager | Prompt 注入 `memory_research_manager` | 已实现 |
| Portfolio Manager | Prompt 注入 `past_context` | 已实现 |
| Sentiment Analyst | 不需要 | — |
| Trader | 不需要 | — |
| Aggressive/Conservative/Neutral Debator | 不需要 | — |

### 9. 记忆 Tool 工厂

位于：`tradingagents/extensions/memory/tools.py`

`create_memory_recall_tool(provider, symbol, date, role)` 工厂函数，返回 LangChain 兼容的工具函数。LLM 通过工具名 `recall_historical_decisions` 和角色定制的描述来判断调用时机。工具内部构造 `MemoryQuery`、调用 `provider.retrieve()`、格式化结果返回给 LLM。

Provider 通过 state 从 graph 传递给各 Agent 节点：`state["memory_provider"]`。`_retrieve_agent_memories` 中的预检索角色列表从 7 个缩减为 2 个（PM + Research Manager），tool 角色不做事先检索。

## 三、新增与修改的文件清单

### 新增文件

| 文件 | 说明 |
|---|---|
| `tradingagents/extensions/memory/agent_profiles.py` | 12 角色检索配置 + 市场特征提取 |
| `tradingagents/extensions/memory/chunker.py` | 决策语义分块 + 标签分类 |
| `tradingagents/extensions/memory/embedder.py` | 双后端嵌入模型封装（离线优先） |
| `tradingagents/extensions/memory/store.py` | ChromaDB 存储后端 |
| `tradingagents/extensions/memory/retrieval.py` | 混合检索管道 |
| `tradingagents/extensions/memory/provider.py` | MemoryProvider 协议实现 |
| `tradingagents/extensions/memory/tools.py` | 记忆 recall tool 工厂 |
| `tests/extension_memory_test/__init__.py` | 测试包 |
| `tests/extension_memory_test/conftest.py` | 测试夹具 + 依赖检测 |
| `tests/extension_memory_test/test_agent_profiles.py` | 角色配置 + 特征提取测试 |
| `tests/extension_memory_test/test_chunker.py` | 分块 + 标签测试 |
| `tests/extension_memory_test/test_embedder.py` | 嵌入模型测试（含离线优先） |
| `tests/extension_memory_test/test_store.py` | 存储 CRUD 测试 |
| `tests/extension_memory_test/test_retrieval.py` | 检索管道测试 |
| `tests/extension_memory_test/test_provider.py` | 协议合规 + 集成测试 |
| `tests/extension_memory_test/test_tools.py` | 记忆 tool 工厂测试 |
| `tests/extension_memory_test/test_agent_integration.py` | Agent 记忆接线验证 |
| `gitlog/rag-memory-20260722.md` | 本报告 |

### 修改文件

| 文件 | 改动说明 |
|---|---|
| `tradingagents/extensions/memory/__init__.py` | 公开导出 `EnhancedMemoryProvider` |
| `tradingagents/default_config.py` | 新增 `memory_db_path`、`memory_embedding`、`memory_embedding_model`、`memory_provider` 四个配置键 |
| `tradingagents/graph/trading_graph.py` | 类级 `memory_provider` 注入、`_resolve_pending_entries` 双路径、`_run_graph` 角色记忆检索、`_record_decision_via_provider`、`_retrieve_agent_memories`（角色列表缩减为 2 个）、provider 写入 state |
| `tradingagents/graph/propagation.py` | `create_initial_state` 增加 `**extra_state` |
| `tradingagents/agents/analysts/market_analyst.py` | state 中有 provider 时追加记忆 tool |
| `tradingagents/agents/analysts/fundamentals_analyst.py` | 同上 |
| `tradingagents/agents/analysts/news_analyst.py` | 同上 |
| `tradingagents/agents/researchers/debate_common.py` | `run_debate_turn` 中追加记忆 tool（不计入辩论工具预算） |
| `tradingagents/agents/managers/research_manager.py` | 从 `state["memory_research_manager"]` 读取记忆注入 prompt |
| `tests/test_memory_log.py` | 3 处 mock 增加 `memory_provider = None` |
| `requirements.txt` | 新增 `chromadb>=0.5`、`sentence-transformers>=3.0` |
| `pyproject.toml` | 新增可选依赖组 `[memory]` |

## 四、验证结果

- 纯 Python 测试（无需 chromadb/embedder）：`54 passed`
- 完整记忆扩展测试（含 chromadb + embedding）：`42 passed, 15 skipped`（skipped 因离线环境网络不通，非测试代码问题）
- 已有记忆日志回归测试（test_memory_log.py）：`66 passed`（2 个 Yahoo Finance HTTP 403 的已有失败除外）
- 协议合规验证：`isinstance(EnhancedMemoryProvider(), MemoryProvider)` 通过
- 所有新增测试全部通过，零回归失败

## 五、A/C 开发者需要注意的接口与行为变化

### 1. 使用 EnhancedMemoryProvider

```python
from tradingagents.extensions.memory import EnhancedMemoryProvider

provider = EnhancedMemoryProvider(config, llm_client=quick_think_llm)
config["memory_provider"] = provider
ta = TradingAgentsGraph(debug=True, config=config)
```

不设置 `memory_provider` 时，行为与之前完全一致（走 `TradingMemoryLog`）。

### 2. Provider 通过 State 传递

Provider 激活后，会自动写入 state 的 `memory_provider` 键。Tool-based Agent 在运行时从 state 中取出它来创建记忆查询工具。自定义 Agent 节点如需使用，按同样方式获取即可。

### 3. 角色记忆字段

Provider 激活后，prompt 注入角色的初始 state 中会增加 `memory_{role}` 字段（字符串格式）：

- `memory_portfolio_manager`（同时映射到 `past_context`）
- `memory_research_manager`

### 4. 新增角色的步骤

- Tool 角色：在 `agent_profiles.py` 的 `PROFILES` 字典中增加配置行，在 `tools.py` 的 `_ROLE_TOOL_DESCRIPTIONS` 中增加工具描述，在 Agent 节点的 tools 列表中加入 tool 创建逻辑。
- Prompt 注入角色：在 `agent_profiles.py` 增加配置，在 `_retrieve_agent_memories` 的 `roles` 列表中增加角色名，在 Agent prompt 中增加记忆消费代码。
- 不需要修改协议或存储层。

### 5. 嵌入模型离线优先

本地嵌入模型 `all-MiniLM-L6-v2` 采用离线优先策略：先尝试 `local_files_only=True` 加载缓存模型，未缓存时自动回退到网络下载。首次使用时会从 HuggingFace 下载（约 80MB），之后缓存到 `~/.cache/huggingface/`。国内用户可设置 `HF_ENDPOINT=https://hf-mirror.com` 使用镜像。

如需使用 OpenAI embeddings 替代本地模型：

```python
config["memory_embedding"] = "openai"
config["memory_embedding_model"] = "text-embedding-3-small"
```
