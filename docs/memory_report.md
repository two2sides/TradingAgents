# TradingAgents 记忆系统优化报告

## 一、背景与目标

TradingAgents 原有记忆系统基于单个 Markdown 文件的时序检索（`TradingMemoryLog`），存在三个核心问题：

1. **检索能力弱**：仅按时间取最近 N 条，无法做语义相似度匹配，也无法按角色筛选
2. **无角色区分**：所有 Agent 共享同一段记忆文本，而各角色关注维度完全不同
3. **无法扩展**：记忆模块与图执行耦合，A 组定义的 `MemoryProvider` 协议无人实现

工作目标：实现 `MemoryProvider` 协议，构建基于 RAG 的语义记忆系统，为不同 Agent 角色提供差异化记忆能力。

## 二、系统架构

### 2.1 技术栈

| 组件 | 选型 | 说明 |
|---|---|---|
| 向量存储 | ChromaDB | 嵌入式运行，HNSW 索引，余弦空间 |
| 嵌入模型 | all-MiniLM-L6-v2（默认） | 384 维，离线优先，fallback 到网络下载 |
| 可选嵌入 | OpenAI text-embedding-3-small | 1536 维，需 API key |
| LLM 反思 | 复用项目 LLM 客户端 | 可选，为 None 时静默跳过 |

### 2.2 存储结构

每条决策被拆分为语义分块，独立嵌入后存入 ChromaDB：

| 分块类型 | 来源 | 截断 |
|---|---|---|
| `thesis` | PM 最终决策 rationale | 800 字符 |
| `market_context` | 决策时 OHLCV 快照 | 500 字符 |
| `portfolio_context` | 持仓状态 | 400 字符 |
| `debate_synthesis` | RM 投资计划 | 600 字符 |
| `reflection` | 事后 LLM 反思 | 600 字符 |

元数据字段：股票代码、Unix 时间戳、自动标签（18 类）、来源 agent、父记录 ID、结果质量。

去重机制：存入前用 thesis chunk 查询同股票记忆，余弦相似度 ≥ 0.95 则跳过。

### 2.3 检索管道

```
查询文本 → embedding → ChromaDB ANN 粗排
  → 元数据过滤（时间安全 / 股票 / 标签 / 分块类型）
  → 加权精排（相似度 × 时效性 × 结果质量 + 同源加成）
  → memory_id 去重 → top-k
```

排序公式：

```
score = w₁ × cos_sim + w₂ × exp(-0.05 × days) + w₃ × outcome_quality + source_boost
```

12 个角色有独立的权重配置。空头研究员重结果质量（0.50），市场分析师重相似度（0.65）。

## 三、Agent 角色记忆方案

经过对全部 12 个 Agent 的逐一分析，确定三条路径：

| Agent | 方式 | 写记忆 | 原因 |
|---|---|---|---|
| Portfolio Manager | Prompt 注入 | 最终决策 | 最终决策者，必须参考历史 |
| Research Manager | Prompt 注入 | 投资计划（中间产物） | 综合辩论，需历史模式 |
| Market Analyst | Tool | 报告（中间产物） | 有 tool loop，异常形态时查询 |
| Fundamentals Analyst | Tool | 报告（中间产物） | 有 tool loop，估值异常时查询 |
| News Analyst | Tool | 报告（中间产物） | 有 tool loop，宏观事件对比 |
| Bull Researcher | Tool | — | 有 tool loop，验证看多论据 |
| Bear Researcher | Tool | — | 有 tool loop，验证看空论据 |
| Sentiment Analyst | 不需要 | — | 职责是测量当下情绪 |
| Trader | 不需要 | — | 模拟环境无执行层记忆需求 |
| Aggressive Debator | 不需要 | — | 记忆会削弱角色立场多样性 |
| Conservative Debator | 不需要 | — | 同上 |
| Neutral Debator | 不需要 | — | 同上 |

**设计原则**：有 tool-calling 能力的 Agent → Tool（按需查询）；使用 `bind_structured` 或无 tools 的 Agent → Prompt 注入；风险辩论者不需要记忆（角色价值在于多样化视角）。

### 3.1 中间产物与结果传播

PM 决策和中间产物（Market / Fundamentals / News / RM 报告）分别作为独立记录存储，通过 `parent` 字段链接。当 PM 决策获得结果后，自动传播到所有子记录，使中间产物也能被后续检索命中并携带验证结果。

### 3.2 记忆 Tool

5 个 tool-based Agent 各自获得 `recall_historical_decisions(query)` 工具。LLM 发出自然语言查询，工具内部经由 provider 检索角色特化的记忆并返回格式化结果。LLM 的查询文本通过 `metadata["llm_query"]` 传递到检索管道，与角色模板混合提升语义匹配精度。

## 四、新增与修改文件

### 新增（核心模块）

| 文件 | 职责 |
|---|---|
| `tradingagents/extensions/memory/agent_profiles.py` | 12 角色检索配置 + 市场特征提取 |
| `tradingagents/extensions/memory/chunker.py` | 语义分块 + 标签分类 |
| `tradingagents/extensions/memory/embedder.py` | 双后端嵌入（离线优先） |
| `tradingagents/extensions/memory/store.py` | ChromaDB 存储 + 去重 + 结果传播 |
| `tradingagents/extensions/memory/retrieval.py` | 混合检索管道 |
| `tradingagents/extensions/memory/provider.py` | `MemoryProvider` 协议实现 |
| `tradingagents/extensions/memory/tools.py` | `recall_historical_decisions` tool 工厂 |

### 新增（测试）

| 文件 | 覆盖范围 |
|---|---|
| `tests/extension_memory_test/conftest.py` | 共享夹具 |
| `tests/extension_memory_test/test_agent_profiles.py` | 角色配置 + 特征提取 + retrieval kwargs |
| `tests/extension_memory_test/test_chunker.py` | 分块 + 标签 + debate_synthesis |
| `tests/extension_memory_test/test_embedder.py` | 嵌入模型 + 离线优先 |
| `tests/extension_memory_test/test_store.py` | CRUD + 去重 + 结果传播 |
| `tests/extension_memory_test/test_retrieval.py` | 检索管道 + 角色检索 + 同源加成 |
| `tests/extension_memory_test/test_provider.py` | 协议合规 + 去重 + 中间产物传播 |
| `tests/extension_memory_test/test_tools.py` | tool 工厂 + 角色覆盖 |
| `tests/extension_memory_test/test_agent_integration.py` | Agent 记忆接线验证 |

### 修改

| 文件 | 改动 |
|---|---|
| `tradingagents/extensions/memory/__init__.py` | 公开导出 + `set_active_provider` / `get_active_provider` 单例 |
| `tradingagents/default_config.py` | 4 个新配置键 |
| `tradingagents/graph/trading_graph.py` | 类级 `memory_provider`、resolve 双路径、角色预检索、中间产物存储、provider 单例注入 |
| `tradingagents/graph/propagation.py` | `create_initial_state` 增加 `**extra_state` |
| `tradingagents/agents/analysts/market_analyst.py` | 添加 memory tool |
| `tradingagents/agents/analysts/fundamentals_analyst.py` | 添加 memory tool |
| `tradingagents/agents/analysts/news_analyst.py` | 添加 memory tool |
| `tradingagents/agents/researchers/debate_common.py` | 添加 memory tool |
| `tradingagents/agents/managers/research_manager.py` | prompt 注入记忆 |
| `tests/test_memory_log.py` | mock 适配 |
| `requirements.txt` | `chromadb>=0.5`、`sentence-transformers>=3.0` |
| `pyproject.toml` | 可选依赖组 `[memory]` |

## 五、数据生命周期

```
propagate("NVDA", "2024-05-10")
  │
  ├─ _resolve_pending_entries
  │   ├─ 读取 markdown log pending entries
  │   ├─ yfinance 获取实际收益（raw + alpha）
  │   ├─ 通过 store.find_by_symbol_and_date() 查找 ChromaDB 中的 UUID
  │   ├─ record_outcome(memory_id, outcome)
  │   │   ├─ update_outcome(parent)          ← ChromaDB 元数据更新
  │   │   ├─ propagate_outcome(children)     ← 自动同步到中间产物
  │   │   └─ LLM 反思 → 追加 reflection chunk
  │   └─ update markdown log（遗留兼容）
  │
  └─ _run_graph
       ├─ set_active_provider(provider)       ← 模块级单例，checkpoint 安全
       ├─ retrieve(PM) + retrieve(RM)         ← prompt 注入角色预检索
       │
       ├─ graph.run()
       │   ├─ Market/Fundamentals/News/Bull/Bear 可调用 memory tool
       │   └─ PM 产出最终决策
       │
       ├─ record_decision(PM)                 ← chunk → embed → dedup → store
       └─ record_decision(child) × 4          ← 中间产物独立存储（带 parent 链接）
```


## 六、测试结果

- 纯 Python 测试（无需 chromadb/embedder）：全部通过
- 含 chromadb 的测试（store + 去重 + 结果传播）：全部通过
- 含 chromadb + embedding 的集成测试（retrieval + provider）：全部通过
- 遗留记忆日志回归测试（`test_memory_log.py`）：全部通过（Yahoo 403 系统性问题除外）
- 全部 101 个测试通过，零回归

## 七、使用方式

```python
from tradingagents.extensions.memory import EnhancedMemoryProvider
from tradingagents.graph.trading_graph import TradingAgentsGraph

config = DEFAULT_CONFIG.copy()
provider = EnhancedMemoryProvider(config, llm_client=quick_think_llm)
config["memory_provider"] = provider

ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate("NVDA", "2024-05-10")
```

不设置 `memory_provider` 时，行为与之前完全一致（走 `TradingMemoryLog`）。

## 八、后续优化方向

1. **市场环境感知**：在元数据中标记牛/熊市、波动率环境，检索时优先匹配同环境记忆
2. **数值特征混合检索**：追加轻量特征向量（10 维），与语义相似度加权混合，解决纯文本 embedding 无法捕捉数值形态的问题
3. **记忆合并**：长时间运行后定期扫描相似记录，合并为带统计摘要的抽象条目，减少 token 消耗
