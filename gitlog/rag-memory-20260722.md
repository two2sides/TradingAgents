# RAG 记忆增强实现报告

## 一、本轮工作的目标

将 TradingAgents 的记忆系统从基于 Markdown 文件的时序检索，升级为基于 RAG（Retrieval-Augmented Generation）的语义检索系统。实现 A 组定义的 `MemoryProvider` 协议，让不同 Agent 角色获得与其职责匹配的差异化记忆，并支持中间产物独立存储与结果传播。

## 二、系统架构

### 存储层

**ChromaDB**（HNSW 索引，余弦空间）。每条记忆拆为语义分块嵌入后存入：

| 分块类型 | 来源 | 截断 |
|---|---|---|
| `thesis` | PM 最终决策 | 800 字符 |
| `market_context` | 决策时 OHLCV 快照 | 500 字符 |
| `portfolio_context` | 持仓状态 | 400 字符 |
| `debate_synthesis` | RM 投资计划 | 600 字符 |
| `reflection` | 事后 LLM 反思 | 600 字符 |

元数据：股票代码、Unix 时间戳、自动标签（18 类）、来源 agent、父记录 ID、结果质量。

两种记录类型：
- **PM 决策记录**（parent）——每次 `propagate()` 产一条
- **中间产物记录**（child）——Market / Fundamentals / News / RM 分析报告，通过 `parent` 链接，结果传播时自动同步 outcome

去重：存入前用 thesis chunk embedding 查询同股票记忆，余弦相似度 ≥ 0.95 则跳过。

### 检索管道

```
查询文本 → embedding → ChromaDB ANN 粗排
  → 元数据过滤（时间安全 / 股票 / 标签 / 分块类型）
  → 加权精排 → memory_id 去重 → top-k
```

排序公式：

```
score = w₁×cos_sim + w₂×exp(-0.05×days) + w₃×outcome_quality + source_boost
```

12 个角色有独立的权重配置。同源记录获得 +0.05 加成（如 Market Analyst 查询时，来自 `market_analyst` 的记录略微优先）。

### 嵌入模型

双后端：本地 `all-MiniLM-L6-v2`（384 维，离线优先）和 OpenAI `text-embedding-3-small`（1536 维）。离线优先策略：先以 `local_files_only=True` 加载缓存模型，未命中时回退到网络下载。

---

## 三、Agent 角色长期记忆方案

| Agent | 方式 | 写记忆？ |
|---|---|---|
| **Portfolio Manager** | Prompt 注入 `past_context` | 存最终决策 |
| **Research Manager** | Prompt 注入 `memory_research_manager` | 存投资计划（中间产物） |
| **Market Analyst** | Tool `recall_historical_decisions` | 存报告（中间产物） |
| **Fundamentals Analyst** | Tool `recall_historical_decisions` | 存报告（中间产物） |
| **News Analyst** | Tool `recall_historical_decisions` | 存报告（中间产物） |
| **Bull Researcher** | Tool `recall_historical_decisions` | — |
| **Bear Researcher** | Tool `recall_historical_decisions` | — |
| **Sentiment Analyst** | 不需要 | — |
| **Trader** | 不需要 | — |
| **Aggressive / Conservative / Neutral Debator** | 不需要 | — |

决策逻辑：
- 有 tool-calling 能力的 Agent → Tool 路径（按需查询，不浪费 token）
- 使用 `bind_structured` 或无 tools 的 Agent → Prompt 注入路径
- 风险辩论者的角色价值在于多样化视角，注入记忆反而削弱差异
- 中间产物（Market / Fundamentals / News / RM 报告）作为独立记录存储，链接到 PM 决策，结果传播时自动同步

---

## 四、数据生命周期

```
propagate("NVDA", "2024-05-10")
  │
  ├─ _resolve_pending_entries
  │   └─ record_outcome(memory_id)
  │       ├─ update_outcome(parent)
  │       ├─ propagate_outcome(children)   ← 自动传播到中间产物
  │       └─ LLM 反思 → 追加 reflection chunk
  │
  └─ _run_graph
       ├─ retrieve(PM) + retrieve(RM)      ← prompt 注入角色预检索
       ├─ provider → state                 ← 供 tool-based agent 使用
       │
       ├─ graph.run()
       │   ├─ 分析师调用 tool 查询记忆
       │   └─ PM 产出最终决策
       │
       └─ record_decision(PM)
            ├─ chunk → embed → dedup 检查
            └─ record_decision(child) × 4   ← 中间产物独立存储
```

---

## 五、新增与修改的文件清单

### 新增文件

| 文件 | 说明 |
|---|---|
| `tradingagents/extensions/memory/agent_profiles.py` | 12 角色检索配置 + 市场特征提取 |
| `tradingagents/extensions/memory/chunker.py` | 语义分块 + 标签分类 |
| `tradingagents/extensions/memory/embedder.py` | 双后端嵌入（离线优先） |
| `tradingagents/extensions/memory/store.py` | ChromaDB 存储 + 去重 + 结果传播 |
| `tradingagents/extensions/memory/retrieval.py` | 混合检索管道 + 同源加成 |
| `tradingagents/extensions/memory/provider.py` | `MemoryProvider` 协议实现 |
| `tradingagents/extensions/memory/tools.py` | `recall_historical_decisions` tool 工厂 |
| `tests/extension_memory_test/__init__.py` | 测试包 |
| `tests/extension_memory_test/conftest.py` | 共享夹具 + 依赖检测 |
| `tests/extension_memory_test/test_agent_profiles.py` | 角色配置 + 特征提取 |
| `tests/extension_memory_test/test_chunker.py` | 分块 + 标签（含 debate_synthesis 和 debate 标签） |
| `tests/extension_memory_test/test_embedder.py` | 嵌入模型（含离线优先） |
| `tests/extension_memory_test/test_store.py` | 存储 CRUD + 去重 + 结果传播 |
| `tests/extension_memory_test/test_retrieval.py` | 检索管道 + 角色检索 + 同源加成 |
| `tests/extension_memory_test/test_provider.py` | 协议合规 + 集成 + 去重 + 中间产物传播 |
| `tests/extension_memory_test/test_tools.py` | tool 工厂 + 角色覆盖 |
| `tests/extension_memory_test/test_agent_integration.py` | Agent 记忆接线验证 |
| `gitlog/rag-memory-20260722.md` | 本报告 |

### 修改文件

| 文件 | 改动说明 |
|---|---|
| `tradingagents/extensions/memory/__init__.py` | 公开导出 `EnhancedMemoryProvider` |
| `tradingagents/default_config.py` | 新增 `memory_db_path`、`memory_embedding`、`memory_embedding_model`、`memory_provider` |
| `tradingagents/graph/trading_graph.py` | 类级 `memory_provider`、双路径 resolve、角色预检索（2 角色）、`_record_decision_via_provider` 返回 memory_id、`_record_intermediate_analyses`（4 个中间产物）、provider 写入 state、去重 |
| `tradingagents/graph/propagation.py` | `create_initial_state` 增加 `**extra_state` |
| `tradingagents/agents/analysts/market_analyst.py` | provider 存在时追加 memory tool |
| `tradingagents/agents/analysts/fundamentals_analyst.py` | 同上 |
| `tradingagents/agents/analysts/news_analyst.py` | 同上 |
| `tradingagents/agents/researchers/debate_common.py` | `run_debate_turn` 追加 memory tool（不占辩论预算） |
| `tradingagents/agents/managers/research_manager.py` | prompt 注入 `memory_research_manager` |
| `tests/test_memory_log.py` | 3 处 mock 增加 `memory_provider = None` |
| `requirements.txt` | `chromadb>=0.5`、`sentence-transformers>=3.0` |
| `pyproject.toml` | 可选依赖组 `[memory]` |

---

## 六、验证结果

- 纯 Python 测试（无需 chromadb/embedder）：全部通过
- 含 chromadb 的测试（store + dedup + 结果传播）：全部通过
- 含 chromadb + embedding 的测试（retrieval + provider 集成）：全部通过
- 已有记忆日志回归测试（test_memory_log.py）：全部通过（Yahoo 403 的系统性问题除外）
- 所有新增测试零回归

---

## 七、A/C 开发者注意事项

### 使用方式

```python
from tradingagents.extensions.memory import EnhancedMemoryProvider

provider = EnhancedMemoryProvider(config, llm_client=quick_think_llm)
config["memory_provider"] = provider
ta = TradingAgentsGraph(debug=True, config=config)
```

不设置 `memory_provider` 时，行为与之前完全一致（走 `TradingMemoryLog`）。

### 新增角色

- Tool 角色：在 `agent_profiles.py` 的 `PROFILES` 加配置行，在 `tools.py` 的 `_ROLE_TOOL_DESCRIPTIONS` 加描述，在 Agent 节点中加 tool 创建逻辑。
- Prompt 注入角色：在 `agent_profiles.py` 加配置，在 `_retrieve_agent_memories` 的 `roles` 列表加角色名，在 Agent prompt 中消费对应字段。
- 不需要修改协议或存储层。
