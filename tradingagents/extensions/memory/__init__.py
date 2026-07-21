"""B-owned implementations of memory storage, retrieval, and context assembly.

Public API
----------
- ``EnhancedMemoryProvider`` — drop-in ``MemoryProvider`` implementation
  backed by ChromaDB with hybrid (vector + metadata) RAG retrieval and
  role-aware agent memory profiles.

Quick start::

    from tradingagents.extensions.memory import EnhancedMemoryProvider

    provider = EnhancedMemoryProvider(config, llm_client=quick_think_llm)
    ctx = provider.retrieve(query)            # MemoryContext
    ref = provider.record_decision(record)     # MemoryReference
    provider.record_outcome(ref, outcome)      # None

To wire into the graph, pass the provider through config::

    config["memory_provider"] = provider
    ta = TradingAgentsGraph(debug=True, config=config)
"""

from .provider import EnhancedMemoryProvider

__all__ = ["EnhancedMemoryProvider"]
