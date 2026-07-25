"""B-owned implementations of memory storage, retrieval, and context assembly.

Public API
----------
- ``EnhancedMemoryProvider`` — drop-in ``MemoryProvider`` implementation
  backed by ChromaDB with hybrid (vector + metadata) RAG retrieval and
  role-aware agent memory profiles.
- ``get_active_provider()`` / ``set_active_provider()`` — module-level
  singleton so agent nodes can access the provider without putting it
  in the LangGraph state dict (which would break checkpoint serialisation).

Quick start::

    from tradingagents.extensions.memory import EnhancedMemoryProvider

    provider = EnhancedMemoryProvider(config, llm_client=quick_think_llm)
    config["memory_provider"] = provider
    ta = TradingAgentsGraph(debug=True, config=config)
"""

from .provider import EnhancedMemoryProvider

# Module-level singleton so agent nodes can get the current provider
# without going through the LangGraph state (checkpoint-safe).
_active_provider = None


def set_active_provider(provider):
    global _active_provider
    _active_provider = provider


def get_active_provider():
    return _active_provider


__all__ = [
    "EnhancedMemoryProvider",
    "get_active_provider",
    "set_active_provider",
]
