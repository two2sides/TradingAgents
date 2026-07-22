"""A-owned implementations of market replay, broker, backtest, and UI services."""

from .backtest import HistoricalBacktestRunner
from .broker import LedgerBroker
from .demo import DemoMemoryProvider, MovingAverageDecisionProvider, generate_demo_market_data
from .ledger import AccountLedger, LedgerEntry
from .market_data import HistoricalMarketDataProvider, MarketDataUnavailable
from .observers import CompositeRunObserver, EventCollector
from .replay import ReplayDecisionProvider, run_execution_what_if
from .service import BacktestApplicationService
from .storage import RunStoreObserver, SQLiteRunStore
from .view_models import DecisionReplayItem, build_decision_replay

__all__ = [
    "AccountLedger",
    "BacktestApplicationService",
    "CompositeRunObserver",
    "DemoMemoryProvider",
    "DecisionReplayItem",
    "EventCollector",
    "HistoricalBacktestRunner",
    "HistoricalMarketDataProvider",
    "LedgerBroker",
    "LedgerEntry",
    "MarketDataUnavailable",
    "MovingAverageDecisionProvider",
    "ReplayDecisionProvider",
    "RunStoreObserver",
    "SQLiteRunStore",
    "build_decision_replay",
    "generate_demo_market_data",
    "run_execution_what_if",
]
