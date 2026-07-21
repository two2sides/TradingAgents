"""Static unavailable vendor — used when Yahoo is disabled and no paid key is set.

Returns explicit sentinel strings so agents degrade honestly instead of calling
yfinance (which fails hard on some networks).
"""

from __future__ import annotations


def _msg(kind: str, symbol: str | None = None) -> str:
    target = f" for '{symbol}'" if symbol else ""
    return (
        f"DATA_UNAVAILABLE: {kind}{target} is not available on this install. "
        "Yahoo Finance / yfinance is disabled by default; set "
        "ALPHA_VANTAGE_API_KEY and switch fundamental_data / "
        "get_insider_transactions to alpha_vantage to enable. "
        "Do not invent figures."
    )


def get_fundamentals_unavailable(ticker: str, curr_date: str = None) -> str:
    return _msg("company fundamentals", ticker)


def get_balance_sheet_unavailable(
    ticker: str, freq: str = "annual", curr_date: str = None
) -> str:
    return _msg("balance sheet", ticker)


def get_cashflow_unavailable(
    ticker: str, freq: str = "annual", curr_date: str = None
) -> str:
    return _msg("cash flow statement", ticker)


def get_income_statement_unavailable(
    ticker: str, freq: str = "annual", curr_date: str = None
) -> str:
    return _msg("income statement", ticker)


def get_insider_transactions_unavailable(ticker: str) -> str:
    return _msg("insider transactions", ticker)
