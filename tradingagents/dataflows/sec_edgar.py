"""SEC EDGAR companyfacts vendor — free US fundamentals without yfinance.

Uses the public ``company_tickers.json`` + ``companyfacts`` JSON APIs.
Requires a descriptive User-Agent per SEC fair-access policy. Only US-listed
tickers with a CIK mapping are supported.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
from datetime import datetime
from functools import lru_cache
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import get_config
from .errors import NoMarketDataError
from .symbol_utils import normalize_symbol

logger = logging.getLogger(__name__)

# SEC fair-access: User-Agent must identify the caller (name + contact).
_UA = "TradingAgents Research student@example.com"
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


def _http_get_json(url: str) -> Any:
    req = Request(
        url,
        headers={
            "User-Agent": _UA,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json",
        },
    )
    with urlopen(req, timeout=45) as resp:
        raw = resp.read()
    try:
        text = gzip.decompress(raw).decode("utf-8")
    except OSError:
        text = raw.decode("utf-8")
    return json.loads(text)


@lru_cache(maxsize=1)
def _ticker_to_cik() -> dict[str, str]:
    data = _http_get_json(_TICKERS_URL)
    mapping: dict[str, str] = {}
    for row in data.values():
        ticker = str(row.get("ticker") or "").upper().strip()
        cik = str(row.get("cik_str") or "").zfill(10)
        if ticker and cik:
            mapping[ticker] = cik
    return mapping


def resolve_cik(symbol: str) -> str:
    canonical = normalize_symbol(symbol).upper()
    # Drop exchange suffixes that SEC map does not use (e.g. BRK-B stays).
    base = canonical.split(".")[0]
    mapping = _ticker_to_cik()
    cik = mapping.get(base) or mapping.get(canonical)
    if not cik:
        raise NoMarketDataError(
            symbol,
            canonical,
            "no SEC CIK mapping (SEC EDGAR covers US-listed tickers only)",
        )
    return cik


def _facts_cache_path(cik: str) -> str:
    cfg = get_config()
    cache_dir = os.path.join(cfg["data_cache_dir"], "sec_edgar")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"companyfacts_{cik}.json")


def load_companyfacts(symbol: str) -> dict[str, Any]:
    cik = resolve_cik(symbol)
    path = _facts_cache_path(cik)
    if os.path.exists(path):
        # Refresh if older than 7 days.
        age_days = (datetime.now().timestamp() - os.path.getmtime(path)) / 86400
        if age_days < 7:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    try:
        facts = _http_get_json(_FACTS_URL.format(cik=cik))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise NoMarketDataError(
            symbol, normalize_symbol(symbol), f"SEC companyfacts fetch failed: {exc}"
        ) from exc
    with open(path, "w", encoding="utf-8") as f:
        json.dump(facts, f)
    return facts


def _parse_as_of(curr_date: str | None) -> datetime | None:
    if not curr_date:
        return None
    return datetime.strptime(curr_date, "%Y-%m-%d")


def _concept_series(facts: dict[str, Any], concept: str) -> list[dict[str, Any]]:
    usgaap = (facts.get("facts") or {}).get("us-gaap") or {}
    dei = (facts.get("facts") or {}).get("dei") or {}
    node = usgaap.get(concept) or dei.get(concept)
    if not node:
        return []
    units = node.get("units") or {}
    # Prefer USD; else first unit bag.
    series = units.get("USD") or units.get("USD/shares") or next(iter(units.values()), [])
    return list(series or [])


def _filter_as_of(series: list[dict[str, Any]], as_of: datetime | None) -> list[dict[str, Any]]:
    if as_of is None:
        return series
    out = []
    for row in series:
        filed = row.get("filed")
        if not filed:
            continue
        try:
            filed_dt = datetime.strptime(filed[:10], "%Y-%m-%d")
        except ValueError:
            continue
        if filed_dt <= as_of:
            out.append(row)
    return out


def _latest(
    facts: dict[str, Any],
    concepts: list[str],
    *,
    as_of: datetime | None,
    forms: set[str] | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    forms = forms or {"10-K", "10-Q", "8-K"}
    best: tuple[str, dict[str, Any]] | None = None
    for concept in concepts:
        series = _filter_as_of(_concept_series(facts, concept), as_of)
        series = [r for r in series if r.get("form") in forms]
        if not series:
            continue
        row = sorted(series, key=lambda r: (r.get("filed", ""), r.get("end", "")))[-1]
        if best is None or (row.get("filed", ""), row.get("end", "")) > (
            best[1].get("filed", ""),
            best[1].get("end", ""),
        ):
            best = (concept, row)
    if best is None:
        return None, None
    return best[0], best[1]


def _fmt_money(val: Any) -> str:
    try:
        n = float(val)
    except (TypeError, ValueError):
        return str(val)
    abs_n = abs(n)
    if abs_n >= 1e12:
        return f"{n / 1e12:.2f}T"
    if abs_n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if abs_n >= 1e6:
        return f"{n / 1e6:.2f}M"
    return f"{n:,.0f}"


def _period_label(row: dict[str, Any]) -> str:
    bits = [
        f"end={row.get('end')}",
        f"form={row.get('form')}",
        f"filed={row.get('filed')}",
    ]
    if row.get("fp"):
        bits.append(f"fp={row['fp']}")
    return ", ".join(bits)


def get_fundamentals_sec(ticker: str, curr_date: str | None = None) -> str:
    """Overview snapshot from the latest filed SEC facts as of ``curr_date``."""
    canonical = normalize_symbol(ticker)
    as_of = _parse_as_of(curr_date)
    facts = load_companyfacts(ticker)
    entity = facts.get("entityName") or canonical

    lines = [
        f"# Company Fundamentals for {canonical}",
        f"# Vendor: sec_edgar",
        f"# Entity: {entity}",
        f"# As-of filter (filed <=): {curr_date or 'latest'}",
        "",
    ]

    overview = [
        ("Assets (latest)", ["Assets"]),
        ("Liabilities (latest)", ["Liabilities"]),
        ("Stockholders Equity", ["StockholdersEquity"]),
        (
            "Net Income (latest period)",
            ["NetIncomeLoss"],
        ),
        (
            "Revenue (latest period)",
            ["Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerExcludingAssessedTax"],
        ),
        (
            "Operating Cash Flow",
            ["NetCashProvidedByUsedInOperatingActivities"],
        ),
        (
            "Cash & Equivalents",
            ["CashAndCashEquivalentsAtCarryingValue"],
        ),
        (
            "Long-term Debt",
            ["LongTermDebt", "LongTermDebtNoncurrent"],
        ),
        (
            "Shares Outstanding",
            ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"],
        ),
    ]

    found = 0
    for label, concepts in overview:
        concept, row = _latest(facts, concepts, as_of=as_of)
        if row is None:
            continue
        found += 1
        lines.append(f"{label}: {_fmt_money(row.get('val'))}  [{concept}; {_period_label(row)}]")

    if found == 0:
        raise NoMarketDataError(ticker, canonical, "no SEC facts available as-of")

    lines.append("")
    lines.append(
        "Note: Values are from SEC XBRL companyfacts (US GAAP). "
        "Market multiples (PE, etc.) are not included."
    )
    return "\n".join(lines)


def _best_concept_series(
    facts: dict[str, Any],
    concepts: list[str],
    *,
    as_of: datetime | None,
    forms: set[str],
    limit: int,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Pick the concept alias whose latest filing is newest, return last ``limit`` rows."""
    best_c = None
    best_series: list[dict[str, Any]] = []
    best_key = ("", "")
    for c in concepts:
        series = _filter_as_of(_concept_series(facts, c), as_of)
        series = [r for r in series if r.get("form") in forms]
        if not series:
            continue
        series_sorted = sorted(series, key=lambda r: (r.get("end", ""), r.get("filed", "")))
        last = series_sorted[-1]
        key = (last.get("filed", ""), last.get("end", ""))
        if key > best_key:
            best_key = key
            best_c = c
            best_series = series_sorted[-limit:]
    return best_c, best_series


def _statement_table(
    facts: dict[str, Any],
    rows_spec: list[tuple[str, list[str]]],
    *,
    as_of: datetime | None,
    freq: str,
    limit: int = 4,
) -> str:
    if str(freq).lower().startswith("a"):
        prefer_forms: set[str] = {"10-K"}
        fallback_forms: set[str] = {"10-K", "10-Q"}
    else:
        prefer_forms = {"10-Q"}
        fallback_forms = {"10-Q", "10-K"}

    anchor_concept, anchor_series = _best_concept_series(
        facts,
        rows_spec[0][1],
        as_of=as_of,
        forms=prefer_forms,
        limit=limit,
    )
    if not anchor_series:
        anchor_concept, anchor_series = _best_concept_series(
            facts,
            rows_spec[0][1],
            as_of=as_of,
            forms=fallback_forms,
            limit=limit,
        )
    if not anchor_series:
        for _, concepts in rows_spec[1:]:
            anchor_concept, anchor_series = _best_concept_series(
                facts, concepts, as_of=as_of, forms=fallback_forms, limit=limit
            )
            if anchor_series:
                break
    if not anchor_series:
        return ""

    # Prefer shorter-duration quarterly frames when duplicates share an end date.
    forms_for_cells = prefer_forms if any(
        r.get("form") in prefer_forms for r in anchor_series
    ) else fallback_forms

    ends: list[str] = []
    for r in anchor_series:
        end = r.get("end")
        if end and end not in ends:
            ends.append(str(end))
    header = "| Item | " + " | ".join(ends) + " |"
    sep = "|---|---" + "|---" * (len(ends) - 1) + "|" if ends else "|---|---|"
    lines = [header, sep]

    for label, concepts in rows_spec:
        cells = []
        for end in ends:
            val = None
            candidates: list[tuple[str, Any]] = []
            for c in concepts:
                series = _filter_as_of(_concept_series(facts, c), as_of)
                matches = [
                    r
                    for r in series
                    if r.get("end") == end and r.get("form") in forms_for_cells
                ]
                if not matches:
                    matches = [
                        r
                        for r in series
                        if r.get("end") == end and r.get("form") in fallback_forms
                    ]
                if matches:
                    matches = sorted(matches, key=lambda r: r.get("filed", ""))
                    candidates.append((matches[-1].get("filed", ""), matches[-1].get("val")))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                val = candidates[-1][1]
            cells.append(_fmt_money(val) if val is not None else "—")
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    if anchor_concept:
        lines.append(f"\n_Anchor concept: {anchor_concept}_")
    return "\n".join(lines)


def get_balance_sheet_sec(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
) -> str:
    canonical = normalize_symbol(ticker)
    as_of = _parse_as_of(curr_date)
    facts = load_companyfacts(ticker)
    table = _statement_table(
        facts,
        [
            ("Cash", ["CashAndCashEquivalentsAtCarryingValue"]),
            ("Total Assets", ["Assets"]),
            ("Total Liabilities", ["Liabilities"]),
            ("Stockholders Equity", ["StockholdersEquity"]),
            ("Long-term Debt", ["LongTermDebt", "LongTermDebtNoncurrent"]),
            ("Current Assets", ["AssetsCurrent"]),
            ("Current Liabilities", ["LiabilitiesCurrent"]),
        ],
        as_of=as_of,
        freq=freq,
    )
    if not table:
        raise NoMarketDataError(ticker, canonical, "no SEC balance sheet rows as-of")
    return (
        f"# Balance Sheet for {canonical}\n"
        f"# Vendor: sec_edgar | freq={freq} | as_of={curr_date or 'latest'}\n\n"
        f"{table}"
    )


def get_cashflow_sec(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
) -> str:
    canonical = normalize_symbol(ticker)
    as_of = _parse_as_of(curr_date)
    facts = load_companyfacts(ticker)
    table = _statement_table(
        facts,
        [
            (
                "Operating Cash Flow",
                ["NetCashProvidedByUsedInOperatingActivities"],
            ),
            (
                "Investing Cash Flow",
                ["NetCashProvidedByUsedInInvestingActivities"],
            ),
            (
                "Financing Cash Flow",
                ["NetCashProvidedByUsedInFinancingActivities"],
            ),
            (
                "CapEx",
                ["PaymentsToAcquirePropertyPlantAndEquipment"],
            ),
        ],
        as_of=as_of,
        freq=freq,
    )
    if not table:
        raise NoMarketDataError(ticker, canonical, "no SEC cashflow rows as-of")
    return (
        f"# Cash Flow for {canonical}\n"
        f"# Vendor: sec_edgar | freq={freq} | as_of={curr_date or 'latest'}\n\n"
        f"{table}"
    )


def get_income_statement_sec(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
) -> str:
    canonical = normalize_symbol(ticker)
    as_of = _parse_as_of(curr_date)
    facts = load_companyfacts(ticker)
    table = _statement_table(
        facts,
        [
            (
                "Revenue",
                [
                    "Revenues",
                    "SalesRevenueNet",
                    "RevenueFromContractWithCustomerExcludingAssessedTax",
                ],
            ),
            ("Gross Profit", ["GrossProfit"]),
            ("Operating Income", ["OperatingIncomeLoss"]),
            ("Net Income", ["NetIncomeLoss"]),
            (
                "EPS Diluted",
                ["EarningsPerShareDiluted"],
            ),
        ],
        as_of=as_of,
        freq=freq,
    )
    if not table:
        raise NoMarketDataError(ticker, canonical, "no SEC income statement rows as-of")
    return (
        f"# Income Statement for {canonical}\n"
        f"# Vendor: sec_edgar | freq={freq} | as_of={curr_date or 'latest'}\n\n"
        f"{table}"
    )
