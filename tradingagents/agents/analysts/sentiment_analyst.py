"""Sentiment analyst — news + lexicon prior (single reliable data path).

Pre-fetches ticker news via the configured ``news_data`` vendor (default:
Google News RSS) and injects a deterministic lexicon score into the prompt.
Does not call StockTwits / Reddit (those endpoints are unreliable on many
networks and are intentionally out of the default path).
"""

from datetime import datetime, timedelta

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.schemas import SentimentReport, render_sentiment_report
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_news,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_with_metadata,
)
from tradingagents.agents.utils.claim_sidecar import append_claim_index
from tradingagents.extensions.decision.credibility import invoke_evidenced
from tradingagents.extensions.decision.credibility.claims import claims_from_invocation


def _seven_days_back(trade_date: str) -> str:
    return (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")


def create_sentiment_analyst(llm):
    """Create a sentiment analyst node for the trading graph."""
    structured_llm = bind_structured(llm, SentimentReport, "Sentiment Analyst")

    def sentiment_analyst_node(state):
        ticker = state["company_of_interest"]
        run_id = state.get("run_id", "legacy-run")
        end_date = state["trade_date"]
        start_date = _seven_days_back(end_date)
        instrument_context = get_instrument_context_from_state(state)

        news_block, news_event = invoke_evidenced(
            run_id=run_id,
            producer_node="sentiment_prefetch",
            tool_name="get_news",
            arguments={"ticker": ticker, "start_date": start_date, "end_date": end_date},
            call=lambda: get_news.func(ticker, start_date, end_date),
        )

        from tradingagents.extensions.decision.sentiment_lexicon import score_documents

        lexicon, lexicon_event = invoke_evidenced(
            run_id=run_id,
            producer_node="sentiment_prefetch",
            tool_name="score_documents",
            arguments={"document_count": 1},
            call=lambda: score_documents([str(news_block)]),
        )
        lexicon_block = (
            f"[artifact_id={lexicon_event['payload']['artifact_id']}] "
            f"Lexicon sentiment score={lexicon.score:.3f} ({lexicon.label}); "
            f"pos_hits={lexicon.positive_hits}, neg_hits={lexicon.negative_hits}. "
            "Treat this as a deterministic numeric prior; reconcile with source text."
        )

        system_message = _build_system_message(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            news_block=(
                f"[artifact_id={news_event['payload']['artifact_id']}]\n{news_block}"
            ),
            lexicon_block=lexicon_block,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}"
                    "\n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=end_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        formatted_messages = prompt.format_messages(messages=state["messages"])

        invocation = invoke_structured_with_metadata(
            structured_llm,
            llm,
            formatted_messages,
            render_sentiment_report,
            "Sentiment Analyst",
        )
        report_text = invocation.text
        claims = claims_from_invocation(
            run_id=run_id,
            agent="Sentiment Analyst",
            stage="sentiment",
            invocation=invocation,
            audit_events=state.get("audit_events", []) + [news_event, lexicon_event],
            trade_date=end_date,
        )
        report_text = append_claim_index(report_text, claims)

        return {
            "messages": [AIMessage(content=report_text)],
            "sentiment_report": report_text,
            "audit_events": [news_event, lexicon_event],
            "structured_invocations": [invocation.model_dump(mode="json")],
            "claims": claims,
        }

    return sentiment_analyst_node


def _build_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    news_block: str,
    lexicon_block: str = "",
) -> str:
    """Assemble the sentiment-analyst system message with news + lexicon prior."""
    lexicon_section = (
        f"\n### Deterministic lexicon sentiment prior\n{lexicon_block}\n"
        if lexicon_block
        else ""
    )
    return f"""You are a financial market sentiment analyst. Your task is to produce a comprehensive sentiment report for {ticker} covering the period from {start_date} to {end_date}, based on pre-fetched news headlines and a deterministic lexicon prior.
{lexicon_section}
## Data sources (pre-fetched, in this prompt)

### News headlines (configured news vendor, default Google News RSS)
Institutional / media framing. Fact-driven signal.

<start_of_news>
{news_block}
<end_of_news>

## How to analyze this data

1. **Read the lexicon prior** as a numeric starting point; cite it when you agree or disagree.
2. **Extract dominant narratives** from headlines (earnings, product, regulation, competition).
3. **Be honest about data limits.** If news is empty or marked unavailable, set confidence to low and say so.
4. **Past sentiment is not predictive.** Frame conclusions as signal for the trader, not a price call.
5. **Do not invent** StockTwits, Reddit, X/Twitter, or other social posts — those sources are not in this prompt.

## Output fields

Fill the following fields:

- **overall_band**: Exactly one of Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. Use Neutral when sources are genuinely silent.
- **overall_score**: A number from 0 (maximally bearish) to 10 (maximally bullish); 5 is neutral. Keep it consistent with overall_band.
- **confidence**: low / medium / high, based on data quality and sample size.
- **narrative**: Headline themes, risks/catalysts, and a markdown summary table of key sentiment signals.

{get_language_instruction()}"""


def create_social_media_analyst(llm):
    """Deprecated alias for :func:`create_sentiment_analyst`."""
    import warnings
    warnings.warn(
        "create_social_media_analyst is deprecated and will be removed in a "
        "future version. Use create_sentiment_analyst instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_sentiment_analyst(llm)
