from datetime import datetime, timedelta

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_global_news,
    get_instrument_context_from_state,
    get_language_instruction,
    get_news,
)
from tradingagents.agents.utils.claim_sidecar import (
    append_claim_index,
    bind_claim_sidecar,
    build_claim_sidecar,
)
from tradingagents.extensions.decision.credibility import invoke_evidenced


def create_news_analyst(llm):
    """News analyst — single path: ticker + global news (default Google News RSS).

    Macro (FRED) and prediction markets are intentionally not bound here so the
    default run does not spam missing-key errors. Enable them later via config
    if needed.
    """

    claim_llm = bind_claim_sidecar(llm, "News Analyst")

    def news_analyst_node(state):
        ticker = state["company_of_interest"]
        run_id = state.get("run_id", "legacy-run")
        current_date = state["trade_date"]
        start_date = (
            datetime.strptime(current_date, "%Y-%m-%d") - timedelta(days=7)
        ).strftime("%Y-%m-%d")
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        instrument_context = get_instrument_context_from_state(state)

        tools = [
            get_news,
            get_global_news,
        ]
        existing = {
            (event.get("payload") or {}).get("tool_name"): event
            for event in state.get("audit_events", [])
            if (event.get("payload") or {}).get("producer_node") == "news_prefetch"
        }
        new_events = []
        if "get_news" in existing:
            company_news = existing["get_news"]["payload"].get("artifact_content", "")
            company_artifact = existing["get_news"]["payload"].get("artifact_id")
        else:
            company_news, company_event = invoke_evidenced(
                run_id=run_id,
                producer_node="news_prefetch",
                tool_name="get_news",
                arguments={"ticker": ticker, "start_date": start_date, "end_date": current_date},
                call=lambda: get_news.func(ticker, start_date, current_date),
            )
            new_events.append(company_event)
            company_artifact = company_event["payload"].get("artifact_id")
        if "get_global_news" in existing:
            global_news = existing["get_global_news"]["payload"].get("artifact_content", "")
            global_artifact = existing["get_global_news"]["payload"].get("artifact_id")
        else:
            global_news, global_event = invoke_evidenced(
                run_id=run_id,
                producer_node="news_prefetch",
                tool_name="get_global_news",
                arguments={"curr_date": current_date, "look_back_days": 7, "limit": 25},
                call=lambda: get_global_news.func(current_date, 7, 25),
            )
            new_events.append(global_event)
            global_artifact = global_event["payload"].get("artifact_id")
        prefetched_evidence = (
            "PREFETCHED NEWS EVIDENCE (tool output; cite only these headlines or "
            "additional tool results):\n\n"
            f"[artifact_id={company_artifact}]\n{company_news}\n\n"
            f"[artifact_id={global_artifact}]\n{global_news}"
        )

        system_message = (
            f"You are a news researcher tasked with analyzing recent news and trends over the past week. "
            f"Write a comprehensive report relevant for trading. Use only these tools: "
            f"get_news(ticker, start_date, end_date) for {asset_label}-specific news, and "
            f"get_global_news(curr_date, look_back_days, limit) for broader market/macro headlines. "
            f"Provide specific, actionable insights with supporting evidence. "
            f"If a tool returns unavailable/empty, say so — do not invent headlines."
            f"\n\n{prefetched_evidence}\n\n"
            "The prefetched block is mandatory evidence. You may call tools again "
            "only to resolve a specific gap. Distinguish company-specific from broad "
            "market news, disclose sparse coverage, and never treat model memory as a source."
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        update = {
            "messages": [result],
            "audit_events": new_events,
        }
        if len(result.tool_calls) == 0 and result.content:
            invocation, claims = build_claim_sidecar(
                structured_llm=claim_llm,
                plain_llm=llm,
                agent_name="News Analyst",
                stage="news",
                draft=str(result.content),
                state={
                    **state,
                    "audit_events": state.get("audit_events", []) + new_events,
                },
            )
            update["news_report"] = append_claim_index(str(result.content), claims)
            update["structured_invocations"] = [invocation]
            update["claims"] = claims
        return update

    return news_analyst_node
