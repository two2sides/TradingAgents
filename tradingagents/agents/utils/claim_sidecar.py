"""Generate a claim sidecar without changing an analyst's human-readable draft."""

from __future__ import annotations

from tradingagents.agents.schemas import AnalystClaimEnvelope
from tradingagents.agents.utils.structured import invoke_structured_with_metadata
from tradingagents.extensions.decision.credibility.claims import claims_from_invocation
from tradingagents.extensions.decision.credibility.models import StructuredInvocationResult


def build_claim_sidecar(
    *,
    structured_llm,
    plain_llm,
    agent_name: str,
    stage: str,
    draft: str,
    state: dict,
) -> tuple[dict, list[dict]]:
    relevant_prefixes = {
        "market": ("tools_market",),
        "news": ("news_prefetch", "tools_news"),
        "fundamentals": ("tools_fundamentals",),
    }.get(stage, ())
    catalog_lines = []
    for event in state.get("audit_events", []):
        payload = event.get("payload") or {}
        if relevant_prefixes and payload.get("producer_node") not in relevant_prefixes:
            continue
        if not payload.get("artifact_id"):
            continue
        preview = str(payload.get("artifact_content") or "")[:1500]
        catalog_lines.append(
            f"ARTIFACT {payload['artifact_id']} TOOL {payload.get('tool_name')}:\n{preview}"
        )
    artifact_catalog = "\n\n".join(catalog_lines[-15:]) or "No artifact catalog available."
    prompt = (
        "Extract a machine-readable sidecar from the analyst draft below. "
        "Do not add facts. For every numeric/date claim include exact value, unit, "
        "window and any artifact_id explicitly present in the draft. Distinguish "
        "fact, causal statement, forecast and opinion. If no artifact ID is shown, "
        "leave evidence_refs empty and disclose the uncertainty.\n\n"
        "Use evidence_refs entries in the form artifact_id#/json/pointer when a "
        "specific JSON field supports the claim. Do not cite an artifact merely "
        "because it is topically related.\n\n"
        f"ARTIFACT CATALOG:\n{artifact_catalog}\n\nDRAFT:\n{draft}"
    )
    if structured_llm is None:
        invocation = StructuredInvocationResult(
            text=draft,
            mode="FALLBACK_UNSTRUCTURED",
            agent_name=f"{agent_name} Claim Sidecar",
            schema_name="AnalystClaimEnvelope",
            parse_error_code="STRUCTURED_UNAVAILABLE",
        )
    else:
        invocation = invoke_structured_with_metadata(
            structured_llm,
            plain_llm,
            prompt,
            lambda _envelope: draft,
            f"{agent_name} Claim Sidecar",
        )
    claims = claims_from_invocation(
        run_id=state.get("run_id", "legacy-run"),
        agent=agent_name,
        stage=stage,
        invocation=invocation,
        audit_events=state.get("audit_events", []),
        trade_date=state.get("trade_date", ""),
    )
    return invocation.model_dump(mode="json"), claims


def append_claim_index(draft: str, claims: list[dict]) -> str:
    if not claims:
        return draft
    lines = [draft.rstrip(), "", "### Evidence Claim Index (machine-readable IDs)"]
    for claim in claims:
        refs = [
            f"{ref.get('artifact_id')}#{ref.get('selector', '/')}"
            for ref in claim.get("evidence_refs", [])
        ]
        suffix = f" | evidence: {', '.join(refs)}" if refs else " | evidence: unresolved"
        lines.append(f"- `{claim['claim_id']}` {claim['text']}{suffix}")
    return "\n".join(lines)


def bind_claim_sidecar(llm, agent_name: str):
    try:
        return llm.with_structured_output(AnalystClaimEnvelope)
    except (NotImplementedError, AttributeError):
        return None
