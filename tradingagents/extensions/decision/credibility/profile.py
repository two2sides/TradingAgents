"""Compute an explainable, non-probabilistic AuditProfile."""

from __future__ import annotations

from typing import Any

from .models import AuditProfile


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def build_audit_profile(
    final_state: dict[str, Any],
    events: list[dict],
    claims: list[dict],
    findings: list[dict],
) -> AuditProfile:
    reasons = list(dict.fromkeys(final_state.get("audit_scope_reasons") or []))
    structured_agents = {
        item.get("agent_name")
        for item in final_state.get("structured_invocations") or []
        if item.get("mode") == "STRUCTURED"
    }
    for report_key, agent_name in (
        ("market_report", "Market Analyst"),
        ("news_report", "News Analyst"),
        ("fundamentals_report", "Fundamentals Analyst"),
    ):
        if (
            final_state.get(report_key)
            and agent_name not in structured_agents
            and f"{agent_name} Claim Sidecar" not in structured_agents
        ):
            reasons.append(f"{agent_name} output is free text without a complete claim sidecar")
    if any(item.get("mode") != "STRUCTURED" for item in final_state.get("structured_invocations") or []):
        reasons.append("one or more structured agents used free-text fallback")
    if any(claim.get("capture_status") == "FALLBACK_CANDIDATE" for claim in claims):
        reasons.append("candidate claims extracted from unstructured text are not complete coverage")

    eligible_claims = [
        claim
        for claim in claims
        if claim.get("importance") in {"CRITICAL", "MAJOR"}
        and claim.get("claim_type") != "OPINION"
    ]
    sourced = sum(bool(claim.get("evidence_refs")) for claim in eligible_claims)
    numeric = [
        claim
        for claim in claims
        if claim.get("claim_type") in {"NUMERIC", "DATE"}
        and claim.get("verification_status") in {"SUPPORTED", "CONTRADICTED"}
    ]
    numeric_supported = sum(
        claim.get("verification_status") == "SUPPORTED" for claim in numeric
    )

    tool_events = [
        event for event in events if event.get("event_type") == "EVIDENCE_OBSERVATION"
    ]
    tool_success = sum(
        (event.get("payload") or {}).get("status") == "SUCCESS" for event in tool_events
    )
    invocations = list(final_state.get("structured_invocations") or [])
    structured_success = sum(item.get("mode") == "STRUCTURED" for item in invocations)

    temporal_events = [
        event
        for event in tool_events
        if any(
            key in ((event.get("payload") or {}).get("arguments_redacted") or {})
            for key in ("end_date", "curr_date", "trade_date", "as_of")
        )
    ]
    temporal_bad = sum(
        finding.get("code") == "TOOL_LOOKAHEAD" for finding in findings
    )

    snapshots = list(final_state.get("decision_snapshots") or [])
    modified = [item for item in snapshots if item.get("alignment") == "MODIFIED"]
    explained = sum(bool(item.get("change_reason_refs")) for item in modified)

    debate_turns = (
        (final_state.get("investment_debate_state") or {}).get("debate_turns") or []
    )
    complete_cycles = {}
    for turn in debate_turns:
        if not turn.get("complete"):
            continue
        complete_cycles.setdefault(turn.get("cycle"), []).append(turn)
    novel_cycles = 0
    for turns in complete_cycles.values():
        identities = {
            ref.get("artifact_id")
            for turn in turns
            for ref in turn.get("added_evidence_refs", [])
        }
        corrections = {
            code for turn in turns for code in turn.get("correction_codes", [])
        }
        if identities or corrections:
            novel_cycles += 1

    critical = [
        finding.get("code") for finding in findings if finding.get("severity") == "CRITICAL"
    ]
    warnings = [
        finding.get("code") for finding in findings if finding.get("severity") == "WARNING"
    ]
    scope = "PARTIAL" if reasons else "FULL"
    status = "AUDIT_FAILED" if critical else (
        "AUDIT_DEGRADED" if reasons or warnings else "AUDIT_PASSED"
    )
    return AuditProfile(
        run_id=str(final_state.get("run_id", "unknown-run")),
        audit_scope=scope,
        audit_scope_reasons=list(dict.fromkeys(reasons)),
        status=status,
        evidence_coverage=_ratio(sourced, len(eligible_claims)),
        numeric_verification_rate=_ratio(numeric_supported, len(numeric)),
        temporal_integrity=_ratio(
            max(0, len(temporal_events) - temporal_bad), len(temporal_events)
        ),
        handoff_explanation_coverage=_ratio(explained, len(modified)),
        tool_success_rate=_ratio(tool_success, len(tool_events)),
        structured_output_integrity=_ratio(structured_success, len(invocations)),
        debate_novelty_rate=_ratio(novel_cycles, len(complete_cycles)),
        metric_counts={
            "eligible_claims": len(eligible_claims),
            "sourced_claims": sourced,
            "evaluated_numeric_date_claims": len(numeric),
            "tool_events": len(tool_events),
            "structured_invocations": len(invocations),
            "modified_handoffs": len(modified),
            "complete_debate_cycles": len(complete_cycles),
        },
        critical_findings=list(dict.fromkeys(filter(None, critical))),
        warnings=list(dict.fromkeys(filter(None, warnings))),
    )
