"""Single credibility verifier orchestrating non-overlapping rule packs."""

from __future__ import annotations

from typing import Any

from .rules import (
    check_numeric_rules,
    check_source_rules,
    check_structured_rules,
    check_temporal_rules,
)


def _dedupe(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for item in items:
        value = item.get(key)
        if value in seen:
            continue
        seen.add(value)
        result.append(item)
    return result


def run_verifier(final_state: dict[str, Any]) -> tuple[list[dict], list[dict], list[dict]]:
    # Event payload is a dict; dedupe with its stable event id.
    event_seen = set()
    unique_events = []
    for event in final_state.get("audit_events") or []:
        event_id = (event.get("payload") or {}).get("event_id")
        if event_id in event_seen:
            continue
        event_seen.add(event_id)
        unique_events.append(event)
    events = unique_events

    claims = _dedupe(list(final_state.get("claims") or []), "claim_id")
    invocations = list(final_state.get("structured_invocations") or [])
    snapshots = _dedupe(list(final_state.get("decision_snapshots") or []), "snapshot_id")

    findings = []
    findings.extend(check_source_rules(claims, events))
    findings.extend(check_numeric_rules(claims, events))
    findings.extend(
        check_temporal_rules(claims, events, str(final_state.get("trade_date")))
    )
    findings.extend(check_structured_rules(invocations, snapshots))
    findings = _dedupe(findings, "finding_id")

    contradicted = {
        finding.get("claim_id")
        for finding in findings
        if finding.get("code") == "NUMERIC_EVIDENCE_MISMATCH"
    }
    unsourced = {
        finding.get("claim_id")
        for finding in findings
        if finding.get("code") == "UNSOURCED_CLAIM"
    }
    for claim in claims:
        if claim.get("claim_id") in contradicted:
            claim["verification_status"] = "CONTRADICTED"
        elif claim.get("claim_id") in unsourced:
            claim["verification_status"] = "UNSOURCED"
        elif claim.get("evidence_refs"):
            claim["verification_status"] = "SUPPORTED"
        elif claim.get("claim_type") in {"CAUSAL", "FORECAST", "OPINION"}:
            claim["verification_status"] = "NOT_DETERMINISTICALLY_VERIFIABLE"
    return events, claims, findings
