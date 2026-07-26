"""Compute an explainable, non-probabilistic AuditProfile."""

from __future__ import annotations

from datetime import date
from typing import Any

from .models import AuditProfile


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _is_iso_date(value: Any) -> bool:
    try:
        date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return False
    return True


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
    if not events:
        reasons.append("no evidence observations were available for deterministic audit")
    if not claims:
        reasons.append("no claims were available for credibility coverage analysis")

    eligible_claims = [
        claim
        for claim in claims
        if claim.get("importance") in {"CRITICAL", "MAJOR"}
        and claim.get("claim_type") != "OPINION"
    ]
    artifact_ids = {
        (event.get("payload") or {}).get("artifact_id")
        for event in events
        if (event.get("payload") or {}).get("artifact_id")
        and not (event.get("payload") or {}).get("post_run_recomputed")
    }

    def _ref_locates_claim(claim: dict[str, Any], ref: dict[str, Any]) -> bool:
        artifact_id = ref.get("artifact_id")
        if not artifact_id or artifact_id not in artifact_ids:
            return False
        # Numeric/date claims need a concrete JSON Pointer, not the whole payload "/".
        if claim.get("claim_type") in {"NUMERIC", "DATE"}:
            selector = str(ref.get("selector") or "").strip()
            return selector not in {"", "/"}
        return True

    sourced = sum(
        any(_ref_locates_claim(claim, ref) for ref in claim.get("evidence_refs") or [])
        for claim in eligible_claims
    )
    evidence_unknown = sum(
        claim.get("capture_status") == "FALLBACK_CANDIDATE"
        for claim in eligible_claims
    )

    numeric_applicable = [
        claim
        for claim in claims
        if claim.get("claim_type") in {"NUMERIC", "DATE"}
    ]
    numeric_eligible = [
        claim
        for claim in numeric_applicable
        if claim.get("capture_status") == "STRUCTURED"
    ]
    numeric_evaluated = [
        claim
        for claim in numeric_eligible
        if claim.get("verification_status") in {"SUPPORTED", "CONTRADICTED"}
    ]
    numeric_supported = sum(
        claim.get("verification_status") == "SUPPORTED" for claim in numeric_evaluated
    )

    tool_events = [
        event
        for event in events
        if event.get("event_type") == "EVIDENCE_OBSERVATION"
        and not (event.get("payload") or {}).get("post_run_recomputed")
    ]
    tool_statuses = [(event.get("payload") or {}).get("status") for event in tool_events]
    tool_success = sum(status in {"SUCCESS", "EXPECTED_EMPTY"} for status in tool_statuses)
    tool_failed = sum(status in {"ERROR", "UNEXPECTED_EMPTY"} for status in tool_statuses)
    tool_unknown = max(0, len(tool_events) - tool_success - tool_failed)
    invocations = list(final_state.get("structured_invocations") or [])
    structured_success = sum(item.get("mode") == "STRUCTURED" for item in invocations)
    structured_unknown = sum(not item.get("mode") for item in invocations)

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
    temporal_unknown = 0
    for event in temporal_events:
        args = (event.get("payload") or {}).get("arguments_redacted") or {}
        values = [
            args.get(key)
            for key in ("end_date", "curr_date", "trade_date", "as_of")
            if args.get(key)
        ]
        if values and not any(_is_iso_date(value) for value in values):
            temporal_unknown += 1
    temporal_evaluated = max(0, len(temporal_events) - temporal_unknown)

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
    advisories = []
    if critical:
        advisories.append(
            "优先复核 CRITICAL finding 指向的字段、来源与截止时间；"
            "审计结论仅作建议，不自动阻止交易或改写 Rating。"
        )
    missing_evidence = max(0, len(eligible_claims) - sourced)
    if missing_evidence:
        advisories.append(
            f"为 {missing_evidence} 条关键主张补充可定位的 ArtifactFieldRef，"
            "优先处理 CRITICAL/MAJOR 主张。"
        )
    numeric_unknown = len(numeric_eligible) - len(numeric_evaluated)
    if numeric_unknown:
        advisories.append(
            f"有 {numeric_unknown} 条结构化数值/日期主张尚未完成确定性核验；"
            "请检查 value、unit，并为数值主张提供非根路径的 JSON Pointer selector。"
        )
    if structured_success < len(invocations):
        advisories.append(
            "检查未通过结构化校验的节点输出；新运行应保持严格结构化，"
            "历史 fallback 仅降低审计覆盖。"
        )
    if tool_failed or tool_unknown:
        advisories.append(
            f"复核工具调用：{tool_failed} 次明确失败，{tool_unknown} 次结果状态未知或为空。"
        )
    if explained < len(modified):
        advisories.append(
            f"为 {len(modified) - explained} 次 MODIFIED 交接补充 reason/constraint 引用。"
        )
    return AuditProfile(
        run_id=str(final_state.get("run_id", "unknown-run")),
        audit_scope=scope,
        audit_scope_reasons=list(dict.fromkeys(reasons)),
        status=status,
        evidence_coverage=_ratio(sourced, len(eligible_claims)),
        numeric_verification_rate=_ratio(numeric_supported, len(numeric_evaluated)),
        temporal_integrity=_ratio(
            max(0, temporal_evaluated - temporal_bad), temporal_evaluated
        ),
        handoff_explanation_coverage=_ratio(explained, len(modified)),
        tool_success_rate=_ratio(tool_success, tool_success + tool_failed),
        structured_output_integrity=_ratio(structured_success, len(invocations)),
        debate_novelty_rate=_ratio(novel_cycles, len(complete_cycles)),
        metric_counts={
            "eligible_claims": len(eligible_claims),
            "sourced_claims": sourced,
            "evaluated_numeric_date_claims": len(numeric_evaluated),
            "tool_events": len(tool_events),
            "structured_invocations": len(invocations),
            "modified_handoffs": len(modified),
            "complete_debate_cycles": len(complete_cycles),
            "evidence_coverage": {
                "eligible_count": len(eligible_claims),
                "supported_count": sourced,
                "failed_count": max(
                    0, len(eligible_claims) - sourced - evidence_unknown
                ),
                "excluded_count": 0,
                "unknown_count": evidence_unknown,
            },
            "numeric_verification_rate": {
                "eligible_count": len(numeric_eligible),
                "supported_count": numeric_supported,
                "failed_count": len(numeric_evaluated) - numeric_supported,
                "excluded_count": len(numeric_applicable) - len(numeric_eligible),
                "unknown_count": len(numeric_eligible) - len(numeric_evaluated),
            },
            "temporal_integrity": {
                "eligible_count": len(temporal_events),
                "supported_count": max(0, temporal_evaluated - temporal_bad),
                "failed_count": temporal_bad,
                "excluded_count": max(0, len(tool_events) - len(temporal_events)),
                "unknown_count": temporal_unknown,
            },
            "tool_success_rate": {
                "eligible_count": len(tool_events),
                "supported_count": tool_success,
                "failed_count": tool_failed,
                "excluded_count": 0,
                "unknown_count": tool_unknown,
            },
            "structured_output_integrity": {
                "eligible_count": len(invocations),
                "supported_count": structured_success,
                "failed_count": max(
                    0, len(invocations) - structured_success - structured_unknown
                ),
                "excluded_count": 0,
                "unknown_count": structured_unknown,
            },
            "handoff_explanation_coverage": {
                "eligible_count": len(modified),
                "supported_count": explained,
                "failed_count": len(modified) - explained,
                "excluded_count": max(0, len(snapshots) - len(modified)),
                "unknown_count": 0,
            },
            "debate_novelty_rate": {
                "eligible_count": len(complete_cycles),
                "supported_count": novel_cycles,
                "failed_count": len(complete_cycles) - novel_cycles,
                "excluded_count": 0,
                "unknown_count": 0,
            },
        },
        critical_findings=list(dict.fromkeys(filter(None, critical))),
        warnings=list(dict.fromkeys(filter(None, warnings))),
        advisories=advisories,
    )
