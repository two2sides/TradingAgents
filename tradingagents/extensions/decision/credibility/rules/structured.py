"""Structured-output and decision-snapshot rules."""

from __future__ import annotations

from typing import Any

from ..models import VerificationFinding, stable_id


def check_structured_rules(
    invocations: list[dict[str, Any]], snapshots: list[dict[str, Any]]
):
    findings = []
    for invocation in invocations:
        if invocation.get("mode") != "STRUCTURED":
            code = "STRUCTURED_OUTPUT_FALLBACK"
            findings.append(
                VerificationFinding(
                    finding_id=stable_id("finding", {"rule": code, "item": invocation}),
                    rule_id="structured.invocation_mode",
                    code=code,
                    severity="WARNING",
                    stage=invocation.get("agent_name", "unknown"),
                    message=(
                        f"{invocation.get('agent_name', 'Agent')} 使用自由文本回退；"
                        "该输出的审计覆盖范围下降。"
                    ),
                    actual=invocation.get("parse_error_code"),
                ).model_dump(mode="json")
            )
    for snapshot in snapshots:
        if not snapshot.get("parsed"):
            code = "DECISION_PARSE_FAILED"
            findings.append(
                VerificationFinding(
                    finding_id=stable_id("finding", {"rule": code, "item": snapshot}),
                    rule_id="structured.decision_parse",
                    code=code,
                    severity="CRITICAL",
                    stage=snapshot.get("stage", "unknown"),
                    message="关键决策字段无法严格解析；审计器不会将其默认成 Hold。",
                    actual=snapshot.get("error"),
                ).model_dump(mode="json")
            )
        if (
            snapshot.get("alignment") == "MODIFIED"
            and not snapshot.get("change_reason_refs")
        ):
            code = "HANDOFF_CHANGE_UNEXPLAINED"
            findings.append(
                VerificationFinding(
                    finding_id=stable_id("finding", {"rule": code, "item": snapshot}),
                    rule_id="handoff.modified_has_reason",
                    code=code,
                    severity="WARNING",
                    stage=snapshot.get("stage", "unknown"),
                    message="节点声明修改上游决策，但没有 reason/constraint 引用。",
                ).model_dump(mode="json")
            )
    return findings
