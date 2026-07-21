"""Decision-cutoff and future-date rules."""

from __future__ import annotations

from datetime import date
from typing import Any

from ..models import VerificationFinding, stable_id


def check_temporal_rules(
    claims: list[dict[str, Any]], events: list[dict[str, Any]], trade_date: str
):
    findings = []
    try:
        cutoff = date.fromisoformat(trade_date)
    except (TypeError, ValueError):
        return findings
    for claim in claims:
        if claim.get("claim_type") != "DATE":
            continue
        try:
            mentioned = date.fromisoformat(str(claim.get("value")))
        except (TypeError, ValueError):
            continue
        if mentioned > cutoff and claim.get("claim_type") not in {"FORECAST", "OPINION"}:
            code = "FUTURE_DATE_MENTION"
            findings.append(
                VerificationFinding(
                    finding_id=stable_id("finding", {"rule": code, "claim": claim.get("claim_id")}),
                    rule_id="temporal.claim_date",
                    code=code,
                    severity="WARNING",
                    stage=claim.get("stage", "unknown"),
                    claim_id=claim.get("claim_id"),
                    expected=f"<= {trade_date} or explicitly forecast",
                    actual=str(mentioned),
                    message="报告出现晚于决策日的日期，且未被结构化标为预测。",
                ).model_dump(mode="json")
            )
    for event in events:
        payload = event.get("payload") or {}
        args = payload.get("arguments_redacted") or {}
        for key in ("end_date", "curr_date", "trade_date", "as_of"):
            raw = args.get(key)
            if not raw:
                continue
            try:
                observed_date = date.fromisoformat(str(raw)[:10])
            except ValueError:
                continue
            if observed_date > cutoff:
                code = "TOOL_LOOKAHEAD"
                findings.append(
                    VerificationFinding(
                        finding_id=stable_id(
                            "finding",
                            {"rule": code, "event": payload.get("event_id"), "key": key},
                        ),
                        rule_id="temporal.tool_arguments",
                        code=code,
                        severity="CRITICAL",
                        stage=payload.get("producer_node", "tool"),
                        artifact_id=payload.get("artifact_id"),
                        expected=f"{key} <= {trade_date}",
                        actual=raw,
                        first_seen_event_id=payload.get("event_id"),
                        message="工具参数越过决策截止日期。",
                    ).model_dump(mode="json")
                )
    return findings
