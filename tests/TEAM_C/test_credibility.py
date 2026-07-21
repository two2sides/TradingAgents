from __future__ import annotations

import json

import pytest

from tradingagents.agents.utils.rating import parse_rating_strict
from tradingagents.extensions.decision.credibility.audit_writer import write_audit_bundle
from tradingagents.extensions.decision.credibility.evidence import (
    invoke_evidenced,
    redact_arguments,
)
from tradingagents.extensions.decision.credibility.models import (
    ArtifactFieldRef,
    ClaimRecord,
    VerificationStatus,
    claim_identity,
    stable_id,
)
from tradingagents.extensions.decision.credibility.verifier import run_verifier
from tradingagents.graph.conditional_logic import ConditionalLogic


@pytest.mark.unit
def test_evidence_event_is_idempotent_and_redacts_secrets():
    kwargs = dict(
        run_id="run-1",
        producer_node="test",
        tool_name="price",
        arguments={"ticker": "AAPL", "api_key": "secret"},
        call=lambda: '{"return": 0.05}',
    )
    _, first = invoke_evidenced(**kwargs)
    _, second = invoke_evidenced(**kwargs)
    assert first["payload"]["event_id"] == second["payload"]["event_id"]
    assert first["payload"]["arguments_redacted"]["api_key"] == "[REDACTED]"
    assert redact_arguments({"authorization": "x"})["authorization"] == "[REDACTED]"


@pytest.mark.unit
def test_strict_rating_never_defaults_to_hold():
    assert parse_rating_strict("No explicit decision.").parsed is None
    parsed = parse_rating_strict("**Recommendation**: Underweight")
    assert parsed.parsed == "Underweight"
    assert parsed.source == "recommendation_label"


@pytest.mark.unit
def test_numeric_verifier_uses_selector_not_text_search():
    result, event = invoke_evidenced(
        run_id="run-2",
        producer_node="market",
        tool_name="returns",
        arguments={},
        call=lambda: json.dumps({"windows": {"20d": {"return": 0.05}}}),
    )
    del result
    payload = event["payload"]
    identity = claim_identity(
        subject="AAPL",
        predicate="20d_return",
        value=30.0,
        unit="percent",
        period_start=None,
        period_end=None,
        as_of="2026-07-15",
        text="20-day return is 30%",
    )
    claim = ClaimRecord(
        claim_id=stable_id("claim", identity),
        identity_key=identity,
        run_id="run-2",
        agent="Market Analyst",
        stage="market",
        text="20-day return is 30%",
        claim_type="NUMERIC",
        value=30.0,
        unit="percent",
        as_of="2026-07-15",
        evidence_refs=[
            ArtifactFieldRef(
                observation_event_id=payload["event_id"],
                artifact_id=payload["artifact_id"],
                selector="/windows/20d/return",
                schema_name="returns",
            )
        ],
        verification_status=VerificationStatus.NOT_EVALUATED,
    ).model_dump(mode="json")
    state = {
        "trade_date": "2026-07-15",
        "audit_events": [event],
        "claims": [claim],
        "structured_invocations": [],
        "decision_snapshots": [],
    }
    _, claims, findings = run_verifier(state)
    assert claims[0]["verification_status"] == "CONTRADICTED"
    assert any(item["code"] == "NUMERIC_EVIDENCE_MISMATCH" for item in findings)


@pytest.mark.unit
def test_audit_bundle_preserves_pm_rating_and_writes_views(tmp_path):
    state = {
        "run_id": "run-3",
        "trade_date": "2026-07-15",
        "final_trade_decision": "**Rating**: Sell",
        "market_report": "20日收益率为 5%。",
        "audit_events": [],
        "claims": [],
        "structured_invocations": [],
        "decision_snapshots": [],
        "investment_debate_state": {"debate_turns": []},
    }
    original = state["final_trade_decision"]
    markdown, profile = write_audit_bundle(state, "AAPL", tmp_path)
    assert state["final_trade_decision"] == original
    assert profile["audit_scope"] == "PARTIAL"
    assert "不修改 Portfolio Manager Rating" in markdown
    for relative in (
        "event_trace.jsonl",
        "claims.jsonl",
        "findings.jsonl",
        "profile.json",
        "manifest.json",
        "credibility.md",
    ):
        assert (tmp_path / "audit" / relative).exists()


@pytest.mark.unit
def test_debate_stops_only_after_complete_no_novelty_cycle():
    router = ConditionalLogic(max_debate_rounds=3)
    base = {
        "count": 2,
        "current_response": "Bear Analyst: no new evidence",
        "no_novelty_cycles": 1,
    }
    assert router.should_continue_debate({"investment_debate_state": base}) == "Research Manager"
    incomplete = {**base, "count": 3, "current_response": "Bull Analyst: next turn"}
    assert router.should_continue_debate({"investment_debate_state": incomplete}) == "Bear Researcher"
