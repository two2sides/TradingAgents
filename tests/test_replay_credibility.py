from datetime import datetime, timezone
from types import SimpleNamespace

from webui.pages.replay import _audit_metric_cards, _resolve_credibility_view


def test_audit_metric_cards_show_denominators_and_unknowns():
    profile = {
        "evidence_coverage": 0.5,
        "numeric_verification_rate": None,
        "metric_counts": {
            "evidence_coverage": {
                "eligible_count": 10,
                "supported_count": 5,
                "failed_count": 3,
                "excluded_count": 1,
                "unknown_count": 2,
            },
            "numeric_verification_rate": {
                "eligible_count": 2,
                "supported_count": 0,
                "failed_count": 0,
                "excluded_count": 3,
                "unknown_count": 2,
            },
        },
    }

    cards = _audit_metric_cards(profile)

    assert len(cards) == 7
    assert cards[0]["value"] == "50.0%"
    assert cards[0]["delta"] == "通过 5/10 · 排除 1 · 未知 2"
    assert cards[0]["tone"] == "negative"
    assert cards[1]["value"] == "N/A"
    assert cards[1]["tone"] == "neutral"


def test_legacy_diagnostics_can_be_rebuilt_for_display_without_persistence():
    decision = SimpleNamespace(
        intent=SimpleNamespace(
            decision_id="legacy-decision",
            as_of=datetime(2026, 7, 21, tzinfo=timezone.utc),
            metadata={"graph_run_id": "legacy-run"},
        ),
        diagnostics={
            "claims": [
                {
                    "claim_id": "claim-1",
                    "importance": "CRITICAL",
                    "claim_type": "OPINION",
                    "capture_status": "STRUCTURED",
                    "verification_status": "NOT_EVALUATED",
                }
            ],
            "audit_events": [],
            "structured_invocations": [],
            "decision_snapshots": [],
            "agent_reports": {},
        },
    )

    profile, findings, claims, rebuilt = _resolve_credibility_view(decision)

    assert rebuilt is True
    assert profile["status"] == "AUDIT_DEGRADED"
    assert profile["audit_scope"] == "PARTIAL"
    assert findings == []
    assert claims[0]["verification_status"] == "NOT_DETERMINISTICALLY_VERIFIABLE"
