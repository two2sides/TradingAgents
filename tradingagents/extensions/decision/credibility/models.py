"""Private, versioned models for credibility audit payloads."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


SCHEMA_VERSION = "1.0"


class VerificationStatus(str, Enum):
    NOT_EVALUATED = "NOT_EVALUATED"
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNSOURCED = "UNSOURCED"
    NOT_DETERMINISTICALLY_VERIFIABLE = "NOT_DETERMINISTICALLY_VERIFIABLE"


class ArtifactFieldRef(BaseModel):
    observation_event_id: str
    artifact_id: str
    selector: str = "/"
    schema_name: str = "unstructured"
    schema_version: str = SCHEMA_VERSION


class ClaimRecord(BaseModel):
    claim_id: str
    identity_key: str
    run_id: str
    agent: str
    stage: str
    text: str
    claim_type: Literal["NUMERIC", "DATE", "FACT", "CAUSAL", "FORECAST", "OPINION"]
    importance: Literal["CRITICAL", "MAJOR", "MINOR"] = "MAJOR"
    subject: str | None = None
    predicate: str | None = None
    value: float | str | None = None
    unit: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    as_of: str | None = None
    evidence_refs: list[ArtifactFieldRef] = Field(default_factory=list)
    tolerance: float | None = None
    polarity: str | None = None
    inherited_claim_ids: list[str] = Field(default_factory=list)
    capture_status: Literal["STRUCTURED", "FALLBACK_CANDIDATE"] = "STRUCTURED"
    verification_status: VerificationStatus = VerificationStatus.NOT_EVALUATED


class StructuredInvocationResult(BaseModel):
    text: str
    mode: Literal["STRUCTURED", "FALLBACK_UNSTRUCTURED"]
    agent_name: str
    schema_name: str
    attempts: int = 1
    parse_error_code: str | None = None
    parse_error: str | None = None
    parsed: dict[str, Any] | None = None

    def trace_payload(self) -> dict[str, Any]:
        return self.model_dump(exclude={"text", "parsed"})


class DecisionSnapshot(BaseModel):
    snapshot_id: str
    run_id: str
    stage: str
    decision_type: str
    value: str | None
    parsed: bool
    source: str
    error: str | None = None
    alignment: Literal["ADOPTED", "MODIFIED", "NOT_COMPARABLE"] = "NOT_COMPARABLE"
    upstream_decision_ref: str | None = None
    change_reason_refs: list[str] = Field(default_factory=list)


class ClaimStatusChange(BaseModel):
    claim_id: str
    old_status: VerificationStatus
    new_status: VerificationStatus


class CorrectionCode(str, Enum):
    WRONG_VALUE = "WRONG_VALUE"
    WRONG_UNIT = "WRONG_UNIT"
    WRONG_WINDOW = "WRONG_WINDOW"
    STALE_EVIDENCE = "STALE_EVIDENCE"
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
    SEMANTIC_MISLABEL = "SEMANTIC_MISLABEL"


class DebateTurnRecord(BaseModel):
    side: Literal["bull", "bear"]
    cycle: int
    target_claim_id: str | None = None
    stance: str
    added_evidence_refs: list[ArtifactFieldRef] = Field(default_factory=list)
    changed_claim_status: list[ClaimStatusChange] = Field(default_factory=list)
    correction_codes: list[CorrectionCode] = Field(default_factory=list)
    complete: bool = True


class VerificationFinding(BaseModel):
    finding_id: str
    rule_id: str
    rule_version: str = SCHEMA_VERSION
    status: Literal["OPEN", "RESOLVED"] = "OPEN"
    code: str
    severity: Literal["INFO", "WARNING", "CRITICAL"]
    stage: str
    message: str
    claim_id: str | None = None
    artifact_id: str | None = None
    expected: Any = None
    actual: Any = None
    location: str | None = None
    first_seen_event_id: str | None = None


class AuditProfile(BaseModel):
    run_id: str
    policy_version: str = SCHEMA_VERSION
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    audit_scope: Literal["FULL", "PARTIAL"]
    audit_scope_reasons: list[str] = Field(default_factory=list)
    status: Literal["AUDIT_PASSED", "AUDIT_DEGRADED", "AUDIT_FAILED"]
    evidence_coverage: float | None = None
    numeric_verification_rate: float | None = None
    temporal_integrity: float | None = None
    handoff_explanation_coverage: float | None = None
    tool_success_rate: float | None = None
    structured_output_integrity: float | None = None
    debate_novelty_rate: float | None = None
    metric_counts: dict[str, Any] = Field(default_factory=dict)
    critical_findings: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def stable_id(prefix: str, payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"


def claim_identity(
    *,
    subject: str | None,
    predicate: str | None,
    value: Any,
    unit: str | None,
    period_start: str | None,
    period_end: str | None,
    as_of: str | None,
    text: str,
) -> str:
    return stable_id(
        "claimkey",
        {
            "subject": subject,
            "predicate": predicate,
            "value": value,
            "unit": unit,
            "period_start": period_start,
            "period_end": period_end,
            "as_of": as_of,
            "text": " ".join(text.lower().split()),
        },
    )
