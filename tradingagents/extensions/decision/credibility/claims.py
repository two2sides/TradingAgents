"""Build typed claims from structured sidecars and conservative text candidates."""

from __future__ import annotations

import re
from typing import Any

from .models import (
    ArtifactFieldRef,
    ClaimRecord,
    StructuredInvocationResult,
    VerificationStatus,
    claim_identity,
    stable_id,
)

_PERCENT_RE = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*%")
_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")


def _refs(raw_refs: list[str], events: list[dict[str, Any]]) -> list[ArtifactFieldRef]:
    result = []
    by_artifact = {
        str((event.get("payload") or {}).get("artifact_id")): event
        for event in events
        if (event.get("payload") or {}).get("artifact_id")
    }
    for raw in raw_refs:
        artifact_id, _, selector = str(raw).partition("#")
        event = by_artifact.get(artifact_id)
        if not event:
            continue
        payload = event.get("payload") or {}
        result.append(
            ArtifactFieldRef(
                observation_event_id=str(payload.get("event_id")),
                artifact_id=artifact_id,
                selector=selector or "/",
                schema_name=str(payload.get("tool_name") or "unstructured"),
                schema_version=str(payload.get("schema_version") or "1.0"),
            )
        )
    return result


def claims_from_invocation(
    *,
    run_id: str,
    agent: str,
    stage: str,
    invocation: StructuredInvocationResult,
    audit_events: list[dict[str, Any]],
    trade_date: str,
) -> list[dict[str, Any]]:
    parsed = invocation.parsed or {}
    raw_claims = list(parsed.get("key_claims") or [])
    for key in ("recommendation", "action", "rating", "overall_band"):
        if key in parsed:
            value = parsed[key]
            if isinstance(value, dict):
                value = value.get("value")
            raw_claims.append(
                {
                    "text": f"{agent} {key}: {value}",
                    "claim_type": "OPINION",
                    "importance": "CRITICAL",
                    "subject": agent,
                    "predicate": key,
                    "value": value,
                    "evidence_refs": [],
                }
            )

    records = []
    for raw in raw_claims:
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        refs = _refs(list(raw.get("evidence_refs") or []), audit_events)
        identity = claim_identity(
            subject=raw.get("subject"),
            predicate=raw.get("predicate"),
            value=raw.get("value"),
            unit=raw.get("unit"),
            period_start=raw.get("period_start"),
            period_end=raw.get("period_end"),
            as_of=raw.get("as_of") or trade_date,
            text=text,
        )
        record = ClaimRecord(
            claim_id=stable_id("claim", {"run_id": run_id, "stage": stage, "identity": identity}),
            identity_key=identity,
            run_id=run_id,
            agent=agent,
            stage=stage,
            text=text,
            claim_type=raw.get("claim_type") or "FACT",
            importance=raw.get("importance") or "MAJOR",
            subject=raw.get("subject"),
            predicate=raw.get("predicate"),
            value=raw.get("value"),
            unit=raw.get("unit"),
            period_start=raw.get("period_start"),
            period_end=raw.get("period_end"),
            as_of=raw.get("as_of") or trade_date,
            evidence_refs=refs,
            capture_status="STRUCTURED",
            verification_status=VerificationStatus.NOT_EVALUATED,
        )
        records.append(record.model_dump(mode="json"))
    return records


def extract_candidate_claims(
    *,
    run_id: str,
    agent: str,
    stage: str,
    text: str,
    trade_date: str,
) -> list[dict[str, Any]]:
    """Extract only obvious numeric/date candidates; never claim full coverage."""
    records = []
    for index, match in enumerate(_PERCENT_RE.finditer(text or "")):
        snippet = (text[max(0, match.start() - 80): match.end() + 80]).strip()
        identity = claim_identity(
            subject=None,
            predicate="percentage_mention",
            value=float(match.group(1)),
            unit="percent",
            period_start=None,
            period_end=None,
            as_of=trade_date,
            text=snippet,
        )
        record = ClaimRecord(
            claim_id=stable_id(
                "claim", {"run_id": run_id, "stage": stage, "identity": identity, "index": index}
            ),
            identity_key=identity,
            run_id=run_id,
            agent=agent,
            stage=stage,
            text=snippet,
            claim_type="NUMERIC",
            importance="MAJOR",
            value=float(match.group(1)),
            unit="percent",
            as_of=trade_date,
            capture_status="FALLBACK_CANDIDATE",
            verification_status=VerificationStatus.UNSOURCED,
        )
        records.append(record.model_dump(mode="json"))
    for index, match in enumerate(_DATE_RE.finditer(text or "")):
        identity = claim_identity(
            subject=None,
            predicate="date_mention",
            value=match.group(1),
            unit="date",
            period_start=None,
            period_end=None,
            as_of=trade_date,
            text=match.group(0),
        )
        records.append(
            ClaimRecord(
                claim_id=stable_id(
                    "claim", {"run_id": run_id, "stage": stage, "date": match.group(1), "index": index}
                ),
                identity_key=identity,
                run_id=run_id,
                agent=agent,
                stage=stage,
                text=match.group(0),
                claim_type="DATE",
                importance="MINOR",
                value=match.group(1),
                unit="date",
                as_of=trade_date,
                capture_status="FALLBACK_CANDIDATE",
                verification_status=VerificationStatus.NOT_EVALUATED,
            ).model_dump(mode="json")
        )
    return records
