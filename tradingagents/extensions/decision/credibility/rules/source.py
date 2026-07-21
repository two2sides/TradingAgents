"""Source and artifact-reference credibility rules."""

from __future__ import annotations

from typing import Any

from ..models import VerificationFinding, stable_id


def check_source_rules(claims: list[dict[str, Any]], events: list[dict[str, Any]]):
    artifacts = {
        (event.get("payload") or {}).get("artifact_id")
        for event in events
        if (event.get("payload") or {}).get("artifact_id")
    }
    findings = []
    for claim in claims:
        if claim.get("claim_type") == "OPINION":
            continue
        refs = claim.get("evidence_refs") or []
        if claim.get("importance") in {"CRITICAL", "MAJOR"} and not refs:
            code = "UNSOURCED_CLAIM"
            findings.append(
                VerificationFinding(
                    finding_id=stable_id("finding", {"rule": code, "claim": claim.get("claim_id")}),
                    rule_id="source.claim_has_reference",
                    code=code,
                    severity="WARNING",
                    stage=claim.get("stage", "unknown"),
                    claim_id=claim.get("claim_id"),
                    message="关键主张没有可定位的 ArtifactFieldRef。",
                ).model_dump(mode="json")
            )
            continue
        for ref in refs:
            if ref.get("artifact_id") not in artifacts:
                code = "MISSING_ARTIFACT"
                findings.append(
                    VerificationFinding(
                        finding_id=stable_id(
                            "finding",
                            {"rule": code, "claim": claim.get("claim_id"), "ref": ref},
                        ),
                        rule_id="source.artifact_exists",
                        code=code,
                        severity="CRITICAL",
                        stage=claim.get("stage", "unknown"),
                        claim_id=claim.get("claim_id"),
                        artifact_id=ref.get("artifact_id"),
                        message="主张引用的 Artifact 不存在于本次事件流。",
                    ).model_dump(mode="json")
                )
    return findings
