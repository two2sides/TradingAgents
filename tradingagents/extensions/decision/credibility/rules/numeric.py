"""Field-selector based numeric verification."""

from __future__ import annotations

import json
from typing import Any

from ..models import VerificationFinding, stable_id


def _json_pointer(value: Any, selector: str) -> Any:
    if selector in {"", "/"}:
        return value
    current = value
    for part in selector.lstrip("/").split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            current = current[int(part)]
        elif isinstance(current, dict):
            current = current[part]
        else:
            raise KeyError(part)
    return current


def check_numeric_rules(claims: list[dict[str, Any]], events: list[dict[str, Any]]):
    artifacts = {
        (event.get("payload") or {}).get("artifact_id"): (event.get("payload") or {})
        for event in events
        if (event.get("payload") or {}).get("artifact_id")
    }
    findings = []
    for claim in claims:
        if claim.get("claim_type") != "NUMERIC" or not claim.get("evidence_refs"):
            continue
        expected = claim.get("value")
        try:
            expected_num = float(expected)
        except (TypeError, ValueError):
            continue
        for ref in claim["evidence_refs"]:
            artifact = artifacts.get(ref.get("artifact_id"))
            if not artifact:
                continue
            try:
                payload = json.loads(artifact.get("artifact_content") or "")
                actual = _json_pointer(payload, ref.get("selector") or "/")
                actual_num = float(actual)
            except (ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError):
                continue
            tolerance = float(claim.get("tolerance") or 0.001)
            if claim.get("unit") == "percent" and abs(actual_num) <= 1:
                actual_num *= 100
            if abs(expected_num - actual_num) > tolerance:
                code = "NUMERIC_EVIDENCE_MISMATCH"
                findings.append(
                    VerificationFinding(
                        finding_id=stable_id(
                            "finding",
                            {"rule": code, "claim": claim.get("claim_id"), "ref": ref},
                        ),
                        rule_id="numeric.field_match",
                        code=code,
                        severity="CRITICAL",
                        stage=claim.get("stage", "unknown"),
                        claim_id=claim.get("claim_id"),
                        artifact_id=ref.get("artifact_id"),
                        expected=expected_num,
                        actual=actual_num,
                        location=ref.get("selector"),
                        message="数值主张与其引用的 Artifact 字段不一致。",
                    ).model_dump(mode="json")
                )
    return findings
