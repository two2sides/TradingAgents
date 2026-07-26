"""Field-selector based numeric verification."""

from __future__ import annotations

import json
from typing import Any

from ..models import VerificationFinding, stable_id


def _is_field_selector(selector: Any) -> bool:
    """True only when the ref points at a concrete JSON field, not the whole payload."""
    text = str(selector or "").strip()
    return text not in {"", "/"}


def _json_pointer(value: Any, selector: str) -> Any:
    if not _is_field_selector(selector):
        raise KeyError("selector must point to a concrete field")
    current = value
    for part in str(selector).lstrip("/").split("/"):
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
        and not (event.get("payload") or {}).get("post_run_recomputed")
    }
    findings = []
    for claim in claims:
        if claim.get("claim_type") != "NUMERIC" or not claim.get("evidence_refs"):
            continue
        expected = claim.get("value")
        try:
            expected_num = float(expected)
        except (TypeError, ValueError):
            code = "NUMERIC_CLAIM_UNREADABLE"
            findings.append(
                VerificationFinding(
                    finding_id=stable_id(
                        "finding", {"rule": code, "claim": claim.get("claim_id")}
                    ),
                    rule_id="numeric.claim_value",
                    code=code,
                    severity="WARNING",
                    stage=claim.get("stage", "unknown"),
                    claim_id=claim.get("claim_id"),
                    actual=expected,
                    message="数值主张无法解析；该主张保持 UNKNOWN，不计为核验通过。",
                ).model_dump(mode="json")
            )
            claim["verification_status"] = "NOT_EVALUATED"
            continue

        evaluated = False
        contradicted = False
        had_artifact = False
        had_field_selector = False
        had_coarse_selector = False

        for ref in claim["evidence_refs"]:
            artifact = artifacts.get(ref.get("artifact_id"))
            if not artifact:
                continue
            had_artifact = True
            selector = ref.get("selector") or "/"
            if not _is_field_selector(selector):
                # Root "/" only says "this artifact is related"; it is not a numeric field ref.
                had_coarse_selector = True
                continue
            had_field_selector = True
            try:
                payload = json.loads(artifact.get("artifact_content") or "")
                actual = _json_pointer(payload, selector)
                actual_num = float(actual)
            except (ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError):
                continue
            evaluated = True
            tolerance = float(claim.get("tolerance") or 0.001)
            if claim.get("unit") == "percent" and abs(actual_num) <= 1:
                actual_num *= 100
            if abs(expected_num - actual_num) > tolerance:
                contradicted = True
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
                        location=selector,
                        message="数值主张与其引用的 Artifact 字段不一致。",
                    ).model_dump(mode="json")
                )

        if contradicted:
            claim["verification_status"] = "CONTRADICTED"
        elif evaluated:
            claim["verification_status"] = "SUPPORTED"
        else:
            claim["verification_status"] = "NOT_EVALUATED"
            if had_artifact and not had_field_selector and had_coarse_selector:
                code = "NUMERIC_SELECTOR_UNSPECIFIED"
                findings.append(
                    VerificationFinding(
                        finding_id=stable_id(
                            "finding",
                            {"rule": code, "claim": claim.get("claim_id")},
                        ),
                        rule_id="numeric.selector_required",
                        code=code,
                        severity="WARNING",
                        stage=claim.get("stage", "unknown"),
                        claim_id=claim.get("claim_id"),
                        message=(
                            "数值主张只引用了 Artifact 整体（selector 为 '/' 或空），"
                            "未给出可核对的字段路径；不计为核验通过。"
                        ),
                    ).model_dump(mode="json")
                )
            elif had_artifact and had_field_selector:
                code = "NUMERIC_EVIDENCE_UNREADABLE"
                findings.append(
                    VerificationFinding(
                        finding_id=stable_id(
                            "finding",
                            {"rule": code, "claim": claim.get("claim_id")},
                        ),
                        rule_id="numeric.artifact_selector",
                        code=code,
                        severity="WARNING",
                        stage=claim.get("stage", "unknown"),
                        claim_id=claim.get("claim_id"),
                        message=(
                            "证据字段无法按 selector 解析为数值；"
                            "该主张保持 UNKNOWN，不计为核验通过。"
                        ),
                    ).model_dump(mode="json")
                )
    return findings
