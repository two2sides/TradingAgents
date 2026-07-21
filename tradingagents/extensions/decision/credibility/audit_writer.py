"""Write the canonical event stream and derived credibility audit views."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tradingagents.extensions.contracts import TraceEvent

from .claims import extract_candidate_claims
from .evidence import artifact_payload
from .models import stable_id
from .profile import build_audit_profile
from .verifier import run_verifier


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_credibility_markdown(profile: dict[str, Any], findings: list[dict]) -> str:
    lines = [
        "## VII. 可信度审计（不修改 Portfolio Manager Rating）",
        "",
        f"- **审计状态**: `{profile['status']}`",
        f"- **审计范围**: `{profile['audit_scope']}`",
        "- 本节检查来源、时间、结构化降级和流程完整性；"
        "不证明数据真伪、金融因果或未来收益。",
        "",
        "### 指标",
        "",
    ]
    labels = {
        "evidence_coverage": "关键主张证据覆盖率",
        "numeric_verification_rate": "数值/日期核验通过率",
        "temporal_integrity": "时间完整性",
        "tool_success_rate": "工具成功率",
        "structured_output_integrity": "结构化输出完整性",
        "handoff_explanation_coverage": "交接修改解释覆盖率",
        "debate_novelty_rate": "辩论完整周期新颖率",
    }
    for key, label in labels.items():
        value = profile.get(key)
        rendered = "N/A" if value is None else f"{100 * value:.1f}%"
        lines.append(f"- {label}: **{rendered}**")
    if profile.get("audit_scope_reasons"):
        lines.extend(["", "### 范围限制", ""])
        lines.extend(f"- {reason}" for reason in profile["audit_scope_reasons"])
    if findings:
        lines.extend(["", "### Findings", ""])
        grouped = Counter(
            (
                finding["severity"],
                finding["code"],
                finding["stage"],
                finding["message"],
            )
            for finding in findings
        )
        for (severity, code, stage, message), count in grouped.items():
            count_text = f" ×{count}" if count > 1 else ""
            lines.append(
                f"- **{severity} · {code}{count_text}** "
                f"({stage}): {message}"
            )
    else:
        lines.extend(["", "未发现适用规则能够确定识别的问题。"])
    return "\n".join(lines)


def write_audit_bundle(
    final_state: dict[str, Any], ticker: str, save_path: Path
) -> tuple[str, dict[str, Any]]:
    audit_dir = Path(save_path) / "audit"
    artifacts_dir = audit_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    working = copy.deepcopy(final_state)
    candidate_claims = []
    structured_stages = {
        claim.get("stage")
        for claim in working.get("claims") or []
        if claim.get("capture_status") == "STRUCTURED"
    }
    for key, agent, stage in (
        ("market_report", "Market Analyst", "market"),
        ("news_report", "News Analyst", "news"),
        ("fundamentals_report", "Fundamentals Analyst", "fundamentals"),
    ):
        if working.get(key) and stage not in structured_stages:
            candidate_claims.extend(
                extract_candidate_claims(
                    run_id=str(working.get("run_id", "unknown-run")),
                    agent=agent,
                    stage=stage,
                    text=str(working[key]),
                    trade_date=str(working.get("trade_date")),
                )
            )
    working["claims"] = list(working.get("claims") or []) + candidate_claims
    if candidate_claims:
        working.setdefault("audit_scope_reasons", []).append(
            "one or more analyst reports lack a complete structured claim sidecar"
        )
    for index, invocation in enumerate(working.get("structured_invocations") or []):
        event_id = stable_id(
            "event",
            {
                "run_id": working.get("run_id"),
                "type": "STRUCTURED_INVOCATION",
                "agent": invocation.get("agent_name"),
                "index": index,
            },
        )
        working.setdefault("audit_events", []).append(
            TraceEvent(
                timestamp=datetime.now(timezone.utc),
                source=str(invocation.get("agent_name") or "structured_agent"),
                event_type="STRUCTURED_INVOCATION",
                summary=(
                    f"{invocation.get('agent_name')}: {invocation.get('mode')}"
                ),
                payload={
                    "schema_version": "1.0",
                    "run_id": working.get("run_id"),
                    "event_id": event_id,
                    **invocation,
                },
            ).model_dump(mode="json")
        )

    events, claims, findings = run_verifier(working)
    profile_model = build_audit_profile(working, events, claims, findings)
    profile = profile_model.model_dump(mode="json")

    event_rows = []
    for event in events:
        row = copy.deepcopy(event)
        payload = row.get("payload") or {}
        artifact = artifact_payload(event)
        if artifact:
            artifact_id, content = artifact
            (artifacts_dir / f"{artifact_id}.txt").write_text(content, encoding="utf-8")
            payload["payload_ref"] = f"artifacts/{artifact_id}.txt"
            payload.pop("artifact_content", None)
        event_rows.append(row)

    _jsonl(audit_dir / "event_trace.jsonl", event_rows)
    _jsonl(audit_dir / "claims.jsonl", claims)
    _jsonl(audit_dir / "findings.jsonl", findings)
    (audit_dir / "profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    derived_files = [
        audit_dir / "event_trace.jsonl",
        audit_dir / "claims.jsonl",
        audit_dir / "findings.jsonl",
        audit_dir / "profile.json",
    ]
    manifest = {
        "schema_version": "1.0",
        "run_id": working.get("run_id"),
        "ticker": ticker,
        "trade_date": working.get("trade_date"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "canonical": ["event_trace.jsonl", "artifacts/"],
        "derived": ["claims.jsonl", "findings.jsonl", "profile.json", "credibility.md"],
        "event_count": len(events),
        "claim_count": len(claims),
        "finding_count": len(findings),
        "structured_fallbacks": [
            item.get("agent_name")
            for item in working.get("structured_invocations") or []
            if item.get("mode") != "STRUCTURED"
        ],
        "file_hashes": {
            path.name: _sha256(path) for path in derived_files if path.exists()
        },
    }
    (audit_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = render_credibility_markdown(profile, findings)
    (audit_dir / "credibility.md").write_text(markdown, encoding="utf-8")
    return markdown, profile
