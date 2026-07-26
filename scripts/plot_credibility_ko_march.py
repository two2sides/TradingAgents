"""Generate credibility appendix figures for DATA2-KO-MARCH (one-off analysis script)."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

RUN_ID = "8d06aa833caa4022a048b6046a41acaf"
ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / "docs"
    / "multi_agent_credibility_report"
    / "figures"
    / "credibility_ko_march"
)
OUT.mkdir(parents=True, exist_ok=True)

CODE_CN = {
    "UNSOURCED_CLAIM": "主张未挂可定位证据",
    "NUMERIC_EVIDENCE_UNREADABLE": "数值证据字段不可读",
    "NUMERIC_SELECTOR_UNSPECIFIED": "数值引用未指定字段路径",
    "NUMERIC_EVIDENCE_MISMATCH": "数值与证据字段不一致",
    "NUMERIC_CLAIM_UNREADABLE": "数值主张本身不可解析",
    "TOOL_LOOKAHEAD": "工具日期可能越过决策时点",
}

STATUS_CN = {
    "AUDIT_PASSED": "通过",
    "AUDIT_DEGRADED": "建议改进",
    "AUDIT_FAILED": "需优先复核",
    "NONE": "无画像",
}
STATUS_COLOR = {
    "AUDIT_PASSED": "#2F6FED",
    "AUDIT_DEGRADED": "#C9842F",
    "AUDIT_FAILED": "#8B3A3A",
    "NONE": "#B0B0B0",
}

DIMS = [
    ("evidence_coverage", "证据覆盖"),
    ("numeric_verification_rate", "数值核验"),
    ("temporal_integrity", "时间完整"),
    ("tool_success_rate", "工具成功"),
    ("structured_output_integrity", "结构化完整"),
    ("debate_novelty_rate", "辩论新颖"),
    ("handoff_explanation_coverage", "交接解释"),
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Microsoft YaHei",
            "axes.unicode_minus": False,
            "figure.facecolor": "#F7F5F1",
            "axes.facecolor": "#FBFaf7",
            "axes.edgecolor": "#D0CBC3",
            "axes.labelcolor": "#2B2B2B",
            "text.color": "#2B2B2B",
            "xtick.color": "#4A4A4A",
            "ytick.color": "#4A4A4A",
            "grid.color": "#E6E1D8",
            "grid.linewidth": 0.8,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "figure.dpi": 160,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
            "savefig.facecolor": "#F7F5F1",
        }
    )


def load_result() -> dict:
    con = sqlite3.connect(ROOT / "local_data" / "runs.sqlite3")
    raw = con.execute("SELECT result_json FROM runs WHERE run_id=?", (RUN_ID,)).fetchone()[0]
    con.close()
    return json.loads(raw)


def build_tables(res: dict) -> tuple[list[dict], list[dict], dict]:
    exec_by_id = {e["decision_id"]: e for e in res["executions"]}
    eq = {p["timestamp"][:10]: p["total_equity"] for p in res["equity_curve"]}
    bh = {
        p["timestamp"][:10]: p["total_equity"]
        for p in res["benchmark_curves"]["BUY_HOLD:KO"]
    }
    # portfolio weight by date
    weights = {}
    for ph in res["portfolio_history"]:
        ko = (ph.get("positions") or {}).get("KO") or {}
        weights[ph["as_of"][:10]] = ko.get("weight")

    daily = []
    findings = []
    for d in res["decisions"]:
        intent = d["intent"]
        diag = d.get("diagnostics") or {}
        profile = diag.get("audit_profile") if isinstance(diag.get("audit_profile"), dict) else None
        eid = intent["decision_id"]
        ex = exec_by_id.get(eid, {})
        day = intent["as_of"][:10]
        row = {
            "date": day,
            "status": d["status"],
            "rating": diag.get("rating"),
            "target_weight": intent.get("target_weight"),
            "achieved_weight": ex.get("achieved_weight"),
            "exec_status": ex.get("status"),
            "fees": ex.get("fees") or 0.0,
            "has_profile": profile is not None,
            "audit_status": None if profile is None else profile.get("status"),
            "audit_scope": None if profile is None else profile.get("audit_scope"),
            "portfolio_weight": weights.get(day),
            "equity": eq.get(day),
            "bh_equity": bh.get(day),
            "n_claims": len(diag.get("claims") or []),
            "n_findings": len(diag.get("audit_findings") or []),
            "n_memory": len(diag.get("memory_provenance") or []),
            "profile": profile,
            "findings": diag.get("audit_findings") or [],
        }
        for key, _ in DIMS:
            row[key] = None if profile is None else profile.get(key)
        daily.append(row)
        for f in diag.get("audit_findings") or []:
            findings.append(
                {
                    "date": day,
                    "code": f.get("code"),
                    "severity": f.get("severity"),
                    "stage": f.get("stage"),
                    "message": (f.get("message") or "")[:160],
                }
            )

    # equity panel includes terminal Apr 1
    equity_panel = []
    for p in res["equity_curve"]:
        day = p["timestamp"][:10]
        equity_panel.append(
            {
                "date": day,
                "strategy": p["total_equity"],
                "bh": bh.get(day),
            }
        )

    metrics = res["metrics"]
    profile_days = [r for r in daily if r["has_profile"]]
    covs = [r["evidence_coverage"] for r in profile_days if r["evidence_coverage"] is not None]
    summary = {
        "run_id": RUN_ID,
        "label": "DATA2-KO-MARCH",
        "n_decisions": len(daily),
        "n_success": sum(r["status"] == "SUCCESS" for r in daily),
        "n_failed_safe": sum(r["status"] == "FAILED_SAFE" for r in daily),
        "n_profiles": len(profile_days),
        "total_return": metrics["total_return"],
        "bh_return": equity_panel[-1]["bh"] / equity_panel[0]["bh"] - 1,
        "alpha": metrics.get("alpha_vs_BUY_HOLD:KO"),
        "max_drawdown": metrics["max_drawdown"],
        "turnover": metrics["turnover"],
        "total_fees": metrics["total_fees"],
        "audit_counts": dict(Counter(r["audit_status"] for r in profile_days)),
        "evidence_coverage_mean": float(np.mean(covs)) if covs else None,
        "critical_findings": sum(f["severity"] == "CRITICAL" for f in findings),
        "warning_findings": sum(f["severity"] == "WARNING" for f in findings),
    }
    return daily, findings, {"summary": summary, "equity_panel": equity_panel}


def caption(ax, text: str) -> None:
    ax.text(
        0.0,
        -0.18,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#5A5A5A",
        wrap=True,
    )


def fig1(daily, equity_panel, summary) -> None:
    dates = [r["date"] for r in equity_panel]
    s0 = equity_panel[0]["strategy"]
    b0 = equity_panel[0]["bh"]
    s_nav = [r["strategy"] / s0 for r in equity_panel]
    b_nav = [r["bh"] / b0 for r in equity_panel]
    audit_by_day = {r["date"]: r["audit_status"] or "NONE" for r in daily}

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10.5, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(dates, s_nav, color="#1F4E79", lw=2.0, label="策略净值")
    ax1.plot(dates, b_nav, color="#6B8F71", lw=2.0, label="KO 买入持有")
    ax1.axhline(1.0, color="#C0C0C0", lw=0.8, ls="--")
    ax1.set_ylabel("归一化净值")
    ax1.set_title("Fig-1  KO 2024-03：净值表现与每日审计状态")
    ax1.grid(True, axis="y")
    ax1.legend(frameon=False, loc="upper left")

    colors = [STATUS_COLOR[audit_by_day.get(d, "NONE")] for d in dates[:-1]]
    # color band for decision days only; terminal mark grey
    ax2.bar(dates[:-1], [1] * (len(dates) - 1), color=colors, width=0.9, align="center")
    ax2.set_yticks([])
    ax2.set_ylabel("审计")
    ax2.set_xlim(-0.5, len(dates) - 0.5)
    for label in ax2.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
        label.set_fontsize(8)
    handles = [Patch(color=c, label=STATUS_CN[k]) for k, c in STATUS_COLOR.items() if k != "AUDIT_PASSED" or STATUS_CN[k]]
    # only show used
    used = set(audit_by_day.values())
    handles = [Patch(color=STATUS_COLOR[k], label=STATUS_CN[k]) for k in STATUS_COLOR if k in used]
    ax2.legend(handles=handles, frameon=False, loc="upper left", ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cred_ko_mar_01_overview.png")
    plt.close(fig)


def fig2(daily) -> None:
    counts = Counter(r["audit_status"] for r in daily if r["has_profile"])
    labels = [STATUS_CN[k] for k in ["AUDIT_DEGRADED", "AUDIT_FAILED", "AUDIT_PASSED"] if k in counts]
    values = [counts[k] for k in ["AUDIT_DEGRADED", "AUDIT_FAILED", "AUDIT_PASSED"] if k in counts]
    colors = [STATUS_COLOR[k] for k in ["AUDIT_DEGRADED", "AUDIT_FAILED", "AUDIT_PASSED"] if k in counts]
    scopes = Counter(r["audit_scope"] for r in daily if r["has_profile"])

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, str(v), ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("决策日数（SUCCESS 且有画像）")
    ax.set_title("Fig-2  审计状态构成")
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(True, axis="y")
    ax.text(
        0.98,
        0.95,
        f"审计范围：完整 {scopes.get('FULL', 0)} / 部分 {scopes.get('PARTIAL', 0)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#4A4A4A",
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_cred_ko_mar_02_audit_status.png")
    plt.close(fig)


def fig3(daily) -> None:
    dates = [r["date"] for r in daily]
    cov = [r["evidence_coverage"] for r in daily]
    n_claims = [r["n_claims"] for r in daily]
    fig, ax1 = plt.subplots(figsize=(10.2, 4.8))
    ax1.plot(dates, cov, color="#1F4E79", lw=2.0, marker="o", ms=4, label="证据覆盖率")
    if len(cov) >= 5:
        kernel = np.ones(5) / 5
        smooth = np.convolve([c if c is not None else np.nan for c in cov], kernel, mode="same")
        ax1.plot(dates, smooth, color="#1F4E79", lw=1.0, alpha=0.35, label="5 日平滑")
    ax1.set_ylabel("证据覆盖率")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, axis="y")
    ax2 = ax1.twinx()
    ax2.bar(dates, n_claims, color="#C4B8A5", alpha=0.45, width=0.7, label="主张条数")
    ax2.set_ylabel("当日主张条数")
    ax1.set_title("Fig-3  证据覆盖随时间变化")
    for label in ax1.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
        label.set_fontsize(8)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cred_ko_mar_03_evidence_coverage.png")
    plt.close(fig)


def fig4(findings) -> None:
    codes = Counter(f["code"] for f in findings)
    items = codes.most_common(8)
    labels = [CODE_CN.get(c, c) for c, _ in items]
    values = [n for _, n in items]
    colors = ["#8B3A3A" if c == "NUMERIC_EVIDENCE_MISMATCH" else "#6E7F8D" for c, _ in items]
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("出现次数（全窗合计）")
    ax.set_title("Fig-4  Findings 类型结构")
    ax.grid(True, axis="x")
    for yi, v in zip(y, values):
        ax.text(v + max(values) * 0.01, yi, str(v), va="center", fontsize=9)
    crit = sum(f["severity"] == "CRITICAL" for f in findings)
    warn = sum(f["severity"] == "WARNING" for f in findings)
    ax.text(
        0.98,
        0.02,
        f"CRITICAL {crit} · WARNING {warn}",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        color="#4A4A4A",
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_cred_ko_mar_04_findings.png")
    plt.close(fig)


def fig5(daily) -> None:
    """Heatmap + mean bars so all seven dims stay visible even when variance is zero."""
    dates = [r["date"][5:] for r in daily]  # MM-DD
    names = [name for _, name in DIMS]
    keys = [key for key, _ in DIMS]
    mat = np.full((len(keys), len(dates)), np.nan, dtype=float)
    for j, r in enumerate(daily):
        for i, key in enumerate(keys):
            v = r.get(key)
            if v is not None:
                mat[i, j] = float(v)

    fig = plt.figure(figsize=(11.2, 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.6], hspace=0.38)
    ax_h = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    cmap = plt.cm.Blues.copy()
    cmap.set_bad("#E8E4DC")
    im = ax_h.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax_h.set_yticks(np.arange(len(names)))
    ax_h.set_yticklabels(names)
    ax_h.set_xticks(np.arange(len(dates)))
    ax_h.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
    ax_h.set_title("Fig-5  七维审计画像：逐日热力图 + 月均水平（灰格=当日无分母/不适用）")
    cbar = fig.colorbar(im, ax=ax_h, fraction=0.025, pad=0.02)
    cbar.set_label("指标取值", fontsize=9)
    # annotate NA cells lightly
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(mat[i, j]):
                ax_h.text(j, i, "NA", ha="center", va="center", fontsize=6, color="#8A847A")

    means, ns, na_rates = [], [], []
    for i, key in enumerate(keys):
        vals = mat[i][~np.isnan(mat[i])]
        means.append(float(np.mean(vals)) if len(vals) else 0.0)
        ns.append(len(vals))
        na_rates.append(1.0 - len(vals) / len(dates))
    y = np.arange(len(names))
    colors = []
    for mean, na in zip(means, na_rates):
        if na >= 0.5:
            colors.append("#B7B1A6")
        elif mean >= 0.85:
            colors.append("#4C7A9A")
        else:
            colors.append("#C9842F")
    ax_b.barh(y, means, color=colors, height=0.62, edgecolor="#FFFFFF", linewidth=0.6)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(names)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 1.05)
    ax_b.set_xlabel("有值日期的月均水平")
    ax_b.grid(True, axis="x")
    for yi, mean, n, na in zip(y, means, ns, na_rates):
        label = f"{mean:.2f}  (n={n}/{len(dates)}"
        if na > 0:
            label += f", NA {na:.0%}"
        label += ")"
        ax_b.text(min(mean + 0.02, 0.72), yi, label, va="center", fontsize=8, color="#333333")

    fig.subplots_adjust(left=0.14, right=0.98, top=0.93, bottom=0.08, hspace=0.40)
    fig.savefig(OUT / "fig_cred_ko_mar_05_profile_dims.png")
    plt.close(fig)


def pick_sample(daily) -> dict:
    scored = []
    for r in daily:
        if not r["profile"]:
            continue
        cov = r["evidence_coverage"] or 0
        present = sum(r.get(k) is not None for k, _ in DIMS)
        score = present * 2
        if 0.25 <= cov <= 0.75:
            score += 2
        if r["audit_status"] in {"AUDIT_DEGRADED", "AUDIT_FAILED"}:
            score += 1
        if any(f.get("code") in CODE_CN for f in r["findings"]):
            score += 1
        scored.append((score, present, cov, r))
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return scored[0][3]


def fig6(sample: dict) -> None:
    """Always render seven dims; NA shown as hollow/hatched placeholder."""
    labels = [name for _, name in DIMS]
    keys = [key for key, _ in DIMS]
    vals = []
    is_na = []
    for key in keys:
        v = sample.get(key)
        if v is None:
            vals.append(0.0)
            is_na.append(True)
        else:
            vals.append(float(v))
            is_na.append(False)

    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    y = np.arange(len(labels))
    colors = ["#D0CBC3" if na else "#3D6B8C" for na in is_na]
    bars = ax.barh(y, vals, color=colors, height=0.58, edgecolor="#FFFFFF")
    for bar, na in zip(bars, is_na):
        if na:
            bar.set_hatch("///")
            bar.set_edgecolor("#9A948A")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0, 1.08)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.7)
    ax.set_xlabel("指标取值（斜线填充 = 当日不适用/无分母）")
    for yi, v, na in zip(y, vals, is_na):
        ax.text(
            0.02 if na else min(v + 0.02, 0.90),
            yi,
            "不适用 (NA)" if na else f"{v:.2f}",
            va="center",
            fontsize=9,
            color="#666666" if na else "#1F1F1F",
            fontweight="regular",
        )
    ax.set_title(
        f"Fig-6  单日深潜 {sample['date']} · {STATUS_CN.get(sample['audit_status'], sample['audit_status'])} · "
        f"Rating={sample['rating']} · 目标仓位 {sample['target_weight']:.0%}"
    )

    # unique finding codes only
    lines = [
        f"证据覆盖：{sample['evidence_coverage']:.1%}" if sample.get("evidence_coverage") is not None else "证据覆盖：NA",
        f"主张 {sample['n_claims']} 条 · findings {sample['n_findings']} 条",
        "主要问题：",
    ]
    seen = []
    for f in sample["findings"]:
        code = f.get("code")
        if code in CODE_CN and code not in seen:
            lines.append(f"· {CODE_CN[code]}（{f.get('severity')}）")
            seen.append(code)
        if len(seen) >= 4:
            break
    ax.text(
        1.03,
        0.5,
        "\n".join(lines),
        transform=ax.transAxes,
        va="center",
        fontsize=9,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#F0EBE3", "edgecolor": "#D0CBC3"},
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_cred_ko_mar_06_sample_day.png")
    plt.close(fig)


def fig7(daily) -> None:
    # rich position changes -> enable alternate
    dates = [r["date"] for r in daily]
    tw = [r["target_weight"] if r["target_weight"] is not None else np.nan for r in daily]
    rating_color = {
        "Sell": "#8B3A3A",
        "Underweight": "#C9842F",
        "Hold": "#2F6FED",
        "Overweight": "#2F6FED",
        "Buy": "#2F6FED",
    }
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    ax.step(dates, tw, where="mid", color="#1F4E79", lw=1.8, label="目标仓位")
    for r in daily:
        ax.scatter(
            r["date"],
            r["target_weight"],
            color=rating_color.get(r["rating"], "#888888"),
            s=46,
            zorder=3,
        )
    ax.set_ylim(-0.05, 0.55)
    ax.set_ylabel("目标仓位")
    ax.set_title("Fig-7  评级与目标仓位路径（候补启用）")
    ax.grid(True, axis="y")
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
        label.set_fontsize(8)
    handles = [
        Patch(color=rating_color["Sell"], label="Sell"),
        Patch(color=rating_color["Underweight"], label="Underweight"),
        Patch(color=rating_color["Hold"], label="Hold"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_cred_ko_mar_07_rating_weight.png")
    plt.close(fig)


def write_tables(summary: dict, sample: dict) -> Path:
    path = OUT / "tables.md"
    ac = summary["audit_counts"]
    lines = [
        "# KO March Tables",
        "",
        "## Table-A 跑次摘要",
        "",
        "| 项目 | 数值 |",
        "|------|------|",
        f"| 跑次 | `{summary['label']}` (`{summary['run_id'][:8]}…`) |",
        "| 窗口 | 2024-03-01 → 2024-04-01 |",
        f"| 决策日 | {summary['n_decisions']}（SUCCESS {summary['n_success']} / FAILED_SAFE {summary['n_failed_safe']}） |",
        f"| 策略收益 | {summary['total_return']:.2%} |",
        f"| KO 买入持有 | {summary['bh_return']:.2%} |",
        f"| 相对 Alpha | {summary['alpha']:.2%} |",
        f"| 最大回撤 | {summary['max_drawdown']:.2%} |",
        f"| 换手 / 费用 | {summary['turnover']:.2f} / ${summary['total_fees']:.2f} |",
        f"| 画像覆盖 | {summary['n_profiles']}/{summary['n_success']} |",
        f"| 审计状态 | 建议改进 {ac.get('AUDIT_DEGRADED',0)} · 需复核 {ac.get('AUDIT_FAILED',0)} · 通过 {ac.get('AUDIT_PASSED',0)} |",
        f"| 证据覆盖均值 | {summary['evidence_coverage_mean']:.1%} |",
        f"| Findings | CRITICAL {summary['critical_findings']} · WARNING {summary['warning_findings']} |",
        "",
        "## Table-B 主样例日",
        "",
        "| 项目 | 数值 |",
        "|------|------|",
        f"| 日期 | {sample['date']} |",
        f"| Rating / 目标仓位 | {sample['rating']} / {sample['target_weight']:.0%} |",
        f"| 审计状态 / 范围 | {STATUS_CN.get(sample['audit_status'])} / {sample['audit_scope']} |",
        f"| 证据覆盖 | {sample['evidence_coverage']:.1%} |",
        f"| 结构化完整 | {sample.get('structured_output_integrity')} |",
        f"| 工具成功 | {sample.get('tool_success_rate')} |",
        f"| 主张 / findings | {sample['n_claims']} / {sample['n_findings']} |",
        "",
        "代表问题：",
    ]
    seen = set()
    for f in sample["findings"]:
        code = f.get("code")
        if code in seen:
            continue
        if code in CODE_CN:
            lines.append(f"- {CODE_CN[code]}（{f.get('severity')}）")
            seen.add(code)
        if len(seen) >= 4:
            break
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    setup_style()
    res = load_result()
    daily, findings, bundle = build_tables(res)
    summary = bundle["summary"]
    equity_panel = bundle["equity_panel"]
    sample = pick_sample(daily)

    fig1(daily, equity_panel, summary)
    fig2(daily)
    fig3(daily)
    fig4(findings)
    fig5(daily)
    fig6(sample)
    fig7(daily)
    write_tables(summary, sample)

    meta = {
        "run_id": RUN_ID,
        "sample_day": sample["date"],
        "summary": summary,
        "figures": sorted(p.name for p in OUT.glob("fig_*.png")),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT)
    print("sample_day", sample["date"], sample["audit_status"], sample["evidence_coverage"])
    print("figs", meta["figures"])


if __name__ == "__main__":
    main()
