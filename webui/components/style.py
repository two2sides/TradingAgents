"""Small HTML helpers for the Decision Lab visual system."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"


def apply_global_style() -> None:
    css = (ASSET_DIR / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_hero(eyebrow: str, title: str, description: str, *, accent: str = "LIVE") -> None:
    st.markdown(
        f"""
        <section class="ta-hero">
          <div class="ta-hero-copy">
            <div class="ta-eyebrow">{escape(eyebrow)}</div>
            <h1>{escape(title)}</h1>
            <p>{escape(description)}</p>
          </div>
          <div class="ta-pulse"><span></span>{escape(accent)}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section(title: str, caption: str | None = None, *, index: str | None = None) -> None:
    prefix = f'<span class="ta-section-index">{escape(index)}</span>' if index else ""
    subtitle = f"<p>{escape(caption)}</p>" if caption else ""
    st.markdown(
        f'<div class="ta-section-title">{prefix}<div><h2>{escape(title)}</h2>{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def render_metric_grid(metrics: list[dict[str, Any]]) -> None:
    cards = []
    for metric in metrics:
        tone = escape(str(metric.get("tone", "neutral")))
        delta = metric.get("delta")
        delta_html = (
            f'<span class="ta-metric-delta">{escape(str(delta))}</span>'
            if delta is not None
            else ""
        )
        cards.append(
            f"""
            <article class="ta-metric-card ta-tone-{tone}">
              <div class="ta-metric-label">{escape(str(metric["label"]))}</div>
              <div class="ta-metric-value">{escape(str(metric["value"]))}</div>
              {delta_html}
            </article>
            """
        )
    st.markdown(f'<div class="ta-metric-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_badges(items: list[tuple[str, str]]) -> None:
    badges = "".join(
        f'<span class="ta-badge ta-badge-{escape(tone)}">{escape(label)}</span>'
        for label, tone in items
    )
    st.markdown(f'<div class="ta-badges">{badges}</div>', unsafe_allow_html=True)


def render_callout(title: str, body: str, *, tone: str = "cyan") -> None:
    st.markdown(
        f"""
        <div class="ta-callout ta-callout-{escape(tone)}">
          <div class="ta-callout-title">{escape(title)}</div>
          <div>{escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="ta-empty">
          <div class="ta-empty-orbit"><span></span></div>
          <h3>{escape(title)}</h3>
          <p>{escape(body)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_percent(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.2%}" if signed else f"{value:.2%}"


def format_money(value: float | None) -> str:
    if value is None:
        return "—"
    return f"${value:,.0f}"


__all__ = [
    "apply_global_style",
    "format_money",
    "format_percent",
    "render_badges",
    "render_callout",
    "render_empty",
    "render_hero",
    "render_metric_grid",
    "render_section",
]
