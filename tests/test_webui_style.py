from webui.components import style


def test_render_metric_grid_uses_html_renderer(monkeypatch):
    rendered: list[str] = []
    monkeypatch.setattr(style.st, "html", rendered.append)

    style.render_metric_grid(
        [
            {
                "label": "Total <return>",
                "value": "+14.55%",
                "tone": "positive",
            }
        ]
    )

    assert len(rendered) == 1
    assert '<article class="ta-metric-card ta-tone-positive">' in rendered[0]
    assert "Total &lt;return&gt;" in rendered[0]
    assert "+14.55%" in rendered[0]
