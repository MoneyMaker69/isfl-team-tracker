"""
One Plotly template and deterministic team colours, so a franchise looks the
same on every chart in every session.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import config

_C = config.COLORS


def install_template() -> None:
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=_C["surface"],
        font=dict(family="Inter, system-ui, sans-serif", color=_C["text"], size=13),
        title=dict(font=dict(family="Barlow Condensed, Inter, sans-serif", size=22),
                   x=0, xanchor="left", pad=dict(b=12)),
        colorway=config.TEAM_PALETTE,
        xaxis=dict(gridcolor=_C["line"], zerolinecolor=_C["line"], linecolor=_C["line"],
                   tickfont=dict(color=_C["text_muted"]),
                   title=dict(font=dict(color=_C["text_muted"], size=12))),
        yaxis=dict(gridcolor=_C["line"], zerolinecolor=_C["line"], linecolor=_C["line"],
                   tickfont=dict(color=_C["text_muted"]),
                   title=dict(font=dict(color=_C["text_muted"], size=12))),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_C["text_muted"], size=12),
                    orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=50, b=10),
        hoverlabel=dict(bgcolor=_C["surface_alt"], bordercolor=_C["line"],
                        font=dict(color=_C["text"], family="Inter, sans-serif")),
    )
    pio.templates[config.PLOTLY_TEMPLATE] = template
    pio.templates.default = config.PLOTLY_TEMPLATE
    px.defaults.template = config.PLOTLY_TEMPLATE


def team_colors(teams: list[str]) -> dict[str, str]:
    out = {}
    palette = config.TEAM_PALETTE
    for i, t in enumerate(sorted(set(teams))):
        out[t] = config.TEAM_COLORS.get(t, palette[i % len(palette)])
    return out


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------------


def lines(frame: pd.DataFrame, x: str, y: str, color: str, colors: dict[str, str], title: str,
          y_label: str | None = None, highlight: list[str] | None = None, height: int = 520,
          hline: float | None = None, markers: bool = False) -> go.Figure:
    """Multi-series line chart. `highlight` dims every other series."""
    fig = go.Figure()
    for name, grp in frame.groupby(color):
        grp = grp.sort_values(x)
        dim = highlight and name not in highlight
        fig.add_trace(go.Scatter(
            x=grp[x], y=grp[y], mode="lines+markers" if markers else "lines", name=str(name),
            line=dict(color=colors.get(name, _C["text_muted"]), width=1.2 if dim else 2.6),
            opacity=0.25 if dim else 1.0,
            hovertemplate=f"{name} · %{{x}}: %{{y:.1f}}<extra></extra>",
        ))
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dot", line_color=_C["line"])
    fig.update_layout(title=title, height=height, hovermode="x unified",
                      xaxis_title=x.capitalize(), yaxis_title=y_label or y)
    return fig


def ranked_bar(frame: pd.DataFrame, label: str, value: str, colors: dict[str, str], title: str,
               fmt: str = "{:,.1f}", height: int | None = None) -> go.Figure:
    ordered = frame.sort_values(value, ascending=True)
    fig = go.Figure(go.Bar(
        x=ordered[value], y=ordered[label], orientation="h",
        marker_color=[colors.get(v, _C["primary"]) for v in ordered[label]],
        text=[fmt.format(v) for v in ordered[value]], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.2f}<extra></extra>",
    ))
    fig.update_layout(title=title, height=height or max(300, 26 * len(ordered) + 120),
                      showlegend=False, yaxis_title="", xaxis_title=value)
    return fig


def scatter(frame: pd.DataFrame, x: str, y: str, label: str, colors: dict[str, str], title: str,
            x_label: str | None = None, y_label: str | None = None, size: str | None = None,
            crosshairs: bool = True, trend: bool = False, height: int = 560,
            hover: list[str] | None = None) -> go.Figure:
    fig = go.Figure()
    sizes = 14
    if size and size in frame.columns and frame[size].notna().any():
        s = frame[size].astype(float)
        s = s.fillna(s.median())
        sizes = (10 + 22 * (s - s.min()) / max(1e-9, (s.max() - s.min()))).tolist()
    hover_text = [
        f"<b>{r[label]}</b><br>{x}: {r[x]:.1f}<br>{y}: {r[y]:.1f}"
        + "".join(f"<br>{h}: {r[h]}" for h in (hover or []) if h in frame.columns)
        for _, r in frame.iterrows()
    ]
    fig.add_trace(go.Scatter(
        x=frame[x], y=frame[y], mode="markers+text", text=frame[label], textposition="top center",
        textfont=dict(size=11, color=_C["text_muted"]),
        marker=dict(size=sizes, color=[colors.get(v, _C["primary"]) for v in frame[label]],
                    line=dict(width=1, color=_C["line"])),
        hovertext=hover_text, hoverinfo="text",
    ))
    if crosshairs:
        fig.add_hline(y=float(frame[y].median()), line_dash="dot", line_color=_C["line"])
        fig.add_vline(x=float(frame[x].median()), line_dash="dot", line_color=_C["line"])
    if trend and len(frame) > 2:
        xs, ys = frame[x].astype(float), frame[y].astype(float)
        ok = xs.notna() & ys.notna()
        if ok.sum() > 2 and xs[ok].std() > 0:
            m, b = np.polyfit(xs[ok], ys[ok], 1)
            xx = np.linspace(xs[ok].min(), xs[ok].max(), 20)
            fig.add_trace(go.Scatter(x=xx, y=m * xx + b, mode="lines", name="trend",
                                     line=dict(color=_C["primary_dim"], dash="dash"), showlegend=False))
    fig.update_layout(title=title, height=height, showlegend=False,
                      xaxis_title=x_label or x, yaxis_title=y_label or y)
    return fig


def heatmap(matrix: pd.DataFrame, title: str, fmt: str = "%{text:.0f}", zmid: float | None = None,
            colorscale: list | str | None = None, height: int | None = None) -> go.Figure:
    z = matrix.to_numpy(dtype=float)
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(c) for c in matrix.columns], y=[str(i) for i in matrix.index],
        colorscale=colorscale or [[0, _C["surface_alt"]], [0.5, _C["primary_dim"]], [1, _C["primary"]]],
        zmid=zmid, text=z, texttemplate=fmt, textfont=dict(size=11),
        hovertemplate="%{y} · %{x}: %{z:.2f}<extra></extra>", showscale=False, xgap=2, ygap=2,
    ))
    fig.update_layout(title=title, height=height or max(260, 30 * len(matrix) + 140),
                      yaxis=dict(autorange="reversed"))
    return fig


def diverging_scale() -> list:
    return [[0, _C["red"]], [0.5, _C["surface_alt"]], [1, _C["green"]]]


def band(frame: pd.DataFrame, x: str, low: str, mid: str, high: str, color: str, name: str,
         fig: go.Figure | None = None) -> go.Figure:
    """A median line with a shaded p10–p90 band."""
    fig = fig or go.Figure()
    f = frame.sort_values(x)
    fig.add_trace(go.Scatter(x=pd.concat([f[x], f[x][::-1]]), y=pd.concat([f[high], f[low][::-1]]),
                             fill="toself", fillcolor=_hex_to_rgba(color, 0.15),
                             line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=f[x], y=f[mid], mode="lines+markers", name=name,
                             line=dict(color=color, width=2.5)))
    return fig


def calibration_chart(frame: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=_C["line"], dash="dot"),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=frame["expected"], y=frame["observed"], mode="markers+lines",
                             marker=dict(size=np.clip(frame["games"] / 20, 6, 30), color=_C["primary"]),
                             line=dict(color=_C["primary_dim"]), name="observed",
                             hovertemplate="expected %{x:.2f} · observed %{y:.2f}<extra></extra>"))
    fig.update_layout(title=title, height=420, xaxis_title="Predicted home win probability",
                      yaxis_title="Observed home win rate", showlegend=False,
                      xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
    return fig


def grouped_bars(frame: pd.DataFrame, x: str, series: list[str], title: str,
                 colors: list[str] | None = None, height: int = 420, y_label: str = "") -> go.Figure:
    fig = go.Figure()
    palette = colors or [_C["primary"], _C["blue"], _C["green"], _C["text_muted"]]
    for i, s in enumerate(series):
        fig.add_trace(go.Bar(name=s, x=frame[x], y=frame[s], marker_color=palette[i % len(palette)]))
    fig.update_layout(barmode="group", title=title, height=height, yaxis_title=y_label)
    return fig


def signed_bar(frame: pd.DataFrame, label: str, value: str, title: str, fmt: str = "{:+.2f}",
               height: int | None = None) -> go.Figure:
    ordered = frame.sort_values(value)
    fig = go.Figure(go.Bar(
        x=ordered[value], y=ordered[label], orientation="h",
        marker_color=[_C["green"] if v >= 0 else _C["red"] for v in ordered[value]],
        text=[fmt.format(v) for v in ordered[value]], textposition="outside", cliponaxis=False,
    ))
    fig.add_vline(x=0, line_color=_C["line"])
    fig.update_layout(title=title, height=height or max(300, 26 * len(ordered) + 120),
                      showlegend=False, yaxis_title="")
    return fig


def strip(frame: pd.DataFrame, group: str, value: str, colors: dict[str, str], title: str,
          hover: list[str] | None = None, height: int = 520) -> go.Figure:
    order = frame.groupby(group)[value].mean().sort_values(ascending=False).index.tolist()
    fig = px.strip(frame, x=group, y=value, color=group, hover_data=hover or [],
                   category_orders={group: order}, color_discrete_map=colors, title=title,
                   labels={group: "", value: value})
    fig.update_traces(marker=dict(size=8, opacity=0.8), jitter=0.5)
    fig.update_layout(showlegend=False, height=height)
    return fig
