"""Shared Streamlit helpers: theming, notices, formatting, exports."""

from __future__ import annotations

from string import Template

import pandas as pd
import streamlit as st

import config
from data import Notice

_C = config.COLORS

# Defined at module level and flush against the left margin on purpose.
# Streamlit's markdown renderer treats any line indented four or more spaces as
# a code block, so an indented <style> block gets partially applied and the
# rest printed to the page as literal text. Injected via st.html, which
# bypasses the markdown parser entirely.

_CSS_TEMPLATE = """
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

.stApp { background: $bg; }

/* font-family on the container only. Enumerating span/div reaches Streamlit's
Material icon elements and overrides their icon font, so icons render as their
ligature names ("warning"). */
.stApp {
font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
.stApp input, .stApp textarea, .stApp select, .stApp button {
font-family: inherit;
}
[data-testid="stIconMaterial"],
span[class*="material-symbols"],
.material-symbols-rounded, .material-symbols-outlined {
font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important;
font-feature-settings: 'liga';
}

h1, h2, h3 {
font-family: 'Barlow Condensed', 'Inter', sans-serif !important;
letter-spacing: 0.01em;
color: $text !important;
}
h1 { font-size: 2.6rem !important; font-weight: 700 !important; }
h2 { font-size: 1.7rem !important; font-weight: 600 !important; }
h3 { font-size: 1.25rem !important; font-weight: 600 !important; }

.stApp, .stApp p, .stApp li, .stApp label { color: $text; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
color: $muted !important;
}

[data-testid="stMetricValue"] {
font-family: 'Barlow Condensed', sans-serif;
font-size: 2.1rem;
color: $text;
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p { color: $muted; }

[data-testid="stSidebar"] {
background: $surface;
border-right: 1px solid $line;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] h3 { color: $text; }

.isfl-status {
display: flex; align-items: baseline; gap: 0.6rem;
padding: 0.5rem 0.85rem; margin-bottom: 1rem;
background: $surface;
border-left: 3px solid $primary;
border-radius: 3px;
color: $muted; font-size: 0.85rem;
}
.isfl-status strong { color: $text; font-weight: 600; }

.isfl-insight {
padding: 0.7rem 0.9rem; margin-bottom: 0.5rem;
background: $surface;
border-left: 3px solid $primary_dim;
border-radius: 3px;
}
.isfl-insight .k { color: $muted; font-size: 0.78rem; display: block; }
.isfl-insight .v { color: $text; font-size: 0.98rem; font-weight: 500; }

.isfl-empty {
padding: 2rem 1.5rem; text-align: left;
background: $surface;
border: 1px dashed $line; border-radius: 4px;
color: $muted;
}
.isfl-empty strong {
color: $text; display: block;
font-family: 'Barlow Condensed', sans-serif; font-size: 1.3rem;
margin-bottom: 0.35rem;
}

.isfl-rank {
display: flex; align-items: center; gap: 0.7rem;
padding: 0.45rem 0.8rem; margin-bottom: 0.35rem;
background: $surface; border-radius: 4px; border: 1px solid $line;
}
.isfl-rank .n {
font-family: 'Barlow Condensed', sans-serif; font-size: 1.3rem; font-weight: 700;
color: $primary; width: 1.6rem;
}
.isfl-rank .t { color: $text; font-weight: 600; }
.isfl-rank .p { color: $muted; font-size: 0.8rem; margin-left: auto; }

.stTabs [data-baseweb="tab"] { font-weight: 500; }
hr { border-color: $line; }
"""

# Template rather than .format(): CSS is full of literal braces.
_CSS = Template(_CSS_TEMPLATE).substitute(
    bg=_C["bg"], surface=_C["surface"], line=_C["line"], text=_C["text"],
    muted=_C["text_muted"], primary=_C["primary"], primary_dim=_C["primary_dim"],
)


def _raw_html(markup: str) -> None:
    if hasattr(st, "html"):
        st.html(markup)
    else:
        st.markdown(markup, unsafe_allow_html=True)


def inject_css() -> None:
    _raw_html("<style>" + _CSS + "</style>")


def render_notices(notices: list[Notice], *, only: set[str] | None = None) -> None:
    for n in notices:
        if only and n.level not in only:
            continue
        body = f"**{n.title}**" + (f"\n\n{n.detail}" if n.detail else "")
        if n.level == "error":
            st.error(body, icon=":material/error:")
        elif n.level == "warning":
            st.warning(body, icon=":material/warning:")
        else:
            st.info(body, icon=":material/info:")


def empty_state(headline: str, guidance: str) -> None:
    _raw_html(f'<div class="isfl-empty"><strong>{headline}</strong>{guidance}</div>')


def status_bar(state: str, detail: str) -> None:
    _raw_html(f'<div class="isfl-status"><strong>{state}</strong>{detail}</div>')


def insight(label: str, value: str) -> None:
    _raw_html(f'<div class="isfl-insight"><span class="k">{label}</span><span class="v">{value}</span></div>')


def rank_row(n: int, team: str, note: str = "") -> None:
    _raw_html(f'<div class="isfl-rank"><span class="n">{n}</span><span class="t">{team}</span>'
              f'<span class="p">{note}</span></div>')


def section(title: str, note: str = "") -> None:
    st.subheader(title)
    if note:
        st.caption(note)


def explain(intro: str, terms: dict[str, str] | None = None, *, expanded: bool = False) -> None:
    """
    A plain-language 'What am I looking at?' box at the top of a page.

    `intro` says what the page is for in one or two sentences; `terms` maps
    each number or word on the page to a one-line explanation.
    """
    with st.expander("What am I looking at?", expanded=expanded):
        st.markdown(intro)
        if terms:
            st.markdown("\n".join(f"- **{k}** — {v}" for k, v in terms.items()))


GLOSSARY = {
    "TPE": "Training Points Earned. The currency players earn by doing weekly tasks and spend on attributes. "
           "More TPE = better player. Rookies start at 50; a strong veteran has 1,200–1,500.",
    "Starter TPE": "The average TPE of the players who would actually start: the best QB, best RB, best 3 WR, "
                   "best TE, best 5 OL (bots included), best 4 DL, 3 LB, 5 DB and the kicker. This ignores "
                   "backups, which is what the sim does too.",
    "Elo": "A rating that goes up when you beat teams and up more when you beat good teams. 1500 is average. "
           "Every offseason a third of the gap to 1500 is wiped, because rosters change.",
    "SRS": "Points per game better (or worse) than an average team, after adjusting for who you played. "
           "+5 means you'd beat an average team by 5. Built from points for/against and the schedule.",
    "Pythagorean wins": "How many games a team 'should' have won given its points scored and allowed. "
                        "If actual wins are higher, they got lucky — and lucky teams usually fall back.",
    "Luck": "Actual wins minus Pythagorean wins.",
    "Career season": "How many seasons since the player's ISFL draft, counting the draft season as 1.",
    "Regression": "Older players lose TPE every offseason: 20% after their 7th season, then 25, 30, 40, 50, 60%, "
                  "and forced retirement after the 13th.",
    "Inactive (IA)": "A player whose user has stopped doing the weekly tasks. They stop earning TPE and eventually "
                     "regress away. A roster full of them is weaker than its TPE suggests.",
    "Strength": "The model's one-number rating of a team for the coming season, in points per game above average. "
                "It's what the simulation uses to decide who wins each game.",
    "Spearman ρ": "How similar two rankings are, from +1 (identical) through 0 (unrelated) to −1 (reversed). "
                  "For a 7-team conference, anything above ~0.5 is genuinely good.",
    "Backtest": "Pretending it's the start of an old season, predicting it with only what was known then, "
                "and checking against what really happened. It's how we know whether the model is any good.",
    "Walk-forward": "A backtest done season by season, so every prediction only ever uses earlier seasons.",
    "GM factor": "How much better or worse a GM's teams did than their TPE said they should. Positive = the "
                 "GM gets more out of the roster than average.",
    "Hazard": "The chance a player retires after a given career season, measured from every player who ever "
              "retired. Rises steeply once regression bites.",
}


def glossary(*keys: str) -> None:
    """Render a subset of the glossary (or all of it) as a bullet list."""
    items = {k: GLOSSARY[k] for k in (keys or GLOSSARY) if k in GLOSSARY}
    st.markdown("\n".join(f"- **{k}** — {v}" for k, v in items.items()))


def download(frame: pd.DataFrame, filename: str, label: str = "Download CSV") -> None:
    if frame is None or frame.empty:
        return
    st.download_button(label, data=frame.to_csv(index=False).encode("utf-8"),
                       file_name=filename, mime="text/csv", key=f"dl_{filename}")


def pct(x: float) -> str:
    try:
        return f"{100 * float(x):.0f}%"
    except (TypeError, ValueError):
        return "—"


def num(x: float, d: int = 0) -> str:
    try:
        return f"{float(x):,.{d}f}"
    except (TypeError, ValueError):
        return "—"


def signed(x: float, d: int = 1) -> str:
    try:
        return f"{float(x):+,.{d}f}"
    except (TypeError, ValueError):
        return "—"


def money(x: float) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    for t, s in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= t:
            return f"{config.CURRENCY_PREFIX}{v / t:,.1f}{s}"
    return f"{config.CURRENCY_PREFIX}{v:,.0f}"


def table(frame: pd.DataFrame, *, height: int | None = None, formats: dict | None = None,
          hide_index: bool = True) -> None:
    """st.dataframe with a column_config built from a {col: format} dict."""
    cfg = {}
    for col, f in (formats or {}).items():
        if col in frame.columns:
            # printf-style ("%.1f", "%+.0f") or the keyword "percent" for a
            # 0–1 fraction. Any "%.N%" shorthand becomes "percent".
            fmt = "percent" if f.endswith("%") and f != "%" else f
            cfg[col] = st.column_config.NumberColumn(col, format=fmt)
    kwargs = {"height": int(height)} if height else {}
    st.dataframe(frame, width="stretch", hide_index=hide_index, column_config=cfg or None, **kwargs)
