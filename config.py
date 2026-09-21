"""
All constants and rule tables for the ISFL Team Tracker. Start here.

Domain rules come from the Official Rulebook (Aug 2026 revision) and were
verified against the live portal API on 2026-09-21 — see DESIGN.md §2–3.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
FIXTURE_DIR = ROOT / "tests" / "fixtures"

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

API_BASE = "https://portal.sim-football.com/api/isfl/v1"
API_TIMEOUT = 30
API_RETRIES = 3
API_DELAY_SECONDS = 0.15          # politeness gap between calls in the builder
USER_AGENT = "isfl-team-tracker/1.0 (+offline cache builder)"

ISFL_LEAGUE_ID = 1

# ---------------------------------------------------------------------------
# League structure
# ---------------------------------------------------------------------------

# Conferences are read from standings at runtime; this is only the fallback
# for seasons where the API has no conference column (none observed so far).
CONFERENCES = {
    "NSFC": ["SAR", "BAL", "OSK", "YKW", "CTC", "COL", "BFB"],
    "ASFC": ["AZ", "NOLA", "HON", "NYS", "SJS", "AUS", "OCO"],
}
HOME_TEAM = "NOLA"

# Relocations. Old abbreviation -> current franchise. Verified from standings:
# each old code's last season is immediately followed by the new code's first,
# in the same conference. Ratings carry across the move.
FRANCHISE_MAP = {
    "LVL": "NOLA",   # S5 -> S6
    "PHI": "CTC",    # S38 -> S39
    "CHI": "OSK",    # S48 -> S49
    "BER": "BFB",    # S50 -> S51
}

GAMES_PER_SEASON = 16
CONFERENCE_GAMES = 12             # 6 conference opponents, home and away
CROSS_CONFERENCE_GAMES = 4        # rotating, one game each
PLAYOFF_TEAMS_PER_CONFERENCE = 3  # #1 seed gets a bye

# The engine changed in S27. Elo runs from S1 (ratings need the run-up) but
# nothing before S27 is used to score or fit anything.
ENGINE_SWITCH_SEASON = 27

# Portal migration. The TPE event log begins here; before this there is only
# a single "Create" snapshot per player.
PORTAL_MIGRATION_DATE = "2025-04-09"
FIRST_TPE_SEASON = 54             # first season with a reliable start-of-season TPE
FIRST_REGRESSION_SEASON = 52      # /player/regression coverage begins

# ---------------------------------------------------------------------------
# Career clock and regression — Rulebook IV.A, verified against the API
# ---------------------------------------------------------------------------
#
# career_season(S) = S - draftSeason + 1, 1-indexed.
# The regression labelled "S<N> Regression" is applied at the start of season N
# and is charged for the career season that finished in N-1, i.e. for
# seasons_finished = N - draftSeason. Loss is floor(oldTPE * pct).

REGRESSION_BY_SEASONS_FINISHED = {
    7: 0.20,
    8: 0.25,
    9: 0.30,
    10: 0.40,
    11: 0.50,
    12: 0.60,
}
RETIRE_AFTER_SEASONS = 13         # auto-retired on conclusion of the 13th
REGRESSION_FLOOR_TPE = 150        # auto-retired if regression takes TPE below this
FIRST_REGRESSION_CAREER_SEASON = 7


def regression_pct(seasons_finished: int) -> float:
    """Fraction of TPE lost at the offseason after `seasons_finished` seasons."""
    if seasons_finished >= RETIRE_AFTER_SEASONS:
        return 1.0
    return REGRESSION_BY_SEASONS_FINISHED.get(seasons_finished, 0.0)


# ---------------------------------------------------------------------------
# Earning — Rulebook I, IV.A. Used only as fallbacks; rates are measured.
# ---------------------------------------------------------------------------

STARTING_TPE = 50
WEEKLY_AC_TPE = 2
WEEKLY_TRAINING_TPE = 5           # the $1M option; $500k buys 3
WEEKS_PER_SEASON = 8
TYPICAL_SEASON_TPE = (150, 200)   # rulebook range for a full participant
ARCHETYPE_CAP_TYPICAL = 1200
MAX_EARNER_PEAK = 1500

# Event-log task types that are earnings (everything else is a snapshot,
# regression or correction and is excluded from rate measurement).
EARNING_TASK_TYPES = {
    "activity check", "training", "point task", "prediction", "training camp",
    "seasonal equipment", "fantasy", "other",
}
SNAPSHOT_TASK_TYPES = {"Create"}
REGRESSION_TASK_TYPES = {"regression"}

# A player with fewer than this many earning events in a season was, in
# practice, inactive for it.
INACTIVE_EVENT_THRESHOLD = 6

# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

# Exact strings as they appear in /player.position.
POSITION_GROUP = {
    "Quarterback": "QB",
    "Running Back": "RB",
    "Wide Receiver": "WR",
    "Tight End": "TE",
    "Offensive Lineman": "OL",
    "Defensive End": "DL",
    "Defensive Tackle": "DL",
    "Linebacker": "LB",
    "Cornerback": "DB",
    "Safety": "DB",
    "Kicker": "K",
}
POSITION_GROUPS = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB", "K"]
OFFENSE_GROUPS = ["QB", "RB", "WR", "TE", "OL"]
DEFENSE_GROUPS = ["DL", "LB", "DB"]

# Base-formation starters. Starter TPE = top-N by TPE in the group. OL includes
# purchased bots. This is what the roster-strength model is fitted on.
STARTERS = {
    "QB": 1, "RB": 1, "WR": 3, "TE": 1, "OL": 5,
    "DL": 4, "LB": 3, "DB": 5, "K": 1,
}

# OL bots — Rulebook III.A. Tier -> TPE.
BOT_TIER_TPE = {0: 50, 1: 150, 2: 350, 3: 550, 4: 750}
BOT_TIER_COST = {0: 500_000, 1: 1_000_000, 2: 2_500_000, 3: 4_500_000, 4: 7_000_000}

# ---------------------------------------------------------------------------
# Ratings
# ---------------------------------------------------------------------------

ELO_START = 1500.0
ELO_K = 40.0                      # grid-tuned on S27+ log-loss (elo.tune_elo)
ELO_HOME_ADVANTAGE = 40.0         # Elo points; home teams win 55.1% (S50–62)
ELO_SEASON_REVERT = 1 / 3         # fraction reverted toward 1500 each offseason
ELO_EXPANSION_START = 1450.0      # expansion teams start a little below par

PYTHAGOREAN_EXPONENT = 2.37       # NFL default; refitted on load
SRS_HOME_ADVANTAGE = 2.0          # points; refitted on load

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

RIDGE_ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
MIN_TRAINING_SEASONS = 3          # roster model needs at least this many

# Blend of preseason signals. Target and ridge alpha chosen by the
# walk-forward (see backtest.walk_forward and the experiment in DESIGN.md).
BLEND_TARGET = "wpct_pts"         # "srs" or "wpct_pts"; ranks better in the walk-forward
BLEND_ALPHA = 100.0               # heavy shrinkage: 84 rows, 5 collinear inputs; weights ~ correlation

# Default blend weights for preseason strength (overridden by the backtest
# when it has enough seasons to fit).
DEFAULT_BLEND = {"elo": 0.3, "srs": 0.15, "wpct": 0.2, "roster": 0.25, "gm": 0.1}

SIMULATIONS = 10_000
SIM_SEED = 63

# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

DEFAULT_HORIZON = 3
MAX_HORIZON = 6
DEFAULT_RATE_WINDOW = 2           # seasons of history used for a player's rate

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

APP_TITLE = "ISFL Team Tracker"
CACHE_TTL_SECONDS = 15 * 60

COLORS = {
    "bg": "#0F1419",
    "surface": "#171E26",
    "surface_alt": "#1F2934",
    "line": "#2E3B48",
    "text": "#E9ECEF",
    "text_muted": "#8E9BAA",
    "primary": "#E0A93A",         # stadium-light amber
    "primary_dim": "#9A7222",
    "green": "#3FA36B",
    "red": "#D0533F",
    "blue": "#4C8FBF",
    "purple": "#8E6FC7",
}
RAMP = ["#D0533F", "#E0A93A", "#8FBF5A", "#3FA36B"]
PLOTLY_TEMPLATE = "isfl"

# Team colours, keyed by abbreviation. Approximations of the franchises' real
# branding; anything not listed falls back to the palette below.
TEAM_COLORS = {
    "AZ": "#B3202E", "AUS": "#B87333", "BAL": "#F2A900", "BFB": "#2E7D32",
    "COL": "#6FA8DC", "CTC": "#7B3FA0", "HON": "#00A3E0", "NOLA": "#5D3FD3",
    "NYS": "#8A8D8F", "OCO": "#F26522", "OSK": "#1C4F9C", "SAR": "#2AB7CA",
    "SJS": "#3E6B48", "YKW": "#5C5C5C",
}
TEAM_PALETTE = [
    "#E0A93A", "#4C8FBF", "#3FA36B", "#D0533F", "#8E6FC7", "#3FB6A8",
    "#E07B39", "#6FA8DC", "#B5C94F", "#D46A9F", "#7FBF6A", "#C9A227",
    "#5E8CD6", "#A5563F", "#4FA8A0", "#9B7FD4", "#DC8A5A", "#6BBF8E",
]

CURRENCY_PREFIX = "$"
