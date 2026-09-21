"""
Build the offline cache from the ISFL portal API.

    python scripts/build_cache.py                # everything, incremental
    python scripts/build_cache.py --stage fast   # standings, players, GMs… (~1 min)
    python scripts/build_cache.py --stage slow   # per-player logs (~20 min first time)
    python scripts/build_cache.py --sample       # slow stage for current ISFL rosters only
    python scripts/build_cache.py --force        # ignore the incremental log

The app reads only what this writes. Nothing here is imported by the app.

Incremental rule: a player's event log and game log are re-fetched only if the
player is active, or retired after the last fetch. Retired players' logs are
final.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import api      # noqa: E402
import config   # noqa: E402

CACHE = config.CACHE_DIR
FETCH_LOG = CACHE / "fetch_log.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def write_csv(name: str, frame: pd.DataFrame) -> None:
    """Atomic write so a crash mid-build never leaves a truncated file."""
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = CACHE / f".{name}.tmp"
    frame.to_csv(tmp, index=False)
    tmp.replace(CACHE / name)
    log(f"  wrote {name}: {len(frame):,} rows")


def read_csv(name: str) -> pd.DataFrame | None:
    path = CACHE / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_fetch_log() -> dict:
    if FETCH_LOG.exists():
        return json.loads(FETCH_LOG.read_text())
    return {}


def save_fetch_log(entries: dict) -> None:
    FETCH_LOG.write_text(json.dumps(entries, indent=0, sort_keys=True))


def polite() -> None:
    time.sleep(config.API_DELAY_SECONDS)


# ---------------------------------------------------------------------------
# Fast stage
# ---------------------------------------------------------------------------


def build_seasons(current: int) -> pd.DataFrame:
    rows = {s: {"season": s, "start_date": None, "end_date": None, "ended": 1}
            for s in range(1, current + 1)}
    for r in api.season_all():
        s = int(r["season"])
        rows.setdefault(s, {"season": s})
        rows[s].update(start_date=r.get("startDate"), end_date=r.get("endDate"),
                       ended=int(r.get("ended", 0)))
    return pd.DataFrame(sorted(rows.values(), key=lambda x: x["season"]))


def _playoff_results(postseason: list[dict]) -> dict[str, str]:
    """
    Team -> 'champion' | 'runner_up' | 'semifinal' | 'first_round'.

    Rounds are inferred from the week numbers: the last week is the final,
    the one before it the conference finals, anything earlier the first round.
    Bye games with fixed outcomes are not in the feed (verified S62).
    """
    if not postseason:
        return {}
    weeks = sorted({int(g["week"]) for g in postseason})
    final_week = weeks[-1]
    semi_week = weeks[-2] if len(weeks) > 1 else None
    result: dict[str, str] = {}
    for g in postseason:
        home, away, w = g["homeTeam"], g["awayTeam"], int(g["week"])
        winner = home if g["winner"] == "home" else away
        loser = away if winner == home else home
        if w == final_week:
            result[winner] = "champion"
            result[loser] = "runner_up"
        elif w == semi_week:
            result.setdefault(loser, "semifinal")
        else:
            result.setdefault(loser, "first_round")
    return result


def build_standings_and_games(current: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    standings_rows: list[dict] = []
    game_rows: list[dict] = []
    for season in range(1, current + 1):
        try:
            payload = api.standings(season)
        except api.ApiError as exc:
            log(f"  standings S{season}: {exc}")
            continue
        polite()
        reg = payload.get("regularSeason") or []
        post = payload.get("postseason") or []
        games = payload.get("games") or []
        n_teams = len(reg)
        per_week = max(1, n_teams // 2)
        results = _playoff_results(post)

        for r in reg:
            standings_rows.append({
                "season": season,
                "team": r["abbreviation"],
                "name": r.get("name"),
                "location": r.get("location"),
                "conference": r.get("conference"),
                "wins": r.get("wins", 0), "losses": r.get("losses", 0), "ties": r.get("ties", 0),
                "pct": r.get("pct"),
                "pf": r.get("pf"), "pa": r.get("pa"), "diff": r.get("diff"),
                "home_w": (r.get("homeRecord") or {}).get("wins"),
                "home_l": (r.get("homeRecord") or {}).get("losses"),
                "away_w": (r.get("awayRecord") or {}).get("wins"),
                "away_l": (r.get("awayRecord") or {}).get("losses"),
                "conf_w": (r.get("confRecord") or {}).get("wins"),
                "conf_l": (r.get("confRecord") or {}).get("losses"),
                "playoff_result": results.get(r["abbreviation"], "missed"),
            })

        for i, g in enumerate(games):
            game_rows.append({
                "season": season, "phase": "regular", "seq": i,
                "week": i // per_week + 1,
                "home": g["homeTeam"], "away": g["awayTeam"],
                "home_score": None, "away_score": None,
                "winner": g.get("winner"),
            })
        for g in post:
            game_rows.append({
                "season": season, "phase": "playoff", "seq": 10_000 + int(g["week"]),
                "week": int(g["week"]),
                "home": g["homeTeam"], "away": g["awayTeam"],
                "home_score": g.get("homeScore"), "away_score": g.get("awayScore"),
                "winner": g.get("winner"),
            })
        if season % 10 == 0 or season == current:
            log(f"  standings through S{season}")
    return pd.DataFrame(standings_rows), pd.DataFrame(game_rows)


PLAYER_SCALARS = [
    "pid", "uid", "username", "name", "firstName", "lastName", "draftSeason",
    "status", "activeStatus", "creationDate", "retirementDate",
    "position", "archetype", "isflTeam", "dsflTeam", "currentLeague",
    "totalTPE", "appliedTPE", "bankedTPE", "secondaryTPE", "tertiaryTPE", "highestTPE",
    "positionChanged", "archetypeChanged", "usedRedistribution",
    "equipmentPurchased", "trainingCamp", "isSuspended", "isRookie", "isCaptain",
    "weeklyActivityCheck", "weeklyTraining", "bankBalance", "jerseyNumber",
    "birthplace", "college", "recruiter", "wfcRegion",
]


def build_players() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = api.players()
    rows, attr_rows = [], []
    for p in raw:
        row = {k: p.get(k) for k in PLAYER_SCALARS}
        traits = p.get("traits") or {}
        row["traits_purchased"] = ";".join(
            sorted(n for n, t in traits.items() if isinstance(t, dict) and t.get("purchased"))
        )
        rows.append(row)
        attrs = p.get("attributes") or {}
        if attrs:
            attr_rows.append({"pid": p["pid"], **attrs})
    players = pd.DataFrame(rows)
    players.columns = [_snake(c) for c in players.columns]
    return players, pd.DataFrame(attr_rows)


def _snake(name: str) -> str:
    """camelCase -> snake_case, keeping acronym runs together (passTD -> pass_td)."""
    return re.sub(r"(?<=[a-z0-9])([A-Z]+)", r"_\1", name).lower()


def build_regressions(current: int) -> pd.DataFrame:
    rows = []
    for season in range(config.FIRST_REGRESSION_SEASON, current + 1):
        try:
            for r in api.regression(season):
                rows.append({
                    "season": season, "pid": r["pid"], "uid": r.get("uid"),
                    "draft_season": r.get("draftSeason"),
                    "old_tpe": r.get("oldTPE"), "pct": r.get("regressionPct"),
                    "regression_tpe": r.get("regressionTPE"), "new_tpe": r.get("newTPE"),
                })
        except api.ApiError as exc:
            log(f"  regression S{season}: {exc}")
        polite()
    return pd.DataFrame(rows)


def build_bots(current: int) -> pd.DataFrame:
    rows = []
    for season in range(config.FIRST_TPE_SEASON, current + 1):
        try:
            for b in api.bots(season):
                rows.append({
                    "season": season, "bot_id": b.get("id"),
                    "team": b.get("isflTeam"), "dsfl_team": b.get("dsflTeam"),
                    "league": b.get("currentLeague"), "tier": b.get("tier"),
                    "position": b.get("position"), "archetype": b.get("archetype"),
                    "status": b.get("status"),
                    "name": f"{b.get('firstName', '')} {b.get('lastName', '')}".strip(),
                })
        except api.ApiError as exc:
            log(f"  bots S{season}: {exc}")
        polite()
    return pd.DataFrame(rows)


def build_gm() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    history = pd.DataFrame(api.gm_history())
    polite()
    records = pd.DataFrame(api.gm_records())
    polite()
    managers = pd.DataFrame(api.managers())
    return history, records, managers


def build_draft_picks(current: int) -> pd.DataFrame:
    rows = []
    empties = 0
    for season in range(1, current + 2):
        try:
            picks = api.draft_picks(season)
        except api.ApiError as exc:
            log(f"  draft S{season}: {exc}")
            continue
        polite()
        if not picks:
            empties += 1
            continue
        for p in picks:
            rows.append({
                "season": season, "round": p.get("round"), "pick": p.get("pick"),
                "overall": p.get("overall"), "pid": p.get("pid"),
                "original_team": p.get("originalTeam"), "owning_team": p.get("owningTeam"),
                "name": f"{p.get('firstName', '')} {p.get('lastName', '')}".strip(),
                "position": p.get("position"), "username": p.get("username"),
            })
    return pd.DataFrame(rows)


def build_awards(current: int) -> pd.DataFrame:
    rows = []
    for season in range(1, current + 1):
        try:
            for a in api.awards(season):
                rows.append({
                    "season": season, "pid": a.get("pid"), "team": a.get("team"),
                    "type": a.get("type"), "award_position": a.get("awardPosition"),
                    "player_position": a.get("playerPosition"),
                    "name": f"{a.get('firstName', '')} {a.get('lastName', '')}".strip(),
                    "username": a.get("username"),
                })
        except api.ApiError as exc:
            log(f"  awards S{season}: {exc}")
        polite()
    return pd.DataFrame(rows)


def run_fast(meta: dict) -> pd.DataFrame:
    log("fast stage")
    current = int(api.season()["season"])
    meta["current_season"] = current
    log(f"  current season S{current}")

    write_csv("seasons.csv", build_seasons(current))

    standings, games = build_standings_and_games(current)
    write_csv("standings.csv", standings)
    write_csv("games.csv", games)

    players, attributes = build_players()
    write_csv("players.csv", players)
    write_csv("attributes.csv", attributes)

    write_csv("regressions.csv", build_regressions(current))
    write_csv("bots.csv", build_bots(current))

    history, records, managers = build_gm()
    write_csv("gm_history.csv", history)
    write_csv("gm_records.csv", records)
    write_csv("managers.csv", managers)

    write_csv("draft_picks.csv", build_draft_picks(current))
    write_csv("awards.csv", build_awards(current))

    try:
        write_csv("analytics.csv", pd.DataFrame(api.analytics()))
    except api.ApiError as exc:
        log(f"  analytics: {exc}")

    return players


def refresh_current() -> dict:
    """
    Light refresh for the Admin page: the current season's standings and
    games, the player table, bots and managers. A few seconds, not minutes.
    Player logs are untouched.
    """
    meta_path = CACHE / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    current = int(api.season()["season"])
    meta["current_season"] = current

    write_csv("seasons.csv", build_seasons(current))
    standings_old = read_csv("standings.csv")
    games_old = read_csv("games.csv")
    new_rows_s, new_rows_g = [], []
    for season in (current - 1, current):
        payload = api.standings(season)
        reg = payload.get("regularSeason") or []
        post = payload.get("postseason") or []
        games = payload.get("games") or []
        per_week = max(1, len(reg) // 2)
        results = _playoff_results(post)
        for r in reg:
            new_rows_s.append({
                "season": season, "team": r["abbreviation"], "name": r.get("name"),
                "location": r.get("location"), "conference": r.get("conference"),
                "wins": r.get("wins", 0), "losses": r.get("losses", 0), "ties": r.get("ties", 0),
                "pct": r.get("pct"), "pf": r.get("pf"), "pa": r.get("pa"), "diff": r.get("diff"),
                "home_w": (r.get("homeRecord") or {}).get("wins"), "home_l": (r.get("homeRecord") or {}).get("losses"),
                "away_w": (r.get("awayRecord") or {}).get("wins"), "away_l": (r.get("awayRecord") or {}).get("losses"),
                "conf_w": (r.get("confRecord") or {}).get("wins"), "conf_l": (r.get("confRecord") or {}).get("losses"),
                "playoff_result": results.get(r["abbreviation"], "missed"),
            })
        for i, g in enumerate(games):
            new_rows_g.append({"season": season, "phase": "regular", "seq": i, "week": i // per_week + 1,
                               "home": g["homeTeam"], "away": g["awayTeam"], "home_score": None,
                               "away_score": None, "winner": g.get("winner")})
        for g in post:
            new_rows_g.append({"season": season, "phase": "playoff", "seq": 10_000 + int(g["week"]),
                               "week": int(g["week"]), "home": g["homeTeam"], "away": g["awayTeam"],
                               "home_score": g.get("homeScore"), "away_score": g.get("awayScore"),
                               "winner": g.get("winner")})
    touched = {current - 1, current}
    if standings_old is not None:
        standings_old = standings_old[~standings_old["season"].isin(touched)]
    if games_old is not None:
        games_old = games_old[~games_old["season"].isin(touched)]
    write_csv("standings.csv", pd.concat([standings_old, pd.DataFrame(new_rows_s)], ignore_index=True)
              .sort_values(["season", "team"]))
    write_csv("games.csv", pd.concat([games_old, pd.DataFrame(new_rows_g)], ignore_index=True)
              .sort_values(["season", "seq"]))

    players, attributes = build_players()
    write_csv("players.csv", players)
    write_csv("attributes.csv", attributes)
    bots_old = read_csv("bots.csv")
    bots_new = build_bots(current)
    if bots_old is not None:
        bots_new = pd.concat([bots_old[~bots_old["season"].isin(set(bots_new["season"]))], bots_new], ignore_index=True)
    write_csv("bots.csv", bots_new)
    write_csv("managers.csv", pd.DataFrame(api.managers()))
    write_csv("regressions.csv", build_regressions(current))

    meta["light_refresh_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta["built_at"] = meta["light_refresh_at"]
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


# ---------------------------------------------------------------------------
# Slow stage — per-player logs
# ---------------------------------------------------------------------------

GAME_STAT_COLUMNS = [
    "passCmp", "passAtt", "passYds", "passTD", "passInt",
    "rushAtt", "rushYds", "rushTD", "recRec", "recYds", "recTD",
    "defTck", "defTFL", "defSack", "defPD", "defInt", "defFF", "defFR", "defTD",
    "kXPM", "kXPA", "kFGMU20", "kFGM2029", "kFGM3039", "kFGM4049", "kFGM50",
    "kFGAU20", "kFGA2029", "kFGA3039", "kFGA4049", "kFGA50",
    "pPunts", "pYds", "otherPancakes", "otherSacksAllowed",
]


def scope_players(players: pd.DataFrame, sample: bool) -> pd.DataFrame:
    """
    Players whose logs we need.

    Anyone who could have been on an ISFL roster from FIRST_TPE_SEASON on: drafted
    within the 13-season career window before it, or still active. --sample
    narrows to the current ISFL rosters, enough to develop every page.
    """
    ds = pd.to_numeric(players["draft_season"], errors="coerce")
    if sample:
        mask = players["isfl_team"].notna() & (players["status"] == "active")
    else:
        earliest = config.FIRST_TPE_SEASON - config.RETIRE_AFTER_SEASONS
        mask = (ds >= earliest) | (players["status"] == "active")
    return players[mask]


def needs_refetch(pid: int, status: str, retired: str | None, fetch_log: dict, force: bool) -> bool:
    if force:
        return True
    entry = fetch_log.get(str(pid))
    if entry is None:
        return True
    if status == "active":
        return True
    # Retired since last fetch: the log grew until retirement, refetch once.
    fetched_at = entry.get("fetched_at", "")
    return bool(retired and str(retired) > fetched_at)


def run_slow(players: pd.DataFrame, meta: dict, *, sample: bool, force: bool) -> None:
    log("slow stage")
    fetch_log = load_fetch_log()
    scope = scope_players(players, sample)
    log(f"  {len(scope):,} players in scope")

    events_old = read_csv("tpe_events.csv")
    games_old = read_csv("player_games.csv")
    keep_events = [] if events_old is None else [events_old]
    keep_games = [] if games_old is None else [games_old]
    refetch_pids: list[int] = []
    for row in scope.itertuples(index=False):
        if needs_refetch(int(row.pid), row.status, row.retirement_date, fetch_log, force):
            refetch_pids.append(int(row.pid))
    log(f"  {len(refetch_pids):,} to fetch ({len(scope) - len(refetch_pids):,} already final)")

    if refetch_pids:
        drop = set(refetch_pids)
        keep_events = [f[~f["pid"].isin(drop)] for f in keep_events]
        keep_games = [f[~f["pid"].isin(drop)] for f in keep_games]

    new_events: list[dict] = []
    new_games: list[dict] = []
    errors: list[str] = []
    started = time.time()
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def flush() -> None:
        """
        Write data and the fetch log together. A checkpoint that recorded a
        player as fetched without writing their rows would make the next
        incremental run skip them forever.
        """
        events = pd.concat(keep_events + [pd.DataFrame(new_events)], ignore_index=True)
        games = pd.concat(keep_games + [pd.DataFrame(new_games)], ignore_index=True)
        if not events.empty:
            events = events.drop_duplicates(subset=["pid", "task_id"]).sort_values(["pid", "date"])
        if not games.empty:
            games = games.drop_duplicates(subset=["pid", "season", "season_state", "week"])
        write_csv("tpe_events.csv", events)
        write_csv("player_games.csv", games)
        save_fetch_log(fetch_log)

    for i, pid in enumerate(refetch_pids, 1):
        try:
            for e in api.tpe_events(pid):
                new_events.append({
                    "pid": pid, "task_id": e.get("taskID"),
                    "date": e.get("submissionDate"), "task_type": e.get("taskType"),
                    "tpe_change": e.get("TPEChange"),
                    "description": e.get("taskDescription"),
                    "submitted_by": e.get("submittedBy"),
                })
            polite()
            for g in api.game_stats(pid):
                row = {
                    "pid": pid, "season": g.get("season"),
                    "season_state": g.get("seasonState"), "week": g.get("week"),
                    "team": g.get("teams"),
                }
                for c in GAME_STAT_COLUMNS:
                    row[_snake(c)] = g.get(c)
                new_games.append(row)
            polite()
            fetch_log[str(pid)] = {"fetched_at": now_iso}
        except api.ApiError as exc:
            errors.append(f"pid {pid}: {exc}")
        if i % 100 == 0 or i == len(refetch_pids):
            rate = i / max(1e-9, time.time() - started)
            remaining = (len(refetch_pids) - i) / max(rate, 1e-9)
            log(f"  {i}/{len(refetch_pids)} players · {remaining / 60:.1f} min left")
            flush()

    if not refetch_pids:
        flush()
    meta["slow_errors"] = errors
    meta["players_fetched"] = len(refetch_pids)
    if errors:
        log(f"  {len(errors)} players failed; first: {errors[0]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["fast", "slow", "all"], default="all")
    parser.add_argument("--sample", action="store_true", help="slow stage for current ISFL rosters only")
    parser.add_argument("--force", action="store_true", help="re-fetch every player log")
    args = parser.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    meta_path = CACHE / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["stage"] = args.stage
    meta["sample"] = args.sample
    started = time.time()

    players: pd.DataFrame | None = None
    if args.stage in ("fast", "all"):
        players = run_fast(meta)
        meta["fast_built_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if args.stage in ("slow", "all"):
        if players is None:
            players = read_csv("players.csv")
            if players is None:
                log("players.csv missing — run the fast stage first")
                return 1
            meta.setdefault("current_season", int(api.season()["season"]))
        run_slow(players, meta, sample=args.sample, force=args.force)
        meta["slow_built_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    meta["built_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta["duration_seconds"] = round(time.time() - started)
    meta["files"] = sorted(p.name for p in CACHE.glob("*.csv"))
    meta_path.write_text(json.dumps(meta, indent=2))
    log(f"done in {meta['duration_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
