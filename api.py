"""
Thin client for the ISFL portal API.

Used only by scripts/build_cache.py and the Admin page's light refresh. The
Streamlit app itself never imports this on a normal page load — see data.py,
which reads the committed cache only.

Every function returns plain Python (lists/dicts) exactly as the API sent it.
Normalisation lives in the builder so that the raw payload is visible when a
shape changes.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import config


class ApiError(RuntimeError):
    pass


def get(path: str, params: dict[str, Any] | None = None, *, retries: int | None = None) -> Any:
    """GET a JSON endpoint with retries. Raises ApiError after the last retry."""
    url = config.API_BASE + path
    if params:
        clean = {k: v for k, v in params.items() if v is not None}
        if clean:
            url += "?" + urllib.parse.urlencode(clean)
    attempts = config.API_RETRIES if retries is None else retries
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT})
            with urllib.request.urlopen(req, timeout=config.API_TIMEOUT) as resp:
                body = resp.read()
            return json.loads(body)
        except urllib.error.HTTPError as exc:
            # 4xx is a caller error; don't hammer the server.
            if 400 <= exc.code < 500:
                raise ApiError(f"{exc.code} for {url}: {exc.read()[:200]!r}") from exc
            last_error = exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_error = exc
        time.sleep(0.5 * (attempt + 1))
    raise ApiError(f"failed after {attempts} attempts: {url}: {last_error}")


# ---------------------------------------------------------------------------
# Endpoints. Names mirror the paths.
# ---------------------------------------------------------------------------


def season() -> dict:
    return get("/season")


def season_all() -> list[dict]:
    return get("/season/all")


def standings(season_num: int, league: int = config.ISFL_LEAGUE_ID) -> dict:
    return get("/standings", {"league": league, "season": season_num})


def players() -> list[dict]:
    """The full player table, every status, all time. Filters are unreliable."""
    return get("/player")


def tpe_events(pid: int) -> list[dict]:
    return get("/tpeevents", {"pid": pid})


def game_stats(pid: int) -> list[dict]:
    return get("/player/game-stats", {"pid": pid})


def regression(season_num: int) -> list[dict]:
    return get("/player/regression", {"season": season_num})


def bots(season_num: int | None = None) -> list[dict]:
    return get("/bots", {"season": season_num})


def gm_history() -> list[dict]:
    return get("/gm-history")


def gm_records() -> list[dict]:
    return get("/gm-history/records")


def managers() -> list[dict]:
    return get("/manager")


def draft_picks(season_num: int, league: int = config.ISFL_LEAGUE_ID) -> list[dict]:
    return get("/draft-picks", {"league": league, "season": season_num})


def awards(season_num: int) -> list[dict]:
    return get("/awards", {"season": season_num})


def team_records() -> dict:
    return get("/team-history/records")


def analytics() -> list[dict]:
    return get("/analytics")
