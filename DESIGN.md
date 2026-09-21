# ISFL Team Tracker — design

Sibling of `ssl-team-tracker`. Same stack, same lessons, different league and a
different question: not "how deep is my squad" but "what wins the Ultimus, and
what will this season's standings look like."

Written 2026-09-21, during S63 preseason. Everything in §2 was verified against
the live API on that date.

---

## 1. Purpose and user

- **User**: Bob, member of NOLA (New Orleans Second Line) with some leadership
  involvement, not a GM. Does the season-rankings prediction task each preseason.
- **Primary job**: before Week 1, produce a defensible ranking of the 7 NSFC and
  7 ASFC teams plus a best-overall-record pick.
- **Secondary jobs**: understand what drives winning (TPE, position groups, GMs),
  see where every org is heading 1–3 seasons out, and NOLA-specific views.
- **Scope**: ISFL only. DSFL ignored except as the source of call-ups.

## 2. Data — verified 2026-09-21

Base `https://portal.sim-football.com/api/isfl/v1`. No auth for reads. Swagger
spec on hand (`swagger (1).json`) but response shapes are loosely typed; trust
the probes below over the spec.

| Endpoint | Coverage | Shape / notes |
|---|---|---|
| `/season` | now | `{season: 63, startDate, endDate, ended}` |
| `/season/all` | S53+ only | Dates of the portal era. Older season dates are not available. |
| `/standings?league=1&season=N` | **S1 → S62** | `regularSeason[14]` (W/L/T, pct, pf, pa, diff, home/away/conf records, conference), `postseason[]` (gid, week, teams, **scores**, winner), `games[112]` (homeTeam, awayTeam, winner — **no score, no week**). S63 `games` is empty until played. |
| `/player` (no params) | all time, **4,823 rows** | 4,059 retired, 753 active. Snapshot only: `draftSeason`, `status`, `retirementDate`, `position`, `archetype`, `isflTeam`, `dsflTeam`, `currentLeague`, `totalTPE`, `appliedTPE`, `bankedTPE`, `highestTPE`, `attributes{}`, `traits{}`, `weeklyActivityCheck`, `weeklyTraining`, `trainingCamp`, `isRookie`, `bankBalance`. Query filters (`leagueID`, `isflTeam`, `status=active`) mostly return empty — filter client-side. |
| `/tpeevents?pid=` | **from 2025-04-09** (S54 start) | Per-player log: `taskType` (activity check, training, point task, prediction, training camp, seasonal equipment, regression, fantasy, correction, other, Create), `TPEChange`, `submissionDate`, `taskDescription`. The `Create` row is the migration snapshot, not an earning. Regressions appear as `"S63 Regression: 360 TPE"` with negative `TPEChange`. Works for retired players too. |
| `/player/regression?season=N` | S52+ | `{pid, draftSeason, oldTPE, regressionPct, regressionTPE, newTPE}`. Explicit — no parsing. |
| `/player/stats?pid=` | per player | Per-season rows with `teams` → **the only source of historical rosters**. Misses players who never recorded a stat that season. |
| `/gm-history` | all time, 1,545 rows | `{season, league, team, uid, username}`. One row per GM per team-season. |
| `/gm-history/records` | all time | Career W/L, playoff record, championships per GM. |
| `/manager` | now | Current GMs. |
| `/bots?season=N` | S62+ | OL bots on ISFL rosters with `tier` (50–750 TPE) and attributes. Invisible in `/player`. |
| `/draft-picks?league=1&season=N` | per season | Pick, original/owning team, drafted player. |
| `/awards?season=N` | per season | Award winners with position. |
| `/team-history/records` | all time | Per-team totals and head-to-head matrices. |
| `/analytics` | weekly | League-wide AC and training counts. |
| `/player/stats-search` | — | Returns empty for every filter tried. Do not rely on it. |
| `index.sim-football.com` | — | Cloudflare-blocked for automation. Not a source. |

### Consequences

- **Elo and margin ratings: S1 → S62.** Full history.
- **TPE-based modelling: S54 → S62, nine completed seasons.** The event log
  starts at the portal migration. Before that there is only the `Create`
  snapshot, which is the player's TPE on 2025-04-09, not at any season start.
- **Engine switch was S27.** Not binding — the TPE window is narrower anyway.
  Elo is computed from S1 but the backtest scores only S27+ and the TPE model
  never sees pre-S54 data.
- **Game order within a season is unknown.** `games` has no week. Treat the
  list order as chronological (probably DB insertion order — verify against
  `/player/game-stats` weeks for a few players during build) and also run the
  Elo pass over several shuffled orderings and average, so the rating is not
  sensitive to the assumption.
- **No per-game scores.** Margin ratings use season PF/PA and the opponent list.

## 3. Domain rules — from the rulebook (updated Aug 2026)

### Seasons
- 8 real weeks: 5 regular season, 1 playoffs, 2 offseason. 16 games per team.
  Conference record is 12 games (6 conference opponents × 2), 4 cross-conference.
- Two conferences of 7: NSFC and ASFC. Top 3 per conference make playoffs; #1
  seeds get a bye. Tiebreak: head-to-head → conference record → points for.

### Career clock
- `draftSeason` is the ISFL draft season. `career_season = current − draftSeason + 1`.
- Regression happens in the offseason **between** career seasons, starting after
  the 7th: 7→20%, 8→25%, 9→30%, 10→40%, 11→50%, 12→60%. Retired after the 13th.
  Applied to total TPE, rounded **down** (player-favourable). Auto-retired if it
  drops below 150.
- The regression posted at the start of S63 (`"S63 Regression"`) is the one
  charged for the S62 career season. Keep the label convention consistent with
  the API, which names it by the season being entered.

### TPE
- Start 50. AC +2/week, training +3 ($500k) or +5 ($1M)/week. 150–200 per season
  for full participants; most archetypes cap ~1,200; max earners ~1,500.
- Two-season claim limit; unclaimed TPE before regression is voided.
- Position switch once per career, offseason only, penalty banked in secondary /
  tertiary banks that unlock over 1–2 seasons. Archetype switch once, $2M.
- Redistribution: up to 50 TPE/season at $100k each.

### Rosters
- Minimum 10 user players per ISFL team.
- OL bots: Tier 0–4 = 50/150/350/550/750 TPE at $0.5M–$7M. Purchasable any time.
- Inactive = no meaningful activity for 14 days. IA players are frozen; can be
  signed at minimum. A roster's *earning* players are the ones that grow.
- DSFL eligibility ends at 250 applied TPE, 16 ISFL games, or 3 seasons post-draft.

### Prediction task
- Due after preseason, before Week 1. Rosters are final: regression, draft, FA,
  send-downs are all done. So the model's "preseason roster" is the roster at
  Week 1, and the backtest must reconstruct exactly that for past seasons.

## 4. Architecture

Flat repo, same as SSL. Entry point is `app.py` (new repo, nothing pinned).

| File | Role |
|---|---|
| `app.py` | Entry: page config, sidebar, routing, cache loading |
| `config.py` | Constants and rule tables: regression schedule, position groups, conferences, team colours, model hyperparameters |
| `data.py` | Reads `cache/` into DataFrames. **Never hits the network.** Schema validation on load. |
| `api.py` | Thin client for the portal API. Used only by the cache builder and the admin refresh. |
| `elo.py` | Game Elo, SRS margin rating, Pythagorean expectation, GM ratings |
| `roster.py` | Historical roster reconstruction, TPE timelines, position grouping, bots merge, starter selection |
| `model.py` | Roster-strength regression, position-group weights, preseason strength blend |
| `simulate.py` | Schedule generator, Monte Carlo season sim, rank distributions |
| `projection.py` | Per-player forward projection (regression, earnings, retirement hazard) rolled up to teams |
| `backtest.py` | Walk-forward evaluation, baselines, scoring |
| `charts.py` | One Plotly template, deterministic team colours |
| `ui.py` | Theme CSS (flush-left, `string.Template`, `.stApp` only — see SSL HANDOFF §5), notices, formatting |
| `page_*.py` | One per page |
| `scripts/build_cache.py` | Pulls everything into `cache/`. Incremental. `--sample` for a fast partial build. |
| `test_offline.py` | pandas-only tests, no network, no Streamlit |
| `.github/workflows/update-cache.yml` | Weekly, quoted `"on":` |
| `cache/` | Committed on purpose |
| `.streamlit/config.toml` | Dark theme |
| `.claude/launch.json` | `streamlit run app.py` for the in-app preview |

### Offline-first — the hard requirement

The app reads **only** `cache/`. Zero network on page load. That gives:

1. **Local run**: `python scripts/build_cache.py` once (~15 min, ~3,000 calls),
   then `streamlit run app.py`. Works with no internet after the first build.
2. **Fast iteration**: `build_cache.py --sample` fetches standings plus event
   logs for one conference only (~5 min) — enough to develop every page.
3. **Tests**: `test_offline.py` runs against a small committed fixture set
   (`tests/fixtures/`) so the analytics are testable without any cache.
4. **Deploy later**: the GitHub Action runs the builder and commits `cache/`;
   Streamlit Cloud just reads. Same pattern as SSL, no filesystem-wipe problem.

Admin page gets a "refresh light endpoints" button (season, standings for the
current season, bots, current rosters — a few seconds) so the deployed app can
pick up mid-season results without waiting for the weekly job.

### Cache files

| File | Source | Grain |
|---|---|---|
| `seasons.csv` | `/season/all` + hardcoded S1–S52 numbering | season |
| `standings.csv` | `/standings` S1–now | team-season |
| `games.csv` | `/standings.games` + `.postseason` | game (with `seq` = list order, `phase`) |
| `players.csv` | `/player` | player (snapshot, refreshed each build) |
| `tpe_events.csv` | `/tpeevents` for every player with `draftSeason ≥ 50` or active | event |
| `tpe_by_season.csv` | derived from events | player-season: TPE at season start, earned, regressed |
| `rosters.csv` | `/player/stats` per player, S54+ | player-season-team |
| `regressions.csv` | `/player/regression` S52+ | player-season |
| `bots.csv` | `/bots` per season | bot-season |
| `gm_history.csv` | `/gm-history` | gm-team-season |
| `draft_picks.csv` | `/draft-picks` S50+ | pick |
| `awards.csv` | `/awards` S50+ | award |
| `meta.json` | builder | build time, counts, API errors |

Incremental rule: re-fetch events and stats only for players who are active or
whose `retirementDate` is after the last build. Retired players' logs are final.

## 5. Models

### 5.1 Elo (game-level, S1 → now)
- Standard Elo. Start 1500. K tuned by backtest (expect ~20–30 for a 16-game
  season). No home advantage term unless the data shows one (check home win
  rate first — sims often have none).
- Playoff games included with the same K; they have scores but we ignore them
  here for consistency.
- **Between seasons**: revert toward 1500 by fraction `r` (start 1/3, tune).
- **Order uncertainty**: average the end-of-season rating over N shuffles of
  each season's game list plus the listed order. Report the spread as an
  honesty check; if it is tiny, drop the shuffling.

### 5.2 Margin rating (SRS) and Pythagorean expectation
- Per season, solve `rating_i = avg_margin_i + mean(rating of opponents_i)` with
  the opponent list from `games` and margin from PF/PA. 14 unknowns, linear,
  centred at 0. Gives points-above-average adjusted for schedule.
- Pythagorean expected win% `= pf^k / (pf^k + pa^k)`, `k` fitted (NFL ≈ 2.37).
  Delta vs actual win% = luck. Teams that over-performed Pythagoras tend to
  fall back next season — a useful predictor feature and a fun chart.
- The "Elo" page shows all three side by side; the season predictor uses Elo
  and SRS as separate features and lets the backtest weigh them.

### 5.3 GM ratings
- `gm_history` gives GMs per team-season. Two ratings:
  - **GM Elo**: each team-season's Elo delta is credited to that season's GMs
    (split evenly). Accumulates over a career, reverts like team Elo.
  - **GM over-performance**: mean residual of actual SRS minus roster-predicted
    SRS (§5.4) across a GM's ISFL team-seasons. This isolates depth-chart and
    strategy skill from roster quality. Only from S54 (needs the TPE model).
- Tenure view: timeline of GM pairs per team, with the team's Elo overlaid.
- Caveat on the page: two GMs per team, so attribution is shared and noisy.
  Present as "teams run by this GM did X," not "this GM is worth X."

### 5.4 Roster strength (S54 → now)
For every team-season, reconstruct the Week 1 roster and compute features:

- **Per position group**, mean TPE and **starter TPE** (top-N by TPE, N from the
  base formation: QB1, RB1, WR3, TE1, OL5 incl. bots, DL4, LB3, CB3, S2, K1).
  Starter TPE is expected to matter more than depth in a sim.
- Overall mean, total, and count of players.
- Age: mean career season; count of players in regression years.
- Activity: count with `weeklyActivityCheck`/`weeklyTraining` = 0 at build
  time is only a current-season signal — for history, use events per season
  (a player with < 10 events in a season was effectively IA).
- Bots merged into OL.

Target: **SRS** (margin), not wins — less noise in 16 games. Fit ridge
regression, leave-one-season-out CV. ~126 team-seasons, so keep it to ~10
features. Report:
- Coefficients per position group → **"which positions matter."**
- Partial effect plots: +100 starter TPE at QB → +X points/game.
- Champion profile: Ultimus winners' group-TPE z-scores vs the league that
  season. Which groups were they elite in? Which didn't matter?
- Correlation table, because ridge coefficients on correlated inputs need a
  sanity companion.

### 5.5 Preseason strength and simulation
- `strength = a·Elo_reverted + b·SRS_last + c·roster_pred + d·GM_overperf`,
  weights from the backtest (or a single ridge on these four).
- Per-game win probability: logistic on strength difference, slope fitted.
- Schedule: S63 games are not published preseason, so generate: 6 conference
  opponents × 2 + 4 cross-conference (random, or last season's cross pairings
  if the league rotates them — check S60–S62). Once the real schedule appears
  in `games`, use it and condition on results so far (in-season updates).
- 10,000 sims → per team: expected wins, P(each finish 1–7 in conference),
  P(best overall record), P(playoffs), P(bye), P(Ultimus) via playoff bracket
  with the same win model.
- Output the modal ranking per conference in the exact format of the prediction
  task, plus a copy button.

### 5.6 Projection (org trajectory, 1–3 seasons)
Per player: apply the regression table on career season, add expected earnings
(the player's own measured rate from events, excluding partial seasons, falling
back to league median by career season), retire at 13, retire when regression
takes TPE under 150, and apply a **retirement hazard** by career season measured
from the 4,059 retired players (`retirementDate` vs `draftSeason`). Roll up to
team: median and 10th–90th band. Same holdings as SSL: rates fixed, no decay,
draft adds little inside the horizon — but verify that here, ISFL draftees
enter at lower TPE relative to the cap than SSL's.

### 5.7 Backtest
Walk-forward: for each target season S in S55…S62 (TPE model) and S27…S62
(Elo-only variants), fit on seasons < S, predict S from its Week 1 roster and
prior ratings, score against the actual regular-season standings:
- Spearman ρ of predicted vs actual rank, per conference and overall.
- Mean absolute error of expected wins.
- Log-loss of P(best record) and P(playoffs).
Baselines that must be beaten: last season's standings; pure Elo; pure
roster-strength; random. Also report **which single feature is most
predictive** per season — the answer to "how much is average TPE" is the
roster-only model's ρ vs the full model's ρ.

Survivorship note: `/player` includes retired players, so no survivorship bias
in roster reconstruction. Bias does exist in **earning-rate** estimates if we
only measure active players — measure from everyone with events.

## 6. Pages

1. **Overview** — current standings, Elo table with movement, Pythagorean luck,
   season status. Quick NOLA card.
2. **Ratings** — Elo timeline S1→now (all teams, toggle), SRS, Pythagorean,
   per-season table, calibration chart ("when Elo said 70%, how often did the
   favourite win").
3. **Teams** — pick a team: roster with TPE, position, archetype, career
   season, next regression %, activity; position-group starter TPE vs league;
   age distribution; TPE-vs-age scatter; bots.
4. **League TPE** — every team's group TPE as a heatmap; timeline of team mean
   TPE S54→now; TPE vs age scatter of all 14 teams with Elo as colour.
5. **What wins** — §5.4 outputs: coefficients, partial effects, champion
   profiles, correlation table, and a "TPE explains X% of margin" headline.
6. **GMs** — GM Elo leaderboard, over-performance, tenure timeline per team,
   current GM pairs, `gm-history/records` careers.
7. **Projection** — §5.6 fan charts per team; league-wide "who's rising/falling"
   ranked by 2-season change.
8. **Predict** — §5.5: sliders for the blend weights (defaults from backtest),
   run sim, rank table per conference, probabilities, copy-ready ranking.
9. **Backtest** — §5.7 tables and charts; this is where the blend weights come
   from and it should say so.
10. **NOLA** — everything above filtered to NOLA: roster and regression
    calendar (who regresses when, by how much), ASFC rivals side by side,
    projection vs ASFC, draft picks owned, GM history, predicted finish
    distribution, and a "what would it take" widget: how much starter TPE at
    which group closes the gap to the ASFC leader.
11. **Admin** — cache meta, build log, light refresh, endpoint probe.

## 7. Build order

Each step ends in something runnable offline.

1. **Cache builder + `data.py`** — standings/games/players/gm_history first
   (fast, ~70 calls). App skeleton with Overview reading them. *Milestone: app
   runs offline with real standings.*
2. **Elo, SRS, Pythagorean** + Ratings page + tests. *Milestone: Elo history
   S1→S62 charted; quick sanity that Ultimus winners have high Elo.*
3. **Events, rosters, regressions, bots** into the cache (the slow part,
   incremental) + `roster.py` + Teams and League TPE pages.
4. **Roster-strength model** + What wins page.
5. **GM ratings** + GMs page.
6. **Projection** + Projection page.
7. **Simulation + backtest** + Predict and Backtest pages. Tune blend weights.
8. **NOLA page**, Admin, polish.
9. Repo, Action, Streamlit Cloud — only after everything above works locally.

Step 7 is the deliverable for the S63 prediction; the deadline is Week 1, so
steps 1–4 and 7 are the critical path. GM ratings and projection can slip.

## 7b. State after the first build (2026-09-21)

Everything in §6 is built and renders. Full cache: 4,823 players, 124,628
TPE events and 89,677 game rows for the 1,858 players in scope, zero fetch
errors. `test_offline.py` 14/14, `scripts/smoke_pages.py` 11/11 pages.
Not yet a git repo, not deployed — offline only, as requested.

What the data said:

- **Starter TPE explains 63% of point margin** (R² on SRS, 126 team-seasons
  S54–S62). The nine-group ridge gets 0.64 out of sample. Per +100 starter
  TPE: OL +0.83 pts/game, DB +0.76, LB +0.72, WR +0.67, RB +0.51, DL +0.49,
  K +0.43, **QB +0.20** (univariate r = 0.10 — QB TPE barely matters in
  this engine), TE −0.17 (noise). Inactive players on the roster: r = −0.60.
- **Preseason ranking tops out at Spearman ≈ 0.5** on a 7-team conference
  whatever the method: copy-last-season 0.52, roster model 0.52, blend
  0.49–0.51, Elo 0.46. Over 6–35 seasons those are indistinguishable. The
  roster model has the lowest points error (MAE 3.9) and the best 14-team
  rank correlation (0.54), so it carries the most information; the blend
  gives it 46% of the weight (SRS 24%, record 17%, Elo 12%, GM 0%).
- **S63, fitted blend**: ASFC HON, NYS, NOLA, AZ, SJS, AUS, OCO; NSFC BAL,
  OSK, SAR, YKW, COL, CTC, BFB; best record BAL. **Roster-only** disagrees
  on the champion: NOLA first in the ASFC (starter TPE 820, 3 IA) and BAL
  fifth in the NSFC (703, 3 IA) — BAL went 14-2 on the second-lowest
  playoff-team TPE, which is either GM skill (RoyRivers leads
  over-performance at +4.7 pts/game) or luck. That disagreement is the
  judgement call for the submission; both views are on the Predict page.
- Retirement hazard by career season: ~1% through 5, 4% at 6, 7% at 7,
  12% at 8, 19% at 9, 34% at 10, 61% at 11, 70% at 12, forced at 13.
- League median earning by career season: 179 (DSFL year) → 160 → 145 →
  … → ~110 at 11+.

Next steps, in order: `git init`, push, enable the Action, deploy to
Streamlit Cloud pointing at `app.py`. Then the ideas that didn't make the
cut: applied-vs-banked TPE, attribute-level model, within-season pacing.

## 8. Verified during the build — decisions that look like bugs but aren't

Each of these was checked against live data on 2026-09-21. Do not undo.

**`/player/regression` is not a historical snapshot.** Its `old_tpe` /
`new_tpe` are the player's *current* applied TPE stamped on every season
(one player showed 1188 for S57 through S62 while his log went 466 → 1443).
Only `pct` is trustworthy. Week 1 TPE therefore comes from the event log
alone, which sums exactly to `total_tpe` for 100% of players with a log
(`roster.reconcile`). Every logged regression matches
`floor(pct × TPE)` for the rulebook table (`roster.verify_regression_pcts`).

**Regression is labelled, not dated.** `"S63 Regression"` is posted one
second after the S63 start timestamp. Bucketing by date would work today
and break the day a regression is posted late, so regressions are assigned
to the season in their label.

**`retirement_date` is unusable for career length.** Only 495 of 4,059
retired players have one, and many of those are the placeholder
`2024-01-01`. The retirement hazard uses the last regular season in the
game logs instead, restricted to players whose whole career is inside the
log window.

**`games` is chronological; `week = index // 7 + 1`.** Verified against
`/player/game-stats` weeks. Elo uses the listed order, no shuffling.

**Home advantage is real: 55.1% over 1,456 games (S50–62).** Elo carries a
fitted home term (40 points) and the win model a fitted `hfa`.

**Elo K=40, revert ⅓** — grid-tuned on S27+ log-loss (0.631 vs 0.693 for a
coin). Calibration is within a few points in every bucket. 23 of 36
post-S27 champions were Elo #1.

**Cross-conference pairings rotate** (28 distinct pairs per season, each
once), so the simulator generates a random circulant pairing rather than
copying last season's.

**Relocations carry ratings**: LVL→NOLA (S6), PHI→CTC (S39), CHI→OSK (S49),
BER→BFB (S51). Each old code's last season is immediately followed by the
new code's first, same conference.

**"Copy last season's record" is a strong baseline** (Spearman ~0.52 on
conference order). Last season's win% is in the blend as `wpct_prev` because
it ranks better than SRS or Elo alone, even though those predict *points*
better. The blend target is win% in points (`wpct_pts`) for the same reason.

**Bots come from `/bots?season=N` back to S56.** ISFL OL bots carry a 0–4
tier index; DSFL filler bots carry raw TPE in the same column. `data._prep_bots`
branches on `<= 4`.

**The preview tool's `launch.json` resolves from the session's original
directory.** Run Streamlit directly if the wrong app comes up.

## 9. Known limits

- Historical rosters come from game logs, so backups who never recorded a
  stat are missing. Starter TPE is unaffected; mean TPE is slightly high.
- Historical TPE is total, not applied. Banked TPE does nothing in the sim.
- ~14 rows per season for the roster model. Coefficients move when a season
  is added; read direction, not decimals.
- Spearman on a 7-team conference has a standard error of ~0.35 per season;
  differences of 0.03 between methods over 35 seasons are noise.
- `wfcRegion`, awards and attributes are loaded but unused.
