# ISFL Team Tracker

Elo, roster TPE, "what wins the Ultimus", org projections and a preseason
ranking simulator for the International Simulation Football League. Built
for the season-rankings prediction task, with a NOLA page.

**New here? Read `GUIDE.md`** — plain step-by-step instructions for running,
updating, using and publishing it. Read `DESIGN.md` before changing code —
it records the domain rules and the data traps.

## Run it offline

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt      # Windows
python scripts/build_cache.py --stage fast          # ~3 min: standings, players, GMs…
python scripts/build_cache.py --stage slow          # ~45 min first time: per-player logs
streamlit run app.py
```

The app reads `cache/` only and never touches the network on a page load.
Once the cache is built it works with no internet at all.

- `--sample` limits the slow stage to the current ISFL rosters (~10 min) —
  enough to develop every page.
- The slow stage is incremental: retired players' logs are final and are
  not re-fetched. Re-run it whenever you want fresh TPE.
- Admin → "Light refresh" pulls this season's standings, players, bots and
  regressions in ~30 s.

## Check it

```bash
python test_offline.py          # analytics checks, no cache needed
python scripts/smoke_pages.py   # renders every page headlessly against the cache
python scripts/experiment_blend.py   # which blend target/alpha ranks best
```

## Layout

Flat. `app.py` is the entry point; `config.py` holds every constant and rule
table; `data.py` reads the cache; `elo.py`, `roster.py`, `model.py`, `gm.py`,
`simulate.py`, `projection.py`, `backtest.py` are the analytics;
`page_*.py` are the pages; `scripts/build_cache.py` is the only thing that
talks to the API.

## Deploy

Push to GitHub with `cache/` committed. The workflow in
`.github/workflows/update-cache.yml` rebuilds the cache weekly and commits
it. Point Streamlit Community Cloud at `app.py`.
