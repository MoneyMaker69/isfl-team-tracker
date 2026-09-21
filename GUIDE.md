# Step-by-step guide

Plain instructions for running, updating, using and publishing the ISFL
Team Tracker. Commands are for Windows PowerShell; run them from the
project folder unless it says otherwise.

---

## A. Run it on this PC (everything is already set up)

1. Open PowerShell (Start → type `powershell` → Enter).
2. Go to the project folder:
   ```
   cd C:\Users\Bob\Downloads\isfl-team-tracker
   ```
3. Start the app:
   ```
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```
4. A browser tab opens at `http://localhost:8501`. The first page takes
   ~15 seconds to compute everything; after that it's instant.
5. To stop it, go back to the PowerShell window and press `Ctrl+C`.

That's it for day-to-day use.

---

## B. Set it up on a different PC from scratch

1. Install Python 3.12 from https://www.python.org/downloads/ — tick
   **"Add python.exe to PATH"** during install.
2. Copy the whole `isfl-team-tracker` folder to the new PC. You can
   leave out `.venv` (it will be recreated) but **keep `cache/`** — it saves
   an hour of downloading.
3. Open PowerShell in that folder and run, one line at a time:
   ```
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe test_offline.py
   ```
   The last line should end with `0 failures`.
4. If you did *not* copy `cache/`, build it (needs internet):
   ```
   .\.venv\Scripts\python.exe scripts\build_cache.py --stage fast
   .\.venv\Scripts\python.exe scripts\build_cache.py --stage slow
   ```
   Fast is ~3 minutes; slow is ~1 hour the first time (it downloads every
   player's TPE history). Both can be interrupted and resumed.
5. Start the app as in section A.

---

## C. Keep the data fresh

The app only reads files in `cache/`; it never contacts the league site by
itself. So before you rely on it (e.g. the night the prediction task is
due), update the cache:

**Quick (30 seconds)** — this season's standings, the player list, bots,
regressions:

- In the app: sidebar → **Admin** → **Light refresh from the API** → then
  sidebar → **Reload cache**.

**Full (a few minutes, incremental)** — also every active player's TPE
history and game log, which is what the Teams / League TPE / Projection /
Predict pages use:

```
.\.venv\Scripts\python.exe scripts\build_cache.py
```

Then press **Reload cache** in the app (or restart it). Only active
players are re-downloaded; retired players' logs are final.

When to do the full update: after the draft, after free agency, and right
before Week 1. Once a week during the season is plenty.

---

## D. Do the prediction task

1. Update the cache (section C, full).
2. Open the app → **Predict**.
3. Read the "What am I looking at?" box once.
4. Leave **Weights** on *Fitted by backtest* for the model's answer. Then
   flip to *Roster model only* to see the pure-TPE answer. Where the two
   agree, be confident. Where they disagree, that's your judgement call —
   the Backtest page says neither is measurably better.
5. The grey box under the two columns is the ranking in task order. Copy
   it. The line "Regular season champion" is the best-record pick.
6. Use the **Finish distribution** tab to see how sure each placement is.
   A team with 35% for 2nd and 30% for 3rd is a coin flip between them —
   order those by gut, or by the **NOLA** page's rivals view.
7. Optional: **Sensitivity** shows how much a team's outlook changes if the
   model is off by a few points. Big swings mean don't over-think it.

---

## E. Put it online (GitHub + Streamlit Cloud + weekly auto-update)

Do this once. Afterwards the site updates itself every Monday.

### E1. Create the GitHub repository

1. Install Git from https://git-scm.com/download/win if `git --version`
   in PowerShell says it isn't recognised.
2. In the project folder:
   ```
   git init
   git add .
   git commit -m "ISFL Team Tracker"
   ```
   `.venv` and log files are excluded by `.gitignore`; `cache/` is
   **included on purpose** — the site reads it.
3. On https://github.com click **New repository**, name it
   `isfl-team-tracker`, leave it empty (no README), create it.
4. Copy the two lines GitHub shows under "push an existing repository" and
   run them. They look like:
   ```
   git remote add origin https://github.com/<you>/isfl-team-tracker.git
   git push -u origin main
   ```
   (If it says the branch is `master`, run `git branch -M main` first.)

### E2. Turn on the weekly cache update

1. On GitHub open the repo → **Settings** → **Actions** → **General** →
   under "Workflow permissions" pick **Read and write permissions** → Save.
2. Open the **Actions** tab → **Update cache** → **Run workflow** → Run.
   It takes a few minutes and commits fresh `cache/` files. From then on it
   runs every Monday 06:00 UTC by itself.

### E3. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. **New app** → pick the `isfl-team-tracker` repo, branch `main`, main
   file path **`app.py`** → Deploy.
3. First build takes a couple of minutes. You get a URL like
   `https://<you>-isfl-team-tracker.streamlit.app` — share it with NOLA.
4. The site redeploys automatically whenever the repo changes, including
   the Monday cache commits.

### E4. Updating the site later

- Data: nothing to do; the Action handles it. For an immediate refresh,
  run the workflow manually (E2 step 2), or run `build_cache.py` locally
  and `git add cache; git commit -m "cache"; git push`.
- Code: edit locally, test with `test_offline.py` and
  `scripts\smoke_pages.py`, then `git add .; git commit -m "..."; git push`.

---

## F. If something goes wrong

| Symptom | Fix |
|---|---|
| `python` not recognised | Reinstall Python with "Add to PATH" ticked, or use the full path `.\.venv\Scripts\python.exe`. |
| App shows "No data yet" | Run `scripts\build_cache.py --stage fast`, then Reload cache. |
| TPE pages say "Player logs not built yet" | Run `scripts\build_cache.py --stage slow`. |
| Numbers look stale | Sidebar → Reload cache. The app caches its computations for the session. |
| A page shows a red error box | Run `scripts\smoke_pages.py` and send the output; the traceback names the file and line. |
| Build script stops mid-way | Just run it again; it resumes where it left off. |
| Streamlit Cloud says "No module named …" | `requirements.txt` is missing something; add it, push. |
| Action says "No event triggers defined in on" | The `"on":` key in the workflow file lost its quotes. Put them back. |

Check the numbers add up any time on **Admin** → *Data quality*: the event
log should reconcile for ~100% of players.
