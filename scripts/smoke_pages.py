"""
Render every page headlessly with Streamlit's AppTest and report exceptions.

    python scripts/smoke_pages.py

Uses the real cache. Slow the first time (context build), then each page is
a re-run against the cached context.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from streamlit.testing.v1 import AppTest  # noqa: E402


def main() -> int:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    t = time.time()
    at.run()
    print(f"initial run {time.time() - t:.1f}s, exceptions: {len(at.exception)}")
    pages = at.sidebar.radio[0].options
    failures = 0
    for page in pages:
        t = time.time()
        at.sidebar.radio[0].set_value(page).run()
        errs = at.exception
        status = "ok " if not errs else "ERR"
        print(f"[{status}] {page:<12} {time.time() - t:5.1f}s")
        for e in errs:
            failures += 1
            print("      ", e.value)
            tb = getattr(e, "stack_trace", None)
            if tb:
                lines = [ln for ln in tb if "isfl-team-tracker" in ln and "site-packages" not in ln]
                for ln in lines[-3:]:
                    print("      ", ln.strip())
    print(f"\n{failures} page exceptions")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
