"""
dashboard.py — minimal terminal table for TokenShark: model, tokens,
cost, latency, refreshed on a timer.

Scope, per the Step 2 priority list this was built against: "working
table ... no fancy sorting or animation needed, just functional and
readable." This deliberately is NOT a rich.Live animated view -- it
clears the screen and reprints a plain rich.Table each refresh. That's
enough for "functional and readable" without the extra edge cases a
live-updating render brings (terminal resize handling, partial-frame
flicker, etc.). Upgrading to rich.Live is a reasonable v0.2 item, not
needed for v0.1.

Not wired into cli.py -- cli.py isn't part of this batch (see
ARCHITECTURE.md file structure / README.md status), so pyproject.toml's
`tokenshark` console script still won't resolve until it exists. Until
then: `python -m tokenshark.dashboard`, or `from tokenshark.dashboard
import show_dashboard; show_dashboard()`.

Depends on tracker._connect(), a name-mangled "private" helper in
another module -- acceptable for a same-package import, but worth
promoting to a proper public query function in tracker.py during v0.2
cleanup rather than leaving two modules coupled through an
underscore-prefixed name.
"""

import os
import sqlite3
import time

from rich.console import Console
from rich.table import Table

from . import config
from .tracker import _connect

# Whitelisted so tokenshark.yaml's dashboard.sort_by value can never be
# concatenated into SQL as-is. config.py's DANGEROUS_KEYS check guards
# secret-shaped *keys*, not arbitrary column-name injection through a
# *value* like this one -- that has to be handled here instead.
_SORT_COLUMNS = {
    "cost": "cost_usd",
    "latency": "latency_ms",
    "recent": "created_at",
}


def _fetch_rows(cfg: dict, limit: int, sort_by: str) -> list[sqlite3.Row]:
    column = _SORT_COLUMNS.get(sort_by, "created_at")
    with _connect(cfg) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            f"SELECT provider, model, prompt_tokens, completion_tokens, "
            f"cost_usd, latency_ms, error, estimated, created_at "
            f"FROM calls ORDER BY {column} DESC, created_at DESC LIMIT ?",
            (limit,),
        )
        return cursor.fetchall()


def _render(rows: list[sqlite3.Row]) -> Table:
    table = Table(title="TokenShark — recent calls")
    table.add_column("Time", style="dim")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Prompt→", justify="right")
    table.add_column("←Compl.", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Latency", justify="right")

    total_cost = 0.0
    for row in rows:
        if row["error"]:
            table.add_row(
                row["created_at"], row["provider"], row["model"],
                "-", "-", "[red]FAILED[/red]", f"{row['latency_ms']}ms",
            )
            continue
        total_cost += row["cost_usd"]
        est = " (est.)" if row["estimated"] else ""
        table.add_row(
            row["created_at"],
            row["provider"],
            row["model"],
            f"{row['prompt_tokens']:,}",
            f"{row['completion_tokens']:,}{est}",
            f"${row['cost_usd']:.4f}",
            f"{row['latency_ms']}ms",
        )
    table.caption = f"Total (rows shown): ${total_cost:.4f}"
    return table


def show_dashboard(once: bool = False) -> None:
    """Render the calls table. Loops on a timer (dashboard.refresh_seconds
    from tokenshark.yaml) until interrupted with Ctrl+C, unless
    once=True (single render — useful for scripting/piping/tests)."""
    cfg = config.load_config()
    limit = cfg.get("dashboard", {}).get("max_rows", 20)
    sort_by = cfg.get("dashboard", {}).get("sort_by", "cost")
    refresh = cfg.get("dashboard", {}).get("refresh_seconds", 2)
    console = Console()

    while True:
        rows = _fetch_rows(cfg, limit, sort_by)
        os.system("cls" if os.name == "nt" else "clear")
        console.print(_render(rows))
        if once:
            return
        try:
            time.sleep(refresh)
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    show_dashboard()
