"""
cli.py — command-line interface for TokenShark.

Wire this up in pyproject.toml:

    [project.scripts]
    tokenshark = "tokenshark.cli:main"

`tokenshark --help` or `tokenshark help` lists every command below;
`tokenshark <command> --help` or `tokenshark help <command>` shows
that command's own options.

Commands:
    report   Summarize logged calls: total spend, tokens, call count,
             broken down by model or provider.
    budget   Show today's spend against your configured daily/session
             budget (tokenshark.yaml's alerts: section).
    export   Dump the local calls table to a CSV or JSON file.
    reset    Clear all logged calls from the local database.
    help     Show top-level help, or help for one specific command.

Every command loads config the same way monitor() does
(config.load_config()), so a project-level tokenshark.yaml is picked
up automatically -- no separate CLI config needed.

Reuses tracker._connect / _resolve_db_path / _today_spend rather than
duplicating connection handling here, and reuses __init__.py's
_Ansi / _supports_color for the same color rules as the startup
banner (auto-detected TTY, NO_COLOR / FORCE_COLOR). Only hand-written
strings (the top-level description/epilog, and every print() below)
get colored -- argparse's own auto-generated option listing is left
in plain text on purpose, since its column widths are computed from
raw string length and embedded escape codes would throw that off.
"""

import argparse
import csv
import json
import sqlite3
import sys

from . import _Ansi, _supports_color, config
from .tracker import _connect, _resolve_db_path, _today_spend


def _require_db(cfg: dict) -> None:
    db_path = _resolve_db_path(cfg)
    if not db_path.exists():
        print(f"No TokenShark database found at {db_path}.")
        print("Run some LLM calls with tokenshark.monitor() active first.")
        sys.exit(1)


def _threshold_color(a: _Ansi, pct: float) -> str:
    """Green under 70% of budget, yellow 70-99%, red at/over 100%."""
    if pct >= 100:
        return a.red
    if pct >= 70:
        return a.yellow
    return a.green


def cmd_report(args) -> None:
    cfg = config.load_config()
    _require_db(cfg)
    a = _Ansi(_supports_color())
    where = "WHERE date(created_at) = date('now')" if args.today else ""

    try:
        with _connect(cfg) as conn:
            conn.row_factory = sqlite3.Row
            totals = conn.execute(
                f"SELECT COUNT(*) AS calls, COALESCE(SUM(cost_usd),0) AS cost, "
                f"COALESCE(SUM(prompt_tokens),0) AS prompt, "
                f"COALESCE(SUM(completion_tokens),0) AS completion "
                f"FROM calls {where}"
            ).fetchone()
            rows = conn.execute(
                f"SELECT {args.group_by} AS key, COUNT(*) AS calls, "
                f"COALESCE(SUM(cost_usd),0) AS cost "
                f"FROM calls {where} GROUP BY {args.group_by} ORDER BY cost DESC"
            ).fetchall()
    except sqlite3.OperationalError:
        print("No calls logged yet.")
        return

    scope = "today" if args.today else "all time"
    print(f"\n{a.bold_cyan}TokenShark report ({scope}){a.reset}")
    print(f"{a.dim}  Calls:       {a.reset}{totals['calls']:,}")
    print(f"{a.dim}  Prompt tok:  {a.reset}{totals['prompt']:,}")
    print(f"{a.dim}  Completion:  {a.reset}{totals['completion']:,}")
    print(f"{a.dim}  Total cost:  {a.reset}{a.green}${totals['cost']:.4f}{a.reset}\n")
    print(f"{a.bold_cyan}By {args.group_by}:{a.reset}")
    for row in rows:
        print(
            f"  {a.cyan}{row['key']:<20}{a.reset} {row['calls']:>6,} calls   "
            f"{a.green}${row['cost']:.4f}{a.reset}"
        )


def cmd_budget(args) -> None:
    cfg = config.load_config()
    _require_db(cfg)
    a = _Ansi(_supports_color())
    try:
        with _connect(cfg) as conn:
            spend = _today_spend(conn)
    except sqlite3.OperationalError:
        spend = 0.0

    daily = cfg["alerts"]["daily_budget"]
    session_cap = cfg["alerts"]["per_session_budget"]
    hard_stop = cfg["alerts"]["hard_stop"]
    pct = (spend / daily * 100) if daily else 0.0
    spend_color = _threshold_color(a, pct)
    hard_stop_color = a.green if hard_stop else a.dim

    print(f"\n{a.dim}Today's spend:   {a.reset}{spend_color}${spend:.4f}{a.reset}")
    print(f"{a.dim}Daily budget:    {a.reset}${daily:.2f}  {spend_color}({pct:.0f}% used){a.reset}")
    print(f"{a.dim}Per-session cap: {a.reset}${session_cap:.2f}")
    print(f"{a.dim}Hard stop:       {a.reset}{hard_stop_color}{'on' if hard_stop else 'off'}{a.reset}\n")


def cmd_export(args) -> None:
    cfg = config.load_config()
    _require_db(cfg)
    a = _Ansi(_supports_color())
    try:
        with _connect(cfg) as conn:
            conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute("SELECT * FROM calls ORDER BY created_at").fetchall()]
    except sqlite3.OperationalError:
        rows = []

    if args.format == "csv":
        with open(args.output, "w", newline="") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    else:
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=2, default=str)

    print(f"{a.green}Exported {len(rows)} rows{a.reset} to {args.output}")


def cmd_reset(args) -> None:
    cfg = config.load_config()
    a = _Ansi(_supports_color())
    db_path = _resolve_db_path(cfg)
    if not db_path.exists():
        print(f"{a.dim}Nothing to reset -- no database found.{a.reset}")
        return

    if not args.yes:
        confirm = input(
            f"{a.yellow}This deletes all logged calls in {db_path}. Type 'yes' to continue: {a.reset}"
        )
        if confirm.strip().lower() != "yes":
            print(f"{a.dim}Cancelled.{a.reset}")
            return

    with _connect(cfg) as conn:
        conn.execute("DELETE FROM calls")
    print(f"{a.green}TokenShark logs cleared.{a.reset}")


def cmd_help(args) -> None:
    if args.topic and args.topic in args.subparsers:
        args.subparsers[args.topic].print_help()
    else:
        args.parser.print_help()


def build_parser() -> argparse.ArgumentParser:
    a = _Ansi(_supports_color())
    epilog = (
        f"{a.dim}examples:{a.reset}\n"
        f"  tokenshark report --today --group-by provider\n"
        f"  tokenshark budget\n"
        f"  tokenshark export calls.json --format json\n"
        f"  tokenshark help report\n"
    )
    parser = argparse.ArgumentParser(
        prog="tokenshark",
        description=f"{a.bold_cyan}TokenShark{a.reset} \u2014 local cost and latency tracking for your LLM API calls.",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    subparsers_by_name = {}

    p_report = sub.add_parser("report", help="Summarize logged calls (spend, tokens, calls)")
    p_report.add_argument("--today", action="store_true", help="Limit to today's calls")
    p_report.add_argument(
        "--group-by", choices=["model", "provider"], default="model",
        help="Group totals by model or provider (default: model)",
    )
    p_report.set_defaults(func=cmd_report)
    subparsers_by_name["report"] = p_report

    p_budget = sub.add_parser("budget", help="Show today's spend against your configured budget")
    p_budget.set_defaults(func=cmd_budget)
    subparsers_by_name["budget"] = p_budget

    p_export = sub.add_parser("export", help="Export logged calls to a CSV or JSON file")
    p_export.add_argument("output", help="Output path, e.g. calls.csv or calls.json")
    p_export.add_argument("--format", choices=["csv", "json"], default="csv")
    p_export.set_defaults(func=cmd_export)
    subparsers_by_name["export"] = p_export

    p_reset = sub.add_parser("reset", help="Clear all logged calls from the local database")
    p_reset.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    p_reset.set_defaults(func=cmd_reset)
    subparsers_by_name["reset"] = p_reset

    p_help = sub.add_parser("help", help="Show top-level help, or help for one specific command")
    p_help.add_argument(
        "topic", nargs="?", default=None,
        help="Command to show help for, e.g. `tokenshark help report`",
    )
    p_help.set_defaults(func=cmd_help, parser=parser, subparsers=subparsers_by_name)
    subparsers_by_name["help"] = p_help

    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
