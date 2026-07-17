"""
alerts.py — Slack budget notifications for TokenShark.

Wired in from tracker.py's log_call(): every successful call computes
today's cumulative spend and passes it here as `current_spend`, behind
a guarded `from .alerts import check_and_alert` import so nothing broke
before this file existed. Nothing in tracker.py, config.py, or
exceptions.py needs to change for this to start working -- they were
already built to call into this the moment it showed up.

DECISION GAP: tracker.log_call() only ever passes *today's* cumulative
spend (current_spend=today_spend) -- there's no per-call cost and no
session id anywhere in the schema (calls has no session column). So
"session" is tracked here, independently, as spend accumulated since
THIS PROCESS started calling into check_and_alert -- computed as the
delta between successive current_spend readings, floor-clamped at 0 so
a day rollover (current_spend dropping back down) never goes negative.
Known limitation this doesn't handle: a process that starts mid-day,
after other processes already logged spend to the same shared logs.db,
will count that pre-existing spend as part of its own first-call
"session" delta. A precise fix needs a real session id threaded through
proxy.py -> tracker.log_call() -> here, which is a bigger change than
this file should make unasked.

Slack notifications are deliberately deduped (see _daily_notified_date
/ _session_notified below) so one crossed budget doesn't spam the
channel on every subsequent call -- but a hard_stop keeps raising every
time regardless, since that's a deliberate per-call guard, not a
one-time heads up.
"""

import time
import urllib.error
import urllib.request
import warnings
from json import dumps

from .exceptions import TokenSharkBudgetExceeded

_session_spend = 0.0
_last_daily_spend = 0.0
_daily_notified_date: str | None = None
_session_notified = False


def _post_to_slack(webhook: str, text: str) -> None:
    """
    Plain stdlib POST to a Slack incoming webhook -- no `requests`
    dependency for one JSON call, matching the rest of the project's
    "zero unnecessary deps" stance (see tracker.py's _print_call_summary
    comment on skipping `rich` for the same reason).
    """
    body = dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        webhook, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        resp.read()


def _notify(cfg: dict, kind: str, spend: float, budget: float, model: str) -> None:
    webhook = cfg["alerts"].get("slack_webhook")
    if not webhook:
        return
    label = "Daily" if kind == "daily" else "Per-session"
    text = (
        f":shark: *TokenShark budget alert*\n"
        f"{label} budget exceeded: *${spend:.2f}* spent vs. *${budget:.2f}* budget "
        f"(triggered by a call to `{model}`)."
    )
    try:
        _post_to_slack(webhook, text)
    except (urllib.error.URLError, OSError) as e:
        # Same rule as tracker.py's DB writes: a broken webhook must
        # never take down the real, already-completed LLM call.
        warnings.warn(f"TokenShark: could not send Slack budget alert ({e}).")


def check_and_alert(cfg: dict, current_spend: float, model: str) -> None:
    """
    Called by tracker.log_call() after every successful, logged call.

    Raises TokenSharkBudgetExceeded only when alerts.hard_stop is true
    AND a budget has actually been exceeded -- tracker.py deliberately
    lets that one exception propagate through the user's real LLM call
    (see the comment on TokenSharkBudgetExceeded in exceptions.py); any
    other exception here is caught and warned on by the caller instead.
    """
    global _session_spend, _last_daily_spend, _daily_notified_date, _session_notified

    delta = current_spend - _last_daily_spend
    _session_spend += delta if delta > 0 else 0.0
    _last_daily_spend = current_spend

    daily_budget = cfg["alerts"]["daily_budget"]
    session_budget = cfg["alerts"]["per_session_budget"]
    hard_stop = cfg["alerts"]["hard_stop"]
    today = time.strftime("%Y-%m-%d")
    exceeded_kind = None

    if daily_budget and current_spend >= daily_budget:
        exceeded_kind = "daily"
        if _daily_notified_date != today:
            _notify(cfg, "daily", current_spend, daily_budget, model)
            _daily_notified_date = today

    if session_budget and _session_spend >= session_budget:
        exceeded_kind = exceeded_kind or "session"
        if not _session_notified:
            _notify(cfg, "session", _session_spend, session_budget, model)
            _session_notified = True

    if exceeded_kind and hard_stop:
        spend = current_spend if exceeded_kind == "daily" else _session_spend
        budget = daily_budget if exceeded_kind == "daily" else session_budget
        raise TokenSharkBudgetExceeded(
            f"TokenShark: {exceeded_kind} budget exceeded (${spend:.2f} / ${budget:.2f}) "
            f"and alerts.hard_stop is true."
        )
