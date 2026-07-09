"""
tracker.py — SQLite persistence + cost calculation for TokenShark.

Owns:
  - COST_PER_1M pricing table (placeholders flagged for Day-3 verification
    — see the header comment on the table itself)
  - calc_cost(): pure cost math, no I/O, easy to unit test in isolation
  - init_db() / log_call(): the schema + the single write path every
    provider patch in proxy.py calls into, on both success and failure
"""

import json
import os
import sqlite3
import warnings
from contextlib import contextmanager
from pathlib import Path

from . import config
from .exceptions import TokenSharkBudgetExceeded, TokenSharkDatabaseError

# ---------------------------------------------------------------------------
# Pricing table
# ---------------------------------------------------------------------------
# DAY 3 ACTION (per ARCHITECTURE.md #3 / your own instruction #4): every
# price below needs to be checked against the official pricing page before
# this ships. Entries marked FILL IN are $0.00 placeholders and WILL
# under-report cost if used as-is. Entries with real-looking numbers still
# need a final check — treat everything here as unverified until you've
# clicked through yourself.
#   OpenAI:     https://openai.com/api/pricing
#   Anthropic:  https://www.anthropic.com/pricing
#   DeepSeek:   https://platform.deepseek.com (pricing page)
#   Qwen:       https://dashscope.aliyun.com
#   Mistral:    https://console.mistral.ai
# All prices are USD per 1,000,000 tokens.

COST_PER_1M = {
    # --- OpenAI --- verify at openai.com/api/pricing
    "gpt-4o": {"in": 0.00, "out": 0.00, "cache_read": 0.00},  # FILL IN
    "gpt-4o-mini": {"in": 0.00, "out": 0.00, "cache_read": 0.00},  # FILL IN
    "gpt-4-turbo": {"in": 10.00, "out": 30.00, "cache_read": 0.00},  # VERIFY

    # --- Anthropic --- verify at anthropic.com/pricing
    # cache_create ~= 1.25x base input, cache_read ~= 0.10x base input —
    # confirm this ratio still holds per-model before shipping, not just
    # for whichever model it was originally checked against.
    "claude-opus-4-6": {
        "in": 5.00, "out": 25.00, "cache_create": 1.25, "cache_read": 0.50,
    },  # VERIFY
    "claude-sonnet-4-6": {
        "in": 3.00, "out": 15.00, "cache_create": 0.75, "cache_read": 0.30,
    },  # VERIFY
    "claude-haiku-4-5": {
        "in": 1.00, "out": 5.00, "cache_create": 0.25, "cache_read": 0.10,
    },  # VERIFY

    # --- DeepSeek --- verify at platform.deepseek.com
    "deepseek-v4-flash": {"in": 0.14, "out": 0.28, "cache_read": 0.00},  # VERIFY
    "deepseek-v3.2": {"in": 0.28, "out": 0.42, "cache_read": 0.00},  # VERIFY

    # --- Qwen --- verify at dashscope.aliyun.com
    "qwen3.5-plus": {"in": 0.50, "out": 3.00, "cache_read": 0.00},  # VERIFY

    # --- Mistral --- verify at console.mistral.ai
    "mistral-large": {"in": 0.00, "out": 0.00, "cache_read": 0.00},  # FILL IN

    # --- Ollama (local) --- always free, nothing to verify
    "gemma3:2b": {"in": 0.00, "out": 0.00, "cache_read": 0.00},
    "gemma3:1b": {"in": 0.00, "out": 0.00, "cache_read": 0.00},
    "llama3.2": {"in": 0.00, "out": 0.00, "cache_read": 0.00},
}


def calc_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cache_create_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Pure function: token counts + price table -> USD. No I/O, so it's
    trivial to unit test on its own (see tests/test_tracker.py in your
    file structure — not part of this batch, but this function is written
    so those tests need zero mocking)."""
    rates = COST_PER_1M.get(model)
    if rates is None:
        warnings.warn(f"TokenShark: unknown model '{model}', cost logged as $0.00")
        return 0.0

    base = (prompt_tokens * rates["in"] + completion_tokens * rates["out"]) / 1_000_000
    cache_cost = (
        cache_create_tokens * rates.get("cache_create", 0)
        + cache_read_tokens * rates.get("cache_read", 0)
    ) / 1_000_000
    return base + cache_cost


# ---------------------------------------------------------------------------
# Token estimation (streaming fallback only)
# ---------------------------------------------------------------------------

_tiktoken_encoding = None
_tiktoken_warned = False


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate, used only by proxy.py's streaming wrappers when
    a provider doesn't return usage data on a stream (ARCHITECTURE.md's
    "Streaming Handling" decision). Tries tiktoken first; falls back to a
    crude chars/4 heuristic with a one-time warning if tiktoken isn't
    installed, so an estimate is never silently mistaken for an exact
    count. tiktoken is an optional dependency
    (pip install tokenshark[estimate]) — see the flag on this in
    pyproject.toml.
    """
    global _tiktoken_encoding, _tiktoken_warned
    if not text:
        return 0
    try:
        import tiktoken

        if _tiktoken_encoding is None:
            _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
        return len(_tiktoken_encoding.encode(text))
    except ImportError:
        if not _tiktoken_warned:
            warnings.warn(
                "TokenShark: tiktoken not installed — using a rough chars/4 "
                "estimate for streamed calls that don't return usage data. "
                "pip install tokenshark[estimate] for a closer estimate."
            )
            _tiktoken_warned = True
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# Kept in sync with migrations/001_init.sql (the human-readable reference
# copy). Embedded here as a string, rather than read from that file at
# runtime, so the installed package doesn't need extra packaging config
# (MANIFEST.in / [tool.hatch.build] package_data) just to ship one .sql
# file inside the wheel — one less thing that can go wrong on a clean
# install. If you change the schema, change both places.
#
# DECISION GAP: deviates from ARCHITECTURE.md's literal CREATE TABLE in
# one way — added the `estimated` column. The doc's "Streaming Handling"
# decision says to log estimated token counts "as estimated, not exact"
# for streams that don't return usage, but the original schema had no
# column to record that flag on. Added here so the two decisions are
# consistent with each other. Flagging since this changes your item-3
# deliverable from what was literally in the architecture doc.

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS calls (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    provider            TEXT NOT NULL,
    model               TEXT NOT NULL,
    prompt_tokens       INTEGER DEFAULT 0,
    completion_tokens   INTEGER DEFAULT 0,
    cache_create_tokens INTEGER DEFAULT 0,
    cache_read_tokens   INTEGER DEFAULT 0,
    cost_usd            REAL DEFAULT 0.0,
    latency_ms          INTEGER DEFAULT 0,
    tags                TEXT DEFAULT '{}',
    error               TEXT DEFAULT NULL,
    estimated           INTEGER DEFAULT 0,
    created_at          DATETIME DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_created_at ON calls(created_at);
CREATE INDEX IF NOT EXISTS idx_model ON calls(model);
CREATE INDEX IF NOT EXISTS idx_provider ON calls(provider);
"""

_db_initialized = False


def _resolve_db_path(cfg: dict | None = None) -> Path:
    cfg = cfg or config.load_config()
    raw_path = cfg.get("storage", {}).get("path", "~/.tokenshark/logs.db")
    return Path(os.path.expanduser(raw_path))


@contextmanager
def _connect(cfg: dict | None = None):
    """
    Short-lived connection per call rather than one long-lived shared
    connection — a bit more overhead per call, but it's the simplest way
    to be safe against concurrent access from multiple processes (the
    user's own script writing while a separate `tokenshark dashboard`
    process reads), especially combined with WAL mode below.
    """
    db_path = _resolve_db_path(cfg)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=5.0)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(cfg: dict | None = None) -> None:
    """
    Idempotent — safe to call every time monitor() runs, every process
    start. Raises TokenSharkDatabaseError on failure (e.g. storage.path
    points somewhere unwritable) because this only ever runs at setup
    time, before any real LLM call has been made — unlike log_call()
    below, there's nothing yet to protect by swallowing the error here.
    """
    global _db_initialized
    try:
        with _connect(cfg) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")  # lets `tokenshark dashboard` tail this live
            conn.executescript(_SCHEMA_SQL)
    except (sqlite3.Error, OSError) as e:
        raise TokenSharkDatabaseError(
            f"Could not initialize TokenShark's database: {e}. "
            f"Check that the storage.path directory is writable."
        ) from e
    _db_initialized = True


def _today_spend(conn: sqlite3.Connection) -> float:
    row = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0.0) FROM calls WHERE date(created_at) = date('now')"
    ).fetchone()
    return row[0] if row else 0.0


# ---------------------------------------------------------------------------
# Public write path
# ---------------------------------------------------------------------------

def log_call(
    *,
    model: str = "unknown",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cache_create_tokens: int = 0,
    cache_read_tokens: int = 0,
    latency_ms: int = 0,
    tags: dict | None = None,
    provider: str = "unknown",
    error: str | None = None,
    estimated: bool = False,
) -> float:
    """
    The single write path every provider patch in proxy.py calls into, on
    both success and failure. Always returns the cost in USD (0.0 on
    error or an unknown model).

    Persistence failures here are caught and warned on, never raised — by
    the time this function runs, the user's *actual* LLM call has already
    succeeded or failed on its own terms. A broken logs.db must never
    turn a successful API call into a crashed script.
    """
    cost = (
        0.0
        if error
        else calc_cost(model, prompt_tokens, completion_tokens, cache_create_tokens, cache_read_tokens)
    )
    tags = tags or {}
    today_spend = None

    try:
        if not _db_initialized:
            init_db()
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO calls (
                    provider, model, prompt_tokens, completion_tokens,
                    cache_create_tokens, cache_read_tokens, cost_usd,
                    latency_ms, tags, error, estimated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    provider, model, prompt_tokens, completion_tokens,
                    cache_create_tokens, cache_read_tokens, cost,
                    latency_ms, json.dumps(tags), error, int(estimated),
                ),
            )
            if error is None:
                today_spend = _today_spend(conn)
    except (TokenSharkDatabaseError, sqlite3.Error, OSError) as e:
        warnings.warn(
            f"TokenShark: could not persist this call to logs.db ({e}). "
            f"Your actual API call was unaffected — only local cost tracking "
            f"for this one call was lost."
        )

    _print_call_summary(model, provider, prompt_tokens, completion_tokens, cost, latency_ms, error, estimated)

    # alerts.py isn't part of this handoff batch (see ARCHITECTURE.md file
    # structure) — guarded import so log_call() works today with no
    # alerts.py on disk, and starts alerting automatically the moment you
    # add that file tomorrow, with no changes needed here.
    if error is None and today_spend is not None:
        try:
            from .alerts import check_and_alert
        except ImportError:
            check_and_alert = None
        if check_and_alert is not None:
            cfg = config.load_config()
            try:
                check_and_alert(cfg, current_spend=today_spend, model=model)
            except TokenSharkBudgetExceeded:
                raise  # deliberate, user-configured hard stop -- must propagate
            except Exception as e:
                warnings.warn(f"TokenShark: alert check failed ({e}) — continuing without alerting.")

    return cost


def _print_call_summary(model, provider, prompt_tokens, completion_tokens, cost, latency_ms, error, estimated) -> None:
    """
    Minimal always-on terminal line, so monitor() alone satisfies "shows
    real-time cost in your terminal" without depending on dashboard.py
    (which owns the full Rich Live table from ARCHITECTURE.md #4, and
    isn't part of this batch). Deliberately plain print + no `rich`
    dependency here, so proxy.py/tracker.py work standalone today, before
    dashboard.py exists.
    """
    if error:
        print(f"[TokenShark] {provider}/{model} FAILED ({latency_ms}ms): {error}")
        return
    est_flag = " (est.)" if estimated else ""
    print(
        f"[TokenShark] {provider}/{model} | "
        f"{prompt_tokens:,}\u2191 {completion_tokens:,}\u2193{est_flag} | "
        f"${cost:.4f} | {latency_ms}ms"
    )
