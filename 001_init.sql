-- TokenShark schema -- migration 001 (initial)
--
-- This is applied automatically by tracker.init_db() every time
-- monitor() runs -- you do not need to run this file by hand. It's kept
-- here as a human-readable reference copy (e.g. for opening logs.db with
-- the sqlite3 CLI or a GUI browser) and MUST be kept in sync with the
-- _SCHEMA_SQL string embedded directly in tracker.py. See the comment
-- above _SCHEMA_SQL in that file for why the schema is duplicated as a
-- Python string rather than loaded from this file at runtime.
--
-- Two things added here beyond ARCHITECTURE.md's original CREATE TABLE
-- (flagging both, since this changes your item-3 deliverable):
--
--   1. `estimated` column -- the architecture doc's "Streaming Handling"
--      decision says token counts should be logged "as estimated, not
--      exact" when a stream doesn't return usage data, but the original
--      schema had no column to record that flag on. Added so the two
--      decisions agree with each other.
--
--   2. PRAGMA journal_mode = WAL -- lets a separate `tokenshark dashboard`
--      process read logs.db live while the user's own script is still
--      writing to it, without "database is locked" errors. Only needs to
--      be set once per database file (it persists in the file itself),
--      but is harmless to re-run.

PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS calls (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    provider            TEXT NOT NULL,           -- 'openai', 'anthropic', 'ollama'
    model               TEXT NOT NULL,
    prompt_tokens       INTEGER DEFAULT 0,
    completion_tokens   INTEGER DEFAULT 0,
    cache_create_tokens INTEGER DEFAULT 0,        -- Anthropic cache-write tokens
    cache_read_tokens   INTEGER DEFAULT 0,        -- Anthropic cache-read + OpenAI cached_tokens
    cost_usd            REAL DEFAULT 0.0,         -- calculated at log time from COST_PER_1M
    latency_ms          INTEGER DEFAULT 0,
    tags                TEXT DEFAULT '{}',        -- JSON-encoded metadata tags
    error               TEXT DEFAULT NULL,        -- NULL if the call succeeded
    estimated           INTEGER DEFAULT 0,        -- 1 if tokens are a tiktoken/heuristic estimate, not exact
    created_at          DATETIME DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_created_at ON calls(created_at);
CREATE INDEX IF NOT EXISTS idx_model ON calls(model);
CREATE INDEX IF NOT EXISTS idx_provider ON calls(provider);
