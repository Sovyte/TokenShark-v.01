"""
config.py — loads, merges, and validates tokenshark.yaml.

Load order (per ARCHITECTURE.md #6):
  1. ./tokenshark.yaml            (project-level, checked first)
  2. ~/.tokenshark/config.yaml    (global user-level defaults)
  3. _DEFAULT_CONFIG (below)      (built-in — nothing breaks with no file)

DECISION GAP: only whichever of (1)/(2) is found FIRST gets read — they
are not layered on top of each other, only on top of the built-in
defaults. The architecture doc doesn't say what should happen if both
exist. "Project wins, global is ignored entirely" was the simplest
reading and is what's implemented below; swap to global-then-project-
overrides in load_config() if you'd rather layer them.
"""

import copy
import os
import warnings
from pathlib import Path

import yaml

from .exceptions import TokenSharkConfigError

# Substrings that should never appear in a config *key* name, because a
# key matching one of these almost certainly holds a secret that belongs
# in an environment variable instead. Kept exactly as specified in
# ARCHITECTURE.md's security review section.
#
# DECISION GAP (flagged, not applied): 'webhook' isn't in this list, so
# `alerts.slack_webhook: "https://hooks.slack.com/..."` pasted directly
# into tokenshark.yaml would NOT be caught, even though a leaked webhook
# URL lets anyone post to your Slack the same as a leaked token would.
# Left exactly as your architecture doc specified — add 'webhook' below
# if you want that caught too.
DANGEROUS_KEYS = ["api_key", "secret", "token", "password", "openai_key", "anthropic_key"]

_DEFAULT_CONFIG = {
    "version": "1",
    "providers": ["openai", "anthropic"],
    "alerts": {
        "slack_webhook": "",
        "daily_budget": 5.00,
        "per_session_budget": 1.00,
        "hard_stop": False,
    },
    "dashboard": {
        "refresh_seconds": 2,
        "max_rows": 20,
        "sort_by": "cost",
    },
    "storage": {
        "path": "~/.tokenshark/logs.db",
    },
    "tags": {},
}

_cached_config: dict | None = None
_runtime_tags: dict = {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge `override` onto `base` key-by-key, recursing into nested
    dicts instead of replacing them wholesale — so setting only
    alerts.daily_budget in tokenshark.yaml doesn't blow away the other
    alerts defaults. Never mutates either input (returns a new dict at
    every level it touches)."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _validate_config(cfg: dict, _path: str = "") -> None:
    """
    Recursively check every key at every nesting level against
    DANGEROUS_KEYS.

    DECISION GAP: extended from ARCHITECTURE.md's example, which only
    checked cfg.keys() at the top level and would miss e.g.
    `alerts: {slack_api_key: '...'}` nested one level down — exactly the
    kind of accidental commit the security review is meant to catch.
    """
    for key, value in cfg.items():
        full_path = f"{_path}.{key}" if _path else key
        if any(dangerous in key.lower() for dangerous in DANGEROUS_KEYS):
            raise TokenSharkConfigError(
                f"API keys must not be stored in tokenshark.yaml. "
                f"Use environment variables instead. "
                f"Found suspicious key: '{full_path}'"
            )
        if isinstance(value, dict):
            _validate_config(value, full_path)


def _find_config_file() -> Path | None:
    project_config = Path("tokenshark.yaml")
    if project_config.is_file():
        return project_config
    global_config = Path(os.path.expanduser("~/.tokenshark/config.yaml"))
    if global_config.is_file():
        return global_config
    return None


def load_config(force_reload: bool = False) -> dict:
    """
    Cached after the first call — proxy.py's patched create() calls this
    on every single LLM request, so re-reading and re-parsing YAML from
    disk on every call would be wasteful. Pass force_reload=True to pick
    up edits made mid-process (e.g. in a long-running notebook session).
    """
    global _cached_config
    if _cached_config is not None and not force_reload:
        return _cached_config

    # Always start from an independent deep copy, never `dict(_DEFAULT_CONFIG)`.
    # A shallow copy only copies the top level and leaves nested dicts
    # (alerts, dashboard, storage) as the SAME objects as the module-level
    # constant. The env-var override below mutates merged["alerts"] in
    # place — with a shallow copy that would silently corrupt
    # _DEFAULT_CONFIG for the rest of the process the first time this
    # function runs without a tokenshark.yaml touching `alerts` at all.
    merged = copy.deepcopy(_DEFAULT_CONFIG)

    config_path = _find_config_file()
    if config_path is not None:
        try:
            with open(config_path, "r") as f:
                user_config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise TokenSharkConfigError(f"Could not parse {config_path}: {e}")
        if not isinstance(user_config, dict):
            raise TokenSharkConfigError(
                f"{config_path} must contain a YAML mapping at the top level, "
                f"got {type(user_config).__name__}."
            )
        _validate_config(user_config)
        merged = _deep_merge(merged, user_config)

    # Slack webhook: env var always wins over config, matching the
    # "never put secrets in the YAML" stance. If someone puts a real
    # webhook in tokenshark.yaml anyway, warn rather than silently accept
    # it — _validate_config() won't catch this on its own (see the
    # DANGEROUS_KEYS note above).
    env_webhook = os.environ.get("TOKENSHARK_SLACK_WEBHOOK")
    if env_webhook:
        merged["alerts"]["slack_webhook"] = env_webhook
    elif merged["alerts"].get("slack_webhook"):
        warnings.warn(
            "TokenShark: slack_webhook is set directly in tokenshark.yaml. "
            "Move it to the TOKENSHARK_SLACK_WEBHOOK environment variable instead — "
            "committing a webhook URL to source control lets anyone post to your Slack."
        )

    _cached_config = merged
    return merged


def set_runtime_tags(tags: dict) -> None:
    """Called by monitor(tags=...) in __init__.py."""
    global _runtime_tags
    _runtime_tags = dict(tags or {})


def get_active_tags() -> dict:
    """
    Tags passed to monitor(tags=...) win for the whole process; otherwise
    fall back to the `tags:` section of tokenshark.yaml.

    DECISION GAP this function resolves: ARCHITECTURE.md has two different
    tag sources that were never reconciled with each other — section 7
    sets tags via a monitor(tags=...) call argument, but the proxy.py
    example in section 1 reads them straight off the loaded config dict
    (`config.get('tags', {})`). This function is the one place both
    __init__.py (which knows about the monitor() argument) and proxy.py
    (which needs the tags at call time) can agree on, without proxy.py
    importing from __init__.py — which would create a circular import
    (__init__.py already imports proxy.py to expose patch_openai/
    patch_anthropic through monitor()).
    """
    if _runtime_tags:
        return dict(_runtime_tags)
    return dict(load_config().get("tags") or {})
