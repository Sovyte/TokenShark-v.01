"""
TokenShark
==========

Real-time LLM token cost tracking with one import.

    import tokenshark
    tokenshark.monitor()
    # OpenAI and Anthropic calls are now logged with cost, latency, and
    # (optionally) your own metadata tags -- to ~/.tokenshark/logs.db by
    # default, or wherever tokenshark.yaml's storage.path points.

Nothing here requires a tokenshark.yaml to exist; see
tokenshark.yaml.example for every optional setting and its default.

DECISION GAP: ARCHITECTURE.md's one-line pitch says TokenShark "patches
your Python LLM client with one import," but its own worked example is
two steps -- `import tokenshark` then `tokenshark.monitor()`. This file
follows the two-step version (the more detailed, later decision in the
doc), which also happens to be the safer choice: patching global SDK
behavior as a side effect of a bare `import` is generally considered
surprising/risky (a linter, test runner, or IDE that imports the module
for introspection would silently trigger patching too) -- this is why
sentry-sdk, ddtrace, and similar instrumentation libraries are designed
the same way. Flagging the inconsistency between your chat message and
your own architecture doc in case you intended literal auto-patch-on-import.
"""

import warnings

from . import config, proxy
from .exceptions import (
    TokenSharkBudgetExceeded,
    TokenSharkConfigError,
    TokenSharkDatabaseError,
    TokenSharkError,
    TokenSharkPatchError,
)
from .tracker import init_db

__version__ = "0.1.0"

__all__ = [
    "monitor",
    "TokenSharkError",
    "TokenSharkConfigError",
    "TokenSharkBudgetExceeded",
    "TokenSharkPatchError",
    "TokenSharkDatabaseError",
]

# "add more as supported" per ARCHITECTURE.md #1 -- deepseek/mistral/qwen
# already have entries in tracker.COST_PER_1M but no patcher yet, since
# the architecture doc only worked through monkey-patch examples for
# openai/anthropic. monitor() warns and skips instead of crashing if one
# of the others shows up in providers: before it has a real patcher.
_PATCHERS = {
    "openai": proxy.patch_openai,
    "anthropic": proxy.patch_anthropic,
}

_monitoring_active = False


def monitor(tags: dict | None = None, providers: list[str] | None = None) -> None:
    """
    Patch every configured, installed provider SDK and start logging
    every call to SQLite.

    Safe to call more than once in the same process (e.g. re-running a
    notebook cell) -- the SDK patch itself is only ever applied once;
    later calls just update the active tags.

    Args:
        tags: metadata attached to every call for the rest of the
              process, e.g. monitor(tags={"feature": "summarizer"}).
              Overrides tokenshark.yaml's `tags:` section for this run
              when given. See ARCHITECTURE.md #7.
        providers: override tokenshark.yaml's `providers:` list for this
                   call, e.g. monitor(providers=["openai"]) to skip
                   anthropic even if both SDKs are installed. Only takes
                   effect on the *first* monitor() call in a process --
                   see the note above about repeat calls.
    """
    global _monitoring_active

    if tags is not None:
        config.set_runtime_tags(tags)

    cfg = config.load_config()
    init_db(cfg)

    if _monitoring_active:
        return

    active_providers = providers if providers is not None else cfg.get("providers", ["openai", "anthropic"])

    patched_any = False
    for name in active_providers:
        patch_fn = _PATCHERS.get(name)
        if patch_fn is None:
            warnings.warn(f"TokenShark: no patcher available yet for provider '{name}' -- skipping.")
            continue
        try:
            patch_fn()
            patched_any = True
        except ImportError:
            continue  # SDK not installed -- zero friction means skip, not crash
        except TokenSharkPatchError as e:
            warnings.warn(f"TokenShark: failed to patch '{name}': {e}")

    if not patched_any:
        warnings.warn(
            "TokenShark: no supported LLM SDKs were found to patch. "
            "Install openai and/or anthropic, or check tokenshark.yaml's providers: list."
        )

    _monitoring_active = True
    print("TokenShark is watching your LLM calls (see ~/.tokenshark/logs.db)")
