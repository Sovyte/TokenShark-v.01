nano pyproject.toml            # version = "0.2.0"
"""
TokenShark
==========

Real-time LLM token cost tracking with one import.

    import tokenshark
    tokenshark.monitor()
    # OpenAI and Anthropic calls are now logged with cost, latency, and
    # (optionally) your own metadata tags -- to ~/.tokenshark/logs.db by
    # default, or wherever tokenshark.yaml's storage.path points.
    #
    # The first successful monitor() call in a process also prints a
    # small startup banner (shark tail included) summarizing which
    # providers got patched and where logs are going -- see
    # _print_startup_banner below. It only ever prints once per
    # process, same as the plain-text line it replaced. Color is
    # auto-detected (real terminal vs. piped/redirected output) and
    # respects NO_COLOR / FORCE_COLOR -- see _supports_color.

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

import os
import sys
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

__version__ = "0.2.0"

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

# Plain ASCII on purpose (no box-drawing/unicode) so the shape itself
# still renders cleanly in any terminal, notebook, or CI log even when
# color is off -- color is layered on top of this, never required to
# make sense of it.
_SHARK_TAIL = r"""
            ___
           /   |
      ____/    |
      \        |
       \_______|
           \  \
            \__\
"""


def _supports_color() -> bool:
    """
    Checked fresh on every banner print (not cached at import time), so
    e.g. a test harness that redirects sys.stdout still gets a correct
    answer. Order matters: NO_COLOR wins unconditionally per
    https://no-color.org ("regardless of its value"), then FORCE_COLOR
    as the opt-in override for piped/CI contexts that do support color,
    then a plain isatty() check as the default -- the same fallback
    chain git, pytest, and most modern CLIs already use.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if os.environ.get("TERM") == "dumb":
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


class _Ansi:
    """Resolves to real escape codes or empty strings depending on
    _supports_color(), so the banner's f-strings never need their own
    if/else -- they just interpolate an _Ansi instance's attributes."""

    def __init__(self, enabled: bool):
        self.reset = "\033[0m" if enabled else ""
        self.cyan = "\033[96m" if enabled else ""
        self.bold_cyan = "\033[1;96m" if enabled else ""
        self.green = "\033[92m" if enabled else ""
        self.yellow = "\033[93m" if enabled else ""
        self.red = "\033[91m" if enabled else ""
        self.dim = "\033[2m" if enabled else ""


def _print_startup_banner(cfg: dict, active_providers: list[str]) -> None:
    """
    Printed once per process, right after the first successful
    monitor() call -- guarded by the same `if _monitoring_active: return`
    check that already made the plain-text line it replaced idempotent,
    so re-running a notebook cell still only shows this once.
    """
    a = _Ansi(_supports_color())
    db_path = cfg.get("storage", {}).get("path", "~/.tokenshark/logs.db")
    providers_label = ", ".join(active_providers) if active_providers else "none patched"

    print(f"{a.cyan}{_SHARK_TAIL}{a.reset}")
    print(f"{a.bold_cyan}TokenShark v{__version__}{a.reset} is circling your LLM calls")
    print(f"{a.dim}  providers   {a.reset}{a.green}{providers_label}{a.reset}")
    print(f"{a.dim}  logging to  {a.reset}{db_path}")


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
    patched_ok = []  # only providers that actually patched successfully -- shown in the banner below
    for name in active_providers:
        patch_fn = _PATCHERS.get(name)
        if patch_fn is None:
            warnings.warn(f"TokenShark: no patcher available yet for provider '{name}' -- skipping.")
            continue
        try:
            patch_fn()
            patched_any = True
            patched_ok.append(name)
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
    _print_startup_banner(cfg, patched_ok)
