"""
exceptions.py — TokenShark's exception hierarchy.

Everything TokenShark raises on purpose inherits from TokenSharkError, so
calling code can do `except tokenshark.TokenSharkError` to catch anything
from this library specifically without swallowing unrelated errors from
the user's own code or the provider SDKs.
"""


class TokenSharkError(Exception):
    """Base class for every exception TokenShark raises on purpose."""


class TokenSharkConfigError(TokenSharkError):
    """Invalid or unsafe tokenshark.yaml content — e.g. an API key or
    Slack webhook accidentally stored in the config file instead of an
    environment variable, or a YAML file that fails to parse."""


class TokenSharkBudgetExceeded(TokenSharkError):
    """Raised when alerts.hard_stop is true and a configured spending
    budget (daily or per-session) has been exceeded. This one is meant
    to propagate — it's a deliberate, user-configured stop, not a bug.
    (Raised by alerts.py, which isn't part of this handoff batch — see
    the flag in tracker.py's log_call() for how this is wired in ahead
    of that file existing.)"""


class TokenSharkPatchError(TokenSharkError):
    """Monkey-patching a provider SDK failed in an unexpected way — most
    likely the SDK's internal class/attribute layout changed since this
    version of TokenShark was released. This is the concrete failure mode
    behind the "uncle review" question in ARCHITECTURE.md about the risk
    of a provider updating their SDK out from under a monkey-patch."""


class TokenSharkDatabaseError(TokenSharkError):
    """Unrecoverable SQLite setup error (e.g. storage.path isn't
    writable), raised at init_db()/monitor() time so failures surface
    immediately, before any real LLM call has happened. Routine per-call
    write failures are caught and warned on instead, never raised — see
    tracker.log_call(). A logging problem must never take down the
    user's actual, already-completed LLM call."""
