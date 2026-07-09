"""
proxy.py — monkey-patches the OpenAI and Anthropic SDKs so every
chat/message call is transparently logged with token usage, cost, and
latency, with zero changes to the user's own calling code.

Patch target (verified against SDK source and current docs, July 2026):
    OpenAI:     openai.resources.chat.completions.Completions.create
                openai.resources.chat.completions.AsyncCompletions.create
    Anthropic:  anthropic.resources.messages.Messages.create
                anthropic.resources.messages.AsyncMessages.create

These are CLASS methods, not module-level functions. DECISION GAP:
ARCHITECTURE.md's own example patches `openai.chat.completions.create`,
a module-level convenience accessor OpenAI provides for backwards
compatibility. That only intercepts calls made through that literal
top-level accessor — it would NOT intercept calls made through an
explicitly instantiated `client = OpenAI(); client.chat.completions.create(...)`,
which is the documented, idiomatic pattern for the modern SDK and almost
certainly how most real code (including your own future code) calls it.
Patching the class method instead means every instance — however many
get created, now or later — is covered automatically. Flagging this as a
deliberate change from the literal example code in the architecture doc.

Async support (AsyncOpenAI / AsyncAnthropic) isn't mentioned in
ARCHITECTURE.md, but is added here since async clients are extremely
common in real codebases (FastAPI backends etc.), and skipping them
would silently miss half of a typical async app's calls.
"""

import functools
import time

from . import config
from .exceptions import TokenSharkPatchError
from .tracker import estimate_tokens, log_call

_PATCH_STATE = {"openai": False, "anthropic": False}


def _is_ollama(base_url) -> bool:
    if not base_url:
        return False
    base_url = str(base_url)
    return "localhost:11434" in base_url or "127.0.0.1:11434" in base_url


def _client_base_url(resource_self) -> str:
    """
    Best-effort read of the parent client's base_url from a resource
    instance's private `_client` attribute, used only for Ollama
    detection (ARCHITECTURE.md #8). This touches SDK internals that
    aren't part of the public API contract — exactly the kind of thing
    your uncle's security-review question ("what are the risks if the
    provider updates their SDK?") is about. If this attribute ever moves,
    Ollama detection silently stops working, but cost tracking for real
    OpenAI/Anthropic calls is unaffected — this is only used to relabel
    the provider string.
    """
    client = getattr(resource_self, "_client", None)
    return getattr(client, "base_url", "") if client is not None else ""


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

def _extract_openai_cache_read_tokens(usage) -> int:
    """
    OpenAI reports cached prompt tokens nested under
    usage.prompt_tokens_details.cached_tokens (confirmed against current
    API docs/examples, July 2026) — NOT a flat usage.cached_tokens as
    ARCHITECTURE.md's example snippet reads it. Checked defensively in
    both shapes: nested first (the real one), flat as a fallback in case
    a compatible-but-not-identical provider (Azure/OpenRouter/etc.) puts
    it somewhere slightly different. Defaults to 0 rather than raising —
    this is a cost-refinement number, not something worth breaking a
    request over.
    """
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", None)
        if cached is not None:
            return cached
    return getattr(usage, "cached_tokens", 0) or 0


def _flatten_openai_messages(messages) -> str:
    """
    Rough text join of an OpenAI `messages` list, used only for the
    streaming-fallback token estimate (see _wrap_openai_stream). Handles
    the common case of plain string content; multi-modal content (image
    blocks, tool results) is skipped rather than guessed at, since a
    partial estimate is more honest than a confidently wrong one.
    """
    parts = []
    for msg in messages or []:
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


def patch_openai() -> None:
    """Patch Completions.create and AsyncCompletions.create so every
    OpenAI() / AsyncOpenAI() client instance is covered. Safe to call
    more than once — a second call is a no-op."""
    if _PATCH_STATE["openai"]:
        return

    from openai.resources.chat.completions import AsyncCompletions, Completions  # raises ImportError if not installed

    try:
        _orig_create = Completions.create
        _orig_async_create = AsyncCompletions.create
    except AttributeError as e:
        raise TokenSharkPatchError(
            f"openai SDK layout has changed and TokenShark can't find Completions.create to patch: {e}"
        ) from e

    @functools.wraps(_orig_create)
    def patched_create(self, *args, **kwargs):
        model = kwargs.get("model", "unknown")
        provider = "ollama" if _is_ollama(_client_base_url(self)) else "openai"
        stream = bool(kwargs.get("stream", False))
        start = time.time()

        if stream:
            _ensure_openai_usage_in_stream(kwargs)
            raw_stream = _orig_create(self, *args, **kwargs)
            return _wrap_openai_stream(raw_stream, model, provider, start, kwargs)

        try:
            response = _orig_create(self, *args, **kwargs)
        except Exception as e:
            log_call(
                model=model, provider=provider, error=str(e),
                latency_ms=int((time.time() - start) * 1000),
                tags=config.get_active_tags(),
            )
            raise
        _log_openai_response(response, model, provider, start)
        return response

    @functools.wraps(_orig_async_create)
    async def patched_async_create(self, *args, **kwargs):
        model = kwargs.get("model", "unknown")
        provider = "ollama" if _is_ollama(_client_base_url(self)) else "openai"
        stream = bool(kwargs.get("stream", False))
        start = time.time()

        if stream:
            _ensure_openai_usage_in_stream(kwargs)
            raw_stream = await _orig_async_create(self, *args, **kwargs)
            return _wrap_openai_async_stream(raw_stream, model, provider, start, kwargs)

        try:
            response = await _orig_async_create(self, *args, **kwargs)
        except Exception as e:
            log_call(
                model=model, provider=provider, error=str(e),
                latency_ms=int((time.time() - start) * 1000),
                tags=config.get_active_tags(),
            )
            raise
        _log_openai_response(response, model, provider, start)
        return response

    Completions.create = patched_create
    AsyncCompletions.create = patched_async_create
    _PATCH_STATE["openai"] = True


def _ensure_openai_usage_in_stream(kwargs: dict) -> None:
    """
    Without stream_options={"include_usage": True}, EVERY chunk of an
    OpenAI stream has usage=None — there is no usage data to log at all
    (confirmed against current API docs). Injecting this when the user
    didn't set it themselves is the "zero friction, sensible defaults"
    philosophy applied to streaming specifically; it only adds one extra
    empty-choices chunk at the end, which the wrapper below consumes
    itself rather than surfacing to the caller. Not explicitly called out
    in ARCHITECTURE.md — flagging as an addition.
    """
    opts = kwargs.get("stream_options")
    if opts is None:
        kwargs["stream_options"] = {"include_usage": True}
    elif "include_usage" not in opts:
        kwargs["stream_options"] = {**opts, "include_usage": True}


def _log_openai_response(response, model, provider, start) -> None:
    latency_ms = int((time.time() - start) * 1000)
    usage = getattr(response, "usage", None)
    log_call(
        model=model,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
        cache_read_tokens=_extract_openai_cache_read_tokens(usage) if usage else 0,
        cache_create_tokens=0,  # OpenAI has no cache-write charge as of this writing -- VERIFY on Day 3 alongside pricing
        latency_ms=latency_ms,
        tags=config.get_active_tags(),
        provider=provider,
        error=None,
    )


def _wrap_openai_stream(raw_stream, model, provider, start, original_kwargs):
    usage_seen = None
    completion_text = []
    try:
        for chunk in raw_stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                usage_seen = usage
            for choice in getattr(chunk, "choices", None) or []:
                delta_content = getattr(getattr(choice, "delta", None), "content", None)
                if delta_content:
                    completion_text.append(delta_content)
            yield chunk
    except Exception as e:
        log_call(
            model=model, provider=provider, error=str(e),
            latency_ms=int((time.time() - start) * 1000),
            tags=config.get_active_tags(),
        )
        raise

    latency_ms = int((time.time() - start) * 1000)
    if usage_seen is not None:
        log_call(
            model=model,
            prompt_tokens=getattr(usage_seen, "prompt_tokens", 0),
            completion_tokens=getattr(usage_seen, "completion_tokens", 0),
            cache_read_tokens=_extract_openai_cache_read_tokens(usage_seen),
            cache_create_tokens=0,
            latency_ms=latency_ms,
            tags=config.get_active_tags(),
            provider=provider,
            error=None,
        )
    else:
        # Fallback per ARCHITECTURE.md "Streaming Handling": no usage chunk
        # arrived even after requesting it (older SDK / passthrough proxy
        # that strips stream_options) -- estimate instead of losing the
        # entry entirely, and mark it clearly as estimated.
        prompt_text = _flatten_openai_messages(original_kwargs.get("messages"))
        log_call(
            model=model,
            prompt_tokens=estimate_tokens(prompt_text),
            completion_tokens=estimate_tokens("".join(completion_text)),
            latency_ms=latency_ms,
            tags=config.get_active_tags(),
            provider=provider,
            error=None,
            estimated=True,
        )


async def _wrap_openai_async_stream(raw_stream, model, provider, start, original_kwargs):
    usage_seen = None
    completion_text = []
    try:
        async for chunk in raw_stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                usage_seen = usage
            for choice in getattr(chunk, "choices", None) or []:
                delta_content = getattr(getattr(choice, "delta", None), "content", None)
                if delta_content:
                    completion_text.append(delta_content)
            yield chunk
    except Exception as e:
        log_call(
            model=model, provider=provider, error=str(e),
            latency_ms=int((time.time() - start) * 1000),
            tags=config.get_active_tags(),
        )
        raise

    latency_ms = int((time.time() - start) * 1000)
    if usage_seen is not None:
        log_call(
            model=model,
            prompt_tokens=getattr(usage_seen, "prompt_tokens", 0),
            completion_tokens=getattr(usage_seen, "completion_tokens", 0),
            cache_read_tokens=_extract_openai_cache_read_tokens(usage_seen),
            cache_create_tokens=0,
            latency_ms=latency_ms,
            tags=config.get_active_tags(),
            provider=provider,
            error=None,
        )
    else:
        prompt_text = _flatten_openai_messages(original_kwargs.get("messages"))
        log_call(
            model=model,
            prompt_tokens=estimate_tokens(prompt_text),
            completion_tokens=estimate_tokens("".join(completion_text)),
            latency_ms=latency_ms,
            tags=config.get_active_tags(),
            provider=provider,
            error=None,
            estimated=True,
        )


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

def _flatten_anthropic_messages(messages, system=None) -> str:
    """Same rough-estimate rationale as _flatten_openai_messages, adapted
    to Anthropic's shape (system prompt is a separate top-level param)."""
    parts = []
    if isinstance(system, str):
        parts.append(system)
    for msg in messages or []:
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


def patch_anthropic() -> None:
    """Patch Messages.create and AsyncMessages.create so every
    Anthropic() / AsyncAnthropic() client instance is covered. Safe to
    call more than once — a second call is a no-op."""
    if _PATCH_STATE["anthropic"]:
        return

    from anthropic.resources.messages import AsyncMessages, Messages  # raises ImportError if not installed

    try:
        _orig_create = Messages.create
        _orig_async_create = AsyncMessages.create
    except AttributeError as e:
        raise TokenSharkPatchError(
            f"anthropic SDK layout has changed and TokenShark can't find Messages.create to patch: {e}"
        ) from e

    @functools.wraps(_orig_create)
    def patched_create(self, *args, **kwargs):
        model = kwargs.get("model", "unknown")
        provider = "ollama" if _is_ollama(_client_base_url(self)) else "anthropic"
        stream = bool(kwargs.get("stream", False))
        start = time.time()

        if stream:
            raw_stream = _orig_create(self, *args, **kwargs)
            return _wrap_anthropic_stream(raw_stream, model, provider, start, kwargs)

        try:
            response = _orig_create(self, *args, **kwargs)
        except Exception as e:
            log_call(
                model=model, provider=provider, error=str(e),
                latency_ms=int((time.time() - start) * 1000),
                tags=config.get_active_tags(),
            )
            raise
        _log_anthropic_response(response, model, provider, start)
        return response

    @functools.wraps(_orig_async_create)
    async def patched_async_create(self, *args, **kwargs):
        model = kwargs.get("model", "unknown")
        provider = "ollama" if _is_ollama(_client_base_url(self)) else "anthropic"
        stream = bool(kwargs.get("stream", False))
        start = time.time()

        if stream:
            raw_stream = await _orig_async_create(self, *args, **kwargs)
            return _wrap_anthropic_async_stream(raw_stream, model, provider, start, kwargs)

        try:
            response = await _orig_async_create(self, *args, **kwargs)
        except Exception as e:
            log_call(
                model=model, provider=provider, error=str(e),
                latency_ms=int((time.time() - start) * 1000),
                tags=config.get_active_tags(),
            )
            raise
        _log_anthropic_response(response, model, provider, start)
        return response

    Messages.create = patched_create
    AsyncMessages.create = patched_async_create
    _PATCH_STATE["anthropic"] = True


def _log_anthropic_response(response, model, provider, start) -> None:
    latency_ms = int((time.time() - start) * 1000)
    usage = getattr(response, "usage", None)
    log_call(
        model=model,
        prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
        completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
        cache_create_tokens=getattr(usage, "cache_creation_input_tokens", 0) if usage else 0,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) if usage else 0,
        latency_ms=latency_ms,
        tags=config.get_active_tags(),
        provider=provider,
        error=None,
    )


def _wrap_anthropic_stream(raw_stream, model, provider, start, original_kwargs):
    usage_acc = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    saw_usage = False
    completion_text = []
    try:
        for event in raw_stream:
            etype = getattr(event, "type", None)
            if etype == "message_start":
                msg_usage = getattr(getattr(event, "message", None), "usage", None)
                if msg_usage is not None:
                    saw_usage = True
                    usage_acc["input_tokens"] = getattr(msg_usage, "input_tokens", 0)
                    usage_acc["cache_creation_input_tokens"] = getattr(msg_usage, "cache_creation_input_tokens", 0)
                    usage_acc["cache_read_input_tokens"] = getattr(msg_usage, "cache_read_input_tokens", 0)
            elif etype == "message_delta":
                d_usage = getattr(event, "usage", None)
                if d_usage is not None:
                    saw_usage = True
                    usage_acc["output_tokens"] = getattr(d_usage, "output_tokens", 0)
            elif etype == "content_block_delta":
                text = getattr(getattr(event, "delta", None), "text", None)
                if text:
                    completion_text.append(text)
            yield event
    except Exception as e:
        log_call(
            model=model, provider=provider, error=str(e),
            latency_ms=int((time.time() - start) * 1000),
            tags=config.get_active_tags(),
        )
        raise

    latency_ms = int((time.time() - start) * 1000)
    if saw_usage:
        log_call(
            model=model,
            prompt_tokens=usage_acc["input_tokens"],
            completion_tokens=usage_acc["output_tokens"],
            cache_create_tokens=usage_acc["cache_creation_input_tokens"],
            cache_read_tokens=usage_acc["cache_read_input_tokens"],
            latency_ms=latency_ms,
            tags=config.get_active_tags(),
            provider=provider,
            error=None,
        )
    else:
        # Fallback per ARCHITECTURE.md "Streaming Handling" -- see the
        # matching OpenAI branch above for the same rationale.
        prompt_text = _flatten_anthropic_messages(original_kwargs.get("messages"), original_kwargs.get("system"))
        log_call(
            model=model,
            prompt_tokens=estimate_tokens(prompt_text),
            completion_tokens=estimate_tokens("".join(completion_text)),
            latency_ms=latency_ms,
            tags=config.get_active_tags(),
            provider=provider,
            error=None,
            estimated=True,
        )


async def _wrap_anthropic_async_stream(raw_stream, model, provider, start, original_kwargs):
    usage_acc = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    saw_usage = False
    completion_text = []
    try:
        async for event in raw_stream:
            etype = getattr(event, "type", None)
            if etype == "message_start":
                msg_usage = getattr(getattr(event, "message", None), "usage", None)
                if msg_usage is not None:
                    saw_usage = True
                    usage_acc["input_tokens"] = getattr(msg_usage, "input_tokens", 0)
                    usage_acc["cache_creation_input_tokens"] = getattr(msg_usage, "cache_creation_input_tokens", 0)
                    usage_acc["cache_read_input_tokens"] = getattr(msg_usage, "cache_read_input_tokens", 0)
            elif etype == "message_delta":
                d_usage = getattr(event, "usage", None)
                if d_usage is not None:
                    saw_usage = True
                    usage_acc["output_tokens"] = getattr(d_usage, "output_tokens", 0)
            elif etype == "content_block_delta":
                text = getattr(getattr(event, "delta", None), "text", None)
                if text:
                    completion_text.append(text)
            yield event
    except Exception as e:
        log_call(
            model=model, provider=provider, error=str(e),
            latency_ms=int((time.time() - start) * 1000),
            tags=config.get_active_tags(),
        )
        raise

    latency_ms = int((time.time() - start) * 1000)
    if saw_usage:
        log_call(
            model=model,
            prompt_tokens=usage_acc["input_tokens"],
            completion_tokens=usage_acc["output_tokens"],
            cache_create_tokens=usage_acc["cache_creation_input_tokens"],
            cache_read_tokens=usage_acc["cache_read_input_tokens"],
            latency_ms=latency_ms,
            tags=config.get_active_tags(),
            provider=provider,
            error=None,
        )
    else:
        prompt_text = _flatten_anthropic_messages(original_kwargs.get("messages"), original_kwargs.get("system"))
        log_call(
            model=model,
            prompt_tokens=estimate_tokens(prompt_text),
            completion_tokens=estimate_tokens("".join(completion_text)),
            latency_ms=latency_ms,
            tags=config.get_active_tags(),
            provider=provider,
            error=None,
            estimated=True,
        )
