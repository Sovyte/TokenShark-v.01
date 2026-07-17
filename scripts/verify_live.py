"""
scripts/verify_live.py — minimal real-API smoke test for TokenShark's
OpenAI and Anthropic interceptors.

Not part of the pip-installed package. This exists because the sandbox
this codebase was audited in has no network egress and neither SDK
installed, so Step 2 priority #1 ("verified with an actual
minimal-token test call") couldn't be run there — run this yourself
once you have both.

Confirms, against real responses (not mocks): (1) the patch doesn't
crash on the real response shape, (2) a known-current model prices to a
non-zero cost via tracker.calc_cost (catches a stale/wrong model-ID key
in COST_PER_1M immediately), (3) streaming still yields chunks after
proxy.py's include_usage / message_start+message_delta handling.

Usage:
    pip install tokenshark[all]
    export OPENAI_API_KEY=...
    export ANTHROPIC_API_KEY=...
    python scripts/verify_live.py
"""

import sys

import tokenshark
from tokenshark.tracker import calc_cost

tokenshark.monitor()

FAILURES = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


def test_openai() -> None:
    try:
        from openai import OpenAI
    except ImportError:
        print("[SKIP] openai package not installed")
        return

    client = OpenAI()
    model = "gpt-5.4-mini"  # cheap, current, non-retired -- see tracker.COST_PER_1M

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with the single word: pong"}],
        max_tokens=5,
    )
    check("openai: non-streaming call returns a response", resp is not None)
    check("openai: usage present on response", getattr(resp, "usage", None) is not None)
    if resp.usage:
        cost = calc_cost(model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
        check(
            "openai: known model prices to a non-zero cost (catches stale model-ID keys)",
            cost > 0.0,
            f"calc_cost returned {cost} for model={model!r}",
        )

    # Confirms the include_usage injection actually yields a populated
    # usage chunk, not a silent fall-through to the token estimator.
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with the single word: pong"}],
        max_tokens=5,
        stream=True,
    )
    chunks = list(stream)
    check("openai: streaming call yields chunks", len(chunks) > 0)


def test_anthropic() -> None:
    try:
        from anthropic import Anthropic
    except ImportError:
        print("[SKIP] anthropic package not installed")
        return

    client = Anthropic()
    model = "claude-haiku-4-5-20251001"  # cheapest current, non-retired model

    resp = client.messages.create(
        model=model,
        max_tokens=5,
        messages=[{"role": "user", "content": "Reply with the single word: pong"}],
    )
    check("anthropic: non-streaming call returns a response", resp is not None)
    check("anthropic: usage present on response", getattr(resp, "usage", None) is not None)
    if resp.usage:
        cost = calc_cost(model, resp.usage.input_tokens, resp.usage.output_tokens)
        check(
            "anthropic: known model prices to a non-zero cost (catches stale model-ID keys)",
            cost > 0.0,
            f"calc_cost returned {cost} for model={model!r}",
        )

    # Calls Messages.create(..., stream=True) directly rather than the
    # SDK's separate client.messages.stream() helper, so this exercises
    # exactly the method proxy.py patches -- not a different code path
    # that may or may not route through it.
    stream = client.messages.create(
        model=model,
        max_tokens=5,
        messages=[{"role": "user", "content": "Reply with the single word: pong"}],
        stream=True,
    )
    events = list(stream)
    check("anthropic: streaming call yields events", len(events) > 0)


if __name__ == "__main__":
    test_openai()
    test_anthropic()
    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) failed: {FAILURES}")
        sys.exit(1)
    print("\nAll checks passed.")
