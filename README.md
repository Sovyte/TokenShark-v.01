# TokenShark

Local cost, token, and latency tracking for your OpenAI and Anthropic API calls — one import, no required config.

## Install

```bash
pip install tokenshark-bin
```

(The PyPI project is `tokenshark-bin`; the module you import is `tokenshark`.)

## Quickstart

```python
import tokenshark
tokenshark.monitor()

# use your OpenAI / Anthropic client exactly as before
```

Every `chat.completions.create` / `messages.create` call — sync, async, and streaming — is now logged locally to `~/.tokenshark/logs.db`, including cost, prompt/completion tokens, cache tokens, latency, and any tags you set.

```python
tokenshark.monitor(tags={"feature": "summarizer"})       # tag every call this run
tokenshark.monitor(providers=["openai"])                  # only patch openai
```

## CLI

```bash
tokenshark report                          # spend + tokens, grouped by model
tokenshark report --today --group-by provider
tokenshark budget                          # today's spend vs. your budget
tokenshark export calls.csv                # add --format json for JSON
tokenshark reset                           # clear the local log database
tokenshark <command> --help                # full options for any command
```

## Configuration

Optional `tokenshark.yaml` in your project root (or `~/.tokenshark/config.yaml` for global defaults):

```yaml
providers: [openai, anthropic]
alerts:
  daily_budget: 5.00
  per_session_budget: 1.00
  hard_stop: false
dashboard:
  refresh_seconds: 2
  max_rows: 20
  sort_by: cost
storage:
  path: ~/.tokenshark/logs.db
tags: {}
```

Nothing here is required — TokenShark works out of the box.

**Never put secrets in `tokenshark.yaml`.** Keys containing `api_key`, `secret`, `token`, `password`, or similar are rejected at load time. Set your Slack alert webhook via the `TOKENSHARK_SLACK_WEBHOOK` environment variable instead of the config file.

## Budget alerts

Set `TOKENSHARK_SLACK_WEBHOOK` (or `alerts.slack_webhook` in `tokenshark.yaml`, for local testing only) and TokenShark posts to that channel the moment `daily_budget` or `per_session_budget` is crossed:

```
🦈 TokenShark budget alert
Daily budget exceeded: $5.42 spent vs. $5.00 budget (triggered by a call to `gpt-4o`).
```

- Each budget notifies **once** per day (daily) or once per process (session) — it won't spam the channel on every call after you're already over.
- Set `hard_stop: true` to make an exceeded budget raise `tokenshark.TokenSharkBudgetExceeded` from the LLM call itself, every time, until the budget resets or you turn it off — use this if you want calls to actually stop, not just alert.
- No webhook configured means no network calls are made at all; alerts are entirely opt-in.

## Supported providers

- **OpenAI** — sync/async, streaming included, local Ollama servers auto-detected via `base_url`
- **Anthropic** — sync/async, streaming included, prompt caching tracked

Pricing entries also exist for DeepSeek, Qwen, and Mistral models, but only OpenAI and Anthropic are actively patched today — `monitor()` warns and skips any configured provider it doesn't yet support, and skips (rather than crashes on) any provider SDK that isn't installed.

## Notes on accuracy

- Streaming calls use the provider's own usage data when available. If a provider doesn't return it, TokenShark estimates using `tiktoken` (optional: `pip install tokenshark-bin[estimate]`) or a rough character-count fallback, and marks those rows `estimated` in the database.
- `monitor()` is safe to call more than once in the same process (e.g. re-running a notebook cell) — the SDK patch is only applied once.

## License

See [LICENSE](LICENSE).
