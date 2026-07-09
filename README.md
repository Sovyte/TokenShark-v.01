# TokenShark

Patch your LLM client with one import and see real-time token costs in
your terminal.

```python
import tokenshark
tokenshark.monitor()
# OpenAI and Anthropic calls are now logged with cost, latency, and
# (optionally) your own metadata tags.
```

## Status

Early build — core patching + SQLite cost tracking (`proxy.py`,
`tracker.py`, `config.py`, `exceptions.py`) is in place. `cli.py`,
`dashboard.py`, `alerts.py`, and tests are still to come.

## Install (once published)

```bash
pip install tokenshark[openai,anthropic]
```

---

*This is a placeholder — `pyproject.toml`'s `readme` field points here,
and hatchling needs the file to exist to build package metadata at all,
so `pip install -e .` would fail on a fresh clone without it. Not one of
the 5 requested files — added purely to keep today's `pip install -e .`
from breaking. Replace with a real README on Day 3 polish.*
