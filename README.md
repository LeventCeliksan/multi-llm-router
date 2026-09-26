# multi-llm-router

Route LLM calls **by task type** across **Gemini**, **OpenRouter** and **local Ollama models**, with **automatic failover** when a provider hits its quota, times out, is down, or has a bad key.

A small standalone implementation of the routing-and-failover approach I use in a 7-agent generative-AI advertising agency system, where each agent type gets its own model chain and most heavy work runs on local models to keep API costs low. This repo is written from scratch as an independent tool; it does not contain code from that project.

## Features
- **Task-based routing** — each task type (`copywriting`, `code`, …) has its own ordered chain of `provider:model` targets, plus a `default` chain.
- **Failover** — HTTP 429 (quota), 5xx, timeouts, connection errors, unknown models, missing or invalid API keys → next target. A genuinely malformed request (other HTTP 400s) stops immediately instead of burning through every provider.
- **Rate-limit cooldown** — a target that returned 429 is skipped until its `Retry-After` (or 60 s) expires.
- **Attempt log** — every result says which targets were tried and why each failed.
- **One small dependency** (`httpx`); providers are ~20 lines each, easy to extend.

## Install
```bash
pip install git+https://github.com/LeventCeliksan/multi-llm-router
export GEMINI_API_KEY=...        # optional
export OPENROUTER_API_KEY=...    # optional
# Ollama: https://ollama.com, then e.g. `ollama pull qwen3:8b`
```
Providers without a key are simply skipped, so it works with Ollama alone.

## Usage
```python
from llm_router import Router

router = Router({
    "default":     ["gemini:gemini-2.5-flash", "openrouter:meta-llama/llama-3.3-70b-instruct", "ollama:qwen3:8b"],
    "copywriting": ["gemini:gemini-2.5-flash", "ollama:qwen3:8b"],
})
result = router.complete("Write a tagline for a coffee brand", task="copywriting", system="Be concise.")
print(result.completion.text, result.completion.provider)
for a in result.attempts:
    print(a.target, "ok" if a.ok else a.error)
```

CLI:
```bash
llm-router "Summarize: ..." --task copywriting --routes routes.example.json -v
# [fail] gemini:gemini-2.5-flash - gemini: HTTP 429: ...
# [ok] ollama:qwen3:8b
```

## Tests
```bash
pip install -e . pytest
pytest -m "not ollama"   # 12 offline tests: routing, request formats, failover, cooldown, error handling
pytest                   # + 2 live tests against a local Ollama model (default qwen3:8b, set OLLAMA_TEST_MODEL)
```
The live tests call the real Gemini and OpenRouter endpoints with deliberately invalid keys and check that the router falls over to the local model.

## License
MIT
