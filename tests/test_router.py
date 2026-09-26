import json
import os

import httpx
import pytest

from llm_router import AllTargetsFailed, Router
from llm_router.cli import main
from llm_router.providers import GeminiProvider, OllamaProvider, OpenRouterProvider


def gemini_ok(text):
    return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": text}]}}]})


def openrouter_ok(text):
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": text}}]})


def ollama_ok(text):
    return httpx.Response(200, json={"message": {"role": "assistant", "content": text}})


def make_router(handlers, routes, **kw):
    """handlers: provider name -> function(request) -> httpx.Response. Records every request."""
    calls = []

    def wrap(name):
        def handler(request):
            calls.append((name, request))
            return handlers[name](request)
        return httpx.Client(transport=httpx.MockTransport(handler))

    providers = {
        "gemini": GeminiProvider(client=wrap("gemini")),
        "openrouter": OpenRouterProvider(client=wrap("openrouter")),
        "ollama": OllamaProvider(client=wrap("ollama"), base_url="http://ollama.test"),
    }
    return Router(routes, providers=providers, **kw), calls


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter")


CHAIN = {"default": ["gemini:gemini-2.5-flash", "openrouter:some/model", "ollama:qwen3:8b"]}


def test_first_target_answers():
    r, calls = make_router({"gemini": lambda q: gemini_ok("hi")}, CHAIN)
    res = r.complete("hello")
    assert res.completion.text == "hi" and res.completion.provider == "gemini"
    assert [c[0] for c in calls] == ["gemini"]


def test_request_formats():
    r, calls = make_router({"gemini": lambda q: gemini_ok("ok")}, CHAIN)
    r.complete("hello", system="be brief")
    req = calls[0][1]
    body = json.loads(req.content)
    assert req.url.path == "/v1beta/models/gemini-2.5-flash:generateContent"
    assert req.headers["x-goog-api-key"] == "test-gemini"
    assert body["systemInstruction"]["parts"][0]["text"] == "be brief"
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]


def test_quota_failover_to_openrouter():
    r, calls = make_router({
        "gemini": lambda q: httpx.Response(429, json={"error": {"status": "RESOURCE_EXHAUSTED"}}),
        "openrouter": lambda q: openrouter_ok("from openrouter"),
    }, CHAIN)
    res = r.complete("hello")
    assert res.completion.provider == "openrouter"
    assert [a.ok for a in res.attempts] == [False, True]
    req = calls[1][1]
    assert req.headers["authorization"] == "Bearer test-openrouter"
    assert json.loads(req.content)["model"] == "some/model"


def test_server_error_and_timeout_fall_through_to_ollama():
    def timeout(q):
        raise httpx.ReadTimeout("slow", request=q)

    r, calls = make_router({
        "gemini": lambda q: httpx.Response(503),
        "openrouter": timeout,
        "ollama": lambda q: ollama_ok("<think>reasoning</think>local answer"),
    }, CHAIN)
    res = r.complete("hello")
    assert res.completion.provider == "ollama" and res.completion.text == "local answer"
    assert json.loads(calls[2][1].content) == {"model": "qwen3:8b", "messages": [{"role": "user", "content": "hello"}], "stream": False}


def test_missing_key_skips_provider(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    r, calls = make_router({"openrouter": lambda q: openrouter_ok("ok")}, CHAIN)
    res = r.complete("hello")
    assert res.completion.provider == "openrouter"
    assert "GEMINI_API_KEY is not set" in res.attempts[0].error
    assert [c[0] for c in calls] == ["openrouter"]


def test_invalid_keys_fail_over():
    # Real responses: Gemini answers an invalid key with 400 API_KEY_INVALID, OpenRouter with 401.
    r, calls = make_router({
        "gemini": lambda q: httpx.Response(400, json={"error": {"code": 400, "message": "API key not valid.",
                                                                "details": [{"reason": "API_KEY_INVALID"}]}}),
        "openrouter": lambda q: httpx.Response(401, json={"error": {"message": "Missing Authentication header"}}),
        "ollama": lambda q: ollama_ok("ok"),
    }, CHAIN)
    assert r.complete("x").completion.provider == "ollama"
    assert [c[0] for c in calls] == ["gemini", "openrouter", "ollama"]


def test_bad_request_does_not_fail_over():
    r, calls = make_router({"gemini": lambda q: httpx.Response(400, json={"error": "bad"})}, CHAIN)
    with pytest.raises(AllTargetsFailed):
        r.complete("hello")
    assert [c[0] for c in calls] == ["gemini"]


def test_all_fail_reports_every_attempt():
    r, _ = make_router({n: (lambda q: httpx.Response(500)) for n in ("gemini", "openrouter", "ollama")}, CHAIN)
    with pytest.raises(AllTargetsFailed) as e:
        r.complete("hello")
    assert len(e.value.attempts) == 3


def test_rate_limited_target_is_on_cooldown_until_retry_after():
    t = [1000.0]
    r, calls = make_router({
        "gemini": lambda q: httpx.Response(429, headers={"retry-after": "30"}),
        "openrouter": lambda q: openrouter_ok("ok"),
    }, CHAIN, clock=lambda: t[0])
    r.complete("a")
    r.complete("b")                      # gemini skipped, not called
    assert [c[0] for c in calls] == ["gemini", "openrouter", "openrouter"]
    t[0] += 31
    r.complete("c")                      # cooldown over, gemini tried again
    assert [c[0] for c in calls][-2:] == ["gemini", "openrouter"]


def test_routes_by_task_type():
    routes = {"default": ["gemini:gemini-2.5-flash"], "code": ["openrouter:coder/model"]}
    r, calls = make_router({"gemini": lambda q: gemini_ok("g"), "openrouter": lambda q: openrouter_ok("o")}, routes)
    assert r.complete("x", task="code").completion.model == "coder/model"
    assert r.complete("x", task="unknown").completion.provider == "gemini"


def test_malformed_response_fails_over():
    r, _ = make_router({"gemini": lambda q: httpx.Response(200, json={"nope": 1}),
                        "openrouter": lambda q: openrouter_ok("ok")}, CHAIN)
    assert r.complete("x").completion.provider == "openrouter"


def test_invalid_target_spec():
    with pytest.raises(ValueError):
        Router({"default": ["gemini"]})


@pytest.mark.ollama
def test_live_ollama_via_cli(tmp_path, monkeypatch, capsys):
    model = os.environ.get("OLLAMA_TEST_MODEL", "qwen3:8b")
    monkeypatch.delenv("GEMINI_API_KEY")
    routes = tmp_path / "routes.json"
    routes.write_text(json.dumps({"default": ["gemini:gemini-2.5-flash", f"ollama:{model}"]}))
    code = main(["Reply with exactly one word: pong", "--routes", str(routes), "-v"])
    out = capsys.readouterr()
    assert code == 0
    assert "pong" in out.out.lower()
    assert "[fail] gemini:gemini-2.5-flash" in out.err and f"[ok] ollama:{model}" in out.err


@pytest.mark.ollama
def test_live_invalid_keys_fail_over_to_local_ollama(monkeypatch):
    """Real Gemini and OpenRouter endpoints with fake keys must fail over to the local model."""
    model = os.environ.get("OLLAMA_TEST_MODEL", "qwen3:8b")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    r = Router({"default": ["gemini:gemini-2.5-flash", "openrouter:meta-llama/llama-3.3-70b-instruct", f"ollama:{model}"]})
    res = r.complete("Reply with exactly one word: pong")
    assert res.completion.provider == "ollama" and "pong" in res.completion.text.lower()
    assert [a.ok for a in res.attempts] == [False, False, True]
