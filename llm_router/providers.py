"""Thin HTTP clients for Gemini, Ollama and OpenRouter that share one call signature."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import httpx

Message = dict  # {"role": "system" | "user" | "assistant", "content": str}


class ProviderError(Exception):
    """A call failed. `retryable` tells the router whether trying another provider makes sense."""

    def __init__(self, message: str, *, retryable: bool, status: int | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status
        self.retry_after = retry_after


@dataclass
class Completion:
    text: str
    provider: str
    model: str


class Provider:
    name = "base"

    def __init__(self, client: httpx.Client | None = None, timeout: float = 60):
        self.client = client or httpx.Client(timeout=timeout)

    def complete(self, model: str, messages: list[Message]) -> Completion:
        try:
            resp = self._post(model, messages)
        except httpx.TimeoutException as e:
            raise ProviderError(f"{self.name}: timeout", retryable=True) from e
        except httpx.TransportError as e:
            raise ProviderError(f"{self.name}: connection error: {e}", retryable=True) from e
        if resp.status_code >= 400:
            raise _http_error(self.name, resp)
        try:
            text = self._parse(resp.json())
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise ProviderError(f"{self.name}: unexpected response format", retryable=True) from e
        if not text:
            raise ProviderError(f"{self.name}: empty response", retryable=True)
        return Completion(text=text, provider=self.name, model=model)

    def _post(self, model: str, messages: list[Message]) -> httpx.Response:
        raise NotImplementedError

    def _parse(self, data: dict) -> str:
        raise NotImplementedError


def _http_error(name: str, resp: httpx.Response) -> ProviderError:
    status = resp.status_code
    retry_after = None
    if "retry-after" in resp.headers:
        try:
            retry_after = float(resp.headers["retry-after"])
        except ValueError:
            pass
    detail = resp.text[:200].replace("\n", " ")
    # Worth trying the next provider: quota/rate limits, outages, unknown models, and credential problems
    # (keys are per provider; Gemini reports an invalid key as HTTP 400 API_KEY_INVALID).
    # A genuinely malformed request (other 400s) would fail everywhere, so it stops the chain.
    auth = status in (401, 403) or (status == 400 and "API_KEY_INVALID" in resp.text)
    retryable = auth or status in (404, 408, 429) or status >= 500
    return ProviderError(f"{name}: HTTP {status}: {detail}", retryable=retryable, status=status, retry_after=retry_after)


def _require_key(env: str, name: str) -> str:
    key = os.environ.get(env)
    if not key:
        raise ProviderError(f"{name}: {env} is not set", retryable=True)
    return key


class GeminiProvider(Provider):
    name = "gemini"
    base_url = "https://generativelanguage.googleapis.com/v1beta"

    def _post(self, model, messages):
        key = _require_key("GEMINI_API_KEY", self.name)
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages if m["role"] != "system"
        ]
        body = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return self.client.post(f"{self.base_url}/models/{model}:generateContent", json=body,
                                headers={"x-goog-api-key": key})

    def _parse(self, data):
        return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"]).strip()


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, client=None, timeout=300, base_url: str | None = None):
        super().__init__(client, timeout)
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")

    def _post(self, model, messages):
        return self.client.post(f"{self.base_url}/api/chat", json={"model": model, "messages": messages, "stream": False})

    def _parse(self, data):
        # Reasoning models (e.g. qwen3, deepseek-r1) may inline their chain of thought; keep only the answer.
        return re.sub(r"<think>.*?</think>", "", data["message"]["content"], flags=re.S).strip()


class OpenRouterProvider(Provider):
    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"

    def _post(self, model, messages):
        key = _require_key("OPENROUTER_API_KEY", self.name)
        return self.client.post(f"{self.base_url}/chat/completions", json={"model": model, "messages": messages},
                                headers={"Authorization": f"Bearer {key}"})

    def _parse(self, data):
        return data["choices"][0]["message"]["content"].strip()


PROVIDERS = {p.name: p for p in (GeminiProvider, OllamaProvider, OpenRouterProvider)}
