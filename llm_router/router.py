"""Route each task type to an ordered chain of provider/model targets, failing over on quota limits and outages."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .providers import PROVIDERS, Completion, Message, Provider, ProviderError


@dataclass(frozen=True)
class Target:
    provider: str
    model: str

    @classmethod
    def parse(cls, spec: str) -> "Target":
        provider, sep, model = spec.partition(":")
        if not sep or not model:
            raise ValueError(f"target must look like 'provider:model', got {spec!r}")
        return cls(provider, model)

    def __str__(self):
        return f"{self.provider}:{self.model}"


@dataclass
class Attempt:
    target: Target
    ok: bool
    error: str | None = None
    skipped: bool = False


@dataclass
class RouteResult:
    completion: Completion
    task: str
    attempts: list[Attempt] = field(default_factory=list)


class AllTargetsFailed(Exception):
    def __init__(self, task: str, attempts: list[Attempt]):
        lines = "; ".join(f"{a.target}: {a.error}" for a in attempts)
        super().__init__(f"all targets failed for task {task!r}: {lines}")
        self.attempts = attempts


class Router:
    """
    routes = {"copywriting": ["gemini:gemini-2.5-flash", "openrouter:...", "ollama:qwen3:8b"], "default": [...]}
    Targets are tried in order. Retryable failures (HTTP 429/5xx/404, timeouts, missing or invalid API key)
    move on to the next target; a malformed request (other HTTP 400s) stops and raises.
    A target that hit a rate limit is put on cooldown (Retry-After, or `cooldown` seconds) and skipped until then.
    """

    def __init__(self, routes: dict[str, list[str]], providers: dict[str, Provider] | None = None,
                 cooldown: float = 60, clock=time.monotonic):
        if "default" not in routes:
            raise ValueError("routes must include a 'default' chain")
        self.routes = {task: [Target.parse(s) for s in chain] for task, chain in routes.items()}
        self.providers = providers or {}
        self.cooldown = cooldown
        self.clock = clock
        self._blocked_until: dict[Target, float] = {}

    def _provider(self, name: str) -> Provider:
        if name not in self.providers:
            if name not in PROVIDERS:
                raise ValueError(f"unknown provider {name!r}; known: {sorted(PROVIDERS)}")
            self.providers[name] = PROVIDERS[name]()
        return self.providers[name]

    def complete(self, prompt: str | list[Message], task: str = "default", system: str | None = None) -> RouteResult:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
        if system:
            messages.insert(0, {"role": "system", "content": system})
        chain = self.routes.get(task, self.routes["default"])
        attempts: list[Attempt] = []
        now = self.clock()
        for target in chain:
            if self._blocked_until.get(target, 0) > now:
                attempts.append(Attempt(target, ok=False, error="on cooldown after rate limit", skipped=True))
                continue
            try:
                completion = self._provider(target.provider).complete(target.model, messages)
            except ProviderError as e:
                attempts.append(Attempt(target, ok=False, error=str(e)))
                if e.status == 429:
                    self._blocked_until[target] = self.clock() + (e.retry_after or self.cooldown)
                if not e.retryable:
                    raise AllTargetsFailed(task, attempts) from e
                continue
            attempts.append(Attempt(target, ok=True))
            return RouteResult(completion, task, attempts)
        raise AllTargetsFailed(task, attempts)
