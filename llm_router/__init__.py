"""Task-based routing across Gemini, Ollama and OpenRouter with automatic failover."""
from .providers import Completion, ProviderError
from .router import AllTargetsFailed, RouteResult, Router, Target

__all__ = ["Router", "Target", "RouteResult", "AllTargetsFailed", "Completion", "ProviderError"]
