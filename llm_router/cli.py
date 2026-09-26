"""llm-router CLI: send one prompt through a task's failover chain."""
import argparse
import json
import sys
from pathlib import Path

from .router import AllTargetsFailed, Router

EXAMPLE_ROUTES = {
    "default": ["gemini:gemini-2.5-flash", "openrouter:meta-llama/llama-3.3-70b-instruct", "ollama:qwen3:8b"],
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="llm-router", description="Route a prompt by task type with automatic failover.")
    p.add_argument("prompt")
    p.add_argument("--task", default="default")
    p.add_argument("--system")
    p.add_argument("--routes", type=Path, help="JSON file mapping task -> [\"provider:model\", ...]")
    p.add_argument("-v", "--verbose", action="store_true", help="print every attempt to stderr")
    a = p.parse_args(argv)

    routes = json.loads(a.routes.read_text()) if a.routes else EXAMPLE_ROUTES
    try:
        result = Router(routes).complete(a.prompt, task=a.task, system=a.system)
    except AllTargetsFailed as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if a.verbose:
        for at in result.attempts:
            print(f"[{'ok' if at.ok else 'skip' if at.skipped else 'fail'}] {at.target}"
                  + (f" - {at.error}" if at.error else ""), file=sys.stderr)
    print(result.completion.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
