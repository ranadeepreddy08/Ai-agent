#!/usr/bin/env python3
"""
Adaptive Agent Runtime — CLI entry point (Phase 1)

Usage:
    python -m backend.main "Your task here"
    python -m backend.main               # interactive prompt
    python -m backend.main --test        # run built-in test suite
"""
from __future__ import annotations

import json
import sys
import os

# Ensure project root is on path regardless of how the script is invoked
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from backend.core.agent import Agent


# ── Built-in test tasks ────────────────────────────────────────────────────────
_TEST_TASKS = [
    "What is 25 * 47 + 100?",
    "Search for information about AI agent frameworks",
    (
        "Calculate the compound interest on $10,000 at 5% annual rate for 3 years "
        "(use the formula A = P * (1 + r)^t), and also search for current high-yield "
        "savings account rates. Tell me how much I would earn and compare it to market rates."
    ),
]


def _run_task(task: str, verbose: bool = True) -> dict:
    """Run a single task and return the result dict."""
    agent = Agent(
        config={
            "max_iterations": 15,
            "max_tool_calls": 20,
            "max_llm_calls": 30,
            "max_retries": 9,
            "use_llm_verification": False,
        },
        verbose=verbose,
    )
    return agent.run(task)


def _print_goal_summary(result: dict) -> None:
    """Print a formatted goal status table after the run."""
    print("\n📋  Goal Summary")
    print("─" * 64)
    _status_emoji = {
        "COMPLETED": "✅",
        "FAILED": "❌",
        "PENDING": "⏳",
        "RUNNING": "🔄",
        "SKIPPED": "⏭️",
        "BLOCKED": "🚫",
    }
    for g in result.get("goals", []):
        emoji = _status_emoji.get(g["status"], "❓")
        attempts = f"({g['attempts']} attempt{'s' if g['attempts'] != 1 else ''})"
        print(f"  {emoji} [{g['id']}] {g['description'][:55]:<55} {g['status']:<12} {attempts}")
    print("─" * 64)
    print(f"  Agent status : {result.get('status', '?')}")
    budget = result.get("budget", {})
    print(f"  Iterations   : {budget.get('iterations', '?')}")
    print(f"  Tool calls   : {budget.get('tool_calls', '?')}")
    print(f"  LLM calls    : {budget.get('llm_calls', '?')}")
    print(f"  Elapsed      : {budget.get('elapsed_seconds', '?')}s")
    tokens = result.get("llm_tokens", {})
    if tokens.get("input") or tokens.get("output"):
        print(f"  Tokens       : {tokens.get('input', 0)} in / {tokens.get('output', 0)} out")
    print()


def _run_tests() -> None:
    """Run the built-in test suite sequentially."""
    print("\n" + "═" * 64)
    print("  ADAPTIVE AGENT RUNTIME — Phase 1 Test Suite")
    print("═" * 64)

    for i, task in enumerate(_TEST_TASKS, 1):
        print(f"\n\n{'━' * 64}")
        print(f"  TEST {i}/{len(_TEST_TASKS)}")
        print(f"{'━' * 64}")
        try:
            result = _run_task(task)
            _print_goal_summary(result)
        except KeyboardInterrupt:
            print("\nTest run interrupted.")
            sys.exit(0)
        except Exception as exc:
            print(f"\n💥 Test {i} raised: {type(exc).__name__}: {exc}")

    print("\n✅  All tests complete.")


def main() -> None:
    args = sys.argv[1:]

    if "--test" in args:
        _run_tests()
        return

    if args:
        task = " ".join(args)
    else:
        print("\nAdaptive Agent Runtime  —  Phase 1")
        print("Type your task and press Enter. Ctrl+C to exit.\n")
        try:
            task = input("Task: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return
        if not task or task.lower() in ("quit", "exit", "q"):
            return

    result = _run_task(task)
    _print_goal_summary(result)


if __name__ == "__main__":
    main()
