#!/usr/bin/env python3
"""
Adaptive Agent Runtime — CLI entry point (Phase 1 + Phase 2)

Usage:
    python -m backend.main "Your task here"
    python -m backend.main               # interactive prompt
    python -m backend.main --test        # Phase 1 test suite
    python -m backend.main --test-p2     # Phase 2 DAG + dependency tests
    python -m backend.main --test-all    # both Phase 1 and Phase 2
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


# ── Phase 1 test tasks ─────────────────────────────────────────────────────────
_P1_TASKS = [
    "What is 25 * 47 + 100?",
    "Search for information about AI agent frameworks",
    (
        "Calculate the compound interest on $10,000 at 5% annual rate for 3 years "
        "(use the formula A = P * (1 + r)^t), and also search for current high-yield "
        "savings account rates. Tell me how much I would earn and compare it to market rates."
    ),
]

# ── Phase 2 test tasks ─────────────────────────────────────────────────────────
_P2_TASKS = [
    # Test 1: Multi-step sequential DAG with explicit dependencies
    # Planner should create: g1 (calculate) → g2 (search) → g3 (compare, depends g1+g2)
    (
        "First calculate what 15% of $8,500 is (use calculator). "
        "Then search for the current US federal income tax rate for that income bracket. "
        "Finally, tell me how the calculated amount compares to what the actual tax would be."
    ),

    # Test 2: Parallel DAG — two independent goals, then one synthesis goal
    # Planner should identify g1 and g2 as independent (parallel), g3 depends on both
    (
        "Simultaneously look up: (a) the formula for the area of a circle, "
        "and (b) search for the diameter of Earth in kilometers. "
        "Then calculate the area if Earth's surface were a flat circle with that diameter."
    ),

    # Test 3: 4-node deep chain to verify dependency-aware execution order
    # g1 → g2 → g3 → g4 must execute strictly in order
    (
        "Step 1: Calculate 2^10. "
        "Step 2: Using the result of step 1, calculate that result divided by 4. "
        "Step 3: Search for what the number from step 2 represents in computing (e.g. KB, MB). "
        "Step 4: Summarize all three results together."
    ),
]


def _run_task(task: str, verbose: bool = True, extra_cfg: dict | None = None) -> dict:
    """Run a single task and return the result dict."""
    cfg = {
        "max_iterations": 15,
        "max_tool_calls": 20,
        "max_llm_calls": 30,
        "max_retries": 9,
        "use_llm_verification": False,
        "use_replanner": True,
    }
    if extra_cfg:
        cfg.update(extra_cfg)
    agent = Agent(config=cfg, verbose=verbose)
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


def _run_suite(tasks: list[str], suite_name: str) -> None:
    """Run a list of test tasks sequentially."""
    print("\n" + "═" * 64)
    print(f"  ADAPTIVE AGENT RUNTIME — {suite_name}")
    print("═" * 64)

    passed = 0
    failed = 0
    for i, task in enumerate(tasks, 1):
        print(f"\n\n{'━' * 64}")
        print(f"  TEST {i}/{len(tasks)}")
        print(f"{'━' * 64}")
        try:
            result = _run_task(task)
            _print_goal_summary(result)
            if result.get("status") == "COMPLETED":
                passed += 1
            else:
                failed += 1
        except KeyboardInterrupt:
            print("\nTest run interrupted.")
            sys.exit(0)
        except Exception as exc:
            print(f"\n💥 Test {i} raised: {type(exc).__name__}: {exc}")
            failed += 1

    print(f"\n{'═' * 64}")
    print(f"  {suite_name} complete: {passed} passed, {failed} failed")
    print("═" * 64)


def main() -> None:
    args = sys.argv[1:]

    if "--test-all" in args:
        _run_suite(_P1_TASKS, "Phase 1 Test Suite")
        _run_suite(_P2_TASKS, "Phase 2 Test Suite (DAG + Dependencies)")
        return

    if "--test-p2" in args:
        _run_suite(_P2_TASKS, "Phase 2 Test Suite (DAG + Dependencies)")
        return

    if "--test" in args:
        _run_suite(_P1_TASKS, "Phase 1 Test Suite")
        return

    if args:
        task = " ".join(args)
    else:
        print("\nAdaptive Agent Runtime  —  Phase 2")
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
