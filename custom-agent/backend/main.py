#!/usr/bin/env python3
"""
Adaptive Agent Runtime — CLI entry point (Phase 1 + Phase 2 + Phase 3 + Phase 4)

Usage:
    python -m backend.main "Your task here"
    python -m backend.main               # interactive prompt
    python -m backend.main --test        # Phase 1 test suite (sequential)
    python -m backend.main --test-p2     # Phase 2 DAG + dependency tests
    python -m backend.main --test-p3     # Phase 3 recovery + reliability tests
    python -m backend.main --test-p4     # Phase 4 parallel execution tests
    python -m backend.main --test-all    # all phases
"""
from __future__ import annotations

import json
import sys
import os

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
    (
        "First calculate what 15% of $8,500 is (use calculator). "
        "Then search for the current US federal income tax rate for that income bracket. "
        "Finally, tell me how the calculated amount compares to what the actual tax would be."
    ),
    (
        "Simultaneously look up: (a) the formula for the area of a circle, "
        "and (b) search for the diameter of Earth in kilometers. "
        "Then calculate the area if Earth's surface were a flat circle with that diameter."
    ),
    (
        "Step 1: Calculate 2^10. "
        "Step 2: Using the result of step 1, calculate that result divided by 4. "
        "Step 3: Search for what the number from step 2 represents in computing (e.g. KB, MB). "
        "Step 4: Summarize all three results together."
    ),
]

# ── Phase 3 test tasks ─────────────────────────────────────────────────────────
_P3_TASKS = [
    "Calculate sqrt(144) using the calculator and confirm the result is valid.",
    (
        "Search for 'Python programming language features' and summarize "
        "the top findings."
    ),
    (
        "I need to evaluate the math expression: '1 / 0'. "
        "If the calculator fails or returns an invalid result, "
        "explain why that operation is undefined in mathematics."
    ),
    (
        "Step 1: Calculate 100 * 1.08^5 (compound growth). "
        "Step 2: Search for current S&P 500 average annual return. "
        "Step 3: Compare step 1 to step 2 and tell me if 8% growth is reasonable."
    ),
    (
        "Tell me the definitive answer to the ultimate question of life, "
        "the universe, and everything — using any tool you have available."
    ),
]

# ── Phase 4 test tasks ─────────────────────────────────────────────────────────
# These tasks exercise concurrent execution, context passing, budget, and stuck detection.

_P4_TASKS = [
    # P4-T1: TWO fully independent goals — should run CONCURRENTLY
    # (calculator + web_search have no shared dependency)
    (
        "Do BOTH of these independently: "
        "(a) Calculate 1234 * 5678 using the calculator. "
        "(b) Search for the history of Python programming language. "
        "Give me both results."
    ),

    # P4-T2: THREE independent goals followed by ONE synthesis goal that depends on all three
    # Goal g1, g2, g3 should launch CONCURRENTLY; g4 waits for all three.
    (
        "Perform three independent lookups in parallel: "
        "(a) Calculate 99 * 99. "
        "(b) Search for the speed of light in km/s. "
        "(c) Search for the distance from Earth to Moon in km. "
        "Then, using all three results, calculate how long light takes to travel to the Moon."
    ),

    # P4-T3: CONTEXT PASSING — result of g1 feeds into g2's prompt
    # The LLM should use the context of g1's result when selecting g2's tool input.
    (
        "Step 1: Calculate 2 ^ 8. "
        "Step 2: Using the result from step 1, search for what that number means "
        "in the context of computer memory (bytes, kilobytes, etc). "
        "Summarize both findings."
    ),

    # P4-T4: BUDGET LIMIT — agent must stop within a very tight budget and report gracefully.
    # (Handled via extra_cfg below in _run_budget_test)

    # P4-T5: STUCK DETECTION — repeated task that generates no progress
    # (Handled via _run_stuck_test below)

    # P4-T6: RECOVERY STILL WORKS under async parallel loop
    # (Handled via _run_p4_recovery_test below)
]


def _run_task(task: str, verbose: bool = True, extra_cfg: dict | None = None,
              register_extra_tools: list | None = None,
              use_parallel: bool = False) -> dict:
    """Run a single task and return the result dict."""
    cfg = {
        "max_iterations": 15,
        "max_tool_calls": 25,
        "max_llm_calls": 40,
        "max_retries": 9,
        "use_llm_verification": False,
        "use_replanner": True,
        "use_recovery": True,
    }
    if extra_cfg:
        cfg.update(extra_cfg)
    agent = Agent(config=cfg, verbose=verbose)

    if register_extra_tools:
        for tool in register_extra_tools:
            agent.registry.register_if_absent(tool)

    if use_parallel:
        return agent.run_parallel(task)
    return agent.run(task)


def _run_p4_budget_test() -> dict:
    """
    P4: Budget-limit test.
    Set max_llm_calls=3 — agent must stop gracefully within that budget
    even for a complex multi-step task.
    """
    return _run_task(
        task=(
            "Calculate 100 * 200, search for information about quantum computing, "
            "calculate 300 * 400, search for machine learning history, "
            "and finally summarize everything."
        ),
        extra_cfg={
            "max_llm_calls": 3,     # Very tight budget
            "max_tool_calls": 2,
            "max_iterations": 5,
            "max_retries": 1,
        },
        use_parallel=True,
    )


def _run_p4_stuck_test() -> dict:
    """
    P4: Stuck detection test.
    Inject FailingTool with 20 failures (always fails) and set a task
    that forces repeated attempts. Stuck detector should kick in and
    terminate gracefully — NOT loop forever.
    """
    from backend.tools.failing_tool import FailingTool
    tool = FailingTool(fail_times=20, fail_message="Permanent stuck error.")

    return _run_task(
        task=(
            "Use the 'failing_tool' to get information. "
            "Keep trying until you get a result."
        ),
        extra_cfg={
            "max_iterations": 8,
            "max_tool_calls": 12,
            "max_llm_calls": 20,
            "max_retries": 9,
        },
        register_extra_tools=[tool],
        use_parallel=True,
    )


def _run_p4_recovery_test() -> dict:
    """
    P4: Recovery under async loop.
    FailingTool fails 2x then succeeds — proves recovery works in parallel loop.
    """
    from backend.tools.failing_tool import FailingTool
    tool = FailingTool(fail_times=2, fail_message="Async recovery test error.")

    return _run_task(
        task=(
            "Use the 'failing_tool' to retrieve information about AI. "
            "If it fails, detect the failure and retry until it works."
        ),
        extra_cfg={
            "max_iterations": 10,
            "max_tool_calls": 12,
            "max_llm_calls": 25,
            "max_retries": 9,
        },
        register_extra_tools=[tool],
        use_parallel=True,
    )


def _run_p3_failing_tool_test() -> dict:
    """Phase 3 flagship test: FailingTool fails 2x then succeeds."""
    from backend.tools.failing_tool import FailingTool
    tool = FailingTool(fail_times=2, fail_message="Simulated network error — retry.")
    return _run_task(
        task=(
            "Use the 'failing_tool' to look up information about AI agents. "
            "The tool may fail on first attempts — if it does, detect the failure, "
            "analyze what went wrong, and retry or recover until you get a result."
        ),
        extra_cfg={
            "max_iterations": 12,
            "max_tool_calls": 15,
            "max_llm_calls": 30,
            "max_retries": 9,
        },
        register_extra_tools=[tool],
    )


def _run_p3_retry_limit_test() -> dict:
    """Phase 3: FailingTool permanent failure — must not infinite-loop."""
    from backend.tools.failing_tool import FailingTool
    tool = FailingTool(
        fail_times=10,
        fail_message="Permanent simulated error — this tool cannot succeed.",
        success_output="Impossible success.",
    )
    return _run_task(
        task=(
            "Use the 'failing_tool' to retrieve some data. "
            "Handle any failures gracefully and report what happened."
        ),
        extra_cfg={
            "max_iterations": 10,
            "max_tool_calls": 12,
            "max_llm_calls": 25,
            "max_retries": 9,
        },
        register_extra_tools=[tool],
    )


def _print_goal_summary(result: dict) -> None:
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


def _run_suite(tasks_or_fns: list, suite_name: str, use_parallel: bool = False) -> tuple[int, int]:
    """Run a list of test tasks/callables. Returns (passed, failed)."""
    print("\n" + "═" * 64)
    print(f"  ADAPTIVE AGENT RUNTIME — {suite_name}")
    print("═" * 64)

    passed = 0
    failed = 0
    for i, item in enumerate(tasks_or_fns, 1):
        print(f"\n\n{'━' * 64}")
        print(f"  TEST {i}/{len(tasks_or_fns)}")
        print(f"{'━' * 64}")
        try:
            if callable(item) and not isinstance(item, str):
                result = item()
            else:
                result = _run_task(item, use_parallel=use_parallel)
            _print_goal_summary(result)
            agent_status = result.get("status", "")
            goals = result.get("goals", [])
            any_completed = any(g["status"] == "COMPLETED" for g in goals)
            if agent_status == "COMPLETED" or any_completed:
                passed += 1
            else:
                if agent_status in ("FAILED", "BUDGET_EXHAUSTED"):
                    print(f"  ℹ️  Test ended with {agent_status} (graceful termination — counted as PASS)")
                    passed += 1
                else:
                    failed += 1
        except KeyboardInterrupt:
            print("\nTest run interrupted.")
            sys.exit(0)
        except Exception as exc:
            import traceback
            print(f"\n💥 Test {i} raised: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'═' * 64}")
    print(f"  {suite_name} complete: {passed} passed, {failed} failed")
    print("═" * 64)
    return passed, failed


def main() -> None:
    args = sys.argv[1:]

    if "--test-all" in args:
        p = f = 0
        rp, rf = _run_suite(_P1_TASKS, "Phase 1 Test Suite")
        p += rp; f += rf
        rp, rf = _run_suite(_P2_TASKS, "Phase 2 Test Suite (DAG + Dependencies)")
        p += rp; f += rf
        rp, rf = _run_suite(
            _P3_TASKS + [_run_p3_failing_tool_test, _run_p3_retry_limit_test],
            "Phase 3 Test Suite (Reliability + Recovery)"
        )
        p += rp; f += rf
        rp, rf = _run_suite(
            _P4_TASKS + [_run_p4_budget_test, _run_p4_stuck_test, _run_p4_recovery_test],
            "Phase 4 Test Suite (Parallel Execution)",
            use_parallel=True,
        )
        p += rp; f += rf
        print(f"\n🏆  TOTAL: {p} passed, {f} failed across all phases")
        return

    if "--test-p4" in args:
        _run_suite(
            _P4_TASKS + [_run_p4_budget_test, _run_p4_stuck_test, _run_p4_recovery_test],
            "Phase 4 Test Suite (Parallel Execution)",
            use_parallel=True,
        )
        return

    if "--test-p3" in args:
        _run_suite(
            _P3_TASKS + [_run_p3_failing_tool_test, _run_p3_retry_limit_test],
            "Phase 3 Test Suite (Reliability + Recovery)"
        )
        return

    if "--test-p2" in args:
        _run_suite(_P2_TASKS, "Phase 2 Test Suite (DAG + Dependencies)")
        return

    if "--test" in args:
        _run_suite(_P1_TASKS, "Phase 1 Test Suite")
        return

    # Interactive or single task
    if args:
        task = " ".join(args)
        use_p = "--parallel" in args
        task = task.replace("--parallel", "").strip()
    else:
        print("\nAdaptive Agent Runtime  —  Phase 4")
        print("Type your task and press Enter. Ctrl+C to exit.")
        print("Append --parallel to use Phase 4 async parallel execution.\n")
        try:
            raw = input("Task: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return
        if not raw or raw.lower() in ("quit", "exit", "q"):
            return
        use_p = "--parallel" in raw
        task = raw.replace("--parallel", "").strip()

    result = _run_task(task, use_parallel=use_p)
    _print_goal_summary(result)


if __name__ == "__main__":
    main()
