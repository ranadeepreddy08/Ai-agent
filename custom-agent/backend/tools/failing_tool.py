"""
Failing Tool — deterministic test tool for Phase 3 recovery scenarios.

This tool intentionally fails a configurable number of times, then succeeds.
It is registered ONLY during Phase 3 test runs — never in production.

Use cases:
  - Tool failure detection (observe failure, not crash)
  - Recovery strategy triggering (RecoveryManager reacts to failures)
  - Retry-limit enforcement (verify agent stops after max attempts)
  - Fallback and replanning tests

Design:
  - Uses a class-level counter dict keyed by instance ID
  - Thread-safe for single-threaded agent loop
  - Output is fully structured ToolResult (never raises)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from backend.tools.base import BaseTool, ToolResult


class FailingTool(BaseTool):
    """
    A test-only tool that fails `fail_times` times before succeeding.

    After `fail_times` failed calls, it returns a successful ToolResult
    with a configurable success output.

    Parameters:
        fail_times:      How many calls should return failure (default 2)
        fail_message:    Error text in the failed ToolResult
        success_output:  Output returned on the eventual successful call
        simulate_timeout: Whether failed calls should set metadata["timeout"]=True
    """

    name = "failing_tool"
    description = (
        "A test tool that intentionally fails a fixed number of times before "
        "succeeding. Used to validate the agent's failure detection and recovery."
    )
    capabilities = [
        "simulate_tool_failure",
        "test_recovery",
        "test_retry_limits",
    ]
    input_schema = {
        "query": "string — any query (the tool ignores it after the failure window)",
    }

    def __init__(
        self,
        fail_times: int = 2,
        fail_message: str = "Simulated tool failure for Phase 3 testing.",
        success_output: str = "Recovery succeeded: tool finally worked after failures.",
        simulate_timeout: bool = False,
    ) -> None:
        self._fail_times = fail_times
        self._fail_message = fail_message
        self._success_output = success_output
        self._simulate_timeout = simulate_timeout
        self._call_count = 0  # Tracks calls across goal retries

    def execute(self, input_data: dict[str, Any]) -> ToolResult:
        """
        Execute the tool — fail the first `fail_times` calls, then succeed.
        Never raises — always returns a ToolResult.
        """
        goal_id = str(input_data.get("goal_id", "unknown"))
        query = str(input_data.get("query", ""))
        start = time.monotonic()

        self._call_count += 1
        elapsed = time.monotonic() - start

        if self._call_count <= self._fail_times:
            return ToolResult(
                success=False,
                output=None,
                error=self._fail_message,
                execution_time=elapsed,
                tool_name=self.name,
                goal_id=goal_id,
                metadata={
                    "timeout": self._simulate_timeout,
                    "call_count": self._call_count,
                    "will_succeed_after": self._fail_times,
                },
            )

        # Success path
        return ToolResult(
            success=True,
            output={
                "result": self._success_output,
                "query": query,
                "call_count": self._call_count,
                "failed_before_success": self._fail_times,
            },
            error=None,
            execution_time=elapsed,
            tool_name=self.name,
            goal_id=goal_id,
            metadata={"call_count": self._call_count},
        )

    def reset(self) -> None:
        """Reset call counter (use between test cases)."""
        self._call_count = 0
