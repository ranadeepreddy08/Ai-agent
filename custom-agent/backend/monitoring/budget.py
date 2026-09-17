"""
Budget Manager — hard limits on all agent resource consumption.

Tracks iterations, tool calls, LLM calls, retries, and wall-clock time.
Raises BudgetExhaustedError the moment any limit is breached.
All limits are configurable via BudgetConfig.

Design: The budget manager is the safety net that prevents runaway loops.
It is called by the loop controller at every tick point.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class BudgetConfig:
    """Configurable resource limits for an agent run."""
    max_iterations: int = 20          # Complete loop cycles
    max_tool_calls: int = 30          # Individual tool executions (+ direct answers)
    max_llm_calls: int = 50           # Total calls to the LLM client
    max_retries: int = 9              # Retry attempts across all goals
    max_execution_time: float = 300.0 # Wall-clock seconds for entire run


class BudgetExhaustedError(Exception):
    """Raised when any configured budget limit is exceeded."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Budget exhausted: {reason}")


@dataclass
class BudgetManager:
    """
    Tracks resource consumption and enforces hard limits.

    Usage pattern:
        budget.tick_iteration()   # call at the top of each loop cycle
        budget.tick_tool_call()   # call before each tool execution
        budget.tick_llm_call()    # call before each LLM request
        budget.tick_retry()       # call on each retry attempt

    Any tick raises BudgetExhaustedError if the corresponding limit is hit.
    """
    config: BudgetConfig = field(default_factory=BudgetConfig)
    iterations: int = 0
    tool_calls: int = 0
    llm_calls: int = 0
    retries: int = 0
    _start_time: float = field(default_factory=time.monotonic, init=False, repr=False)

    def __post_init__(self) -> None:
        self._start_time = time.monotonic()

    def tick_iteration(self) -> None:
        """Called at the start of each main loop cycle."""
        self.iterations += 1
        self._enforce()

    def tick_tool_call(self) -> None:
        """Called before each tool dispatch (including direct answers)."""
        self.tool_calls += 1
        self._enforce()

    def tick_llm_call(self) -> None:
        """Called before each LLM request."""
        self.llm_calls += 1
        self._enforce()

    def tick_retry(self) -> None:
        """Called each time a goal attempt is retried."""
        self.retries += 1
        self._enforce()

    def elapsed(self) -> float:
        """Elapsed wall-clock time in seconds since agent started."""
        return time.monotonic() - self._start_time

    def remaining(self) -> dict[str, float | int]:
        """How much budget is left on each dimension."""
        return {
            "iterations": self.config.max_iterations - self.iterations,
            "tool_calls": self.config.max_tool_calls - self.tool_calls,
            "llm_calls": self.config.max_llm_calls - self.llm_calls,
            "retries": self.config.max_retries - self.retries,
            "time_seconds": self.config.max_execution_time - self.elapsed(),
        }

    def summary(self) -> dict:
        """Usage summary dict (logged at end of run)."""
        return {
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "llm_calls": self.llm_calls,
            "retries": self.retries,
            "elapsed_seconds": round(self.elapsed(), 2),
        }

    def is_ok(self) -> bool:
        """Returns True if no budget limit has been reached yet."""
        try:
            self._enforce()
            return True
        except BudgetExhaustedError:
            return False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _enforce(self) -> None:
        """Check all limits. Raise immediately if any is exceeded."""
        if self.iterations > self.config.max_iterations:
            raise BudgetExhaustedError(
                f"Max iterations ({self.config.max_iterations}) exceeded "
                f"(used {self.iterations})."
            )
        if self.tool_calls > self.config.max_tool_calls:
            raise BudgetExhaustedError(
                f"Max tool calls ({self.config.max_tool_calls}) exceeded "
                f"(used {self.tool_calls})."
            )
        if self.llm_calls > self.config.max_llm_calls:
            raise BudgetExhaustedError(
                f"Max LLM calls ({self.config.max_llm_calls}) exceeded "
                f"(used {self.llm_calls})."
            )
        if self.retries > self.config.max_retries:
            raise BudgetExhaustedError(
                f"Max retries ({self.config.max_retries}) exceeded "
                f"(used {self.retries})."
            )
        elapsed = self.elapsed()
        if elapsed > self.config.max_execution_time:
            raise BudgetExhaustedError(
                f"Max execution time ({self.config.max_execution_time}s) exceeded "
                f"({elapsed:.1f}s elapsed)."
            )
