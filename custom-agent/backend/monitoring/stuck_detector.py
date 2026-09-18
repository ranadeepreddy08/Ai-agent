"""
Stuck Detector — Phase 4 standalone component.

Detects when the agent is making no useful progress and provides
a structured signal to the loop controller.

Three detection modes:
  1. No-completion stagnation: N iterations with zero new goal completions
  2. Repeated tool failure: same tool fails on same goal K times in a row
  3. Repeated observation: identical output produced multiple consecutive times
     (indicates tool is stuck in a rut returning the same useless answer)

Design:
  - Pure Python, no LLM calls, zero side effects.
  - Called each iteration by the loop controller.
  - Returns a StuckDetectionResult — caller decides how to react.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from backend.core.state import AgentState, GoalStatus


class StuckReason(str, Enum):
    NOT_STUCK              = "not_stuck"
    NO_COMPLETION_PROGRESS = "no_completion_progress"  # No new completions in N iters
    REPEATED_TOOL_FAILURE  = "repeated_tool_failure"    # Same tool fails K times on same goal
    REPEATED_OBSERVATION   = "repeated_observation"     # Same output returned multiple times
    BUDGET_APPROACHING     = "budget_approaching"       # < 20% budget remaining


@dataclass
class StuckDetectionResult:
    is_stuck: bool
    reason: StuckReason
    detail: str = ""
    iterations_stagnant: int = 0


class StuckDetector:
    """
    Detects lack of useful progress in the agent loop.

    Instantiate once per agent run and call check() each iteration.
    Maintains internal counters to track progress across iterations.
    """

    def __init__(
        self,
        max_stagnant_iterations: int = 4,
        max_repeated_tool_failures: int = 3,
        max_repeated_observations: int = 3,
    ) -> None:
        self._max_stagnant = max_stagnant_iterations
        self._max_repeated_failures = max_repeated_tool_failures
        self._max_repeated_obs = max_repeated_observations

        # Internal counters
        self._last_completed_count: int = 0
        self._stagnant_iterations: int = 0
        self._obs_fingerprints: list[str] = []   # Recent observation fingerprints

    def check(self, state: AgentState) -> StuckDetectionResult:
        """
        Run all stuck checks for the current state.
        Call once per loop iteration AFTER goal processing.

        Returns the first StuckDetectionResult with is_stuck=True,
        or a NOT_STUCK result if all checks pass.
        """
        # ── 1. No completion progress ────────────────────────────────────────
        current_completed = len(state.completed_goal_ids())
        if current_completed > self._last_completed_count:
            self._last_completed_count = current_completed
            self._stagnant_iterations = 0
        else:
            # Only count stagnation if there are still pending goals to complete
            pending = any(g.status == GoalStatus.PENDING for g in state.goals)
            if pending:
                self._stagnant_iterations += 1
            else:
                self._stagnant_iterations = 0

        if self._stagnant_iterations >= self._max_stagnant:
            return StuckDetectionResult(
                is_stuck=True,
                reason=StuckReason.NO_COMPLETION_PROGRESS,
                detail=(
                    f"No new goal completed in {self._stagnant_iterations} iterations."
                ),
                iterations_stagnant=self._stagnant_iterations,
            )

        # ── 2. Repeated tool failure on same goal ────────────────────────────
        for goal in state.goals:
            if goal.status in (GoalStatus.RUNNING, GoalStatus.PENDING):
                goal_obs = state.observations_for_goal(goal.id)
                if len(goal_obs) >= self._max_repeated_failures:
                    recent = goal_obs[-self._max_repeated_failures:]
                    all_same_fail = (
                        all(not o.success for o in recent)
                        and len({o.tool_name for o in recent}) == 1
                    )
                    if all_same_fail:
                        tool = recent[0].tool_name
                        return StuckDetectionResult(
                            is_stuck=True,
                            reason=StuckReason.REPEATED_TOOL_FAILURE,
                            detail=(
                                f"Goal '{goal.id}': tool '{tool}' failed "
                                f"{self._max_repeated_failures} consecutive times."
                            ),
                        )

        # ── 3. Repeated identical observations ───────────────────────────────
        if len(state.observations) >= self._max_repeated_obs:
            recent_obs = state.observations[-self._max_repeated_obs:]
            fingerprints = [
                f"{o.tool_name}:{str(o.output)[:100]}"
                for o in recent_obs if o.success
            ]
            if (
                len(fingerprints) >= self._max_repeated_obs
                and len(set(fingerprints)) == 1
            ):
                return StuckDetectionResult(
                    is_stuck=True,
                    reason=StuckReason.REPEATED_OBSERVATION,
                    detail=(
                        f"Identical observation repeated {self._max_repeated_obs} times: "
                        f"{fingerprints[0][:80]}"
                    ),
                )

        return StuckDetectionResult(
            is_stuck=False,
            reason=StuckReason.NOT_STUCK,
            iterations_stagnant=self._stagnant_iterations,
        )

    def reset_stagnation(self) -> None:
        """Call after recovery/replan to reset stagnation counter."""
        self._stagnant_iterations = 0
