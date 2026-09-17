"""
Context Manager — builds minimal, relevant LLM context.

Design principle: Never send the full conversation history to the LLM.
Instead, select only what's relevant to the current decision:
  - Current goal
  - Tool catalogue
  - Previous successful results (capped)
  - Recent observations for this goal
  - Recent failures (for recovery context)

This keeps prompt sizes predictable and avoids distraction from irrelevant history.
"""
from __future__ import annotations

from typing import Any

from backend.core.state import AgentState, Goal, GoalStatus
from backend.tools.registry import ToolRegistry


class ContextManager:
    """
    Assembles minimal, decision-relevant context for each LLM call.

    Two context-building methods:
      build_tool_selection_context() — for the action selection LLM call
      build_synthesis_context()      — for the final answer synthesis call
    """

    # Tuneable limits to keep prompts focused
    _MAX_COMPLETED_RESULTS = 5     # Max completed goals to include
    _MAX_OBSERVATIONS_PER_GOAL = 3  # Recent observations for current goal
    _MAX_RESULT_CHARS = 400        # Truncate long results
    _MAX_FAILED_ATTEMPTS = 3       # Recent failures to show for recovery context

    def build_tool_selection_context(
        self,
        state: AgentState,
        goal: Goal,
        registry: ToolRegistry,
    ) -> dict[str, Any]:
        """
        Context for the tool selection LLM call.

        Contains:
          - Overall task (for grounding)
          - Current goal details and attempt number
          - Full tool catalogue (for selection)
          - Completed goals and their result snippets
          - Recent observations for this specific goal
          - Recent failures for this goal (recovery context)
        """
        return {
            "task": state.task,
            "current_goal": {
                "id": goal.id,
                "description": goal.description,
                "attempt_number": goal.attempts + 1,
            },
            "available_tools": registry.list_for_llm(),
            "completed_goals": self._summarize_completed(state),
            "recent_observations_for_goal": self._recent_obs(state, goal.id),
            "recent_failures_for_goal": self._recent_failures(state, goal.id),
        }

    def build_synthesis_context(self, state: AgentState) -> dict[str, Any]:
        """
        Context for the final answer synthesis call.

        Contains all completed goal results and any failures
        (to allow the synthesizer to acknowledge gaps).
        """
        completed = [
            {
                "goal_id": g.id,
                "description": g.description,
                "result": self._truncate(str(g.result), 2000),
                "verification": g.verification_status.value,
            }
            for g in state.goals
            if g.status == GoalStatus.COMPLETED and g.result is not None
        ]
        failed = [
            {
                "goal_id": g.id,
                "description": g.description,
                "error": g.error or "unknown error",
            }
            for g in state.goals
            if g.status == GoalStatus.FAILED
        ]
        return {
            "task": state.task,
            "completed_goals": completed,
            "failed_goals": failed,
            "total_iterations": state.iterations,
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _summarize_completed(self, state: AgentState) -> list[dict]:
        completed = [
            g for g in state.goals
            if g.status == GoalStatus.COMPLETED and g.result is not None
        ]
        # Most recent first, capped
        recent = completed[-self._MAX_COMPLETED_RESULTS:]
        return [
            {
                "id": g.id,
                "description": g.description,
                "result_snippet": self._truncate(str(g.result), self._MAX_RESULT_CHARS),
            }
            for g in recent
        ]

    def _recent_obs(self, state: AgentState, goal_id: str) -> list[dict]:
        obs = [o for o in state.observations if o.goal_id == goal_id]
        recent = obs[-self._MAX_OBSERVATIONS_PER_GOAL:]
        return [
            {
                "tool": o.tool_name,
                "success": o.success,
                "output_snippet": (
                    self._truncate(str(o.output), self._MAX_RESULT_CHARS)
                    if o.success and o.output else None
                ),
                "error": o.error,
                "verification": o.verification_status.value,
            }
            for o in recent
        ]

    def _recent_failures(self, state: AgentState, goal_id: str) -> list[dict]:
        failed_obs = [
            o for o in state.observations
            if o.goal_id == goal_id and not o.success
        ]
        return [
            {"tool": o.tool_name, "error": o.error}
            for o in failed_obs[-self._MAX_FAILED_ATTEMPTS:]
        ]

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        return text[:max_len] + "…" if len(text) > max_len else text
