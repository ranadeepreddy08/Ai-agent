"""
Replanner — dynamic goal graph modification after goal failure.

When a goal fails, the Replanner asks the LLM whether recovery is possible
and, if so, what new goals to inject into the graph to replace or work around
the failed goal.

Design:
  - One focused LLM call per failed goal.
  - Returns a list of new Goal objects to append to state.goals.
  - Existing goals are NEVER modified (only new ones are appended).
  - The DAGManager is responsible for re-validating after injection.
  - If recovery is impossible or the LLM call fails, returns [] (caller keeps
    the goal FAILED and continues).
"""
from __future__ import annotations

import json
from typing import Any

from backend.core.state import AgentState, Goal, GoalStatus
from backend.llm.client import LLMClient, LLMError
from backend.monitoring.logger import AgentLogger

# ── Replanning prompt ─────────────────────────────────────────────────────────
_REPLAN_SYSTEM = """You are the recovery/replanning module of an AI agent runtime.

A goal has failed. Your job: decide whether recovery is possible and, if so,
propose replacement goals that can achieve the same objective via a different approach.

Return ONLY valid JSON:
{
  "recoverable": true | false,
  "reasoning": "one-sentence explanation",
  "new_goals": [
    {
      "id": "<new unique id, e.g. g1_retry or g4>",
      "description": "<specific, actionable description>",
      "priority": <int 1-5>,
      "dependencies": ["<existing or new goal ids>"]
    }
  ]
}

Rules:
  - If recoverable=false, set new_goals=[].
  - New goal IDs MUST NOT duplicate existing IDs in the plan.
  - Dependencies can reference existing COMPLETED goal IDs or other new goal IDs.
  - Keep new goals ATOMIC (one tool call each).
  - A goal is NOT recoverable if: it requires real-time data that a mock cannot provide,
    it hit a hard resource limit, or all reasonable alternatives are exhausted.
  - Do not propose more than 3 new goals.
"""


class Replanner:
    """
    Proposes recovery goals when a goal fails.

    Usage (called by LoopController on goal failure):
        new_goals = replanner.replan(failed_goal, state)
        if new_goals:
            state.goals.extend(new_goals)
            dag.validate()  # re-validate after injection
    """

    def __init__(self, llm: LLMClient, logger: AgentLogger) -> None:
        self._llm = llm
        self._logger = logger

    def replan(self, failed_goal: Goal, state: AgentState) -> list[Goal]:
        """
        Ask the LLM to propose recovery goals for a failed goal.

        Returns a (possibly empty) list of new Goal objects.
        Returns [] when recovery is not possible or the LLM call fails.
        """
        self._logger.info(
            f"Replanning: goal '{failed_goal.id}' failed — querying LLM for recovery options."
        )

        context = self._build_context(failed_goal, state)
        user_prompt = json.dumps(context, indent=2, default=str)

        try:
            response = self._llm.complete_json(_REPLAN_SYSTEM, user_prompt)
        except LLMError as exc:
            self._logger.warn(f"Replanner LLM call failed: {exc}. No recovery.")
            return []

        if not response.get("recoverable", False):
            self._logger.info(
                f"Replanner: goal '{failed_goal.id}' not recoverable. "
                f"Reason: {response.get('reasoning', 'unknown')}"
            )
            return []

        new_goals = self._parse_new_goals(response.get("new_goals", []), state)
        if new_goals:
            self._logger.info(
                f"Replanner injected {len(new_goals)} recovery goal(s) for '{failed_goal.id}': "
                + ", ".join(f"'{g.id}'" for g in new_goals)
            )
        return new_goals

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_context(self, failed_goal: Goal, state: AgentState) -> dict[str, Any]:
        """Build the context dict sent to the replanner LLM."""
        completed_summaries = [
            {"id": g.id, "description": g.description, "result_snippet": str(g.result)[:200]}
            for g in state.goals
            if g.status == GoalStatus.COMPLETED
        ]
        failed_obs = [
            {
                "tool": o.tool_name,
                "success": o.success,
                "output_snippet": str(o.output)[:200] if o.output else None,
                "error": o.error,
            }
            for o in state.observations_for_goal(failed_goal.id)
        ]
        existing_ids = [g.id for g in state.goals]

        return {
            "original_task": state.task,
            "failed_goal": {
                "id": failed_goal.id,
                "description": failed_goal.description,
                "attempts": failed_goal.attempts,
                "last_error": failed_goal.error,
                "failed_observations": failed_obs,
            },
            "completed_goals": completed_summaries,
            "existing_goal_ids": existing_ids,
            "instruction": (
                "Propose recovery goals using NEW IDs not in existing_goal_ids. "
                "Dependencies may reference IDs from existing_goal_ids (completed ones) "
                "or from new_goals within this response."
            ),
        }

    def _parse_new_goals(
        self, raw_goals: list[dict], state: AgentState
    ) -> list[Goal]:
        """Parse and deduplicate new goals from the LLM response."""
        existing_ids = {g.id for g in state.goals}
        result: list[Goal] = []
        seen_new: set[str] = set()

        for i, raw in enumerate(raw_goals):
            gid = str(raw.get("id", f"rg{i+1}")).strip()

            # Avoid ID collisions with existing goals or duplicates in this batch
            if gid in existing_ids or gid in seen_new:
                gid = f"{gid}_r{i}"
            seen_new.add(gid)

            goal = Goal(
                id=gid,
                description=str(raw.get("description", "Recovery goal")).strip(),
                priority=int(raw.get("priority", 2)),
                dependencies=[str(d) for d in raw.get("dependencies", [])],
                status=GoalStatus.PENDING,
            )
            result.append(goal)

        return result
