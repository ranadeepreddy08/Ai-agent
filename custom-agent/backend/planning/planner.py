"""
Planner — single-LLM-call task decomposition.

One LLM call does everything: complexity analysis, task decomposition,
dependency identification, priority assignment, and strategy selection.

Design rationale: Separate LLM calls for "analyze", "decompose", "prioritize"
would triple API costs for no improvement. A well-crafted single prompt with
structured JSON output captures all planning information in one round-trip.
"""
from __future__ import annotations

from typing import Any

from backend.core.state import AgentState, Goal, GoalStatus
from backend.llm.client import LLMClient
from backend.monitoring.logger import AgentLogger

# ── System prompt for the planner LLM call ────────────────────────────────────
_PLANNER_SYSTEM = """You are a task planning module for an AI agent runtime.
Analyze the user's task and produce a structured execution plan.

Return ONLY valid JSON — no explanation, no markdown:
{
  "complexity": "simple" | "complex",
  "reasoning": "one-sentence explanation of your analysis",
  "strategy": "direct" | "sequential" | "parallel",
  "goals": [
    {
      "id": "g1",
      "description": "Specific, actionable description of one unit of work",
      "priority": 1,
      "dependencies": []
    }
  ]
}

Strategy rules:
  "direct"     — The task requires only a single action or direct LLM reasoning; one goal.
  "sequential" — Goals must run in order because later goals need earlier results; use dependencies.
  "parallel"   — Goals are independent and can run concurrently; leave dependencies empty.

Goal rules:
  - Use lowercase IDs: "g1", "g2", "g3", ...
  - Keep each goal ATOMIC — ideally one tool call per goal.
  - "priority" is 1 (highest) to 5 (lowest). Sort goals by natural execution order.
  - "dependencies" lists goal IDs that MUST complete before this goal can start.
  - If a goal requires a previous goal's output, add the previous goal's ID to dependencies.

Task classification:
  - Pure arithmetic or well-known facts → "direct", one goal, no tools
  - Web lookup + no calculation → "direct" or "sequential", one or two goals
  - Calculation + web lookup (independent) → "parallel", two goals
  - Multi-step research (step B needs step A's result) → "sequential" with dependencies

Examples of good goals:
  "Search for current high-yield savings account interest rates"
  "Calculate compound interest: 10000 * (1 + 0.05)^3"
  "Find the current price of gold per ounce"

Do NOT create goals like "research and then calculate" — split into separate atomic goals.
"""


class Planner:
    """
    Single-LLM-call task planner.

    Takes the user task string and populates state.goals with a prioritized,
    dependency-aware list of Goal objects ready for the loop controller.
    """

    def __init__(self, llm: LLMClient, logger: AgentLogger) -> None:
        self._llm = llm
        self._logger = logger

    def plan(self, state: AgentState) -> AgentState:
        """
        Decompose the task into goals and update state.
        A single LLM call handles all planning logic.
        """
        self._logger.planning(state.task)

        user_prompt = f'Task: "{state.task}"'

        try:
            plan_data = self._llm.complete_json(_PLANNER_SYSTEM, user_prompt)
        except Exception as exc:
            self._logger.warn(f"Planner LLM call failed ({exc}). Using single fallback goal.")
            plan_data = _fallback_plan(state.task)

        goals = self._parse_goals(plan_data)
        state.goals = goals
        state.execution_strategy = plan_data.get("strategy", "sequential")

        self._logger.goals_created(
            goals,
            state.execution_strategy,
            plan_data.get("reasoning", ""),
        )
        return state

    # ── Private helpers ───────────────────────────────────────────────────────

    def _parse_goals(self, plan_data: dict[str, Any]) -> list[Goal]:
        """Convert raw LLM JSON into validated Goal objects."""
        raw_goals = plan_data.get("goals", [])
        if not raw_goals:
            # Defensive: always have at least one goal
            raw_goals = [{"id": "g1", "description": plan_data.get("task", "Complete task"), "priority": 1, "dependencies": []}]

        goals: list[Goal] = []
        seen_ids: set[str] = set()

        for i, raw in enumerate(raw_goals):
            gid = str(raw.get("id", f"g{i+1}")).strip()
            # Deduplicate IDs (LLM sometimes returns duplicates)
            if gid in seen_ids:
                gid = f"{gid}_{i}"
            seen_ids.add(gid)

            goal = Goal(
                id=gid,
                description=str(raw.get("description", "")).strip() or "Complete assigned task",
                priority=int(raw.get("priority", 1)),
                dependencies=[str(d) for d in raw.get("dependencies", [])],
                status=GoalStatus.PENDING,
            )
            goals.append(goal)

        # Sort by priority so the loop controller processes higher-priority goals first
        goals.sort(key=lambda g: g.priority)
        return goals


def _fallback_plan(task: str) -> dict[str, Any]:
    """Minimal single-goal plan used when the planner LLM call fails."""
    return {
        "complexity": "simple",
        "reasoning": "Fallback plan due to planner failure.",
        "strategy": "direct",
        "goals": [
            {
                "id": "g1",
                "description": task,
                "priority": 1,
                "dependencies": [],
            }
        ],
    }
