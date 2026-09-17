"""
DAG Manager — explicit Goal dependency graph management.

Responsibilities:
  1. Validate the goal graph on creation (no unknown deps, no cycles).
  2. Compute topological execution order.
  3. Mark BLOCKED goals whose dependencies have failed.
  4. Detect dependency deadlocks and fail stranded goals.
  5. Surface ready goals (PENDING + all deps COMPLETED).

Design: Pure Python, no external libraries.
The LoopController calls into DAGManager to query what is executable next.
"""
from __future__ import annotations

from collections import deque
from typing import Iterator

from backend.core.state import AgentState, Goal, GoalStatus
from backend.monitoring.logger import AgentLogger


class DAGValidationError(Exception):
    """Raised when the goal graph is structurally invalid."""


class DAGManager:
    """
    Manages the goal dependency graph for an agent run.

    The DAG is defined by the `dependencies` field on each Goal.
    Execution order is determined by Kahn's algorithm (topological sort).

    Usage:
        dag = DAGManager(state.goals, logger)
        dag.validate()               # call once after planner sets goals
        for goal_id in dag.topo_order():
            ...
        dag.update_blocked(state)    # call each iteration
    """

    def __init__(self, goals: list[Goal], logger: AgentLogger) -> None:
        self._goals = goals
        self._logger = logger
        self._id_to_goal: dict[str, Goal] = {g.id: g for g in goals}

    # ── Validation ─────────────────────────────────────────────────────────────

    def validate(self) -> None:
        """
        Validate the dependency graph.

        Raises DAGValidationError on:
          - A goal that references a non-existent dependency ID
          - A cycle in the dependency graph
        """
        known_ids = set(self._id_to_goal.keys())

        # Check for unknown dependency IDs
        for goal in self._goals:
            for dep_id in goal.dependencies:
                if dep_id not in known_ids:
                    raise DAGValidationError(
                        f"Goal '{goal.id}' references unknown dependency '{dep_id}'. "
                        f"Known IDs: {sorted(known_ids)}"
                    )

        # Check for cycles (Kahn's algorithm)
        if self._has_cycle():
            raise DAGValidationError(
                "Goal dependency graph contains a cycle. "
                "Execution would deadlock. Goals: "
                + ", ".join(f"{g.id}←{g.dependencies}" for g in self._goals)
            )

    def _has_cycle(self) -> bool:
        """Return True if the dependency graph has a cycle (Kahn's in-degree check)."""
        in_degree: dict[str, int] = {g.id: 0 for g in self._goals}
        for goal in self._goals:
            for dep_id in goal.dependencies:
                in_degree[goal.id] = in_degree.get(goal.id, 0) + 1
                # We already validated dep_id exists above; safe to access

        # Recount properly
        in_degree = {g.id: len(g.dependencies) for g in self._goals}

        queue = deque(gid for gid, deg in in_degree.items() if deg == 0)
        visited = 0
        while queue:
            gid = queue.popleft()
            visited += 1
            # Find goals that depend on gid
            for goal in self._goals:
                if gid in goal.dependencies:
                    in_degree[goal.id] -= 1
                    if in_degree[goal.id] == 0:
                        queue.append(goal.id)

        return visited != len(self._goals)

    # ── Topological ordering ───────────────────────────────────────────────────

    def topo_order(self) -> list[str]:
        """
        Return goal IDs in valid topological execution order (Kahn's algorithm).

        Goals with lower priority numbers come first among peers at the same
        dependency level.
        """
        in_degree = {g.id: len(g.dependencies) for g in self._goals}
        # Use priority as tiebreaker — min-heap by (priority, id)
        queue: list[Goal] = sorted(
            [g for g in self._goals if in_degree[g.id] == 0],
            key=lambda g: (g.priority, g.id),
        )
        order: list[str] = []

        while queue:
            goal = queue.pop(0)
            order.append(goal.id)
            # Reduce in-degree for dependents
            for other in self._goals:
                if goal.id in other.dependencies:
                    in_degree[other.id] -= 1
                    if in_degree[other.id] == 0:
                        queue.append(other)
                        queue.sort(key=lambda g: (g.priority, g.id))

        return order

    # ── Live status management ─────────────────────────────────────────────────

    def update_blocked(self, state: AgentState) -> list[str]:
        """
        Scan all PENDING goals and mark those whose dependencies have FAILED
        as BLOCKED (they can never execute).

        Returns list of newly blocked goal IDs.
        """
        failed_ids = state.failed_goal_ids()
        newly_blocked: list[str] = []

        for goal in state.goals:
            if goal.status != GoalStatus.PENDING:
                continue
            # If any dependency has FAILED or is BLOCKED, this goal is deadlocked
            for dep_id in goal.dependencies:
                dep = state.get_goal(dep_id)
                if dep and dep.status in (GoalStatus.FAILED, GoalStatus.BLOCKED):
                    goal.status = GoalStatus.BLOCKED
                    goal.error = (
                        f"Blocked: dependency '{dep_id}' did not complete "
                        f"(status={dep.status.value})."
                    )
                    newly_blocked.append(goal.id)
                    self._logger.warn(
                        f"Goal '{goal.id}' BLOCKED — dependency '{dep_id}' "
                        f"is {dep.status.value}."
                    )
                    break

        return newly_blocked

    def ready_goals(self, state: AgentState) -> list[Goal]:
        """
        Return goals that are PENDING and have all dependencies COMPLETED,
        sorted by priority.
        """
        completed = state.completed_goal_ids()
        return sorted(
            [
                g for g in state.goals
                if g.status == GoalStatus.PENDING and g.is_ready(completed)
            ],
            key=lambda g: (g.priority, g.id),
        )

    def is_deadlocked(self, state: AgentState) -> bool:
        """
        True if there are PENDING goals but none are ready AND the cause
        is not a running goal (i.e., nothing can ever unblock them).
        """
        if not any(g.status == GoalStatus.PENDING for g in state.goals):
            return False
        if any(g.status == GoalStatus.RUNNING for g in state.goals):
            return False  # A running goal may complete and unblock others
        ready = self.ready_goals(state)
        return len(ready) == 0

    # ── Summaries ─────────────────────────────────────────────────────────────

    def summary(self, state: AgentState) -> dict:
        """Compact status summary for logging/display."""
        counts: dict[str, int] = {}
        for g in state.goals:
            counts[g.status.value] = counts.get(g.status.value, 0) + 1
        return {
            "total": len(state.goals),
            "topo_order": self.topo_order(),
            "status_counts": counts,
        }
