"""
Agent state — dataclasses and enumerations.

AgentState is the single source of truth passed through the entire agent loop.
All components READ from state; the loop controller WRITES to it.
No global state anywhere else in the codebase.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Enumerations ──────────────────────────────────────────────────────────────

class GoalStatus(str, Enum):
    PENDING = "PENDING"               # Not yet started
    RUNNING = "RUNNING"               # Currently executing
    COMPLETED = "COMPLETED"           # Successfully finished and verified
    FAILED = "FAILED"                 # Exhausted retries or unrecoverable error
    BLOCKED = "BLOCKED"               # Dependencies not yet satisfied
    NEEDS_VERIFICATION = "NEEDS_VERIFICATION"  # Executed, pending verification
    SKIPPED = "SKIPPED"               # Intentionally bypassed


class AgentStatus(str, Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class VerificationStatus(str, Enum):
    VALID = "VALID"               # Fully satisfies the goal
    INCOMPLETE = "INCOMPLETE"     # Partial result; more information needed
    CONTRADICTORY = "CONTRADICTORY"  # Conflicts with known/other information
    INVALID = "INVALID"           # Wrong, irrelevant, or errored
    UNRELIABLE = "UNRELIABLE"     # Source/method questionable
    SKIPPED = "SKIPPED"           # Verification was not performed


# ── Core dataclasses ──────────────────────────────────────────────────────────

@dataclass
class Goal:
    """
    A single unit of work within an agent run.

    Goals form a DAG: a goal is only eligible to execute once all goals
    listed in its `dependencies` list are COMPLETED.
    """
    id: str                              # Unique, e.g. "g1", "g2"
    description: str                     # What needs to be accomplished
    priority: int = 1                    # 1 = highest priority
    dependencies: list[str] = field(default_factory=list)  # Goal IDs that must complete first

    status: GoalStatus = GoalStatus.PENDING
    result: Any = None                   # Stored on COMPLETED
    error: str | None = None            # Stored on FAILED
    attempts: int = 0                    # Execution attempt count
    verification_status: VerificationStatus = VerificationStatus.SKIPPED
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_ready(self, completed_ids: set[str]) -> bool:
        """
        Returns True if all dependencies have been completed.
        A goal with no dependencies is always ready.
        """
        return all(dep in completed_ids for dep in self.dependencies)

    def __repr__(self) -> str:
        return f"<Goal {self.id}: {self.description[:40]}... [{self.status.value}]>"


@dataclass
class Observation:
    """
    Normalized record of one tool execution, stored in agent state.

    Every tool call produces exactly one Observation. The Verifier
    and ContextManager use observations to reason about goal progress.
    """
    goal_id: str
    tool_name: str
    success: bool
    output: Any
    error: str | None
    execution_time: float              # Seconds
    iteration: int                     # Which agent loop iteration produced this
    verification_status: VerificationStatus = VerificationStatus.SKIPPED
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """
    Central state object for an entire agent run.

    Passed by reference through all components. The loop controller
    is the primary writer; all other components read and suggest updates.
    """
    task: str
    goals: list[Goal] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    iterations: int = 0
    status: AgentStatus = AgentStatus.IDLE
    final_answer: str | None = None
    execution_strategy: str = "sequential"  # direct | sequential | parallel
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── Goal accessors ────────────────────────────────────────────────────────

    def get_goal(self, goal_id: str) -> Goal | None:
        """Return the goal with the given ID, or None."""
        for g in self.goals:
            if g.id == goal_id:
                return g
        return None

    def completed_goal_ids(self) -> set[str]:
        return {g.id for g in self.goals if g.status == GoalStatus.COMPLETED}

    def failed_goal_ids(self) -> set[str]:
        return {g.id for g in self.goals if g.status == GoalStatus.FAILED}

    def pending_goals(self) -> list[Goal]:
        return [g for g in self.goals if g.status == GoalStatus.PENDING]

    def ready_goals(self) -> list[Goal]:
        """Goals that are PENDING and have all dependencies completed."""
        completed = self.completed_goal_ids()
        return [
            g for g in self.goals
            if g.status == GoalStatus.PENDING and g.is_ready(completed)
        ]

    def all_goals_terminal(self) -> bool:
        """True if every goal is in a terminal state (no more work to do)."""
        terminal = {GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.SKIPPED, GoalStatus.BLOCKED}
        return bool(self.goals) and all(g.status in terminal for g in self.goals)

    # ── Observation accessors ─────────────────────────────────────────────────

    def observations_for_goal(self, goal_id: str) -> list[Observation]:
        return [o for o in self.observations if o.goal_id == goal_id]

    def last_observation(self) -> Observation | None:
        return self.observations[-1] if self.observations else None

    def successful_observations(self) -> list[Observation]:
        return [o for o in self.observations if o.success]
