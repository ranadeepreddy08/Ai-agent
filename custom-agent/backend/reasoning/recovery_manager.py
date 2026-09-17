"""
Recovery Manager — Phase 3 dynamic failure analysis and strategy selection.

When a goal fails after exhausting its attempt budget, the RecoveryManager:
  1. Classifies the failure type from the observation error and metadata.
  2. Asks the LLM to pick a recovery strategy from a structured set.
  3. Returns a RecoveryDecision the LoopController acts on.

Recovery strategies (LLM chooses):
  retry          — Retry the same goal with the same tool (e.g. transient error)
  modify_input   — Retry with a rephrased/modified tool input
  switch_tool    — Try a different registered tool for the same goal
  fallback       — Accept a partial/degraded result and mark goal done
  replan         — Invoke Replanner to inject new goals and modify the plan
  terminate      — Goal is unrecoverable; fail it and continue with dependents if any

Design constraints:
  - Strategy selection is LLM-driven → NOT hardcoded if/else keyword routing.
  - RecoveryManager never executes tools itself; it only recommends.
  - LoopController reads the decision and acts.
  - RecoveryManager is stateless per call (no internal counters).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from backend.core.state import AgentState, Goal, Observation, GoalStatus
from backend.llm.client import LLMClient, LLMError
from backend.monitoring.logger import AgentLogger
from backend.tools.registry import ToolRegistry


# ── Recovery strategy enumeration ─────────────────────────────────────────────

class RecoveryStrategy(str, Enum):
    RETRY         = "retry"         # Same goal, same tool, same input
    MODIFY_INPUT  = "modify_input"  # Same goal, same tool, different input
    SWITCH_TOOL   = "switch_tool"   # Same goal, different tool
    FALLBACK      = "fallback"      # Accept degraded/partial result
    REPLAN        = "replan"        # Invoke Replanner to inject recovery goals
    TERMINATE     = "terminate"     # Unrecoverable — fail the goal


@dataclass
class RecoveryDecision:
    """
    Output of RecoveryManager.decide().

    The LoopController reads this and applies the strategy.
    """
    strategy: RecoveryStrategy
    reasoning: str                            # LLM's one-sentence justification
    modified_input: dict[str, Any] | None = None   # For MODIFY_INPUT
    suggested_tool: str | None = None         # For SWITCH_TOOL
    fallback_result: str | None = None        # For FALLBACK (synthetic result text)
    is_deterministic: bool = False            # True if no LLM was needed


# ── Failure type classification ────────────────────────────────────────────────

class FailureType(str, Enum):
    TOOL_ERROR       = "tool_error"       # Tool raised / returned error
    TIMEOUT          = "timeout"          # Tool timed out
    EMPTY_RESULT     = "empty_result"     # Tool succeeded but output was empty
    INVALID_RESULT   = "invalid_result"   # Tool output failed verification
    LLM_ERROR        = "llm_error"        # LLM call failed (action selection)
    UNKNOWN          = "unknown"          # Could not classify


def classify_failure(goal: Goal, obs: Observation | None) -> FailureType:
    """
    Deterministically classify the failure type from a failed goal's last observation.
    No LLM needed — pure Python logic on error messages and metadata.
    """
    if obs is None:
        if goal.error and "llm" in goal.error.lower():
            return FailureType.LLM_ERROR
        return FailureType.UNKNOWN

    if obs.metadata.get("timeout"):
        return FailureType.TIMEOUT

    if not obs.success:
        error_lower = (obs.error or "").lower()
        if any(k in error_lower for k in ("timeout", "timed out", "deadline")):
            return FailureType.TIMEOUT
        if any(k in error_lower for k in ("not found", "404", "missing", "no such")):
            return FailureType.TOOL_ERROR
        return FailureType.TOOL_ERROR

    # Tool succeeded but verifier said INVALID/INCOMPLETE
    if obs.output is None or str(obs.output).strip() in ("", "{}", "[]"):
        return FailureType.EMPTY_RESULT

    return FailureType.INVALID_RESULT


# ── Recovery system prompt ─────────────────────────────────────────────────────

_RECOVERY_SYSTEM = """You are the recovery manager for an AI agent runtime.

A goal has failed. Analyze the failure and choose the best recovery strategy.

Available strategies:
  retry         — Retry with same tool + input (use ONLY for transient errors)
  modify_input  — Retry with the same tool but a different/rephrased input
  switch_tool   — Try a completely different tool to achieve the same goal
  fallback      — Accept a partial/approximated result and move on
  replan        — Create new sub-goals to work around the failure
  terminate     — This goal cannot be recovered; accept the failure

Return ONLY valid JSON:
{
  "strategy": "<one of: retry | modify_input | switch_tool | fallback | replan | terminate>",
  "reasoning": "one-sentence justification",
  "modified_input": null | { <new tool input fields> },
  "suggested_tool": null | "<exact tool name from available_tools>",
  "fallback_result": null | "<brief fallback answer text if strategy=fallback>"
}

Rules:
  - modified_input is REQUIRED when strategy=modify_input.
  - suggested_tool is REQUIRED when strategy=switch_tool; must be from available_tools.
  - fallback_result should be set when strategy=fallback.
  - Choose retry ONLY for transient errors (connection, rate limit).
  - Choose terminate when all reasonable alternatives are exhausted.
  - Do NOT suggest retrying if the goal already has 3+ attempts.
"""


class RecoveryManager:
    """
    LLM-driven recovery strategy selector.

    Called by LoopController after a goal exhausts its retry budget.

    Usage:
        decision = recovery_manager.decide(goal, state, last_obs)
        # LoopController then acts on decision.strategy
    """

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        logger: AgentLogger,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._logger = logger

    def decide(
        self,
        goal: Goal,
        state: AgentState,
        last_obs: Observation | None,
    ) -> RecoveryDecision:
        """
        Analyze the failure and return a RecoveryDecision.

        Always returns a valid decision — never raises.
        Falls back to TERMINATE if LLM call fails.
        """
        failure_type = classify_failure(goal, last_obs)
        self._logger.info(
            f"RecoveryManager: goal '{goal.id}' failed "
            f"(type={failure_type.value}, attempts={goal.attempts})"
        )

        # ── Fast deterministic short-circuits ─────────────────────────────────
        # If all tools have been tried or we have too many attempts, terminate early
        available = self._registry.names()
        if goal.attempts >= 5:
            return RecoveryDecision(
                strategy=RecoveryStrategy.TERMINATE,
                reasoning=f"Goal exhausted {goal.attempts} attempts — terminating.",
                is_deterministic=True,
            )

        # LLM call failures → retry once, then terminate
        if failure_type == FailureType.LLM_ERROR and goal.attempts <= 1:
            return RecoveryDecision(
                strategy=RecoveryStrategy.RETRY,
                reasoning="LLM call failed transiently — retrying once.",
                is_deterministic=True,
            )

        # ── LLM-driven strategy selection ──────────────────────────────────────
        context = self._build_context(goal, state, last_obs, failure_type, available)
        user_prompt = json.dumps(context, indent=2, default=str)

        try:
            response = self._llm.complete_json(_RECOVERY_SYSTEM, user_prompt)
        except LLMError as exc:
            self._logger.warn(f"RecoveryManager LLM call failed: {exc}. Defaulting to TERMINATE.")
            return RecoveryDecision(
                strategy=RecoveryStrategy.TERMINATE,
                reasoning=f"Recovery LLM call failed: {exc}",
                is_deterministic=True,
            )

        # Parse strategy
        raw_strategy = str(response.get("strategy", "terminate")).lower().strip()
        try:
            strategy = RecoveryStrategy(raw_strategy)
        except ValueError:
            self._logger.warn(f"RecoveryManager: unknown strategy '{raw_strategy}', defaulting to TERMINATE.")
            strategy = RecoveryStrategy.TERMINATE

        # Validate tool for switch_tool
        suggested_tool = response.get("suggested_tool")
        if strategy == RecoveryStrategy.SWITCH_TOOL:
            if not suggested_tool or suggested_tool not in available:
                self._logger.warn(
                    f"RecoveryManager: switch_tool requested but tool '{suggested_tool}' not in registry. "
                    "Falling back to REPLAN."
                )
                strategy = RecoveryStrategy.REPLAN
                suggested_tool = None

        decision = RecoveryDecision(
            strategy=strategy,
            reasoning=str(response.get("reasoning", "LLM recovery decision.")),
            modified_input=response.get("modified_input") if isinstance(response.get("modified_input"), dict) else None,
            suggested_tool=suggested_tool,
            fallback_result=response.get("fallback_result"),
            is_deterministic=False,
        )

        self._logger.recovery_triggered(
            f"Strategy={decision.strategy.value} for '{goal.id}' — {decision.reasoning}"
        )
        return decision

    # ── Context builder ────────────────────────────────────────────────────────

    def _build_context(
        self,
        goal: Goal,
        state: AgentState,
        last_obs: Observation | None,
        failure_type: FailureType,
        available_tools: list[str],
    ) -> dict[str, Any]:
        """Build the context dict sent to the recovery LLM."""
        failed_obs_list = [
            {
                "attempt": i + 1,
                "tool": o.tool_name,
                "success": o.success,
                "output_snippet": str(o.output)[:200] if o.output else None,
                "error": o.error,
                "metadata": o.metadata,
            }
            for i, o in enumerate(state.observations_for_goal(goal.id))
        ]

        return {
            "original_task": state.task,
            "failed_goal": {
                "id": goal.id,
                "description": goal.description,
                "attempts": goal.attempts,
                "failure_type": failure_type.value,
                "last_error": goal.error,
                "dependencies_completed": goal.dependencies,
            },
            "failed_observations": failed_obs_list,
            "available_tools": [
                self._registry.get(t).to_registry_entry()
                for t in available_tools
            ],
            "completed_goals_count": len(state.completed_goal_ids()),
            "instruction": (
                "Choose the most appropriate recovery strategy. "
                "Prefer replan or switch_tool over retry unless the failure is clearly transient."
            ),
        }
