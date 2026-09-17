"""
Loop Controller — the custom agent runtime core.

Implements the PLAN → ACT → OBSERVE → VERIFY → RECOVER loop.

This is the most important file in the framework. Every step of the agent loop
is explicit Python code — no external framework, no hidden abstractions.

Loop cycle:
  1. Pick the highest-priority ready goal.
  2. Build minimal context (ContextManager).
  3. Ask LLM to select an action: tool_call or direct_answer.
  4. Execute the action (ToolExecutor).
  5. Record the Observation in state.
  6. Verify: does the observation satisfy the goal? (Verifier)
  7a. VALID → mark goal COMPLETED, continue to next goal.
  7b. INCOMPLETE/INVALID → recovery: retry with same or different approach.
  7c. Exhausted attempts → mark goal FAILED.
  8. Check termination: all goals terminal, or budget exhausted.

Recovery (Phase 1):
  - Retry up to MAX_GOAL_ATTEMPTS times per goal.
  - The LLM sees previous failed observations in context → naturally adapts.
  - Phase 3 will add explicit tool-switching and replanning logic.
"""
from __future__ import annotations

import json
from typing import Any

from backend.context.manager import ContextManager
from backend.core.state import (
    AgentState,
    AgentStatus,
    Goal,
    GoalStatus,
    Observation,
    VerificationStatus,
)
from backend.execution.executor import ToolExecutor
from backend.llm.client import LLMClient, LLMError
from backend.monitoring.budget import BudgetExhaustedError, BudgetManager
from backend.monitoring.logger import AgentLogger
from backend.reasoning.verifier import Verifier
from backend.tools.registry import ToolRegistry

# ── Tool selection prompt ─────────────────────────────────────────────────────
_ACTION_SELECTION_SYSTEM = """You are the action selection module of an AI agent runtime.

Given the current goal and available tools, decide the NEXT action to take.

You MUST return ONLY valid JSON in one of these two formats:

Option A — Use a tool:
{
  "action": "tool_call",
  "tool": "<exact tool name from available_tools list>",
  "input": { <input matching the tool's input_schema> },
  "goal_id": "<current goal id>",
  "reasoning": "why this tool and input"
}

Option B — Answer directly (no tool needed):
{
  "action": "direct_answer",
  "answer": "<complete answer text>",
  "goal_id": "<current goal id>",
  "reasoning": "why no tool is needed"
}

Rules:
  1. ONLY use tool names from the "available_tools" list. Never invent names.
  2. Choose the tool whose "capabilities" best match the goal.
  3. If previous attempts for this goal failed, learn from them:
     - Try a different tool if the previous one failed.
     - Rephrase the input if the tool worked but returned a bad result.
  4. Use "direct_answer" ONLY when you can confidently answer from knowledge
     (e.g. pure math reasoning, well-known facts). When unsure, use a tool.
  5. For calculator: "input" MUST have an "expression" key (math expression string).
  6. For web_search: "input" MUST have a "query" key (search query string).
  7. The "goal_id" in your response MUST match the current_goal.id exactly.
"""

# ── Synthesis prompt ──────────────────────────────────────────────────────────
_SYNTHESIS_SYSTEM = """You are the response synthesizer for an AI agent runtime.

Given the original task and the results from completed goals, write a clear,
accurate, and concise final answer for the user.

Rules:
  - Integrate all relevant results naturally.
  - Be specific: include numbers, names, and facts from the results.
  - If some goals failed, acknowledge what could not be determined.
  - Do NOT fabricate information not present in the goal results.
  - Use plain, readable prose. Format numbers clearly.
"""


class LoopController:
    """
    The custom agent runtime — PLAN→ACT→OBSERVE→VERIFY loop.

    All agent orchestration logic lives here as explicit Python.
    No external framework, no hidden magic.
    """

    MAX_GOAL_ATTEMPTS = 3  # Retries per goal before marking FAILED

    def __init__(
        self,
        llm: LLMClient,
        tool_registry: ToolRegistry,
        executor: ToolExecutor,
        verifier: Verifier,
        context_manager: ContextManager,
        budget_manager: BudgetManager,
        logger: AgentLogger,
    ) -> None:
        self._llm = llm
        self._registry = tool_registry
        self._executor = executor
        self._verifier = verifier
        self._ctx = context_manager
        self._budget = budget_manager
        self._logger = logger

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, state: AgentState) -> AgentState:
        """
        Execute the agent loop until termination.

        Termination conditions:
          - All goals are in terminal status (COMPLETED / FAILED / SKIPPED)
          - BudgetExhaustedError is raised by BudgetManager
          - Unexpected exception (caught, logged, status set to FAILED)
        """
        state.status = AgentStatus.RUNNING

        try:
            while not self._should_stop(state):
                state.iterations += 1
                self._budget.tick_iteration()

                # ── Pick next ready goal ──────────────────────────────────
                ready = state.ready_goals()

                if not ready:
                    if state.pending_goals():
                        # Pending goals exist but none are ready — dependency deadlock
                        self._logger.warn(
                            "No ready goals despite pending goals — possible dependency cycle."
                        )
                        self._fail_blocked_goals(state)
                    break  # Nothing left to do

                # Process highest-priority ready goal
                goal = min(ready, key=lambda g: g.priority)
                goal.status = GoalStatus.RUNNING
                self._logger.goal_start(goal.id, goal.description)

                # ── ACT → OBSERVE → VERIFY cycle ─────────────────────────
                self._process_goal(goal, state)

        except BudgetExhaustedError as exc:
            self._logger.error(f"Budget exhausted: {exc.reason}")
            state.status = AgentStatus.BUDGET_EXHAUSTED
            return state

        except Exception as exc:
            self._logger.error(f"Unexpected loop error: {type(exc).__name__}: {exc}")
            state.status = AgentStatus.FAILED
            return state

        # ── Set final status ──────────────────────────────────────────────
        if state.status == AgentStatus.RUNNING:
            completed = sum(1 for g in state.goals if g.status == GoalStatus.COMPLETED)
            if completed > 0:
                state.status = AgentStatus.COMPLETED
            else:
                state.status = AgentStatus.FAILED

        return state

    # ── Goal processing ───────────────────────────────────────────────────────

    def _process_goal(self, goal: Goal, state: AgentState) -> None:
        """
        ACT → OBSERVE → VERIFY cycle for a single goal.

        Retries up to MAX_GOAL_ATTEMPTS times. On each retry the LLM
        receives the history of previous failed observations in context,
        allowing it to naturally adapt its tool choice or inputs.
        """
        while goal.attempts < self.MAX_GOAL_ATTEMPTS:
            goal.attempts += 1

            # Build context (minimal, relevant)
            context = self._ctx.build_tool_selection_context(state, goal, self._registry)

            # ── ACT: Ask LLM to select an action ─────────────────────────
            action = self._select_action(context, state)

            if action is None:
                self._logger.error(f"Action selection failed for goal {goal.id}. Will retry.")
                if goal.attempts >= self.MAX_GOAL_ATTEMPTS:
                    break
                self._budget.tick_retry()
                continue

            # Log the selected action
            action_type = action.get("action", "?")
            if action_type == "tool_call":
                self._logger.tool_selected(
                    action.get("tool", "?"),
                    action.get("input", {}),
                )
            elif action_type == "direct_answer":
                self._logger.tool_selected("direct_answer", action.get("answer", "")[:80])

            # ── Handle direct answer (no tool needed) ─────────────────────
            if action_type == "direct_answer":
                answer_text = str(action.get("answer", "")).strip()
                if answer_text:
                    goal.result = answer_text
                    goal.status = GoalStatus.COMPLETED
                    goal.verification_status = VerificationStatus.VALID
                    obs = Observation(
                        goal_id=goal.id,
                        tool_name="direct_answer",
                        success=True,
                        output=answer_text,
                        error=None,
                        execution_time=0.0,
                        iteration=state.iterations,
                        verification_status=VerificationStatus.VALID,
                    )
                    state.observations.append(obs)
                    self._budget.tick_tool_call()
                    self._logger.observation_received(obs)
                    self._logger.goal_completed(goal.id)
                    return
                else:
                    # LLM returned empty direct_answer — treat as failure
                    self._logger.warn(f"direct_answer was empty for goal {goal.id}, retrying.")
                    self._budget.tick_retry()
                    continue

            # ── OBSERVE: Execute the selected tool ─────────────────────────
            self._budget.tick_tool_call()
            obs = self._executor.execute(action, state)
            state.observations.append(obs)

            # ── VERIFY: Does the observation satisfy the goal? ─────────────
            verification = self._verifier.verify(goal, obs)
            obs.verification_status = verification.status
            goal.verification_status = verification.status

            if verification.status == VerificationStatus.VALID:
                # ✅ Goal satisfied
                goal.result = obs.output
                goal.status = GoalStatus.COMPLETED
                self._logger.goal_completed(goal.id)
                return

            elif verification.status == VerificationStatus.INCOMPLETE:
                # 🔄 Partial — retry (LLM will see previous obs in context)
                self._logger.recovery_triggered(
                    f"Incomplete result for [{goal.id}] — retrying (attempt {goal.attempts}/{self.MAX_GOAL_ATTEMPTS})"
                )
                if goal.attempts < self.MAX_GOAL_ATTEMPTS:
                    self._budget.tick_retry()
                    continue

            else:
                # INVALID / CONTRADICTORY / UNRELIABLE — retry with context
                self._logger.recovery_triggered(
                    f"Verification={verification.status.value} for [{goal.id}] "
                    f"— attempt {goal.attempts}/{self.MAX_GOAL_ATTEMPTS}"
                )
                if goal.attempts < self.MAX_GOAL_ATTEMPTS:
                    self._budget.tick_retry()
                    continue

            break  # Exhausted attempts inside loop

        # ── All attempts exhausted ─────────────────────────────────────────
        goal.status = GoalStatus.FAILED
        goal.error = (
            f"Failed after {goal.attempts} attempt(s). "
            f"Last verification: {goal.verification_status.value}"
        )
        self._logger.goal_failed(goal.id, goal.error)

    # ── Action selection ──────────────────────────────────────────────────────

    def _select_action(
        self, context: dict[str, Any], state: AgentState
    ) -> dict[str, Any] | None:
        """
        Ask the LLM to select the next action.

        Returns a parsed action dict, or None if the LLM call fails.
        The LLM receives the full tool catalogue and current context;
        it autonomously decides tool + input based on capabilities.
        """
        user_prompt = json.dumps(context, indent=2, default=str)
        try:
            self._budget.tick_llm_call()
            action = self._llm.complete_json(_ACTION_SELECTION_SYSTEM, user_prompt)
            # Validate minimal structure
            if "action" not in action:
                self._logger.warn("LLM action missing 'action' key. Retrying selection.")
                return None
            return action
        except LLMError as exc:
            self._logger.error(f"Action selection LLM call failed: {exc}")
            return None

    # ── Final answer synthesis ────────────────────────────────────────────────

    def synthesize_answer(self, state: AgentState) -> str:
        """
        Generate the final user-facing answer from completed goal results.

        Uses one LLM call to integrate all results coherently.
        Falls back to manual assembly if the LLM call fails.
        """
        self._logger.synthesizing()

        synthesis_context = self._ctx.build_synthesis_context(state)
        user_prompt = (
            f'Original task: "{state.task}"\n\n'
            f"Goal results:\n"
            f"{json.dumps(synthesis_context, indent=2, default=str)}"
        )

        try:
            self._budget.tick_llm_call()
            answer = self._llm.complete(_SYNTHESIS_SYSTEM, user_prompt)
            return answer.strip()
        except Exception as exc:
            self._logger.warn(f"Synthesis LLM call failed: {exc}. Using fallback assembly.")
            return self._fallback_assemble(state)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _should_stop(self, state: AgentState) -> bool:
        """True when the loop should exit."""
        if state.all_goals_terminal():
            return True
        if state.status in (AgentStatus.FAILED, AgentStatus.BUDGET_EXHAUSTED):
            return True
        return False

    def _fail_blocked_goals(self, state: AgentState) -> None:
        """Mark all remaining pending goals as FAILED due to dependency deadlock."""
        for goal in state.goals:
            if goal.status == GoalStatus.PENDING:
                goal.status = GoalStatus.FAILED
                goal.error = "Blocked: dependency goals never completed."
                self._logger.goal_failed(goal.id, goal.error)

    @staticmethod
    def _fallback_assemble(state: AgentState) -> str:
        """Manually assemble answer from goal results when synthesis LLM call fails."""
        parts: list[str] = []
        for goal in state.goals:
            if goal.status == GoalStatus.COMPLETED and goal.result:
                parts.append(f"[{goal.id}] {goal.description}:\n{goal.result}")
            elif goal.status == GoalStatus.FAILED:
                parts.append(f"[{goal.id}] Could not complete: {goal.description}")
        return "\n\n".join(parts) if parts else "No results available."
