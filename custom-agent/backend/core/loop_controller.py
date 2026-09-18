"""
Loop Controller â€” Phase 3: full reliability pipeline.

Phase 3 additions over Phase 2:
  1. RecoveryManager integration: after goal exhausts attempts, ask recovery manager
     for a strategy (retry/modify_input/switch_tool/fallback/replan/terminate).
  2. Strategy execution: LoopController acts on the RecoveryDecision.
  3. Stuck detection: tracks "last completed count"; if no progress for
     MAX_STUCK_ITERATIONS iterations, triggers recovery or terminates.
  4. Per-goal attempt budget enforced by MAX_GOAL_ATTEMPTS before escalating.
  5. All existing Phase 1+2 behavior (DAG, dependency tracking, replanning) preserved.

Loop cycle:
  1. DAGManager: compute ready goals (PENDING + all deps COMPLETED).
  2. Stuck detection: if no progress in recent iterations, handle.
  3. Build minimal context (ContextManager).
  4. Ask LLM to select an action: tool_call or direct_answer.
  5. Execute the action (ToolExecutor).
  6. Record the Observation in state.
  7. Verify: does the observation satisfy the goal? (Verifier â€” Phase 3 enhanced)
  8a. VALID â†’ mark goal COMPLETED, continue.
  8b. INCOMPLETE/INVALID â†’ retry up to MAX_GOAL_ATTEMPTS.
  8c. Exhausted â†’ RecoveryManager decides strategy:
       - retry/modify_input/switch_tool â†’ reset goal and re-queue
       - fallback â†’ accept partial result, mark COMPLETED
       - replan â†’ Replanner injects new goals
       - terminate â†’ mark FAILED, DAGManager will BLOCK dependents
  9. Check termination: all goals terminal, or budget exhausted.
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
from backend.planning.dag_manager import DAGManager, DAGValidationError
from backend.planning.replanner import Replanner
from backend.reasoning.recovery_manager import RecoveryManager, RecoveryStrategy
from backend.reasoning.verifier import Verifier
from backend.tools.registry import ToolRegistry

# â”€â”€ Tool selection prompt â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_ACTION_SELECTION_SYSTEM = """You are the action selection module of an AI agent runtime.

Given the current goal and available tools, decide the NEXT action to take.

You MUST return ONLY valid JSON in one of these two formats:

Option A â€” Use a tool:
{
  "action": "tool_call",
  "tool": "<exact tool name from available_tools list>",
  "input": { <input matching the tool's input_schema> },
  "goal_id": "<current goal id>",
  "reasoning": "why this tool and input"
}

Option B â€” Answer directly (no tool needed):
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

# â”€â”€ Synthesis prompt â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_SYNTHESIS_SYSTEM = """You are the response synthesizer for an AI agent runtime.

Given the original task and the results from completed goals, write a clear,
complete, and well-formatted final answer for the user.

Formatting rules:
  - Use Markdown: **bold** for key results, ## headings to separate sections.
  - For calculations: show the step-by-step arithmetic, then state the final
    answer prominently with **bold**.
  - For multi-goal tasks: use clear section headings for each part.
  - Be concise — do not pad simple answers with unnecessary explanation.
  - Include the actual numbers, names, and facts from the goal results.
  - If a goal failed or was blocked, acknowledge what could not be determined.
  - Do NOT fabricate information not present in the goal results.
  - Do NOT truncate your response — always complete every sentence and the
    final answer statement.
  - End with a clear, prominent statement of the result.

Example for a calculation task:
  ## Calculation
  25 x 47 = **1,175**
  1,175 + 100 = **1,275**
  ## Answer
  **1,275**
"""


class LoopController:
    """
    Phase 3 agent runtime: PLANâ†’ACTâ†’OBSERVEâ†’VERIFYâ†’RECOVER/REPLAN loop.

    Integrates: DAGManager, Replanner, RecoveryManager, stuck detection.
    All orchestration is explicit Python â€” no external framework.
    """

    MAX_GOAL_ATTEMPTS = 3    # Inner retry attempts before escalating to RecoveryManager
    MAX_REPLAN_ROUNDS = 2    # Max times Replanner may inject new goals
    MAX_STUCK_ITERATIONS = 4 # Iterations with zero goal completions before intervention

    def __init__(
        self,
        llm: LLMClient,
        tool_registry: ToolRegistry,
        executor: ToolExecutor,
        verifier: Verifier,
        context_manager: ContextManager,
        budget_manager: BudgetManager,
        logger: AgentLogger,
        replanner: Replanner | None = None,
        recovery_manager: RecoveryManager | None = None,
    ) -> None:
        self._llm = llm
        self._registry = tool_registry
        self._executor = executor
        self._verifier = verifier
        self._ctx = context_manager
        self._budget = budget_manager
        self._logger = logger
        self._replanner = replanner
        self._recovery = recovery_manager

    # â”€â”€ Main loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def run(self, state: AgentState) -> AgentState:
        """
        Execute the agent loop until termination.

        Phase 3: RecoveryManager, stuck detection, and strategy execution.
        """
        state.status = AgentStatus.RUNNING
        replan_rounds = 0
        last_completed_count = 0
        stuck_iteration_count = 0

        # â”€â”€ Build and validate DAG â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        dag = DAGManager(state.goals, self._logger)
        try:
            dag.validate()
            topo = dag.topo_order()
            self._logger.info(
                f"DAG validated. Topological order: {' â†’ '.join(topo)}"
            )
        except DAGValidationError as exc:
            self._logger.error(f"DAG validation failed: {exc}")
            state.status = AgentStatus.FAILED
            return state

        try:
            while not self._should_stop(state):
                state.iterations += 1
                self._budget.tick_iteration()

                # â”€â”€ Update BLOCKED goals after any failure â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                dag.update_blocked(state)

                # â”€â”€ Stuck detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                current_completed = len(state.completed_goal_ids())
                if current_completed == last_completed_count:
                    stuck_iteration_count += 1
                else:
                    stuck_iteration_count = 0
                    last_completed_count = current_completed

                if stuck_iteration_count >= self.MAX_STUCK_ITERATIONS:
                    self._logger.warn(
                        f"Stuck detected: no progress for {stuck_iteration_count} iterations. "
                        f"Failing remaining pending goals."
                    )
                    self._fail_stranded_goals(state)
                    break

                # â”€â”€ Pick next ready goal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                ready = dag.ready_goals(state)

                if not ready:
                    if dag.is_deadlocked(state):
                        self._logger.warn(
                            "DAG deadlock detected â€” no ready goals and no running goals."
                        )
                        self._fail_stranded_goals(state)
                    break

                # Process highest-priority ready goal
                goal = ready[0]
                goal.status = GoalStatus.RUNNING
                self._logger.goal_start(goal.id, goal.description)

                # â”€â”€ ACT â†’ OBSERVE â†’ VERIFY cycle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                completed = self._process_goal(goal, state)

                # â”€â”€ Phase 3: Recovery on failure â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                if not completed and goal.status == GoalStatus.FAILED:
                    did_recover = self._handle_recovery(goal, state, dag, replan_rounds)
                    if did_recover:
                        replan_rounds += 1
                        # Rebuild DAG after potential goal injection
                        dag = DAGManager(state.goals, self._logger)
                        try:
                            dag.validate()
                        except DAGValidationError as exc:
                            self._logger.warn(f"Post-recovery DAG invalid: {exc}")

        except BudgetExhaustedError as exc:
            self._logger.error(f"Budget exhausted: {exc.reason}")
            state.status = AgentStatus.BUDGET_EXHAUSTED
            return state

        except Exception as exc:
            self._logger.error(f"Unexpected loop error: {type(exc).__name__}: {exc}")
            state.status = AgentStatus.FAILED
            return state

        # â”€â”€ Set final status â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if state.status == AgentStatus.RUNNING:
            completed = sum(1 for g in state.goals if g.status == GoalStatus.COMPLETED)
            state.status = AgentStatus.COMPLETED if completed > 0 else AgentStatus.FAILED

        return state

    # â”€â”€ Goal processing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _process_goal(self, goal: Goal, state: AgentState) -> bool:
        """
        ACT â†’ OBSERVE â†’ VERIFY cycle for a single goal.
        Returns True if goal completed, False if it failed.
        """
        while goal.attempts < self.MAX_GOAL_ATTEMPTS:
            goal.attempts += 1

            context = self._ctx.build_tool_selection_context(state, goal, self._registry)
            action = self._select_action(context, state)

            if action is None:
                self._logger.error(f"Action selection failed for goal {goal.id}. Will retry.")
                if goal.attempts >= self.MAX_GOAL_ATTEMPTS:
                    break
                self._budget.tick_retry()
                continue

            action_type = action.get("action", "?")
            if action_type == "tool_call":
                self._logger.tool_selected(action.get("tool", "?"), action.get("input", {}))
            elif action_type == "direct_answer":
                self._logger.tool_selected("direct_answer", action.get("answer", "")[:80])

            # â”€â”€ Direct answer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if action_type == "direct_answer":
                answer_text = str(action.get("answer", "")).strip()
                if answer_text:
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

                    # Run verifier even on direct answers (Phase 3)
                    verification = self._verifier.verify(goal, obs)
                    if verification.status in (VerificationStatus.VALID,):
                        goal.result = answer_text
                        goal.status = GoalStatus.COMPLETED
                        goal.verification_status = VerificationStatus.VALID
                        self._logger.goal_completed(goal.id)
                        return True
                    else:
                        self._logger.warn(
                            f"Direct answer verification: {verification.status.value} â€” retrying."
                        )
                        if goal.attempts < self.MAX_GOAL_ATTEMPTS:
                            self._budget.tick_retry()
                            continue
                        break
                else:
                    self._logger.warn(f"direct_answer was empty for goal {goal.id}, retrying.")
                    self._budget.tick_retry()
                    continue

            # â”€â”€ Tool call â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            self._budget.tick_tool_call()
            obs = self._executor.execute(action, state)
            state.observations.append(obs)

            # â”€â”€ Verify â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            verification = self._verifier.verify(goal, obs)
            obs.verification_status = verification.status
            goal.verification_status = verification.status

            if verification.status == VerificationStatus.VALID:
                goal.result = obs.output
                goal.status = GoalStatus.COMPLETED
                self._logger.goal_completed(goal.id)
                return True

            # Not valid â€” retry if attempts remain
            self._logger.recovery_triggered(
                f"Verification={verification.status.value} for [{goal.id}] "
                f"â€” attempt {goal.attempts}/{self.MAX_GOAL_ATTEMPTS}"
            )
            if goal.attempts < self.MAX_GOAL_ATTEMPTS:
                self._budget.tick_retry()
                continue

            break  # Exhausted inner attempts

        # â”€â”€ All inner attempts exhausted â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        goal.status = GoalStatus.FAILED
        goal.error = (
            f"Failed after {goal.attempts} attempt(s). "
            f"Last verification: {goal.verification_status.value}"
        )
        self._logger.goal_failed(goal.id, goal.error)
        return False

    # â”€â”€ Recovery strategy execution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _handle_recovery(
        self,
        goal: Goal,
        state: AgentState,
        dag: DAGManager,
        replan_rounds: int,
    ) -> bool:
        """
        Phase 3 recovery: query RecoveryManager, then execute the strategy.
        Returns True if a meaningful recovery action was taken (for replan_rounds tracking).
        """
        if self._recovery is None:
            return False  # Recovery disabled

        last_obs = state.observations_for_goal(goal.id)
        last_obs_obj = last_obs[-1] if last_obs else None

        decision = self._recovery.decide(goal, state, last_obs_obj)

        strategy = decision.strategy
        self._logger.info(
            f"Recovery strategy for '{goal.id}': {strategy.value} â€” {decision.reasoning}"
        )

        # â”€â”€ RETRY â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if strategy == RecoveryStrategy.RETRY:
            if goal.attempts < 6:  # Hard cap even with recovery
                goal.status = GoalStatus.PENDING
                goal.attempts = 0
                self._logger.info(f"Recovery RETRY: resetting goal '{goal.id}' for another attempt.")
                return False  # Not a replan

        # â”€â”€ MODIFY INPUT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif strategy == RecoveryStrategy.MODIFY_INPUT:
            if decision.modified_input:
                goal.status = GoalStatus.PENDING
                goal.attempts = 0
                goal.metadata["recovery_modified_input"] = decision.modified_input
                self._logger.info(
                    f"Recovery MODIFY_INPUT: resetting goal '{goal.id}' with new input: "
                    f"{decision.modified_input}"
                )
                return False

        # â”€â”€ SWITCH TOOL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif strategy == RecoveryStrategy.SWITCH_TOOL:
            if decision.suggested_tool:
                goal.status = GoalStatus.PENDING
                goal.attempts = 0
                goal.metadata["recovery_forced_tool"] = decision.suggested_tool
                self._logger.info(
                    f"Recovery SWITCH_TOOL: resetting goal '{goal.id}' to use '{decision.suggested_tool}'."
                )
                return False

        # â”€â”€ FALLBACK â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif strategy == RecoveryStrategy.FALLBACK:
            fallback_text = decision.fallback_result or f"[Fallback] Could not complete: {goal.description}"
            goal.result = fallback_text
            goal.status = GoalStatus.COMPLETED  # Accept degraded result
            goal.verification_status = VerificationStatus.UNRELIABLE
            self._logger.info(
                f"Recovery FALLBACK: goal '{goal.id}' accepted with degraded result."
            )
            return False

        # â”€â”€ REPLAN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif strategy == RecoveryStrategy.REPLAN:
            if self._replanner is not None and replan_rounds < self.MAX_REPLAN_ROUNDS:
                new_goals = self._replanner.replan(goal, state)
                if new_goals:
                    state.goals.extend(new_goals)
                    self._logger.replanning(
                        f"Injected {len(new_goals)} recovery goal(s) after '{goal.id}' failed."
                    )
                    return True  # Signals a replan round was consumed

        # â”€â”€ TERMINATE (or default) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # goal.status is already FAILED â€” nothing to do
        self._logger.info(f"Recovery TERMINATE: goal '{goal.id}' marked as unrecoverable.")
        return False

    # â”€â”€ Action selection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _select_action(
        self, context: dict[str, Any], state: AgentState
    ) -> dict[str, Any] | None:
        """Ask the LLM to select the next action. Returns None on failure."""
        # Honour recovery-forced tool if set
        goal_id = context.get("current_goal", {}).get("id", "")
        goal = state.get_goal(goal_id)
        forced_tool = goal.metadata.get("recovery_forced_tool") if goal else None

        user_prompt = json.dumps(context, indent=2, default=str)

        # If a tool was forced by recovery, inject a hint into the prompt
        if forced_tool:
            user_prompt = (
                f"[RECOVERY INSTRUCTION: You MUST use the tool '{forced_tool}' for this goal.]\n\n"
                + user_prompt
            )

        try:
            self._budget.tick_llm_call()
            action = self._llm.complete_json(_ACTION_SELECTION_SYSTEM, user_prompt)
            if "action" not in action:
                self._logger.warn("LLM action missing 'action' key.")
                return None
            return action
        except LLMError as exc:
            self._logger.error(f"Action selection LLM call failed: {exc}")
            return None

    # â”€â”€ Final answer synthesis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def synthesize_answer(self, state: AgentState) -> str:
        """Generate the final answer from completed goal results."""
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

    # â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _should_stop(self, state: AgentState) -> bool:
        if state.all_goals_terminal():
            return True
        if state.status in (AgentStatus.FAILED, AgentStatus.BUDGET_EXHAUSTED):
            return True
        return False

    def _fail_stranded_goals(self, state: AgentState) -> None:
        for goal in state.goals:
            if goal.status == GoalStatus.PENDING:
                goal.status = GoalStatus.FAILED
                goal.error = "Stranded: no progress detected â€” stuck detection triggered."
                self._logger.goal_failed(goal.id, goal.error)

    @staticmethod
    def _fallback_assemble(state: AgentState) -> str:
        parts: list[str] = []
        for goal in state.goals:
            if goal.status == GoalStatus.COMPLETED and goal.result:
                parts.append(f"[{goal.id}] {goal.description}:\n{goal.result}")
            elif goal.status == GoalStatus.FAILED:
                parts.append(f"[{goal.id}] Could not complete: {goal.description}")
            elif goal.status == GoalStatus.BLOCKED:
                parts.append(f"[{goal.id}] Blocked (dependency failed): {goal.description}")
        return "\n\n".join(parts) if parts else "No results available."
