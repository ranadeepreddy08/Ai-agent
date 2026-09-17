"""
Loop Controller — Phase 3: full reliability pipeline.

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
  7. Verify: does the observation satisfy the goal? (Verifier — Phase 3 enhanced)
  8a. VALID → mark goal COMPLETED, continue.
  8b. INCOMPLETE/INVALID → retry up to MAX_GOAL_ATTEMPTS.
  8c. Exhausted → RecoveryManager decides strategy:
       - retry/modify_input/switch_tool → reset goal and re-queue
       - fallback → accept partial result, mark COMPLETED
       - replan → Replanner injects new goals
       - terminate → mark FAILED, DAGManager will BLOCK dependents
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
  - If some goals failed or were blocked, acknowledge what could not be determined.
  - Do NOT fabricate information not present in the goal results.
  - Use plain, readable prose. Format numbers clearly.
"""


class LoopController:
    """
    Phase 3 agent runtime: PLAN→ACT→OBSERVE→VERIFY→RECOVER/REPLAN loop.

    Integrates: DAGManager, Replanner, RecoveryManager, stuck detection.
    All orchestration is explicit Python — no external framework.
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

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, state: AgentState) -> AgentState:
        """
        Execute the agent loop until termination.

        Phase 3: RecoveryManager, stuck detection, and strategy execution.
        """
        state.status = AgentStatus.RUNNING
        replan_rounds = 0
        last_completed_count = 0
        stuck_iteration_count = 0

        # ── Build and validate DAG ────────────────────────────────────────────
        dag = DAGManager(state.goals, self._logger)
        try:
            dag.validate()
            topo = dag.topo_order()
            self._logger.info(
                f"DAG validated. Topological order: {' → '.join(topo)}"
            )
        except DAGValidationError as exc:
            self._logger.error(f"DAG validation failed: {exc}")
            state.status = AgentStatus.FAILED
            return state

        try:
            while not self._should_stop(state):
                state.iterations += 1
                self._budget.tick_iteration()

                # ── Update BLOCKED goals after any failure ────────────────────
                dag.update_blocked(state)

                # ── Stuck detection ───────────────────────────────────────────
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

                # ── Pick next ready goal ──────────────────────────────────────
                ready = dag.ready_goals(state)

                if not ready:
                    if dag.is_deadlocked(state):
                        self._logger.warn(
                            "DAG deadlock detected — no ready goals and no running goals."
                        )
                        self._fail_stranded_goals(state)
                    break

                # Process highest-priority ready goal
                goal = ready[0]
                goal.status = GoalStatus.RUNNING
                self._logger.goal_start(goal.id, goal.description)

                # ── ACT → OBSERVE → VERIFY cycle ─────────────────────────────
                completed = self._process_goal(goal, state)

                # ── Phase 3: Recovery on failure ──────────────────────────────
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

        # ── Set final status ──────────────────────────────────────────────────
        if state.status == AgentStatus.RUNNING:
            completed = sum(1 for g in state.goals if g.status == GoalStatus.COMPLETED)
            state.status = AgentStatus.COMPLETED if completed > 0 else AgentStatus.FAILED

        return state

    # ── Goal processing ───────────────────────────────────────────────────────

    def _process_goal(self, goal: Goal, state: AgentState) -> bool:
        """
        ACT → OBSERVE → VERIFY cycle for a single goal.
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

            # ── Direct answer ─────────────────────────────────────────────────
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
                            f"Direct answer verification: {verification.status.value} — retrying."
                        )
                        if goal.attempts < self.MAX_GOAL_ATTEMPTS:
                            self._budget.tick_retry()
                            continue
                        break
                else:
                    self._logger.warn(f"direct_answer was empty for goal {goal.id}, retrying.")
                    self._budget.tick_retry()
                    continue

            # ── Tool call ─────────────────────────────────────────────────────
            self._budget.tick_tool_call()
            obs = self._executor.execute(action, state)
            state.observations.append(obs)

            # ── Verify ───────────────────────────────────────────────────────
            verification = self._verifier.verify(goal, obs)
            obs.verification_status = verification.status
            goal.verification_status = verification.status

            if verification.status == VerificationStatus.VALID:
                goal.result = obs.output
                goal.status = GoalStatus.COMPLETED
                self._logger.goal_completed(goal.id)
                return True

            # Not valid — retry if attempts remain
            self._logger.recovery_triggered(
                f"Verification={verification.status.value} for [{goal.id}] "
                f"— attempt {goal.attempts}/{self.MAX_GOAL_ATTEMPTS}"
            )
            if goal.attempts < self.MAX_GOAL_ATTEMPTS:
                self._budget.tick_retry()
                continue

            break  # Exhausted inner attempts

        # ── All inner attempts exhausted ──────────────────────────────────────
        goal.status = GoalStatus.FAILED
        goal.error = (
            f"Failed after {goal.attempts} attempt(s). "
            f"Last verification: {goal.verification_status.value}"
        )
        self._logger.goal_failed(goal.id, goal.error)
        return False

    # ── Recovery strategy execution ───────────────────────────────────────────

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
            f"Recovery strategy for '{goal.id}': {strategy.value} — {decision.reasoning}"
        )

        # ── RETRY ─────────────────────────────────────────────────────────────
        if strategy == RecoveryStrategy.RETRY:
            if goal.attempts < 6:  # Hard cap even with recovery
                goal.status = GoalStatus.PENDING
                goal.attempts = 0
                self._logger.info(f"Recovery RETRY: resetting goal '{goal.id}' for another attempt.")
                return False  # Not a replan

        # ── MODIFY INPUT ──────────────────────────────────────────────────────
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

        # ── SWITCH TOOL ───────────────────────────────────────────────────────
        elif strategy == RecoveryStrategy.SWITCH_TOOL:
            if decision.suggested_tool:
                goal.status = GoalStatus.PENDING
                goal.attempts = 0
                goal.metadata["recovery_forced_tool"] = decision.suggested_tool
                self._logger.info(
                    f"Recovery SWITCH_TOOL: resetting goal '{goal.id}' to use '{decision.suggested_tool}'."
                )
                return False

        # ── FALLBACK ──────────────────────────────────────────────────────────
        elif strategy == RecoveryStrategy.FALLBACK:
            fallback_text = decision.fallback_result or f"[Fallback] Could not complete: {goal.description}"
            goal.result = fallback_text
            goal.status = GoalStatus.COMPLETED  # Accept degraded result
            goal.verification_status = VerificationStatus.UNRELIABLE
            self._logger.info(
                f"Recovery FALLBACK: goal '{goal.id}' accepted with degraded result."
            )
            return False

        # ── REPLAN ────────────────────────────────────────────────────────────
        elif strategy == RecoveryStrategy.REPLAN:
            if self._replanner is not None and replan_rounds < self.MAX_REPLAN_ROUNDS:
                new_goals = self._replanner.replan(goal, state)
                if new_goals:
                    state.goals.extend(new_goals)
                    self._logger.replanning(
                        f"Injected {len(new_goals)} recovery goal(s) after '{goal.id}' failed."
                    )
                    return True  # Signals a replan round was consumed

        # ── TERMINATE (or default) ────────────────────────────────────────────
        # goal.status is already FAILED — nothing to do
        self._logger.info(f"Recovery TERMINATE: goal '{goal.id}' marked as unrecoverable.")
        return False

    # ── Action selection ──────────────────────────────────────────────────────

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

    # ── Final answer synthesis ────────────────────────────────────────────────

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

    # ── Helpers ───────────────────────────────────────────────────────────────

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
                goal.error = "Stranded: no progress detected — stuck detection triggered."
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
