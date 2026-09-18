"""
Async Loop Controller — Phase 4 parallel execution engine.

Implements TRUE parallel execution of independent goals using asyncio:
  - Goals with no dependencies between them run CONCURRENTLY via asyncio.gather()
  - Goals with dependencies WAIT for their prerequisites before starting
  - DAG correctness is preserved: a goal starts only when all deps are COMPLETED
  - Thread-safe state access via asyncio.Lock()

Architecture:
  - Uses AsyncToolExecutor to run sync tools in a thread pool
  - LLM action-selection calls are also dispatched to thread pool
    (Groq client is sync; we don't block the event loop)
  - The main loop continually asks: "What goals are READY RIGHT NOW?"
    and launches all ready goals as concurrent asyncio Tasks
  - Each goal task runs its own ACT → OBSERVE → VERIFY → RECOVER cycle
  - When a goal completes/fails, the loop rechecks for newly unblocked goals

Phases 1-3 behavior is FULLY PRESERVED:
  - DAGManager (validation, topo sort, BLOCKED propagation)
  - Verifier (Phase 3 enhanced)
  - RecoveryManager (Phase 3 LLM-driven strategies)
  - Replanner (Phase 2 goal injection)
  - BudgetManager (hard limits on iterations, LLM calls, time)
  - StuckDetector (Phase 4 standalone — replaces inline stuck detection)

Entry points:
  run()       — synchronous wrapper (calls asyncio.run on run_async)
  run_async() — pure async; can be awaited from other async code
"""
from __future__ import annotations

import asyncio
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
from backend.execution.async_executor import AsyncToolExecutor
from backend.llm.client import LLMClient, LLMError
from backend.monitoring.budget import BudgetExhaustedError, BudgetManager
from backend.monitoring.logger import AgentLogger
from backend.monitoring.stuck_detector import StuckDetector, StuckReason
from backend.planning.dag_manager import DAGManager, DAGValidationError
from backend.planning.replanner import Replanner
from backend.reasoning.recovery_manager import RecoveryManager, RecoveryStrategy
from backend.reasoning.verifier import Verifier
from backend.tools.registry import ToolRegistry

# ── Prompts (same as Phase 3 LoopController) ─────────────────────────────────
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
  3. If previous attempts for this goal failed, try a different approach.
  4. Use "direct_answer" ONLY when you can confidently answer from knowledge.
  5. For calculator: "input" MUST have an "expression" key.
  6. For web_search: "input" MUST have a "query" key.
  7. The "goal_id" in your response MUST match the current_goal.id exactly.
"""

_SYNTHESIS_SYSTEM = """You are the response synthesizer for an AI agent runtime.

Given the original task and the results from completed goals, write a clear,
accurate, and concise final answer for the user.

Rules:
  - Integrate all relevant results naturally.
  - Be specific: include numbers, names, and facts from the results.
  - If some goals failed or were blocked, acknowledge what could not be determined.
  - Do NOT fabricate information not present in the goal results.
"""

# How many times a goal may attempt before escalating to RecoveryManager
_MAX_GOAL_ATTEMPTS = 3
# Max replan rounds per run
_MAX_REPLAN_ROUNDS = 2


class AsyncLoopController:
    """
    Phase 4: asyncio-based agent loop with true parallel goal execution.

    Key properties:
      - Independent goals (no dep path between them) run CONCURRENTLY.
      - Dependent goals wait for their prerequisites via asyncio event.
      - A single asyncio.Lock serializes state mutations (goal status writes,
        observation appends) to prevent races.
      - BudgetManager and StuckDetector are checked from the coordinator task.
    """

    def __init__(
        self,
        llm: LLMClient,
        tool_registry: ToolRegistry,
        async_executor: AsyncToolExecutor,
        verifier: Verifier,
        context_manager: ContextManager,
        budget_manager: BudgetManager,
        logger: AgentLogger,
        replanner: Replanner | None = None,
        recovery_manager: RecoveryManager | None = None,
    ) -> None:
        self._llm = llm
        self._registry = tool_registry
        self._executor = async_executor
        self._verifier = verifier
        self._ctx = context_manager
        self._budget = budget_manager
        self._logger = logger
        self._replanner = replanner
        self._recovery = recovery_manager
        self._stuck = StuckDetector(
            max_stagnant_iterations=4,
            max_repeated_tool_failures=3,
            max_repeated_observations=3,
        )

    # ── Public entry points ────────────────────────────────────────────────────

    def run(self, state: AgentState) -> AgentState:
        """Synchronous wrapper — calls asyncio.run(run_async(state))."""
        return asyncio.run(self.run_async(state))

    async def run_async(self, state: AgentState) -> AgentState:
        """
        Main async entry point.

        Validates the DAG, then launches the parallel coordinator loop.
        """
        state.status = AgentStatus.RUNNING

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

        lock = asyncio.Lock()
        replan_rounds = [0]  # mutable int for coroutine closure

        try:
            await self._coordinator(state, dag, lock, replan_rounds)
        except BudgetExhaustedError as exc:
            self._logger.error(f"Budget exhausted: {exc.reason}")
            state.status = AgentStatus.BUDGET_EXHAUSTED
            return state
        except Exception as exc:
            self._logger.error(f"Unexpected loop error: {type(exc).__name__}: {exc}")
            state.status = AgentStatus.FAILED
            return state

        if state.status == AgentStatus.RUNNING:
            completed = sum(1 for g in state.goals if g.status == GoalStatus.COMPLETED)
            state.status = AgentStatus.COMPLETED if completed > 0 else AgentStatus.FAILED

        return state

    # ── Coordinator ────────────────────────────────────────────────────────────

    async def _coordinator(
        self,
        state: AgentState,
        dag: DAGManager,
        lock: asyncio.Lock,
        replan_rounds: list[int],
    ) -> None:
        """
        Main scheduling loop.

        Each tick:
          1. Update BLOCKED goals.
          2. Run stuck detector.
          3. Find all READY goals (PENDING + all deps COMPLETED).
          4. Launch each ready goal as an asyncio.Task concurrently.
          5. Wait for at least one task to finish (asyncio.FIRST_COMPLETED).
          6. Repeat until all goals are terminal.
        """
        running_tasks: dict[str, asyncio.Task] = {}  # goal_id → Task

        while True:
            self._budget.tick_iteration()
            state.iterations += 1

            # ── Phase 4 iteration marker ──────────────────────────────────────
            self._logger.p4_loop(state.iterations)

            # ── Update BLOCKED ────────────────────────────────────────────────
            async with lock:
                dag.update_blocked(state)

            # ── Stuck detection ───────────────────────────────────────────────
            stuck_result = self._stuck.check(state)
            # Log stuck status each iteration for demo clarity
            self._logger.p4_stuck(
                stuck_result.is_stuck,
                stuck_result.reason.value,
                stuck_result.detail,
            )
            if stuck_result.is_stuck:
                async with lock:
                    self._fail_stranded_goals(state)
                break

            # ── Budget checkpoint ─────────────────────────────────────────────
            bused = self._budget.summary()
            bcfg  = self._budget.config
            self._logger.p4_budget(
                bused,
                {
                    "max_llm_calls":   bcfg.max_llm_calls,
                    "max_tool_calls":  bcfg.max_tool_calls,
                    "max_iterations":  bcfg.max_iterations,
                },
            )

            # ── Check termination ─────────────────────────────────────────────
            if state.all_goals_terminal() and not running_tasks:
                self._logger.p4_loop(state.iterations, "All goals completed")
                break

            # ── Find and launch ready goals ───────────────────────────────────
            async with lock:
                ready = dag.ready_goals(state)
                ready_ids = [g.id for g in ready if g.id not in running_tasks]

            if ready_ids:
                self._logger.p4_dag(
                    f"Ready goals: {', '.join(ready_ids)}"
                )
                if len(ready_ids) > 1:
                    self._logger.p4_executor(
                        f"Running {len(ready_ids)} goals concurrently: {', '.join(ready_ids)}"
                    )

            async with lock:
                for goal in ready:
                    if goal.id not in running_tasks:
                        goal.status = GoalStatus.RUNNING
                        task = asyncio.create_task(
                            self._process_goal(goal, state, dag, lock, replan_rounds),
                            name=f"goal-{goal.id}",
                        )
                        running_tasks[goal.id] = task
                        if len(ready_ids) == 1:
                            self._logger.p4_executor(f"Launching {goal.id}")
                        self._logger.goal_start(goal.id, goal.description)

            if not running_tasks:
                # Deadlock or all done
                if dag.is_deadlocked(state):
                    self._logger.warn("DAG deadlock — failing stranded goals.")
                    async with lock:
                        self._fail_stranded_goals(state)
                break

            # ── Wait for at least one task to complete ────────────────────────
            done, _ = await asyncio.wait(
                running_tasks.values(),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                goal_id = task.get_name().removeprefix("goal-")
                running_tasks.pop(goal_id, None)
                exc = task.exception()
                if exc:
                    self._logger.error(
                        f"Goal task '{goal_id}' raised unexpectedly: {exc}"
                    )

    # ── Per-goal task ──────────────────────────────────────────────────────────

    async def _process_goal(
        self,
        goal: Goal,
        state: AgentState,
        dag: DAGManager,
        lock: asyncio.Lock,
        replan_rounds: list[int],
    ) -> None:
        """
        Async per-goal runner: ACT → OBSERVE → VERIFY → RECOVER cycle.

        Runs concurrently with other independent goal tasks.
        Uses the lock only when mutating shared state.
        """
        while goal.attempts < _MAX_GOAL_ATTEMPTS:
            goal.attempts += 1

            # ── Build context (read-only — no lock needed) ────────────────────
            context = self._ctx.build_tool_selection_context(state, goal, self._registry)

            # Honour recovery-forced tool
            forced_tool = goal.metadata.get("recovery_forced_tool")
            user_prompt = json.dumps(context, indent=2, default=str)
            if forced_tool:
                user_prompt = (
                    f"[RECOVERY INSTRUCTION: You MUST use the tool '{forced_tool}' "
                    f"for this goal.]\n\n" + user_prompt
                )

            # ── Action selection (LLM call in thread pool) ────────────────────
            action = await self._select_action_async(user_prompt)

            if action is None:
                self._logger.error(f"Action selection failed for goal {goal.id}.")
                if goal.attempts >= _MAX_GOAL_ATTEMPTS:
                    break
                await asyncio.sleep(0)  # yield
                continue

            action_type = action.get("action", "?")
            if action_type == "tool_call":
                self._logger.tool_selected(action.get("tool", "?"), action.get("input", {}))
            else:
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
                    verification = self._verifier.verify(goal, obs)
                    async with lock:
                        state.observations.append(obs)
                        self._budget.tick_tool_call()

                    if verification.status == VerificationStatus.VALID:
                        async with lock:
                            goal.result = answer_text
                            goal.status = GoalStatus.COMPLETED
                            goal.verification_status = VerificationStatus.VALID
                        self._logger.goal_completed(goal.id)
                        return
                    else:
                        self._logger.warn(
                            f"Direct answer verification {verification.status.value} — retrying."
                        )
                        if goal.attempts < _MAX_GOAL_ATTEMPTS:
                            async with lock:
                                self._budget.tick_retry()
                            continue
                        break
                else:
                    if goal.attempts < _MAX_GOAL_ATTEMPTS:
                        async with lock:
                            self._budget.tick_retry()
                        continue
                    break

            # ── Tool call ─────────────────────────────────────────────────────
            async with lock:
                self._budget.tick_tool_call()

            obs = await self._executor.execute_async(action, state.iterations)

            verification = self._verifier.verify(goal, obs)
            obs.verification_status = verification.status
            goal.verification_status = verification.status

            async with lock:
                state.observations.append(obs)

            if verification.status == VerificationStatus.VALID:
                async with lock:
                    goal.result = obs.output
                    goal.status = GoalStatus.COMPLETED
                self._logger.goal_completed(goal.id)
                return

            self._logger.recovery_triggered(
                f"Verification={verification.status.value} for [{goal.id}] "
                f"— attempt {goal.attempts}/{_MAX_GOAL_ATTEMPTS}"
            )
            if goal.attempts < _MAX_GOAL_ATTEMPTS:
                async with lock:
                    self._budget.tick_retry()
                await asyncio.sleep(0)  # yield to other tasks
                continue

            break  # Exhausted inner attempts

        # ── Inner attempts exhausted — escalate to RecoveryManager ────────────
        async with lock:
            goal.status = GoalStatus.FAILED
            goal.error = (
                f"Failed after {goal.attempts} attempt(s). "
                f"Last verification: {goal.verification_status.value}"
            )
        self._logger.goal_failed(goal.id, goal.error)

        # Recovery
        if self._recovery is not None:
            last_obs_list = state.observations_for_goal(goal.id)
            last_obs = last_obs_list[-1] if last_obs_list else None
            decision = self._recovery.decide(goal, state, last_obs)
            self._logger.info(
                f"Recovery strategy for '{goal.id}': {decision.strategy.value}"
            )

            strategy = decision.strategy

            async with lock:
                if strategy == RecoveryStrategy.RETRY and goal.attempts < 6:
                    goal.status = GoalStatus.PENDING
                    goal.attempts = 0
                    self._stuck.reset_stagnation()
                    return  # Will be re-picked by coordinator

                elif strategy == RecoveryStrategy.MODIFY_INPUT and decision.modified_input:
                    goal.status = GoalStatus.PENDING
                    goal.attempts = 0
                    goal.metadata["recovery_modified_input"] = decision.modified_input
                    self._stuck.reset_stagnation()
                    return

                elif strategy == RecoveryStrategy.SWITCH_TOOL and decision.suggested_tool:
                    goal.status = GoalStatus.PENDING
                    goal.attempts = 0
                    goal.metadata["recovery_forced_tool"] = decision.suggested_tool
                    self._stuck.reset_stagnation()
                    return

                elif strategy == RecoveryStrategy.FALLBACK:
                    fallback_text = (
                        decision.fallback_result
                        or f"[Fallback] Could not complete: {goal.description}"
                    )
                    goal.result = fallback_text
                    goal.status = GoalStatus.COMPLETED
                    goal.verification_status = VerificationStatus.UNRELIABLE
                    self._logger.info(f"Recovery FALLBACK accepted for '{goal.id}'.")
                    return

                elif (
                    strategy == RecoveryStrategy.REPLAN
                    and self._replanner is not None
                    and replan_rounds[0] < _MAX_REPLAN_ROUNDS
                ):
                    new_goals = self._replanner.replan(goal, state)
                    if new_goals:
                        state.goals.extend(new_goals)
                        replan_rounds[0] += 1
                        self._logger.replanning(
                            f"Injected {len(new_goals)} recovery goal(s) after '{goal.id}'."
                        )
                        # Rebuild DAGManager with extended goals list
                        new_dag = DAGManager(state.goals, self._logger)
                        try:
                            new_dag.validate()
                            # Swap _goals ref in original dag to include new goals
                            dag._goals = state.goals
                            dag._id_to_goal = {g.id: g for g in state.goals}
                        except DAGValidationError as exc:
                            self._logger.warn(f"Post-recovery DAG invalid: {exc}")
                        self._stuck.reset_stagnation()
                    return

                # TERMINATE or unhandled — goal stays FAILED

    # ── Action selection (async LLM) ──────────────────────────────────────────

    async def _select_action_async(self, user_prompt: str) -> dict[str, Any] | None:
        """Run the sync LLM call in a thread pool."""
        loop = asyncio.get_event_loop()
        try:
            self._budget.tick_llm_call()
            from functools import partial
            result = await loop.run_in_executor(
                None,
                partial(self._llm.complete_json, _ACTION_SELECTION_SYSTEM, user_prompt),
            )
            if "action" not in result:
                self._logger.warn("LLM action missing 'action' key.")
                return None
            return result
        except BudgetExhaustedError:
            raise
        except LLMError as exc:
            self._logger.error(f"Action selection LLM call failed: {exc}")
            return None

    # ── Final answer synthesis ────────────────────────────────────────────────

    async def synthesize_answer_async(self, state: AgentState) -> str:
        """Async synthesis: run sync LLM call in thread pool."""
        self._logger.synthesizing()
        synthesis_context = self._ctx.build_synthesis_context(state)
        user_prompt = (
            f'Original task: "{state.task}"\n\n'
            f"Goal results:\n"
            f"{json.dumps(synthesis_context, indent=2, default=str)}"
        )
        loop = asyncio.get_event_loop()
        try:
            self._budget.tick_llm_call()
            from functools import partial
            answer = await loop.run_in_executor(
                None,
                partial(self._llm.complete, _SYNTHESIS_SYSTEM, user_prompt),
            )
            return answer.strip()
        except Exception as exc:
            self._logger.warn(f"Synthesis failed: {exc}. Using fallback.")
            return self._fallback_assemble(state)

    def synthesize_answer(self, state: AgentState) -> str:
        """Sync wrapper for synthesize_answer_async."""
        return asyncio.run(self.synthesize_answer_async(state))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fail_stranded_goals(self, state: AgentState) -> None:
        for goal in state.goals:
            if goal.status == GoalStatus.PENDING:
                goal.status = GoalStatus.FAILED
                goal.error = "Stranded: stuck detection triggered."
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
                parts.append(f"[{goal.id}] Blocked: {goal.description}")
        return "\n\n".join(parts) if parts else "No results available."
