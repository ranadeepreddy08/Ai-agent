"""
Agent — top-level orchestrator.

Wires all components together and exposes:
  - run(task)              → sync (Phase 1-3 LoopController)
  - run_async(task)        → async (Phase 4 AsyncLoopController with parallel execution)
  - run_parallel(task)     → sync wrapper around run_async

Phase 4 new components:
  - AsyncLoopController    → asyncio-based loop with true parallel goal execution
  - AsyncToolExecutor      → dispatches sync tools to a thread pool
  - StuckDetector          → standalone stuck/stagnation detection

Component assembly happens here; loop controllers stay pure business logic.

Tool registration:
  Add new tools by calling self.registry.register(MyTool()) here.
  Nothing else changes — the agent loop is tool-agnostic.
"""
from __future__ import annotations

import asyncio

from backend.context.manager import ContextManager
from backend.core.async_loop_controller import AsyncLoopController
from backend.core.loop_controller import LoopController
from backend.core.state import AgentState, AgentStatus
from backend.execution.async_executor import AsyncToolExecutor
from backend.execution.executor import ToolExecutor
from backend.llm.client import LLMClient
from backend.monitoring.budget import BudgetConfig, BudgetManager
from backend.monitoring.logger import AgentLogger
from backend.planning.planner import Planner
from backend.planning.replanner import Replanner
from backend.reasoning.recovery_manager import RecoveryManager
from backend.reasoning.verifier import Verifier
from backend.tools.calculator import CalculatorTool
from backend.tools.registry import ToolRegistry
from backend.tools.web_search import WebSearchTool


class Agent:
    """
    Top-level agent entry point.

    Pipeline (both sync and async):
      1. Planner → decompose task into Goals
      2. [Loop] → PLAN→ACT→OBSERVE→VERIFY until all goals terminal
      3. synthesize_answer() → final user-facing answer

    Phase 4 adds:
      - run_async() / run_parallel() → uses AsyncLoopController, true parallel
        goal execution (independent goals run concurrently via asyncio)
      - run() (existing) → uses LoopController (Phase 1-3), sequential execution

    To add a new tool:
      self.registry.register(MyNewTool())
    """

    def __init__(
        self,
        config: dict | None = None,
        verbose: bool = True,
    ) -> None:
        cfg = config or {}

        # Logger is created first and injected everywhere
        self.logger = AgentLogger(verbose=verbose)

        # LLM client — Groq SDK used ONLY as thin LLM interface
        self.llm = LLMClient(model=cfg.get("model"))
        self.logger.info(f"LLM: {self.llm._model_name}")

        # ── Tool registry — add tools here ────────────────────────────────
        self.registry = ToolRegistry()
        self.registry.register(CalculatorTool())
        self.registry.register(WebSearchTool())
        self.logger.info(
            f"Tools registered: {self.registry.names()} "
            f"(web_search backend: {WebSearchTool().backend_name})"
        )

        # ── Budget limits ─────────────────────────────────────────────────
        budget_cfg = BudgetConfig(
            max_iterations=cfg.get("max_iterations", 20),
            max_tool_calls=cfg.get("max_tool_calls", 30),
            max_llm_calls=cfg.get("max_llm_calls", 50),
            max_retries=cfg.get("max_retries", 9),
            max_execution_time=cfg.get("max_execution_time", 300.0),
        )
        self.budget = BudgetManager(config=budget_cfg)

        # ── Core components ───────────────────────────────────────────────
        self.context_manager = ContextManager()
        self.executor = ToolExecutor(registry=self.registry, logger=self.logger)
        self.verifier = Verifier(
            llm=self.llm,
            logger=self.logger,
            use_llm_verification=cfg.get("use_llm_verification", False),
        )
        self.planner = Planner(llm=self.llm, logger=self.logger)

        # Phase 2: Replanner — enabled by default
        _use_replanner = cfg.get("use_replanner", True)
        self.replanner = Replanner(llm=self.llm, logger=self.logger) if _use_replanner else None

        # Phase 3: RecoveryManager — enabled by default
        _use_recovery = cfg.get("use_recovery", True)
        self.recovery_manager = (
            RecoveryManager(llm=self.llm, registry=self.registry, logger=self.logger)
            if _use_recovery else None
        )

        # Phase 1-3: Sync LoopController (preserved unchanged)
        self.loop = LoopController(
            llm=self.llm,
            tool_registry=self.registry,
            executor=self.executor,
            verifier=self.verifier,
            context_manager=self.context_manager,
            budget_manager=self.budget,
            logger=self.logger,
            replanner=self.replanner,
            recovery_manager=self.recovery_manager,
        )

        # Phase 4: Async executor + AsyncLoopController (parallel execution)
        self.async_executor = AsyncToolExecutor(
            registry=self.registry,
            logger=self.logger,
        )
        self.async_loop = AsyncLoopController(
            llm=self.llm,
            tool_registry=self.registry,
            async_executor=self.async_executor,
            verifier=self.verifier,
            context_manager=self.context_manager,
            budget_manager=self.budget,
            logger=self.logger,
            replanner=self.replanner,
            recovery_manager=self.recovery_manager,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, task: str) -> dict:
        """
        Synchronous execution (Phase 1-3 LoopController — sequential).

        Preserved unchanged for full backward compatibility.
        """
        return self._execute(task, use_async=False)

    def run_parallel(self, task: str) -> dict:
        """
        Synchronous wrapper around async parallel execution (Phase 4).

        Uses asyncio.run() to drive the AsyncLoopController.
        Independent goals execute CONCURRENTLY.
        """
        return self._execute(task, use_async=True)

    async def run_async(self, task: str) -> dict:
        """
        Async entry point (Phase 4).

        Awaitable — can be used from other async code.
        Independent goals execute CONCURRENTLY via asyncio.
        """
        return await self._execute_async(task)

    # ── Internal execution ─────────────────────────────────────────────────────

    def _execute(self, task: str, use_async: bool) -> dict:
        """Shared setup → plan → loop → synthesize pipeline."""
        print(f"\n{'═' * 64}")
        self.logger.info(f"Task received: {task!r}")
        print(f"{'═' * 64}\n")

        state = AgentState(task=task)
        state = self.planner.plan(state)
        print()

        if use_async:
            # Phase 4: true parallel loop
            state = asyncio.run(self.async_loop.run_async(state))
            print()
            if state.final_answer is None:
                state.final_answer = asyncio.run(
                    self.async_loop.synthesize_answer_async(state)
                )
        else:
            # Phase 1-3: sequential loop (preserved)
            state = self.loop.run(state)
            print()
            if state.final_answer is None:
                state.final_answer = self.loop.synthesize_answer(state)

        self.logger.final_answer(state.final_answer)

        budget_summary = self.budget.summary()
        budget_summary["llm_calls"] = self.llm.total_calls
        self.logger.budget_summary(budget_summary)

        return self._build_result(state, budget_summary)

    async def _execute_async(self, task: str) -> dict:
        """Pure async execution path."""
        print(f"\n{'═' * 64}")
        self.logger.info(f"Task received: {task!r}")
        print(f"{'═' * 64}\n")

        state = AgentState(task=task)

        # Plan is sync but fast; wrap in executor for async compatibility
        loop = asyncio.get_event_loop()
        from functools import partial
        state = await loop.run_in_executor(None, partial(self.planner.plan, state))
        print()

        state = await self.async_loop.run_async(state)
        print()

        if state.final_answer is None:
            state.final_answer = await self.async_loop.synthesize_answer_async(state)

        self.logger.final_answer(state.final_answer)

        budget_summary = self.budget.summary()
        budget_summary["llm_calls"] = self.llm.total_calls
        self.logger.budget_summary(budget_summary)

        return self._build_result(state, budget_summary)

    def _build_result(self, state: AgentState, budget_summary: dict) -> dict:
        return {
            "answer": state.final_answer,
            "status": state.status.value,
            "goals": [
                {
                    "id": g.id,
                    "description": g.description,
                    "status": g.status.value,
                    "attempts": g.attempts,
                    "verification": g.verification_status.value,
                    "result_snippet": str(g.result)[:200] if g.result else None,
                    "error": g.error,
                }
                for g in state.goals
            ],
            "budget": budget_summary,
            "llm_tokens": {
                "input": self.llm.total_input_tokens,
                "output": self.llm.total_output_tokens,
            },
        }
