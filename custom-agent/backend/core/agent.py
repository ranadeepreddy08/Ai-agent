"""
Agent — top-level orchestrator.

Wires all components together and exposes a single run(task) method.
Component assembly happens here; the loop controller stays pure business logic.

Tool registration:
  Add new tools by calling self.registry.register(MyTool()) here.
  Nothing else changes — the agent loop is tool-agnostic.
"""
from __future__ import annotations

from backend.context.manager import ContextManager
from backend.core.loop_controller import LoopController
from backend.core.state import AgentState, AgentStatus
from backend.execution.executor import ToolExecutor
from backend.llm.client import LLMClient
from backend.monitoring.budget import BudgetConfig, BudgetManager
from backend.monitoring.logger import AgentLogger
from backend.planning.planner import Planner
from backend.reasoning.verifier import Verifier
from backend.tools.calculator import CalculatorTool
from backend.tools.registry import ToolRegistry
from backend.tools.web_search import WebSearchTool


class Agent:
    """
    Top-level agent entry point.

    Pipeline:
      1. Planner → decompose task into Goals
      2. LoopController → PLAN→ACT→OBSERVE→VERIFY until all goals terminal
      3. LoopController.synthesize_answer() → final user-facing answer

    To add a new tool for future phases:
      self.registry.register(MyNewTool())
    That's the only change needed anywhere in the codebase.
    """

    def __init__(
        self,
        config: dict | None = None,
        verbose: bool = True,
    ) -> None:
        cfg = config or {}

        # Logger is created first and injected everywhere
        self.logger = AgentLogger(verbose=verbose)

        # LLM client — only Gemini SDK usage in the entire project
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
        self.loop = LoopController(
            llm=self.llm,
            tool_registry=self.registry,
            executor=self.executor,
            verifier=self.verifier,
            context_manager=self.context_manager,
            budget_manager=self.budget,
            logger=self.logger,
        )

    def run(self, task: str) -> dict:
        """
        Execute the agent on a task string.

        Returns:
            {
              "answer": str,               # Final synthesized answer
              "status": str,               # AgentStatus value
              "goals": list[dict],         # Goal summaries
              "budget": dict,              # Resource usage
              "llm_tokens": dict,          # Token counts
            }
        """
        print(f"\n{'═' * 64}")
        self.logger.info(f"Task received: {task!r}")
        print(f"{'═' * 64}\n")

        # ── 1. Initialize state ───────────────────────────────────────────
        state = AgentState(task=task)

        # ── 2. Plan ───────────────────────────────────────────────────────
        state = self.planner.plan(state)

        # ── 3. Loop (ACT → OBSERVE → VERIFY) ─────────────────────────────
        print()
        state = self.loop.run(state)

        # ── 4. Synthesize final answer ────────────────────────────────────
        print()
        if state.final_answer is None:
            state.final_answer = self.loop.synthesize_answer(state)

        self.logger.final_answer(state.final_answer)

        # ── 5. Budget summary ─────────────────────────────────────────────
        budget_summary = self.budget.summary()
        # Sync actual LLM call count from client (may differ due to retries)
        budget_summary["llm_calls"] = self.llm.total_calls
        self.logger.budget_summary(budget_summary)

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
