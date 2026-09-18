"""
Async Tool Executor — Phase 4.

Wraps the synchronous ToolExecutor and LLM calls in asyncio-compatible
executor threads so that multiple independent goals can run concurrently
without blocking the event loop.

Design:
  - Sync tools/LLM → run in ThreadPoolExecutor (via asyncio.get_event_loop().run_in_executor)
  - Never blocks the main asyncio event loop
  - Same Observation output contract as the sync executor
"""
from __future__ import annotations

import asyncio
import json
import time
from functools import partial
from typing import Any

from backend.core.state import AgentState, Observation, VerificationStatus
from backend.monitoring.logger import AgentLogger
from backend.tools.registry import ToolRegistry, ToolRegistryError


class AsyncToolExecutor:
    """
    Async wrapper around the synchronous ToolExecutor.

    Dispatches tool calls to a thread pool so the asyncio event loop
    can schedule multiple independent goal executions concurrently.

    Usage:
        obs = await executor.execute_async(action, state, iteration)
    """

    def __init__(self, registry: ToolRegistry, logger: AgentLogger) -> None:
        self._registry = registry
        self._logger = logger

    async def execute_async(
        self,
        action: dict[str, Any],
        iteration: int,
    ) -> Observation:
        """
        Async dispatch: runs the sync tool.execute() in a thread pool.

        Returns a normalized Observation — never raises.
        """
        tool_name = str(action.get("tool", "")).strip()
        goal_id = str(action.get("goal_id", "unknown")).strip()
        raw_input = action.get("input", {})

        if not isinstance(raw_input, dict):
            try:
                raw_input = json.loads(str(raw_input))
            except (json.JSONDecodeError, TypeError):
                raw_input = {"value": str(raw_input)}

        input_data: dict[str, Any] = {**raw_input, "goal_id": goal_id}
        self._logger.executing(tool_name, raw_input, goal_id)

        # Resolve tool (fast — no IO)
        try:
            tool = self._registry.get(tool_name)
        except ToolRegistryError as exc:
            return Observation(
                goal_id=goal_id,
                tool_name=tool_name,
                success=False,
                output=None,
                error=f"Tool '{tool_name}' not found. Available: {self._registry.names()}",
                execution_time=0.0,
                iteration=iteration,
            )

        # Run sync tool in thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        start = time.monotonic()
        try:
            result = await loop.run_in_executor(None, partial(tool.execute, input_data))
        except Exception as exc:
            elapsed = time.monotonic() - start
            self._logger.error(f"Tool '{tool_name}' raised: {exc}")
            return Observation(
                goal_id=goal_id,
                tool_name=tool_name,
                success=False,
                output=None,
                error=f"Unexpected exception in '{tool_name}': {exc}",
                execution_time=elapsed,
                iteration=iteration,
            )

        obs = Observation(
            goal_id=result.goal_id,
            tool_name=result.tool_name,
            success=result.success,
            output=result.output,
            error=result.error,
            execution_time=result.execution_time,
            iteration=iteration,
            metadata=result.metadata,
        )

        if result.success:
            self._logger.observation_received(obs)
        else:
            self._logger.tool_failed(tool_name, result.error or "no error message")

        return obs
