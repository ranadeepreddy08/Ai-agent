"""
Tool Executor — dispatches actions to tools and returns Observations.

Design:
  - Receives an action dict from the loop controller (originally from LLM).
  - Looks up the tool by name in the registry.
  - Calls tool.execute(input_data).
  - Wraps the ToolResult into a normalized Observation.
  - Zero hardcoded routing: dispatching is done entirely by tool.name lookup.
"""
from __future__ import annotations

import json
import time
from typing import Any

from backend.core.state import AgentState, Observation, VerificationStatus
from backend.monitoring.logger import AgentLogger
from backend.tools.base import ToolResult
from backend.tools.registry import ToolRegistry, ToolRegistryError


class ToolExecutor:
    """
    Dispatches LLM-selected actions to registered tools.

    Action format expected:
    {
      "action": "tool_call",
      "tool": "<name registered in ToolRegistry>",
      "input": { <keys matching tool's input_schema> },
      "goal_id": "<goal id>",
      "reasoning": "..."   (optional, for logging)
    }
    """

    def __init__(self, registry: ToolRegistry, logger: AgentLogger) -> None:
        self._registry = registry
        self._logger = logger

    def execute(self, action: dict[str, Any], state: AgentState) -> Observation:
        """
        Execute one action and return a normalized Observation.

        Never raises — any failure is captured in a failed Observation.
        """
        tool_name = str(action.get("tool", "")).strip()
        goal_id = str(action.get("goal_id", "unknown")).strip()
        raw_input = action.get("input", {})

        # Normalize input to dict
        if not isinstance(raw_input, dict):
            try:
                raw_input = json.loads(str(raw_input))
            except (json.JSONDecodeError, TypeError):
                raw_input = {"value": str(raw_input)}

        # Always inject goal_id so tools can reference it in their ToolResult
        input_data: dict[str, Any] = {**raw_input, "goal_id": goal_id}

        self._logger.executing(tool_name, raw_input, goal_id)

        # ── Resolve tool ────────────────────────────────────────────────────
        try:
            tool = self._registry.get(tool_name)
        except ToolRegistryError as exc:
            self._logger.error(f"Tool lookup failed: {exc}")
            return Observation(
                goal_id=goal_id,
                tool_name=tool_name,
                success=False,
                output=None,
                error=f"Tool '{tool_name}' not found. Available: {self._registry.names()}",
                execution_time=0.0,
                iteration=state.iterations,
            )

        # ── Execute ─────────────────────────────────────────────────────────
        start = time.monotonic()
        try:
            result: ToolResult = tool.execute(input_data)
        except Exception as exc:
            # Tool raised unexpectedly — wrap it
            elapsed = time.monotonic() - start
            self._logger.error(f"Tool '{tool_name}' raised: {exc}")
            return Observation(
                goal_id=goal_id,
                tool_name=tool_name,
                success=False,
                output=None,
                error=f"Unexpected exception in '{tool_name}': {exc}",
                execution_time=elapsed,
                iteration=state.iterations,
            )

        # ── Wrap in Observation ─────────────────────────────────────────────
        obs = Observation(
            goal_id=result.goal_id,
            tool_name=result.tool_name,
            success=result.success,
            output=result.output,
            error=result.error,
            execution_time=result.execution_time,
            iteration=state.iterations,
            metadata=result.metadata,
        )

        if result.success:
            self._logger.observation_received(obs)
        else:
            self._logger.tool_failed(tool_name, result.error or "no error message")

        return obs
