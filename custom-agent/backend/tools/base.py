"""
Tool base classes.

Every tool in the framework implements BaseTool.
The agent runtime dispatches ONLY by tool.name — no isinstance checks,
no hardcoded routing, no tool-specific logic in the loop.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """
    Normalized output from any tool execution.

    Every tool returns this exact structure so the Observation layer
    and Verifier can work uniformly regardless of which tool ran.
    """
    success: bool
    output: Any                      # Tool-specific payload
    error: str | None                # Error message if success=False
    execution_time: float            # Wall-clock seconds
    tool_name: str                   # Matches BaseTool.name
    goal_id: str                     # Which goal triggered this execution
    metadata: dict[str, Any] = field(default_factory=dict)  # Optional extras


class BaseTool(ABC):
    """
    Abstract base class for all agent tools.

    Subclass contract:
      1. Declare class-level attributes: name, description, capabilities, input_schema.
      2. Implement execute(input_data) -> ToolResult.
      3. Register with ToolRegistry — nothing else in the agent loop changes.

    The LLM sees to_registry_entry() output when selecting tools.
    """

    # ── Class-level declarations (override in subclass) ──────────────────────
    name: str           # Unique identifier used by the registry and LLM
    description: str    # Human/LLM-readable explanation of what this tool does
    capabilities: list[str]     # Short capability tags for LLM tool selection
    input_schema: dict[str, Any]  # Describes expected input fields

    # ── Required implementation ──────────────────────────────────────────────

    @abstractmethod
    def execute(self, input_data: dict[str, Any]) -> ToolResult:
        """
        Execute the tool.

        input_data always contains at least {"goal_id": str}.
        Additional keys defined by the tool's input_schema.
        Must never raise — return a failed ToolResult instead.
        """
        ...

    # ── Provided methods ─────────────────────────────────────────────────────

    def to_registry_entry(self) -> dict[str, Any]:
        """Serialize this tool for inclusion in LLM prompts."""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "input_schema": self.input_schema,
        }

    def validate_input(self, input_data: dict[str, Any]) -> tuple[bool, str]:
        """
        Lightweight input validation. Override for strict checking.
        Returns (is_valid, error_message).
        Default: always valid (tools handle their own error cases).
        """
        return True, ""

    def __repr__(self) -> str:
        return f"<Tool: {self.name}>"
