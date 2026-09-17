"""
Tool Registry — plugin-style tool store.

Design principles:
  - Adding a new tool = call registry.register(MyTool()). That's it.
  - The agent runtime queries by name. No tool-specific code anywhere else.
  - list_for_llm() feeds the full tool catalogue into every LLM tool-selection prompt.
"""
from __future__ import annotations

from typing import Any

from backend.tools.base import BaseTool


class ToolRegistryError(Exception):
    """Raised on registry misuse (duplicate name, tool not found)."""


class ToolRegistry:
    """
    Immutable-after-registration plugin store for agent tools.

    Usage:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(WebSearchTool())

        tool = registry.get("calculator")
        all_tools = registry.list_for_llm()   # passed to LLM prompts
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    # ── Mutation ─────────────────────────────────────────────────────────────

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool instance.
        Raises ToolRegistryError if a tool with the same name already exists.
        """
        if tool.name in self._tools:
            raise ToolRegistryError(
                f"Tool '{tool.name}' is already registered. "
                "Each tool name must be unique."
            )
        self._tools[tool.name] = tool

    def register_if_absent(self, tool: BaseTool) -> None:
        """Register a tool only if not already registered (useful in tests)."""
        if tool.name not in self._tools:
            self._tools[tool.name] = tool

    # ── Queries ───────────────────────────────────────────────────────────────

    def get(self, name: str) -> BaseTool:
        """
        Retrieve a registered tool by name.
        Raises ToolRegistryError with available names if not found.
        """
        if name not in self._tools:
            raise ToolRegistryError(
                f"Tool '{name}' not found in registry. "
                f"Available: {self.names()}"
            )
        return self._tools[name]

    def has(self, name: str) -> bool:
        """Return True if a tool with this name is registered."""
        return name in self._tools

    def list_tools(self) -> list[BaseTool]:
        """Return all registered tool instances."""
        return list(self._tools.values())

    def list_for_llm(self) -> list[dict[str, Any]]:
        """
        Return all tools serialized for LLM prompt inclusion.
        The LLM sees this list when deciding which tool to call.
        """
        return [tool.to_registry_entry() for tool in self._tools.values()]

    def names(self) -> list[str]:
        """Return list of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"<ToolRegistry tools={self.names()}>"
