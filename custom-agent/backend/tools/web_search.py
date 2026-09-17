"""
Web Search Tool — pluggable backend architecture.

Phase 1: MockSearchBackend — no API key required, deterministic results.
Production: Set TAVILY_API_KEY in .env → TavilyBackend activates automatically.

Architecture:
  ┌─────────────────┐
  │  WebSearchTool  │  ← agent runtime only talks to this class
  └────────┬────────┘
           │ delegates to
  ┌────────▼────────────────────────────────────────┐
  │  SearchBackend (ABC)                            │
  │  ├── MockSearchBackend   (no API key needed)    │
  │  ├── TavilyBackend       (TAVILY_API_KEY set)   │
  │  └── SerperBackend       (future, Phase 2+)     │
  └─────────────────────────────────────────────────┘

The agent runtime (loop_controller.py, executor.py) never changes
when the backend is swapped — only .env changes.
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any

from backend.tools.base import BaseTool, ToolResult


# ── Backend protocol ─────────────────────────────────────────────────────────

class SearchBackend(ABC):
    """Abstract backend protocol for web search providers."""

    @abstractmethod
    def search(self, query: str) -> list[dict[str, str]]:
        """
        Execute a search query.
        Returns a list of result dicts, each with: title, url, snippet.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name for logging."""
        ...


# ── Mock backend ─────────────────────────────────────────────────────────────

class MockSearchBackend(SearchBackend):
    """
    Deterministic mock search results for Phase 1 testing.
    Keyword-matching against a small in-memory database.
    No network calls. No API keys needed.
    """

    provider_name = "mock"

    # Keyword → result mapping for demo scenarios
    _RESULTS: dict[str, list[dict[str, str]]] = {
        "ai agent": [
            {
                "title": "What Are AI Agents? A Complete Guide",
                "url": "https://example.com/ai-agents-guide",
                "snippet": (
                    "AI agents are autonomous software systems that perceive their environment, "
                    "reason about goals, select actions, and execute them in a loop. "
                    "Modern agents use large language models for planning and tool use."
                ),
            },
            {
                "title": "Building Custom AI Agent Frameworks",
                "url": "https://example.com/custom-agents",
                "snippet": (
                    "Custom agent frameworks built from scratch give developers full control "
                    "over the agent loop: planning, tool selection, observation, verification, "
                    "recovery, and replanning — without depending on third-party orchestration libraries."
                ),
            },
        ],
        "savings rate": [
            {
                "title": "Best High-Yield Savings Accounts 2024",
                "url": "https://example.com/savings-rates",
                "snippet": (
                    "Top high-yield savings accounts are currently offering 4.5% to 5.1% APY "
                    "as of late 2024. Traditional bank savings accounts average 0.45% APY. "
                    "Online banks like Marcus, Ally, and SoFi lead with competitive rates."
                ),
            },
        ],
        "compound interest": [
            {
                "title": "Compound Interest Formula & Calculator",
                "url": "https://example.com/compound-interest",
                "snippet": (
                    "Compound interest formula: A = P(1 + r/n)^(nt), where P=principal, "
                    "r=annual rate, n=compounds per year, t=years. "
                    "For $10,000 at 5% compounded annually for 3 years: A = 10000 × 1.05³ = $11,576.25"
                ),
            },
        ],
        "python": [
            {
                "title": "Python Programming Language",
                "url": "https://python.org",
                "snippet": (
                    "Python is a high-level, general-purpose programming language known for its "
                    "readability, extensive standard library, and wide ecosystem. "
                    "Latest stable version: Python 3.12."
                ),
            },
        ],
        "weather": [
            {
                "title": "Weather Forecast API",
                "url": "https://example.com/weather",
                "snippet": (
                    "Note: This is a mock result. Real weather data requires a live API. "
                    "Current conditions: unavailable in mock mode."
                ),
            },
        ],
    }

    _DEFAULT_RESULTS = [
        {
            "title": "Search Result (Mock)",
            "url": "https://example.com/result",
            "snippet": (
                "This is a mock search result for testing the agent loop. "
                "Set TAVILY_API_KEY in .env to enable real web search."
            ),
        },
        {
            "title": "About Mock Mode",
            "url": "https://example.com/mock-mode",
            "snippet": (
                "The agent is running with the mock web search backend. "
                "Results are synthetic and used to validate the agent orchestration loop."
            ),
        },
    ]

    def search(self, query: str) -> list[dict[str, str]]:
        query_lower = query.lower()
        for keyword, results in self._RESULTS.items():
            if keyword in query_lower:
                return results
        return self._DEFAULT_RESULTS


# ── Tavily backend ────────────────────────────────────────────────────────────

class TavilyBackend(SearchBackend):
    """
    Real web search via Tavily API.
    Activated automatically when TAVILY_API_KEY is set in .env.
    """

    provider_name = "tavily"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def search(self, query: str) -> list[dict[str, str]]:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "The 'requests' package is required for Tavily search. "
                "Add 'requests' to requirements.txt and reinstall."
            ) from exc

        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self._api_key,
                    "query": query,
                    "max_results": 5,
                    "search_depth": "basic",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", ""),
                }
                for r in data.get("results", [])
            ]
        except Exception as exc:
            raise RuntimeError(f"Tavily API error: {exc}") from exc


# ── Backend factory ───────────────────────────────────────────────────────────

def _select_backend() -> SearchBackend:
    """
    Auto-select the best available search backend.
    Priority: Tavily > Mock
    Adding Serper or another backend: add its check here only.
    """
    if api_key := os.getenv("TAVILY_API_KEY"):
        return TavilyBackend(api_key=api_key)
    return MockSearchBackend()


# ── Public tool class ─────────────────────────────────────────────────────────

class WebSearchTool(BaseTool):
    """
    Web search tool with a pluggable backend.

    The agent runtime always interacts with this class.
    The backend (mock vs real) is selected at construction time
    based on environment variables — zero agent loop changes needed.
    """

    name = "web_search"
    description = (
        "Searches the web for current information, news, facts, documentation, "
        "or research. Returns a list of relevant results with titles, URLs, and "
        "content snippets. Use for questions requiring up-to-date or external information "
        "that cannot be answered from training knowledge alone."
    )
    capabilities = [
        "web search",
        "current events",
        "factual lookup",
        "news retrieval",
        "research",
        "finding sources and URLs",
        "real-world data retrieval",
        "financial rates and market data",
    ]
    input_schema = {
        "query": {
            "type": "string",
            "description": (
                "Search query string. Be specific and descriptive. "
                "Example: 'current high-yield savings account interest rates 2024'"
            ),
            "required": True,
        }
    }

    def __init__(self) -> None:
        self._backend: SearchBackend = _select_backend()

    @property
    def backend_name(self) -> str:
        return self._backend.provider_name

    def execute(self, input_data: dict[str, Any]) -> ToolResult:
        start = time.monotonic()
        goal_id = input_data.get("goal_id", "unknown")
        query = str(input_data.get("query", "")).strip()

        if not query:
            return ToolResult(
                success=False,
                output=None,
                error="No query provided. Supply a 'query' key.",
                execution_time=time.monotonic() - start,
                tool_name=self.name,
                goal_id=goal_id,
            )

        try:
            results = self._backend.search(query)
            return ToolResult(
                success=True,
                output={
                    "query": query,
                    "results": results,
                    "result_count": len(results),
                },
                error=None,
                execution_time=time.monotonic() - start,
                tool_name=self.name,
                goal_id=goal_id,
                metadata={"backend": self._backend.provider_name},
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output=None,
                error=str(exc),
                execution_time=time.monotonic() - start,
                tool_name=self.name,
                goal_id=goal_id,
                metadata={"backend": self._backend.provider_name},
            )
