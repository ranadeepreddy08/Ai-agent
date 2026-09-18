"""
Agent trace logger — structured, emoji-annotated terminal output.

Phase 1: Colored terminal trace.
Phase 4: Added p4_loop/p4_dag/p4_executor/p4_budget/p4_stuck for the
         structured demo trace that makes the parallel execution visible.
Phase 5: Each _emit() call will also push a JSON event to a WebSocket queue
         for the React dashboard (hook point marked with TODO: PHASE 5).

Design: Logger is injected into every component. Never imported as a singleton.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.core.state import Goal, Observation
    from backend.reasoning.verifier import VerificationResult

# ── ANSI color codes ──────────────────────────────────────────────────────────
RST   = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
CYAN  = "\033[96m"
GREEN = "\033[92m"
YELL  = "\033[93m"
RED   = "\033[91m"
MAG   = "\033[95m"
BLUE  = "\033[94m"
WHITE = "\033[97m"


class AgentLogger:
    """
    Structured trace logger for the agent runtime.

    Every significant decision, action, and state change in the agent loop
    produces an emoji-annotated log line. This is what makes the agent
    observable without adding external telemetry.
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self._events: list[dict[str, Any]] = []  # For Phase 5 WebSocket export

    # ── Core emit ─────────────────────────────────────────────────────────────

    def _emit(
        self,
        emoji: str,
        color: str,
        message: str,
        data: dict | None = None,
    ) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        if self.verbose:
            print(f"{DIM}[{ts}]{RST} {emoji}  {color}{message}{RST}")
        event: dict[str, Any] = {"ts": ts, "emoji": emoji, "message": message}
        if data:
            event["data"] = data
        self._events.append(event)
        # TODO: PHASE 5 — push event to asyncio.Queue for WebSocket broadcast

    # ── Planning ──────────────────────────────────────────────────────────────

    def planning(self, task: str) -> None:
        self._emit("🧠", CYAN + BOLD, f"Planning: {task!r}")

    def goals_created(
        self,
        goals: "list[Goal]",
        strategy: str,
        reasoning: str = "",
    ) -> None:
        self._emit(
            "🎯", GREEN + BOLD,
            f"Created {len(goals)} goal(s) | strategy: {strategy}"
        )
        if reasoning and self.verbose:
            print(f"   {DIM}└─ Reasoning: {reasoning}{RST}")
        for g in goals:
            deps = f" → requires: {', '.join(g.dependencies)}" if g.dependencies else ""
            print(f"   {BLUE}• [{g.id}] p{g.priority}: {g.description}{deps}{RST}")

    # ── Goal lifecycle ────────────────────────────────────────────────────────

    def goal_start(self, goal_id: str, description: str) -> None:
        self._emit("▶️ ", YELL + BOLD, f"[{goal_id}] {description}")

    def goal_completed(self, goal_id: str) -> None:
        self._emit("✅", GREEN + BOLD, f"Goal {goal_id} completed")

    def goal_failed(self, goal_id: str, reason: str) -> None:
        self._emit("❌", RED + BOLD, f"Goal {goal_id} failed: {reason}")

    # ── Action / tool ─────────────────────────────────────────────────────────

    def tool_selected(self, tool: str, input_summary: Any) -> None:
        summary = str(input_summary)
        if len(summary) > 120:
            summary = summary[:117] + "..."
        self._emit("🔧", MAG, f"Selected: {BOLD}{tool}{RST}{MAG} | {summary}")

    def executing(self, tool: str, input_data: Any, goal_id: str) -> None:
        self._emit("⚡", YELL, f"Executing {BOLD}{tool}{RST}{YELL} → [{goal_id}]")

    def tool_failed(self, tool: str, error: str) -> None:
        self._emit("❌", RED, f"Tool '{tool}' failed: {error[:120]}")

    # ── Observation ───────────────────────────────────────────────────────────

    def observation_received(self, obs: "Observation") -> None:
        output_str = str(obs.output) if obs.output else "(empty)"
        snippet = output_str[:100] + "…" if len(output_str) > 100 else output_str
        self._emit(
            "✅", GREEN,
            f"Observation: {obs.tool_name} | {snippet}"
        )

    # ── Verification ──────────────────────────────────────────────────────────

    def verification(self, goal_id: str, result: "VerificationResult") -> None:
        _status_color = {
            "VALID":         GREEN,
            "INCOMPLETE":    YELL,
            "CONTRADICTORY": RED,
            "INVALID":       RED,
            "UNRELIABLE":    YELL,
            "SKIPPED":       DIM,
        }
        color = _status_color.get(result.status.value, WHITE)
        self._emit(
            "🔍", color,
            f"Verification [{goal_id}]: {BOLD}{result.status.value}{RST}{color}"
            f" — {result.reason}"
        )

    # ── Recovery / replanning ─────────────────────────────────────────────────

    def recovery_triggered(self, reason: str) -> None:
        self._emit("🔄", YELL + BOLD, f"Recovery: {reason}")

    def retry(self, attempt: int, tool: str) -> None:
        self._emit("🔁", YELL, f"Retry #{attempt} with '{tool}'")

    def replanning(self, reason: str) -> None:
        self._emit("🔀", MAG + BOLD, f"Replanning: {reason}")

    # ── Synthesis / completion ────────────────────────────────────────────────

    def synthesizing(self) -> None:
        self._emit("📝", CYAN + BOLD, "Synthesizing final answer…")

    def final_answer(self, answer: str) -> None:
        bar = "─" * 60
        print(f"\n{GREEN}{BOLD}{bar}{RST}")
        print(f"{GREEN}{BOLD}FINAL ANSWER:{RST}")
        print(f"{WHITE}{answer}{RST}")
        print(f"{GREEN}{BOLD}{bar}{RST}")

    def budget_summary(self, summary: dict) -> None:
        self._emit(
            "📊", BLUE,
            f"Budget used — "
            f"iterations: {summary.get('iterations', '?')} | "
            f"tool calls: {summary.get('tool_calls', '?')} | "
            f"LLM calls: {summary.get('llm_calls', '?')} | "
            f"elapsed: {summary.get('elapsed_seconds', '?')}s"
        )

    # ── General ───────────────────────────────────────────────────────────────

    def error(self, message: str) -> None:
        self._emit("💥", RED + BOLD, f"ERROR: {message}")

    def warn(self, message: str) -> None:
        self._emit("⚠️ ", YELL, message)

    def info(self, message: str) -> None:
        self._emit("ℹ️ ", BLUE, message)

    # ── Phase 4 structured labels ─────────────────────────────────────────────
    # These produce the visible [LOOP]/[DAG]/[EXECUTOR]/[BUDGET]/[STUCK] labels
    # in the terminal trace — critical for buildathon demo clarity.

    def p4_loop(self, iteration: int, message: str = "") -> None:
        """Marks the start of each async coordinator iteration."""
        self._emit(
            "🔁", CYAN + BOLD,
            f"[LOOP] Iteration {iteration}" + (f" — {message}" if message else ""),
        )

    def p4_dag(self, message: str) -> None:
        """DAG scheduling decision (ready goals, blocked goals, etc.)."""
        self._emit("🗺 ", BLUE + BOLD, f"[DAG] {message}")

    def p4_executor(self, message: str) -> None:
        """Async executor concurrency events (launch/concurrent/gather)."""
        self._emit("⚙️ ", MAG + BOLD, f"[EXECUTOR] {message}")

    def p4_budget(self, used: dict, limits: dict) -> None:
        """Budget checkpoint — shows usage vs. limits at each iteration."""
        self._emit(
            "💰", BLUE,
            f"[BUDGET] LLM calls: {used.get('llm_calls', '?')}/{limits.get('max_llm_calls', '?')} | "
            f"Tool calls: {used.get('tool_calls', '?')}/{limits.get('max_tool_calls', '?')} | "
            f"Iterations: {used.get('iterations', '?')}/{limits.get('max_iterations', '?')}",
        )

    def p4_stuck(self, is_stuck: bool, reason: str = "", detail: str = "") -> None:
        """Stuck detection result for each iteration."""
        if is_stuck:
            self._emit("🔴", RED + BOLD, f"[STUCK] {reason}: {detail}")
        else:
            self._emit(
                "🟢", GREEN,
                f"[STUCK] NOT_STUCK" + (f" — {detail}" if detail else ""),
            )

    # ── Event export (Phase 5 hook) ───────────────────────────────────────────

    def get_events(self) -> list[dict[str, Any]]:
        """Return all logged events (for WebSocket broadcast in Phase 5)."""
        return self._events
