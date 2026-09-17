"""
Verifier — determines whether an observation satisfies a goal.

Verification strategy (Phase 1):
  1. Deterministic checks first (fast, zero LLM cost):
     - Tool failed → INVALID immediately
     - Output empty → INCOMPLETE
  2. Optional LLM semantic verification (disabled by default in Phase 1,
     enabled via use_llm_verification=True):
     - Uses LLM to judge whether output truly satisfies the goal description

Phase 3 will expand to all 5 statuses: VALID, INCOMPLETE, CONTRADICTORY,
INVALID, UNRELIABLE — with richer deterministic heuristics before LLM fallback.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from backend.core.state import Goal, Observation, VerificationStatus
from backend.llm.client import LLMClient
from backend.monitoring.logger import AgentLogger

# ── LLM verification prompt ───────────────────────────────────────────────────
_VERIFY_SYSTEM = """You are a verification module for an AI agent runtime.

Given a goal description and a tool observation, determine whether the observation
satisfactorily achieves the goal.

Return ONLY valid JSON:
{
  "status": "VALID" | "INCOMPLETE" | "CONTRADICTORY" | "INVALID" | "UNRELIABLE",
  "reason": "one-sentence explanation"
}

Status definitions:
  VALID         — The observation fully and correctly satisfies the goal.
  INCOMPLETE    — Partial answer; more information is still needed.
  CONTRADICTORY — Observation conflicts with other known information.
  INVALID       — Observation is wrong, irrelevant, or based on an error.
  UNRELIABLE    — Source is questionable or method is unreliable; treat with caution.
"""


@dataclass
class VerificationResult:
    """Outcome of verifying one observation against one goal."""
    status: VerificationStatus
    reason: str


class Verifier:
    """
    Verifies whether an observation satisfies its goal.

    Phase 1 mode (use_llm_verification=False, default):
      - Failed tool → INVALID
      - Empty output → INCOMPLETE
      - Non-empty successful output → VALID (deterministic, zero LLM cost)

    LLM mode (use_llm_verification=True):
      - Same deterministic pre-checks
      - Falls through to LLM semantic judgment for successful outputs
    """

    def __init__(
        self,
        llm: LLMClient,
        logger: AgentLogger,
        use_llm_verification: bool = False,
    ) -> None:
        self._llm = llm
        self._logger = logger
        self._use_llm = use_llm_verification

    def verify(self, goal: Goal, observation: Observation) -> VerificationResult:
        """
        Verify observation against goal.
        Always returns a VerificationResult — never raises.
        """
        # ── Fast deterministic checks ────────────────────────────────────────

        if not observation.success:
            result = VerificationResult(
                status=VerificationStatus.INVALID,
                reason=(
                    f"Tool '{observation.tool_name}' execution failed: "
                    f"{observation.error or 'unknown error'}"
                ),
            )
            self._logger.verification(goal.id, result)
            return result

        if observation.output is None:
            result = VerificationResult(
                status=VerificationStatus.INCOMPLETE,
                reason="Tool returned no output (None).",
            )
            self._logger.verification(goal.id, result)
            return result

        output_str = str(observation.output).strip()
        if not output_str or output_str in ("", "{}", "[]"):
            result = VerificationResult(
                status=VerificationStatus.INCOMPLETE,
                reason="Tool returned empty output.",
            )
            self._logger.verification(goal.id, result)
            return result

        # ── LLM semantic check (optional) ─────────────────────────────────
        if self._use_llm:
            return self._llm_verify(goal, observation)

        # ── Default: non-empty successful output → VALID ──────────────────
        result = VerificationResult(
            status=VerificationStatus.VALID,
            reason="Tool executed successfully with non-empty output (deterministic check).",
        )
        self._logger.verification(goal.id, result)
        return result

    def _llm_verify(self, goal: Goal, observation: Observation) -> VerificationResult:
        """Use LLM to semantically verify the observation satisfies the goal."""
        payload = json.dumps(
            {
                "goal": goal.description,
                "tool_used": observation.tool_name,
                "observation_output": str(observation.output)[:1500],
            },
            ensure_ascii=False,
        )
        try:
            data = self._llm.complete_json(_VERIFY_SYSTEM, payload)
            raw_status = data.get("status", "VALID").upper()
            # Guard against LLM returning an unrecognized status
            try:
                status = VerificationStatus(raw_status)
            except ValueError:
                status = VerificationStatus.UNRELIABLE
            result = VerificationResult(
                status=status,
                reason=str(data.get("reason", "LLM verification completed.")),
            )
        except Exception as exc:
            result = VerificationResult(
                status=VerificationStatus.UNRELIABLE,
                reason=f"LLM verification call failed: {exc}",
            )
        self._logger.verification(goal.id, result)
        return result
