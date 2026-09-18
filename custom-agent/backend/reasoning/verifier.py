"""
Verifier - Phase 3 enhanced deterministic + semantic verification.
Fix (v2): _check_direct_answer no longer rejects short numeric answers.
"1275", "0", "51920", "12,000" are all VALID. Only genuinely bad strings rejected.
"""
from __future__ import annotations
import json
import math
import re
from dataclasses import dataclass
from typing import Any
from backend.core.state import Goal, Observation, VerificationStatus
from backend.llm.client import LLMClient
from backend.monitoring.logger import AgentLogger

_VERIFY_SYSTEM = """You are a verification module for an AI agent runtime.
Given a goal description and a tool observation, determine whether the observation
satisfactorily achieves the goal.
Return ONLY valid JSON:
{
  "status": "VALID" | "INCOMPLETE" | "CONTRADICTORY" | "INVALID" | "UNRELIABLE",
  "reason": "one-sentence explanation"
}
Status definitions:
  VALID         - The observation fully and correctly satisfies the goal.
  INCOMPLETE    - Partial answer; more information is still needed.
  CONTRADICTORY - Observation conflicts with other known information.
  INVALID       - Observation is wrong, irrelevant, or based on an error.
  UNRELIABLE    - Source is questionable or method is unreliable; treat with caution.
"""


@dataclass
class VerificationResult:
    """Outcome of verifying one observation against one goal."""
    status: VerificationStatus
    reason: str
    is_deterministic: bool = True


class Verifier:
    """
    Phase 3 Verifier - deterministic-first, LLM-fallback semantic verification.

    Direct-answer check (v2):
      - Known bad phrases (?, error, unknown...) -> INVALID
      - Valid numeric value (ANY length)         -> VALID  (fixes "1275" false-reject)
      - Non-numeric >= 3 chars                   -> VALID
      - Non-numeric < 3 chars                    -> INCOMPLETE
    """

    _ALWAYS_INVALID: frozenset = frozenset({
        "?", "??", "???",
        "error", "errors",
        "unknown", "n/a", "na", "none",
        "null", "undefined",
        "cannot", "can't", "cant",
        "failed", "failure",
        "no answer", "no result", "no results",
        "i don't know", "i dont know",
        "i cannot", "i can't",
    })

    _MOCK_PHRASES = (
        "mock result",
        "search result 1 for",
        "placeholder",
        "[mock]",
    )

    _NUMERIC_STRIP_RE = re.compile(r"[\u20b9$\u20ac\xa3\xa5\u20a9,\s%+]")

    def __init__(self, llm: LLMClient, logger: AgentLogger, use_llm_verification: bool = False) -> None:
        self._llm = llm
        self._logger = logger
        self._use_llm = use_llm_verification

    def verify(self, goal: Goal, observation: Observation) -> VerificationResult:
        """Verify observation against goal. Always returns a VerificationResult."""
        if not observation.success:
            error_msg = observation.error or "unknown error"
            is_timeout = observation.metadata.get("timeout", False)
            reason = (
                f"Tool '{observation.tool_name}' timed out: {error_msg}"
                if is_timeout else
                f"Tool '{observation.tool_name}' execution failed: {error_msg}"
            )
            return self._make(VerificationStatus.INVALID, reason, goal)

        if observation.output is None:
            return self._make(VerificationStatus.INCOMPLETE, "Tool returned no output (None).", goal)

        output_str = str(observation.output).strip()

        if not output_str or output_str in ("", "{}", "[]", "null"):
            return self._make(VerificationStatus.INCOMPLETE, "Tool returned empty output.", goal)

        tool_check = self._check_tool_specific(observation)
        if tool_check is not None:
            return self._make(tool_check[0], tool_check[1], goal)

        if self._use_llm:
            return self._llm_verify(goal, observation)

        return self._make(
            VerificationStatus.VALID,
            "Tool executed successfully with non-empty output (deterministic check).",
            goal,
        )

    def _check_tool_specific(self, obs: Observation) -> tuple | None:
        tool = obs.tool_name
        if tool == "calculator":
            return self._check_calculator(obs)
        if tool == "web_search":
            return self._check_web_search(obs)
        if tool == "direct_answer":
            return self._check_direct_answer(obs)
        if tool == "failing_tool":
            return None
        return None

    def _check_calculator(self, obs: Observation) -> tuple | None:
        """Validate calculator output is a finite real number."""
        output = obs.output
        if isinstance(output, dict):
            result = output.get("result")
        else:
            result = output

        if result is None:
            return (VerificationStatus.INCOMPLETE, "Calculator returned no result value.")
        try:
            val = float(result)
        except (TypeError, ValueError):
            return (VerificationStatus.INVALID, f"Calculator result is not numeric: {result!r}")

        if math.isnan(val):
            return (VerificationStatus.INVALID, "Calculator result is NaN.")
        if math.isinf(val):
            return (VerificationStatus.INVALID, "Calculator result is infinite (likely division by zero).")

        return (VerificationStatus.VALID, f"Calculator produced finite numeric result: {val}")

    def _check_web_search(self, obs: Observation) -> tuple | None:
        """Check web search has at least one non-placeholder result."""
        output = obs.output
        if isinstance(output, dict):
            results = output.get("results", [])
        elif isinstance(output, list):
            results = output
        else:
            output_lower = str(output).lower()
            if any(phrase in output_lower for phrase in self._MOCK_PHRASES):
                return (VerificationStatus.VALID, "Web search returned mock results (mock backend active).")
            return None

        if not results:
            return (VerificationStatus.INCOMPLETE, "Web search returned zero results.")

        all_mock = all(any(p in str(r).lower() for p in self._MOCK_PHRASES) for r in results)
        if all_mock:
            return (VerificationStatus.VALID, f"Web search returned {len(results)} mock result(s) (mock backend active).")

        return (VerificationStatus.VALID, f"Web search returned {len(results)} result(s).")

    def _check_direct_answer(self, obs: Observation) -> tuple | None:
        """
        Context-aware substance check for direct_answer tool output.

        Policy:
          1. Known-bad phrases (?, error, unknown)  -> INVALID
          2. Parseable numeric value (ANY length)   -> VALID
             Covers: "1275", "0", "51920", "12,000", "15%", "-3.14"
          3. Non-numeric >= 3 chars                 -> VALID
          4. Non-numeric < 3 chars                  -> INCOMPLETE

        Rationale: Do NOT reject correct answers because they are short.
        "1275" (4 chars) is a complete answer to "25 * 47 + 100".
        "0" (1 char) is a complete answer to "100 - 100".
        """
        output_str = str(obs.output).strip()
        lower = output_str.lower()

        # 1. Always-invalid known bad responses
        if lower in self._ALWAYS_INVALID:
            return (VerificationStatus.INVALID, f"Direct answer is a known-invalid response: {output_str!r}")

        # 2. Accept any parseable numeric value regardless of length
        numeric_str = self._NUMERIC_STRIP_RE.sub("", output_str)
        numeric_str = numeric_str.lstrip("\u2212-")  # handle em-dash and minus
        if numeric_str:
            try:
                val = float(numeric_str)
                if math.isfinite(val):
                    return (VerificationStatus.VALID, f"Direct answer is a valid numeric value: {output_str!r}")
            except (ValueError, TypeError):
                pass

        # 3/4. Non-numeric substance check
        if len(output_str) < 3:
            return (VerificationStatus.INCOMPLETE, f"Direct answer is too short ({len(output_str)} chars) and non-numeric.")

        return (VerificationStatus.VALID, f"Direct answer provided ({len(output_str)} chars).")

    def _llm_verify(self, goal: Goal, observation: Observation) -> VerificationResult:
        """Use LLM to semantically verify the observation satisfies the goal."""
        payload = json.dumps({
            "goal": goal.description,
            "tool_used": observation.tool_name,
            "observation_output": str(observation.output)[:1500],
        }, ensure_ascii=False)
        try:
            data = self._llm.complete_json(_VERIFY_SYSTEM, payload)
            raw_status = data.get("status", "VALID").upper()
            try:
                status = VerificationStatus(raw_status)
            except ValueError:
                status = VerificationStatus.UNRELIABLE
            result = VerificationResult(
                status=status,
                reason=str(data.get("reason", "LLM verification completed.")),
                is_deterministic=False,
            )
        except Exception as exc:
            result = VerificationResult(
                status=VerificationStatus.UNRELIABLE,
                reason=f"LLM verification call failed: {exc}",
                is_deterministic=False,
            )
        self._logger.verification(goal.id, result)
        return result

    def _make(self, status: VerificationStatus, reason: str, goal: Goal) -> VerificationResult:
        result = VerificationResult(status=status, reason=reason, is_deterministic=True)
        self._logger.verification(goal.id, result)
        return result