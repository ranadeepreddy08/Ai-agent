"""
RunManager — in-memory store for concurrent agent runs.

Manages the lifecycle of each dashboard run:
  - Creates an asyncio.Queue per run for live event streaming
  - Stores final result once the agent completes
  - Provides clean-up to prevent memory leaks

Thread safety: All state is stored in a plain dict. Access from
the FastAPI event loop is safe because FastAPI is single-threaded
per event loop. Agent runs execute in a thread-pool executor, but
they communicate only via the thread-safe asyncio.Queue mechanism.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunRecord:
    """One active or completed agent run."""
    run_id: str
    task: str
    status: str = "RUNNING"          # RUNNING | COMPLETED | FAILED
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None

    def age_seconds(self) -> float:
        return time.monotonic() - self.started_at


class RunManager:
    """
    Central registry for all active / recent dashboard agent runs.

    Runs are kept in memory and pruned after TTL_SECONDS to prevent
    unbounded growth during a demo session.
    """

    TTL_SECONDS = 3600  # keep run records for 1 hour

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}

    # ── Run lifecycle ──────────────────────────────────────────────────────────

    def create_run(self, task: str) -> RunRecord:
        """Create a new run record and return it."""
        run_id = str(uuid.uuid4())
        record = RunRecord(run_id=run_id, task=task)
        self._runs[run_id] = record
        self._evict_old()
        return record

    def get_run(self, run_id: str) -> RunRecord | None:
        """Return the record for run_id, or None if not found."""
        return self._runs.get(run_id)

    def complete_run(self, run_id: str, result: dict[str, Any]) -> None:
        """Mark a run as COMPLETED and store its final result."""
        record = self._runs.get(run_id)
        if record:
            record.result = result
            record.status = result.get("status", "COMPLETED")
            record.finished_at = time.monotonic()
            # Signal WebSocket consumers that the run is done
            record.queue.put_nowait({"type": "done", "result": result})

    def fail_run(self, run_id: str, error: str) -> None:
        """Mark a run as FAILED."""
        record = self._runs.get(run_id)
        if record:
            record.status = "FAILED"
            record.error = error
            record.finished_at = time.monotonic()
            record.queue.put_nowait({"type": "error", "error": error})

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _evict_old(self) -> None:
        """Remove completed runs older than TTL_SECONDS."""
        now = time.monotonic()
        stale = [
            rid for rid, rec in self._runs.items()
            if rec.status != "RUNNING" and (now - rec.started_at) > self.TTL_SECONDS
        ]
        for rid in stale:
            del self._runs[rid]

    def active_count(self) -> int:
        return sum(1 for r in self._runs.values() if r.status == "RUNNING")
