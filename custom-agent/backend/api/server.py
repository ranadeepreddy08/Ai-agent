"""
FastAPI server — Phase 5 Dashboard API.

Endpoints:
    GET  /api/health              — liveness check
    POST /api/run                 — start an agent run, returns {run_id}
    GET  /api/runs/{run_id}       — poll final result (after completion)
    WS   /ws/{run_id}             — live event stream while agent is running

Architecture:
    - Agent.run_async() executes in asyncio.run_in_executor() (thread pool)
    - AgentLogger.set_event_queue() is called before the run starts
    - _emit() in the logger pushes JSON events via call_soon_threadsafe
    - WebSocket handler drains the queue and forwards to the browser
    - RunManager stores the result once the agent finishes

CORS is configured to allow the Vite dev server (localhost:5173) and any
local port. Restrict origins in production.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Path fix so we can import backend.* from this module ─────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from backend.api.run_manager import RunManager
from backend.core.agent import Agent


# ── Application lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hook."""
    print("🚀  Phase 5 Dashboard API starting…")
    yield
    print("🛑  Phase 5 Dashboard API shutting down.")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Adaptive Agent Runtime — Dashboard API",
    description="Phase 5: Live execution trace dashboard for the custom AI agent.",
    version="5.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Vite dev server + any local origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global RunManager (shared across requests)
_run_manager = RunManager()


# ── Request / response models ─────────────────────────────────────────────────

class RunRequest(BaseModel):
    task: str
    use_parallel: bool = False
    max_iterations: int = 15
    max_tool_calls: int = 25
    max_llm_calls: int = 40
    max_retries: int = 9


class RunStarted(BaseModel):
    run_id: str
    task: str
    message: str = "Agent run started"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Liveness check — returns server status and active run count."""
    return {
        "status": "ok",
        "active_runs": _run_manager.active_count(),
        "version": "5.0.0",
    }


@app.post("/api/run", response_model=RunStarted)
async def start_run(req: RunRequest) -> RunStarted:
    """
    Start an agent run asynchronously.

    Returns immediately with a run_id. Connect to /ws/{run_id} to
    receive live events, or poll /api/runs/{run_id} for the final result.
    """
    task = req.task.strip()
    if not task:
        raise HTTPException(status_code=400, detail="Task must not be empty.")

    record = _run_manager.create_run(task)
    run_id = record.run_id
    event_queue = record.queue

    # Build agent config
    cfg = {
        "max_iterations": req.max_iterations,
        "max_tool_calls": req.max_tool_calls,
        "max_llm_calls": req.max_llm_calls,
        "max_retries": req.max_retries,
        "use_llm_verification": False,
        "use_replanner": True,
        "use_recovery": True,
    }

    # Capture the running event loop for thread-safe queue access
    loop = asyncio.get_running_loop()

    async def _run_agent() -> None:
        """Run the agent in the default thread pool executor."""
        def _blocking_run() -> dict:
            agent = Agent(config=cfg, verbose=True)
            # Attach the live event queue BEFORE running
            agent.logger.set_event_queue(event_queue, loop=loop)
            try:
                if req.use_parallel:
                    return agent.run_parallel(task)
                return agent.run(task)
            finally:
                # Detach queue when done
                agent.logger.set_event_queue(None)

        try:
            result = await loop.run_in_executor(None, _blocking_run)
            _run_manager.complete_run(run_id, result)
        except Exception as exc:
            _run_manager.fail_run(run_id, str(exc))

    # Fire and forget — the WebSocket / poll endpoints track progress
    asyncio.create_task(_run_agent())

    return RunStarted(run_id=run_id, task=task)


@app.get("/api/runs/{run_id}")
async def get_run_result(run_id: str) -> dict[str, Any]:
    """
    Poll the result of a completed run.

    Returns the full result dict once the agent finishes,
    or a status snapshot if still running.
    """
    record = _run_manager.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")

    if record.status == "RUNNING":
        return {
            "run_id": run_id,
            "status": "RUNNING",
            "message": "Agent is still running — connect to /ws/{run_id} for live events.",
        }

    return {
        "run_id": run_id,
        "status": record.status,
        "task": record.task,
        "result": record.result,
        "error": record.error,
        "elapsed_seconds": round(record.finished_at - record.started_at, 2)
        if record.finished_at else None,
    }


@app.websocket("/ws/{run_id}")
async def websocket_events(websocket: WebSocket, run_id: str) -> None:
    """
    WebSocket — live event stream for a run.

    Protocol:
        Server → Client: JSON event objects as text frames
            { "ts": "HH:MM:SS.mmm", "emoji": "🔧", "message": "...", "data": {...} }
            { "type": "done", "result": {...} }   ← terminal frame
            { "type": "error", "error": "..." }   ← terminal frame on failure
        Client → Server: ignored (read-only stream)

    The connection stays open until a terminal frame is sent or the
    agent run finishes.
    """
    record = _run_manager.get_run(run_id)
    if record is None:
        await websocket.close(code=4004, reason=f"Run {run_id!r} not found.")
        return

    await websocket.accept()

    # If the run already finished before the WS connected, replay all events
    # then send the terminal frame immediately.
    if record.status != "RUNNING":
        # Drain any queued events first
        while not record.queue.empty():
            try:
                event = record.queue.get_nowait()
                await websocket.send_text(json.dumps(event, default=str))
            except Exception:
                break
        # Send terminal frame
        if record.result:
            await websocket.send_text(json.dumps({"type": "done", "result": record.result}, default=str))
        else:
            await websocket.send_text(json.dumps({"type": "error", "error": record.error or "Unknown error"}, default=str))
        await websocket.close()
        return

    # Stream events until a terminal frame arrives
    try:
        while True:
            try:
                event = await asyncio.wait_for(record.queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send a heartbeat to keep the connection alive
                await websocket.send_text(json.dumps({"type": "heartbeat"}))
                continue

            await websocket.send_text(json.dumps(event, default=str))

            # Terminal frames end the stream
            if isinstance(event, dict) and event.get("type") in ("done", "error"):
                break

    except WebSocketDisconnect:
        pass  # Client disconnected — that's fine
    except Exception as exc:
        try:
            await websocket.send_text(json.dumps({"type": "error", "error": str(exc)}, default=str))
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
