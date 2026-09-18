/**
 * useAgentRun — custom hook managing the full lifecycle of one agent run.
 *
 * Flow:
 *  1. POST /api/run  → receive run_id
 *  2. Open WebSocket /ws/{run_id}
 *  3. Stream events → append to trace
 *  4. On "done" frame → store final result
 *  5. On "error" frame → store error
 */
import { useState, useRef, useCallback } from 'react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const WS_BASE  = API_BASE.replace(/^http/, 'ws');

export function useAgentRun() {
  const [status, setStatus]       = useState('idle');   // idle | running | done | error
  const [events, setEvents]       = useState([]);
  const [result, setResult]       = useState(null);
  const [errorMsg, setErrorMsg]   = useState(null);
  const [runId, setRunId]         = useState(null);
  const wsRef = useRef(null);

  const clearRun = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setStatus('idle');
    setEvents([]);
    setResult(null);
    setErrorMsg(null);
    setRunId(null);
  }, []);

  const startRun = useCallback(async (task, options = {}) => {
    // Reset state
    clearRun();
    setStatus('running');

    let runIdLocal;
    try {
      const resp = await fetch(`${API_BASE}/api/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task, ...options }),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(err.detail || `HTTP ${resp.status}`);
      }
      const data = await resp.json();
      runIdLocal = data.run_id;
      setRunId(runIdLocal);
    } catch (err) {
      setStatus('error');
      setErrorMsg(`Failed to start run: ${err.message}`);
      return;
    }

    // Open WebSocket
    const ws = new WebSocket(`${WS_BASE}/ws/${runIdLocal}`);
    wsRef.current = ws;

    ws.onmessage = (evt) => {
      let frame;
      try { frame = JSON.parse(evt.data); }
      catch { return; }

      if (frame.type === 'heartbeat') return;

      if (frame.type === 'done') {
        setResult(frame.result);
        setStatus('done');
        ws.close();
        return;
      }

      if (frame.type === 'error') {
        setErrorMsg(frame.error || 'Agent run failed');
        setStatus('error');
        ws.close();
        return;
      }

      // Regular log event
      setEvents(prev => [...prev, frame]);
    };

    ws.onerror = () => {
      setStatus('error');
      setErrorMsg('WebSocket connection failed. Is the API server running on port 8000?');
    };

    ws.onclose = () => {
      wsRef.current = null;
    };
  }, [clearRun]);

  return { status, events, result, errorMsg, runId, startRun, clearRun };
}
