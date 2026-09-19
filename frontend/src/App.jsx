/**
 * App - main shell for the Adaptive Agent Runtime dashboard.
 *
 * Layout (desktop):
 *   +--header---------------------------------------------------+
 *   | LEFT PANEL                | RIGHT PANEL                   |
 *   |  [Task input]             | [Answer Panel]                |
 *   |  ─────────────            |   Final Answer (scrollable)   |
 *   |  [Trace Panel]            |   ─────────────────────────   |
 *   |   live events             | [Stats Panel]                 |
 *   |   (scrollable)            |   goals, budget, last tool    |
 *   +-----------------------------------------------------------+
 *
 * The trace and answer panels scroll INDEPENDENTLY.
 */
import { useState, useCallback } from 'react';
import { useAgentRun } from './hooks/useAgentRun';
import { TracePanel } from './components/TracePanel';
import { AnswerPanel } from './components/AnswerPanel';
import { StatusPanel } from './components/StatusPanel';

const EXAMPLE_TASKS = [
  'What is 25 * 47 + 100?',
  'Calculate 15% of 80,000',
  'Calculate compound interest on $10,000 at 5% for 3 years, then search for current savings rates',
  'Calculate 99 * 99 and search for the speed of light in km/s simultaneously',
];

function StatusDot({ status }) {
  const label = {
    idle: 'Ready',
    running: 'Running',
    done: 'Completed',
    error: 'Error',
  }[status] || 'Ready';

  return (
    <div className="app-header-status">
      <div className={`status-dot ${status}`} />
      {label}
    </div>
  );
}

export default function App() {
  const [task, setTask]            = useState('');
  const [useParallel, setParallel] = useState(false);
  const { status, events, result, errorMsg, startRun, clearRun } = useAgentRun();

  const isRunning = status === 'running';

  const handleRun = useCallback(async () => {
    const t = task.trim();
    if (!t || isRunning) return;
    await startRun(t, { use_parallel: useParallel });
  }, [task, isRunning, startRun, useParallel]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleRun();
    }
  }, [handleRun]);

  const handleExample = useCallback((ex) => {
    setTask(ex);
  }, []);

  return (
    <div className="app">
      {/* ── Header ── */}
      <div className="app-header">
        <span className="app-header-logo">🤖</span>
        <div>
          <div className="app-header-title">
            Adaptive Agent Runtime
            <span className="app-header-subtitle"> · Custom AI Brain</span>
          </div>
        </div>
        <div className="app-header-spacer" />
        <StatusDot status={status} />
      </div>

      {/* ── Body: two main columns ── */}
      <div className="app-body">

        {/* ── LEFT: Task input + Execution Trace ── */}
        <div className="left-panel">
          {/* Input section */}
          <div className="input-section">
            <div className="input-label">Task</div>
            <div className="input-row">
              <textarea
                id="task-input"
                className="task-textarea"
                value={task}
                onChange={e => setTask(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Enter a task… (Ctrl+Enter to run)"
                disabled={isRunning}
                rows={3}
              />
              <button
                id="run-btn"
                className="run-btn"
                onClick={handleRun}
                disabled={isRunning || !task.trim()}
              >
                {isRunning ? (
                  <><div className="spinner" style={{ borderTopColor: '#fff', borderColor: 'rgba(255,255,255,0.25)' }} /> Running</>
                ) : '▶ Run'}
              </button>
              {(events.length > 0 || status !== 'idle') && (
                <button
                  id="clear-btn"
                  className="clear-btn"
                  onClick={clearRun}
                  disabled={isRunning}
                >
                  Clear
                </button>
              )}
            </div>

            {/* Options + examples */}
            <div className="options-bar">
              <label className="option-toggle" htmlFor="parallel-toggle">
                <input
                  id="parallel-toggle"
                  type="checkbox"
                  checked={useParallel}
                  onChange={e => setParallel(e.target.checked)}
                  disabled={isRunning}
                />
                Parallel (Phase 4)
              </label>

              <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>
                Examples:
              </span>
              {EXAMPLE_TASKS.slice(0, 2).map((ex, i) => (
                <button
                  key={i}
                  id={`example-${i}`}
                  onClick={() => handleExample(ex)}
                  disabled={isRunning}
                  style={{
                    background: 'var(--bg-elevated)',
                    border: '1px solid var(--border)',
                    borderRadius: 'var(--radius-sm)',
                    color: 'var(--text-secondary)',
                    cursor: 'pointer',
                    fontSize: 11,
                    padding: '3px 9px',
                    transition: 'all 0.15s',
                    maxWidth: 180,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                  title={ex}
                >
                  {ex.slice(0, 28)}…
                </button>
              ))}
            </div>
          </div>

          {/* Error banner */}
          {errorMsg && (
            <div className="error-banner" id="error-banner">
              💥 {errorMsg}
            </div>
          )}

          {/* Left panel label */}
          <div className="panel-section-label">🔄 Execution Trace</div>

          {/* Live trace */}
          <TracePanel events={events} status={status} />
        </div>

        {/* ── RIGHT: Answer + compact stats ── */}
        <div className="right-panel">
          {/* Final Answer — main content, independently scrollable */}
          <AnswerPanel result={result} status={status} />

          {/* Compact stats strip */}
          <StatusPanel events={events} result={result} status={status} />
        </div>

      </div>
    </div>
  );
}