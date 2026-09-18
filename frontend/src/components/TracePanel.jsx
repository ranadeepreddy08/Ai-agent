/**
 * TracePanel — scrollable live event feed.
 *
 * Classifies each event by its emoji prefix into a type that drives
 * border-left color and text color via CSS data-type attributes.
 */
import { useEffect, useRef } from 'react';

/** Classify an event into a CSS data-type string. */
function classify(event) {
  const { emoji = '', message = '' } = event;
  const m = message.toLowerCase();
  if (emoji === '🧠' || m.includes('planning'))         return 'plan';
  if (emoji === '🎯' || m.includes('created') && m.includes('goal')) return 'goal';
  if (emoji === '▶️' || emoji === '▶️ ')                return 'goal';
  if (emoji === '✅' && m.includes('goal'))              return 'goal';
  if (emoji === '❌' && m.includes('goal'))              return 'error';
  if (emoji === '🔧' || m.includes('selected:'))        return 'tool';
  if (emoji === '⚡' || m.includes('executing'))         return 'exec';
  if ((emoji === '✅' || emoji === '☑️') && m.includes('observation')) return 'observe';
  if (emoji === '🔍' || m.includes('verification'))     return 'verify';
  if (emoji === '🔄' || m.includes('recovery'))         return 'recover';
  if (emoji === '🔁' && !m.includes('[loop]'))           return 'recover';
  if (emoji === '🔀' || m.includes('replanning'))       return 'replan';
  if (emoji === '📊' || emoji === '💰' || m.includes('[budget]')) return 'budget';
  if (emoji === '🗺' || m.includes('[dag]'))             return 'budget';
  if (emoji === '⚙️' || m.includes('[executor]'))       return 'budget';
  if (emoji === '🔁' && m.includes('[loop]'))            return 'budget';
  if (emoji === '🏁' || m.includes('final answer'))     return 'final';
  if (emoji === '📝' || m.includes('synthesizing'))     return 'final';
  if (emoji === '💥' || emoji === '❌')                  return 'error';
  return 'default';
}

/** Strip ANSI color codes from message text. */
function stripAnsi(str) {
  return str.replace(/\u001b\[[0-9;]*m/g, '');
}

/** Compute a phase label badge from message text. */
function phaseBadge(message) {
  const m = message.toUpperCase();
  if (m.includes('[LOOP]'))     return { label: 'LOOP',     cls: 'badge-budget' };
  if (m.includes('[DAG]'))      return { label: 'DAG',      cls: 'badge-dag'    };
  if (m.includes('[EXECUTOR]')) return { label: 'EXECUTOR', cls: 'badge-exec'   };
  if (m.includes('[BUDGET]'))   return { label: 'BUDGET',   cls: 'badge-budget' };
  if (m.includes('[STUCK]'))    return { label: 'STUCK',    cls: 'badge-recover'};
  return null;
}

export function TracePanel({ events, status }) {
  const bottomRef = useRef(null);

  // Auto-scroll to bottom on new events
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [events.length]);

  if (events.length === 0 && status === 'idle') {
    return (
      <div className="trace-panel">
        <div className="trace-empty">
          <div className="trace-empty-icon">🤖</div>
          <div className="trace-empty-text">Ready to run</div>
          <div className="trace-empty-hint">
            Enter a task above and click Run to see the live execution trace
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="trace-panel" id="trace-panel">
      {events.map((ev, i) => {
        const type = classify(ev);
        const msg  = stripAnsi(ev.message || '');
        const badge = phaseBadge(msg);

        return (
          <div
            key={i}
            className="trace-event"
            data-type={type}
          >
            <span className="trace-event-emoji">{ev.emoji}</span>
            <div className="trace-event-body">
              <div className="trace-event-ts">{ev.ts}</div>
              <div className="trace-event-msg">
                {badge && (
                  <span className={`phase-badge ${badge.cls}`}>{badge.label}</span>
                )}
                {msg}
              </div>
            </div>
          </div>
        );
      })}

      {status === 'running' && (
        <div className="trace-running-indicator">
          <div className="spinner" />
          Agent is thinking…
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
