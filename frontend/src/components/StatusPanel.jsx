/**
 * StatusPanel - compact stats strip below the AnswerPanel.
 * Shows: run overview, last action, budget usage, and goal list.
 * Final answer is NO LONGER here - it lives in AnswerPanel.
 */

function parseBudget(events) {
  const budget = { iterations: 0, toolCalls: 0, llmCalls: 0, elapsed: 0 };
  for (let i = events.length - 1; i >= 0; i--) {
    const m = events[i].message || '';
    if (m.includes('Budget used')) {
      const iter = m.match(/iterations:\s*(\d+)/i);
      const tool = m.match(/tool calls:\s*(\d+)/i);
      const llm  = m.match(/LLM calls:\s*(\d+)/i);
      const elap = m.match(/elapsed:\s*([\d.]+)/i);
      if (iter) budget.iterations = parseInt(iter[1]);
      if (tool) budget.toolCalls  = parseInt(tool[1]);
      if (llm)  budget.llmCalls   = parseInt(llm[1]);
      if (elap) budget.elapsed    = parseFloat(elap[1]);
      break;
    }
    if (m.includes('[BUDGET]')) {
      const iter = m.match(/Iterations:\s*(\d+)\//i);
      const tool = m.match(/Tool calls:\s*(\d+)\//i);
      const llm  = m.match(/LLM calls:\s*(\d+)\//i);
      if (iter) budget.iterations = parseInt(iter[1]);
      if (tool) budget.toolCalls  = parseInt(tool[1]);
      if (llm)  budget.llmCalls   = parseInt(llm[1]);
    }
  }
  return budget;
}

function countRecoveries(events) {
  return events.filter(e =>
    (e.emoji === '🔄' || e.emoji === '🔁') &&
    (e.message || '').toLowerCase().includes('recov')
  ).length;
}

function getLastTool(events) {
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i];
    if (e.emoji === '🔧') {
      const m = e.message || '';
      const match = m.match(/Selected:\s*([^\s|]+)\s*\|?\s*(.*)/);
      if (match) return { name: match[1].trim(), input: match[2]?.trim() || '' };
    }
  }
  return null;
}

function getLastVerification(events) {
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i];
    if (e.emoji === '🔍') {
      const m = e.message || '';
      const match = m.match(/\b(VALID|INCOMPLETE|INVALID|CONTRADICTORY|UNRELIABLE|SKIPPED)\b/);
      if (match) return match[1];
    }
  }
  return null;
}

function BudgetBar({ label, value, max, barClass }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  const isHigh = pct > 80;
  return (
    <div className="budget-bar-row">
      <div className="budget-bar-header">
        <span>{label}</span>
        <span style={{ color: isHigh ? 'var(--accent-red)' : undefined }}>
          {value} / {max}
        </span>
      </div>
      <div className="budget-bar-track">
        <div
          className={`budget-bar-fill ${isHigh ? 'bar-orange' : barClass}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

const GOAL_STATUS_ICON = {
  COMPLETED:          '✅',
  FAILED:             '❌',
  RUNNING:            '🔄',
  PENDING:            '⏳',
  BLOCKED:            '🚫',
  NEEDS_VERIFICATION: '🔍',
  SKIPPED:            '⏭️',
};

export function StatusPanel({ events, result, status }) {
  const budget     = parseBudget(events);
  const recoveries = countRecoveries(events);
  const lastTool   = getLastTool(events);
  const lastVerify = getLastVerification(events);

  const goals = result?.goals || [];

  const budgetResult = result?.budget || {};
  const displayBudget = {
    iterations: budgetResult.iterations  ?? budget.iterations,
    toolCalls:  budgetResult.tool_calls  ?? budget.toolCalls,
    llmCalls:   budgetResult.llm_calls   ?? budget.llmCalls,
    elapsed:    budgetResult.elapsed_seconds ?? budget.elapsed,
  };

  return (
    <div className="status-panel">

      {/* ── Overview stats (4 mini cards) ── */}
      <div className="status-section">
        <div className="status-section-title">Run Overview</div>
        <div className="stat-grid">
          <div className="stat-card">
            <div className="stat-label">Goals</div>
            <div className="stat-value accent-blue">{goals.length || '—'}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Recovered</div>
            <div className="stat-value" style={{ color: recoveries > 0 ? 'var(--accent-red)' : 'var(--text-muted)' }}>
              {recoveries}
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-label">LLM Calls</div>
            <div className="stat-value accent-purple">{displayBudget.llmCalls}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Elapsed</div>
            <div className="stat-value accent-yellow">
              {displayBudget.elapsed > 0 ? `${displayBudget.elapsed}s` : '—'}
            </div>
          </div>
        </div>
      </div>

      {/* ── Last action ── */}
      <div className="status-section">
        <div className="status-section-title">Last Action</div>
        <div className="info-row">
          <div className="info-field">
            <span className="info-key">Tool</span>
            <span className="info-val mono">{lastTool?.name || '—'}</span>
          </div>
          {lastTool?.input && (
            <div className="info-field">
              <span className="info-key">Input</span>
              <span className="info-val" style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                {lastTool.input.slice(0, 120)}
              </span>
            </div>
          )}
          <div className="info-field">
            <span className="info-key">Verification</span>
            <span>
              {lastVerify ? (
                <span className={`verify-badge verify-${lastVerify}`}>
                  {lastVerify === 'VALID' ? '✓' : lastVerify === 'INVALID' ? '✗' : '~'} {lastVerify}
                </span>
              ) : (
                <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>—</span>
              )}
            </span>
          </div>
        </div>
      </div>

      {/* ── Budget bars ── */}
      <div className="status-section">
        <div className="status-section-title">Budget Usage</div>
        <div className="budget-bars">
          <BudgetBar label="LLM Calls"  value={displayBudget.llmCalls}   max={40} barClass="bar-purple" />
          <BudgetBar label="Tool Calls" value={displayBudget.toolCalls}  max={25} barClass="bar-blue"   />
          <BudgetBar label="Iterations" value={displayBudget.iterations} max={15} barClass="bar-green"  />
        </div>
      </div>

      {/* ── Goals ── */}
      {goals.length > 0 && (
        <div className="status-section">
          <div className="status-section-title">Goals</div>
          <div className="goal-list">
            {goals.map(g => (
              <div key={g.id} className="goal-item">
                <span className="goal-status-icon">
                  {GOAL_STATUS_ICON[g.status] || '❓'}
                </span>
                <div className="goal-body">
                  <div className="goal-desc">{g.description}</div>
                  <div className="goal-meta">
                    <span className={`goal-status-badge badge-${g.status}`}>{g.status}</span>
                    {g.attempts > 0 && (
                      <span>{g.attempts} attempt{g.attempts !== 1 ? 's' : ''}</span>
                    )}
                    {g.verification && g.verification !== 'SKIPPED' && (
                      <span style={{ color: g.verification === 'VALID' ? 'var(--accent-green)' : 'var(--accent-yellow)' }}>
                        {g.verification}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

    </div>
  );
}