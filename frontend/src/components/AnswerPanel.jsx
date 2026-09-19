/**
 * AnswerPanel - Right-side final answer display with inline Markdown rendering.
 *
 * Renders the final synthesized answer with:
 *   - Headings (## / ###)
 *   - Bold text (**bold**)
 *   - Bullet lists (- item)
 *   - Numbered lists (1. item)
 *   - Horizontal rules (---)
 *   - Preserved line spacing
 *
 * Independently scrollable from the trace panel.
 */

/** Minimal Markdown -> React elements renderer. No external dependencies. */
function renderMarkdown(text) {
  if (!text) return null;

  const lines = text.split('\n');
  const elements = [];
  let i = 0;
  let listBuffer = [];
  let listType = null; // 'ul' or 'ol'

  const flushList = (key) => {
    if (listBuffer.length === 0) return;
    const Tag = listType === 'ol' ? 'ol' : 'ul';
    elements.push(
      <Tag key={`list-${key}`} className="md-list">
        {listBuffer.map((item, idx) => (
          <li key={idx} className="md-li">{inlineMarkdown(item)}</li>
        ))}
      </Tag>
    );
    listBuffer = [];
    listType = null;
  };

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    // Heading h2
    if (trimmed.startsWith('## ')) {
      flushList(i);
      elements.push(<h2 key={i} className="md-h2">{inlineMarkdown(trimmed.slice(3))}</h2>);
      i++; continue;
    }
    // Heading h3
    if (trimmed.startsWith('### ')) {
      flushList(i);
      elements.push(<h3 key={i} className="md-h3">{inlineMarkdown(trimmed.slice(4))}</h3>);
      i++; continue;
    }
    // Heading h1
    if (trimmed.startsWith('# ')) {
      flushList(i);
      elements.push(<h1 key={i} className="md-h1">{inlineMarkdown(trimmed.slice(2))}</h1>);
      i++; continue;
    }
    // Horizontal rule
    if (/^---+$/.test(trimmed) || /^\*\*\*+$/.test(trimmed)) {
      flushList(i);
      elements.push(<hr key={i} className="md-hr" />);
      i++; continue;
    }
    // Unordered list item
    if (/^[-*]\s/.test(trimmed)) {
      if (listType !== 'ul') { flushList(i); listType = 'ul'; }
      listBuffer.push(trimmed.slice(2));
      i++; continue;
    }
    // Ordered list item
    if (/^\d+\.\s/.test(trimmed)) {
      if (listType !== 'ol') { flushList(i); listType = 'ol'; }
      listBuffer.push(trimmed.replace(/^\d+\.\s/, ''));
      i++; continue;
    }
    // Empty line
    if (trimmed === '') {
      flushList(i);
      elements.push(<div key={i} className="md-spacer" />);
      i++; continue;
    }
    // Regular paragraph line
    flushList(i);
    elements.push(<p key={i} className="md-p">{inlineMarkdown(trimmed)}</p>);
    i++;
  }
  flushList('end');

  return elements;
}

/** Render inline Markdown: **bold**, *italic*, `code` */
function inlineMarkdown(text) {
  if (!text) return '';
  // Split by bold, italic, or inline code patterns
  const parts = [];
  let remaining = text;
  const pattern = /(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`)/g;
  let lastIndex = 0;
  let match;

  pattern.lastIndex = 0;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    if (match[2] !== undefined) {
      parts.push(<strong key={match.index}>{match[2]}</strong>);
    } else if (match[3] !== undefined) {
      parts.push(<em key={match.index}>{match[3]}</em>);
    } else if (match[4] !== undefined) {
      parts.push(<code key={match.index} className="md-code">{match[4]}</code>);
    }
    lastIndex = pattern.lastIndex;
  }
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }
  return parts.length > 0 ? parts : text;
}

export function AnswerPanel({ result, status }) {
  const finalAnswer = result?.answer;
  const agentStatus = result?.status;

  const isRunning = status === 'running';
  const isDone = status === 'done';
  const isError = status === 'error';

  return (
    <div className="answer-panel" id="answer-panel">
      {/* Panel header */}
      <div className="answer-panel-header">
        <span className="answer-panel-icon">📝</span>
        <span className="answer-panel-title">Final Answer</span>
        {isDone && finalAnswer && (
          <span className="answer-done-badge">✓ Completed</span>
        )}
        {isRunning && (
          <span className="answer-waiting-badge">Waiting…</span>
        )}
      </div>

      {/* Answer content */}
      <div className="answer-panel-body">
        {finalAnswer ? (
          <div className="answer-content" id="final-answer-content">
            {renderMarkdown(finalAnswer)}
          </div>
        ) : isRunning ? (
          <div className="answer-placeholder">
            <div className="answer-placeholder-spinner">
              <div className="spinner" style={{ width: 20, height: 20 }} />
            </div>
            <div className="answer-placeholder-text">
              Agent is working on your task…
            </div>
            <div className="answer-placeholder-hint">
              The final answer will appear here when synthesis completes.
            </div>
          </div>
        ) : isError ? (
          <div className="answer-placeholder error">
            No answer — agent encountered an error.
          </div>
        ) : (
          <div className="answer-placeholder">
            <div className="answer-placeholder-icon">💡</div>
            <div className="answer-placeholder-text">No answer yet</div>
            <div className="answer-placeholder-hint">
              Run a task to see the synthesized answer here.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}