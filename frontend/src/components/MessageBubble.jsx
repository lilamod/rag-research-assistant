export default function MessageBubble({ message }) {
  const { role, content, sources, status } = message

  const bubbleClass = [
    'bubble',
    status === 'error' ? 'error' : '',
    status === 'thinking' ? 'thinking' : '',
  ].filter(Boolean).join(' ')

  return (
    <div className={`msg ${role}`}>
      <div className={bubbleClass}>{content}</div>
      {sources && sources.length > 0 && (
        <div className="sources">
          {sources.map((s) => (
            <div className="source" key={s.rank}>
              <span className="src-head">
                [{s.rank}] {s.filename} · score {s.relevance_score}
              </span>
              {s.text_preview}…
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
