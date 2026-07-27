import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble.jsx'

export default function ChatThread({ messages }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  if (!messages.length) {
    return (
      <div className="thread">
        <div className="empty-state">
          <h2>Nothing indexed yet</h2>
          Upload a document to begin. Every answer below will cite
          exactly where it came from.
        </div>
      </div>
    )
  }

  return (
    <div className="thread">
      {messages.map((m) => (
        <MessageBubble message={m} key={m.id} />
      ))}
      <div ref={bottomRef} />
    </div>
  )
}
