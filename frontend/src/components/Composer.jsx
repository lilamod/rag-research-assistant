import { useRef, useState } from 'react'

export default function Composer({ onAsk, disabled }) {
  const [value, setValue] = useState('')
  const textareaRef = useRef(null)

  function submit() {
    const question = value.trim()
    if (!question || disabled) return
    onAsk(question)
    setValue('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  function handleInput(e) {
    setValue(e.target.value)
    const el = textareaRef.current
    if (el) {
      el.style.height = 'auto'
      el.style.height = Math.min(el.scrollHeight, 160) + 'px'
    }
  }

  return (
    <div className="composer">
      <textarea
        ref={textareaRef}
        rows={1}
        placeholder="Ask a question about your uploaded documents…"
        value={value}
        onChange={handleInput}
        onKeyDown={handleKeyDown}
      />
      <button onClick={submit} disabled={disabled}>Ask</button>
    </div>
  )
}
