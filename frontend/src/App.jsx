import { useCallback, useEffect, useRef, useState } from 'react'
import Sidebar from './components/Sidebar.jsx'
import ChatThread from './components/ChatThread.jsx'
import Composer from './components/Composer.jsx'
import Toast from './components/Toast.jsx'
import {
  askQuestionStream,
  createConversation,
  clearConversation,
  deleteDocument,
  getStats,
  listDocuments,
  uploadFiles,
} from './api.js'

let nextId = 1
const makeId = () => nextId++

export default function App() {
  const [documents, setDocuments] = useState([])
  const [stats, setStats] = useState({ total_documents: 0, total_chunks: 0 })
  const [messages, setMessages] = useState([])
  const [asking, setAsking] = useState(false)
  const [conversationId, setConversationId] = useState(null)
  const [toast, setToast] = useState({ message: '', visible: false })
  const toastTimer = useRef(null)

  const showToast = useCallback((message) => {
    setToast({ message, visible: true })
    if (toastTimer.current) clearTimeout(toastTimer.current)
    toastTimer.current = setTimeout(() => setToast((t) => ({ ...t, visible: false })), 3200)
  }, [])

  const refreshDocuments = useCallback(async () => {
    try {
      const [docsRes, statsRes] = await Promise.all([listDocuments(), getStats()])
      setDocuments(docsRes.documents || [])
      setStats(statsRes)
    } catch (err) {
      console.error(err)
    }
  }, [])

  useEffect(() => {
    refreshDocuments()
  }, [refreshDocuments])

  async function handleUpload(fileList) {
    showToast(`Indexing ${fileList.length} file(s)…`)
    try {
      const data = await uploadFiles(fileList)
      if (data.ingested?.length) showToast(`Indexed ${data.ingested.length} document(s).`)
      if (data.errors?.length) {
        showToast(`${data.errors.length} file(s) failed — see console.`)
        console.warn('Upload errors:', data.errors)
      }
      await refreshDocuments()
    } catch (err) {
      console.error(err)
      showToast('Upload failed. Is the backend running?')
    }
  }

  async function handleDelete(docId) {
    try {
      await deleteDocument(docId)
      showToast('Document removed.')
      await refreshDocuments()
    } catch (err) {
      console.error(err)
      showToast('Could not remove document.')
    }
  }

  async function handleClearConversation() {
    if (conversationId) {
      try { await clearConversation(conversationId) } catch {}
    }
    setMessages([])
    setConversationId(null)
    showToast('Conversation cleared.')
  }

  async function handleAsk(question) {
    // Auto-create conversation on first question
    let convId = conversationId
    if (!convId) {
      try {
        const conv = await createConversation()
        convId = conv.conversation_id
        setConversationId(convId)
      } catch {
        // Fall back to stateless if conversation creation fails
      }
    }

    const userMsg = { id: makeId(), role: 'user', content: question }
    const thinkingMsg = { id: makeId(), role: 'assistant', content: 'Searching sources…', status: 'thinking' }
    setMessages((prev) => [...prev, userMsg, thinkingMsg])
    setAsking(true)

    let buffer = ''
    let errored = false
    let tokensSinceRender = 0

    await askQuestionStream(question, null, convId, {
      onToken(token) {
        buffer += token
        tokensSinceRender++
        // Render at most every 5 tokens or on newlines — reduces React reconciliation
        // overhead during fast streaming while keeping the UI feeling responsive
        if (tokensSinceRender >= 5 || token.includes('\n')) {
          tokensSinceRender = 0
          setMessages((prev) =>
            prev.map((m) =>
              m.id === thinkingMsg.id ? { ...m, content: buffer } : m
            )
          )
        }
      },
      onSources(sources) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === thinkingMsg.id
              ? { ...m, content: buffer, sources, status: undefined }
              : m
          )
        )
      },
      onError(message) {
        errored = true
        setMessages((prev) =>
          prev.map((m) =>
            m.id === thinkingMsg.id
              ? { ...m, content: message || 'Something went wrong.', status: 'error' }
              : m
          )
        )
      },
    })

    if (!errored) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === thinkingMsg.id ? { ...m, content: buffer } : m
        )
      )
    }
    setAsking(false)
  }

  return (
    <div className="app">
      <Sidebar
        documents={documents}
        stats={stats}
        onUpload={handleUpload}
        onDelete={handleDelete}
      />
      <main className="main">
        <div className="header">
          <h1>Ask your sources</h1>
          {messages.length > 0 && (
            <button className="clear-conv-btn" onClick={handleClearConversation} title="Clear conversation">
              Clear chat
            </button>
          )}
        </div>
        <ChatThread messages={messages} hasDocuments={documents.length > 0} />
        <Composer onAsk={handleAsk} disabled={asking} />
      </main>
      <Toast message={toast.message} visible={toast.visible} />
    </div>
  )
}
