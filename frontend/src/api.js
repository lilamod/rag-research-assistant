// Thin wrapper around the backend REST API.
// All calls go through /api/*, which Vite proxies to FastAPI in dev
// and which FastAPI serves directly in production.

async function handle(res) {
  let data = null
  try {
    data = await res.json()
  } catch {
    // no JSON body
  }
  if (!res.ok) {
    const message = data && data.detail ? data.detail : `Request failed (${res.status})`
    throw new Error(typeof message === 'string' ? message : JSON.stringify(message))
  }
  return data
}

export async function getStats() {
  const res = await fetch('/api/stats')
  return handle(res)
}

export async function listDocuments() {
  const res = await fetch('/api/documents')
  return handle(res)
}

export async function deleteDocument(docId) {
  const res = await fetch(`/api/documents/${docId}`, { method: 'DELETE' })
  return handle(res)
}

export async function uploadFiles(fileList) {
  const formData = new FormData()
  for (const file of fileList) formData.append('files', file)
  const res = await fetch('/api/upload', { method: 'POST', body: formData })
  return handle(res)
}

export async function askQuestion(question, topK) {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: topK ?? null }),
  })
  return handle(res)
}
