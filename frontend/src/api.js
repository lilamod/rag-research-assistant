// Thin wrapper around the backend REST API.
//
// Locally: leave VITE_API_URL unset. Vite's dev proxy forwards /api/* to
// FastAPI on :8000 (see vite.config.js), and relative paths work as-is.
//
// Deployed separately (e.g. frontend on Vercel, backend on Render/Railway):
// set VITE_API_URL to the backend's full origin, e.g.
// https://rag-backend.onrender.com — no trailing slash.
const API_BASE = import.meta.env.VITE_API_URL || ''

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
  const res = await fetch(`${API_BASE}/api/stats`)
  return handle(res)
}

export async function listDocuments() {
  const res = await fetch(`${API_BASE}/api/documents`)
  return handle(res)
}

export async function deleteDocument(docId) {
  const res = await fetch(`${API_BASE}/api/documents/${docId}`, { method: 'DELETE' })
  return handle(res)
}

export async function uploadFiles(fileList) {
  const formData = new FormData()
  for (const file of fileList) formData.append('files', file)
  const res = await fetch(`${API_BASE}/api/upload`, { method: 'POST', body: formData })
  return handle(res)
}

export async function askQuestion(question, topK) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: topK ?? null }),
  })
  return handle(res)
}
