import UploadZone from './UploadZone.jsx'
import DocumentList from './DocumentList.jsx'

export default function Sidebar({ documents, stats, onUpload, onDelete }) {
  return (
    <aside className="sidebar">
      <div className="brand">Archive <small>RAG Assistant</small></div>

      <div>
        <div className="section-label">Add sources</div>
        <UploadZone onUpload={onUpload} />
      </div>

      <div>
        <div className="section-label">Sources</div>
        <DocumentList documents={documents} onDelete={onDelete} />
      </div>

      <div className="stats-line">
        {stats.total_documents ?? 0} documents · {stats.total_chunks ?? 0} chunks indexed
      </div>
    </aside>
  )
}
