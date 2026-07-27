export default function DocumentList({ documents, onDelete }) {
  if (!documents.length) {
    return <div className="empty-note">No documents yet.</div>
  }

  return (
    <div className="doc-list">
      {documents.map((doc) => (
        <div className="doc-item" key={doc.doc_id}>
          <span className="name" title={doc.filename}>{doc.filename}</span>
          <span className="chunks">{doc.chunks}</span>
          <button title="Remove" onClick={() => onDelete(doc.doc_id)}>×</button>
        </div>
      ))}
    </div>
  )
}
