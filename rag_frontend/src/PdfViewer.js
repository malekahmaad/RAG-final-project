import React, { useState } from 'react';
import { Document, Page, pdfjs } from 'react-pdf'; 
import 'react-pdf/dist/Page/AnnotationLayer.css'; 
import 'react-pdf/dist/Page/TextLayer.css';

pdfjs.GlobalWorkerOptions.workerSrc = `${process.env.PUBLIC_URL}/pdfjs-workers/pdf.worker.min.js`;

function PdfViewer({ fileUrl, onClose, chunks}) {
  const [numPages, setNumPages] = useState(null);
  const [loadingError, setLoadingError] = useState(null);

  function onDocumentLoadSuccess({ numPages }) {
    setNumPages(numPages);
    setLoadingError(null); 
  }

  function onDocumentLoadError(error) {
    console.error('Failed to load PDF document:', error);
    setLoadingError(error);
    setNumPages(0);
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>PDF Preview</h2>
          <button onClick={onClose}>×</button>
        </div>
        <div className="modal-body" style={{ overflowY: 'auto' }}>
          <Document
            file={fileUrl} 
            onLoadSuccess={onDocumentLoadSuccess}
            onLoadError={onDocumentLoadError}
            loading={<p>Loading PDF...</p>} 
            error={loadingError ? (
                <p>Error loading PDF: {loadingError.message || 'Unknown error'}</p>
            ) : null} 
          >
            {numPages && Array.from(new Array(numPages), (el, index) => (
              <div key={`page_${index + 1}`} style={{ marginBottom: '10px', border: '1px solid #eee', borderRadius: '4px', background: '#fff' }}>
                <p style={{ textAlign: 'center', margin: '5px 0', fontSize: '0.9em', color: '#555' }}>
                  Page {index + 1} of {numPages}
                </p>
                <Page
                  pageNumber={index + 1}
                  width={830} 
                  renderTextLayer={true}
                  renderAnnotationLayer={true} 
                />
              </div>
            ))}
            {!fileUrl && <p>No PDF file selected for preview.</p>}
          </Document>
        </div>
      </div>
    </div>
  );
}

export default PdfViewer;