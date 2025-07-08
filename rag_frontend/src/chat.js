import './chat.css';
import { useState, useRef } from 'react';
import axios from 'axios';
import PdfViewer from './PdfViewer.js';

function Chat() {
  const [chatMessages, setChatMessages] = useState([]);
  const [answering, setAnswering] = useState(false);
  const [first, setFirst] = useState(true);
  const [input, setInput] = useState("");
  const [expandedIndexes, setExpandedIndexes] = useState([]);
  const placeholder = useRef(null);
  const chat = useRef(null);
  const [viewerContent, setViewerContent] = useState(null);
  const [viewerTitle, setViewerTitle] = useState("");
  const [viewerVisible, setViewerVisible] = useState(false);
  const [pdfViewer, setPdfViewer] = useState({ visible: false, fileUrl: "" });
  const [chunks, setChunks] = useState([]);

  const toggleSources = (index) => {
    setExpandedIndexes(prev =>
      prev.includes(index)
        ? prev.filter(i => i !== index)
        : [...prev, index]
    );
  };

  const handle_send = ()=>{
    setAnswering(true);
    const trimmed = input.trim();
    if (trimmed === ""){
      setInput("");
      return;
    }
    const encodedQuestion = encodeURIComponent(trimmed);
    if(first === true){
      placeholder.current.style.display = "none";
      chat.current.style.display = "flex";
      setFirst(false);
    }
    setChatMessages(prev => [...prev, { question: trimmed, answer: "AI thinking . . .", source_files: []}]);
    setInput("");
    axios.get("http://localhost:5000/answer/"+encodedQuestion)
    .then(response => {
      let a = response.data["answer"];
      const fullAnswer = a.replace(/\n\s+/g, '\n');
      const sources = response.data["source_files"];
      const data = response.data["data"];

      let charIndex = 0;

      const typeChar = () => {
        setChatMessages(prev => {
          const updated = [...prev];
          const current = updated[updated.length - 1];
      
          const text = fullAnswer.slice(0, charIndex + 1);
          current.answer = text + (charIndex < fullAnswer.length - 1 ? "|" : "");
          updated[updated.length - 1] = { ...current };
      
          return updated;
        });
      
        charIndex++;
      
        if (charIndex < fullAnswer.length) {
          const delay = fullAnswer[charIndex] === "\n" ? 80 : 25;
          setTimeout(typeChar, delay);
        } else {
          setChatMessages(prev => {
            const updated = [...prev];
            updated[updated.length - 1] = {
              ...updated[updated.length - 1],
              answer: fullAnswer, 
              source_files: sources,
              files_data: data
            };
            return updated;
          });
          setAnswering(false);
        }
      };
      typeChar();
    })
    .catch(error => {
      console.error(error);
      setAnswering(false);
    })
  }

  function highlightChunks(content, chunks) {
    if (!chunks || chunks.length === 0) return content;

    const escapeRegExp = (string) => string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const validChunks = chunks
      .map(chunk => chunk.trim())
      .filter(chunk => chunk.length > 0)
      .sort((a, b) => b.length - a.length);

    let highlighted = content;

    validChunks.forEach(chunk => {
      const regex = new RegExp(escapeRegExp(chunk), 'gi');
      highlighted = highlighted.replace(regex, `<mark>${chunk}</mark>`);
    });

    return highlighted;
  }

  const handleViewFile = async (fileName, data) => {
    try {
    setViewerVisible(false); 
    setPdfViewer({ visible: false, fileUrl: "" }); 

    if (!(fileName.toLowerCase().endsWith(".pdf"))) {
      const response = await axios.get(`http://localhost:5000/file_content/${fileName}`);
      let fileContent = response.data.content;

      const highlightedContent = highlightChunks(fileContent, data);
      setViewerContent(highlightedContent);
      setViewerTitle(fileName);
      setViewerVisible(true); 
      document.body.style.overflow = "hidden";
    } else {
      const fileUrl = `http://localhost:5000/files/${fileName}?download=false`;
      setPdfViewer({ visible: true, fileUrl }); 
      setChunks(data);
      document.body.style.overflow = "hidden";
    }
  } catch (error) {
    console.error("Error fetching file content:", error);
    setViewerVisible(false);
    setPdfViewer({ visible: false, fileUrl: "" });
    document.body.style.overflow = "auto";
  }
  }

  return (
    <div className='chat-section'>
        <div className='messages'>
          <div ref={placeholder} className='placeholder'>What can I help with? 😊</div>
          <div ref={chat} className='chat-log'>
            {chatMessages.map((item, index) => (
              <div key={index} className="message-pair">
                <div className='message-user'>{item.question}</div>
                <div className="message-ai-wrapper">
                  <div
                    className={`message-ai ${item.answer === "AI thinking . . ." ? "thinking" : ""}`}
                    style={{ whiteSpace: "pre-wrap" }}
                  >
                    {item.answer}
                  </div>
                  {item.source_files.length > 0 && (
                    <button 
                      onClick={() => toggleSources(index)} 
                      className="source-toggle-arrow"
                    >
                      {expandedIndexes.includes(index) ? "sources ▲" : "sources ▼"}
                    </button>
                  )}
                  {expandedIndexes.includes(index) && (
                    <div className="source-files-tab">
                      <strong>Sources:</strong>
                      <ul>
                        {item.source_files.map((file, i) => (
                          <li key={i}>
                            <button className="file-download-button" title={file} onClick={() => handleViewFile(file, item.files_data)}>
                              {file}
                            </button>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {viewerVisible && (
                    <div 
                      className="modal-overlay" 
                      onClick={() => {
                        setViewerVisible(false);
                        document.body.style.overflow = "auto";
                      }}
                    >
                      <div 
                        className="modal-content" 
                        onClick={(e) => e.stopPropagation()}
                        style={{ direction: /[\u0590-\u05FF]/.test(viewerContent) ? 'rtl' : 'ltr' }}
                      >
                        <div className="modal-header">
                          <h2>{viewerTitle}</h2>
                          <button onClick={() => {
                            setViewerVisible(false);
                            document.body.style.overflow = "auto";
                          }}>×</button>
                        </div>
                        <div className="modal-body" dangerouslySetInnerHTML={{ __html: viewerContent }} />
                      </div>
                    </div>
                  )}
                  {pdfViewer.visible && (
                    <PdfViewer 
                      fileUrl={pdfViewer.fileUrl} 
                      onClose={() => {
                        setPdfViewer({ visible: false, fileUrl: "" });
                        document.body.style.overflow = "auto";
                      }}
                      chunks={chunks} 
                    />
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className='footer'>
          <textarea 
          placeholder='Ask anything' 
          rows={1} 
          className='query-input' 
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e)=> {
              if (answering){
                if(e.key === "Enter" && !e.shiftKey)
                  e.preventDefault();
                return;
              }
              if(e.key === "Enter" && !e.shiftKey){
                e.preventDefault();
                handle_send();
              }
            }
          }
          />
          <button 
          className='submit-button' 
          title="Send message"
          onClick={handle_send}
          disabled={answering}
          style={{
            cursor: answering ? 'not-allowed' : 'pointer',
            opacity: answering ? 0.6 : 1
          }}
          >
            📨
          </button>
        </div>
    </div>
  );
}

export default Chat;