import './chat.css';
import { useState, useRef } from 'react';
import axios from 'axios';

function Chat() {
  const [chatMessages, setChatMessages] = useState([]);
  const [answering, setAnswering] = useState(false);
  const [first, setFirst] = useState(true);
  const [input, setInput] = useState("");
  const [expandedIndexes, setExpandedIndexes] = useState([]);
  const placeholder = useRef(null);
  const chat = useRef(null);

  const toggleSources = (index) => {
    setExpandedIndexes(prev =>
      prev.includes(index)
        ? prev.filter(i => i !== index)
        : [...prev, index]
    );
  };

  const handle_send = ()=>{
    setAnswering(true);
    console.log(input);
    console.log(chatMessages)
    const trimmed = input.trim();
    if (trimmed === ""){
      console.log("cancelled");
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

      let charIndex = 0;

      const typeChar = () => {
        setChatMessages(prev => {
          const updated = [...prev];
          const current = updated[updated.length - 1];
      
          const text = fullAnswer.slice(0, charIndex + 1);
          current.answer = text + (charIndex < fullAnswer.length - 1 ? "|" : "");
          // console.log(current.answer)
          updated[updated.length - 1] = { ...current };
          // console.log(updated[updated.length - 1]);
      
          return updated;
        });
      
        charIndex++;
      
        if (charIndex < fullAnswer.length) {
          const delay = fullAnswer[charIndex] === "\n" ? 80 : 25;
          setTimeout(typeChar, delay);
        } else {
          // Final update to remove the "|" cursor and add sources
          setChatMessages(prev => {
            const updated = [...prev];
            updated[updated.length - 1] = {
              ...updated[updated.length - 1],
              answer: fullAnswer, // remove cursor
              source_files: sources
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
    // .finally(() => {
    //   setAnswering(false);
    // });
  }

  const handleDownloadFile = (fileName) => {
    console.log("download "+fileName);
    axios.get(`http://localhost:5000//files/${fileName}?download=true`, {
     responseType: 'blob'
    })
    .then((response) =>{ 
      console.log(response)
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', fileName);
      document.body.appendChild(link);
      link.click();
      link.remove();
    })
    .catch(error=>console.error(error));
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
                            <button className="file-download-button" onClick={() => handleDownloadFile(file)}>
                              {file}
                            </button>
                          </li>
                        ))}
                      </ul>
                    </div>
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