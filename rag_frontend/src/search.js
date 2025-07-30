import './search.css';
import axios from 'axios';
import { useEffect, useState } from 'react';

function Search() {
  const [files, setFiles] = useState([]);
  const [allFiles, setAllFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");
  useEffect(()=>{
    get_files();
  }, []);

  const get_files = () =>{
    axios.get("http://localhost:5000/files")
    .then(response=>{
      const sortedFiles = response.data["files names"].sort();
      setFiles(sortedFiles);
      setAllFiles(sortedFiles);
    })
    .catch(error=>console.error(error))
  }

  const handleFileChange = (event) => {
    setUploading(true);
    const files = Array.from(event.target.files);
    const form = new FormData();
    if (files.length > 0) {
      console.log("Selected Files:", files);
    };
    
    files.forEach(file => {
      form.append("files", file);
    });
    axios.post("http://localhost:5000/saveFiles", form)
    .then(res => {
      get_files();
      setUploadMessage("✅ File(s) uploaded successfully!");
      setTimeout(() => setUploadMessage(""), 3000);
    })
    .catch(error => console.error(error))
    .finally(() => {
      setUploading(false); 
    });
  };

  const handleViewFile = (fileName) => {
    const fileUrl = `http://localhost:5000/files/${fileName}?download=false`;
    window.open(fileUrl, "_blank");
  }

  const handleDownloadFile = (fileName) => {
    axios.get(`http://localhost:5000//files/${fileName}?download=true`, {
     responseType: 'blob'
    })
    .then((response) =>{ 
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

  const handleInputChange = (event) => {
    const value = event.target.value.toLowerCase();
    if(value === ""){
      setFiles(allFiles)
    }
    else{
      const filteredFiles = allFiles.filter(file =>
        file.toLowerCase().includes(value)
      );
      setFiles(filteredFiles);
    }
  };

  return (
    <div className='search'>
      <div className='search-title'>Source files</div>
      <div className='search-bar'>
        <input
          placeholder='Write file name...'
          className='file_input'
          type="text"
          onChange={(e) => handleInputChange(e)}
        />
      </div>
      <div className={` ${files.length === 0 ? 'hidden' : 'files'}`} >
        <ul className='file-list'>
          {files.map((fileName, index) => (
            <li key={index} className='file-item'>
              <span className='file-name' title={fileName}>{fileName}</span>
              <div className='file-actions'>
                <span
                  className='file-action'
                  data-file={fileName}
                  onClick={() => handleViewFile(fileName)}
                  title="View file"
                >
                  View
                </span>
                <span
                  className='file-action'
                  data-file={fileName}
                  onClick={() => handleDownloadFile(fileName)}
                  title="Download file"
                >
                  Download
                </span>
              </div>
            </li>
          ))}
        </ul>
      </div>
      <input
        type="file"
        id="file-upload"
        style={{ display: "none" }}
        onChange={handleFileChange}
        multiple
      />
      <button 
        className='upload_button' 
        title="Upload new file"
        onClick={() => document.getElementById("file-upload").click()}
        disabled={uploading}
        style={{
          cursor: uploading ? 'not-allowed' : 'pointer',
          opacity: uploading ? 0.6 : 1
        }}
      >
        {uploading ? (
          <>
            <span className="spinner"></span>
            Uploading...
          </>
        ) : (
          <>
            📤 Upload File
          </>
        )}
      </button>
      {uploadMessage && (
        <div className="upload-message">
          {uploadMessage}
        </div>
      )}
    </div>
  );
}

export default Search;