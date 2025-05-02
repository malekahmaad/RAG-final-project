import './search.css';
import axios from 'axios';
import { useEffect, useState } from 'react';

function Search() {
  const [files, setFiles] = useState([]);
  const [selectedFiles, setSelectedFiles] = useState([]);
  useEffect(()=>{
    get_files();
    // needs to put it in the files list for the selected when i do th input function
    setSelectedFiles(files);
  }, []);

  const get_files = () =>{
    axios.get("http://localhost:5000/files")
    .then(response=>setFiles(response.data["files names"].sort()))
    .catch(error=>console.error(error))
  }

  const handleFileChange = (event) => {
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
      console.log(res)
      get_files();
    })
    .catch(error => console.error(error));
    // console.log(form);
    // console.log(form["files"]);
  };

  const handleViewFile = (fileName) => {
    console.log("view "+fileName);
    const fileUrl = `http://localhost:5000/files/${fileName}?download=false`;
    window.open(fileUrl, "_blank");
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
    <div className='search'>
      <div className='search-title'>Source files</div>
      <div className='search-bar'>
        <input
          placeholder='Write file name...'
          className='file_input'
          type="text"
        />
        {/* <button className='search_file' title="Submit">🔍</button> */}
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
      >
        📤 Upload File
      </button>
    </div>
  );
}

export default Search;