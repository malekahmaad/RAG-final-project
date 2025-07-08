from flask import Flask, jsonify, send_file, request
import langchain
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForCausalLM
import torch
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import numpy as np
import time
from bs4 import BeautifulSoup
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from flask_cors import CORS
import hashlib
import json
import shutil
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


server = Flask(__name__)
CORS(server)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
persist_directory = 'chroma_db/'
index_folder = "faiss_index"
data_file = "data_file.txt"
mainFolder = "./files"
tempFolder = "./temp"
hash_json = f"{index_folder}/hash.json"

# the embedding class
class embedding_object:
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            inputs = embedding_model_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = embedding_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                selected_states = hidden_states[-1]
                embedding = selected_states.mean(dim=1)
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                embedding_np = np.vstack(embedding).astype("float32")
                embedding_list = [element for element in embedding_np[0]]
                embeddings.append(embedding_list)

        return embeddings

    def embed_query(self, text):
        inputs = embedding_model_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = embedding_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            selected_states = hidden_states[-1]
            embedding = selected_states.mean(dim=1)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            embedding_np = np.vstack(embedding).astype("float32")
            embedding_list = [element for element in embedding_np[0]]

        return embedding_list
    
    def __call__(self, text):
        return self.embed_query(text)

# hash function to give every file a hash value
def hash_function(data):
    hasher = hashlib.sha256()
    for chunk in data:
        hasher.update(chunk.encode("utf-8"))

    return hasher.hexdigest()

# function to get the file name from its hash value
def get_dict_key(dict, v):
    for key, value in dict.items():
        if value == v:
            return key
    
    return None

# function to convert the html files to text
def html_reader(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    stack = list()
    stack.append(soup.body)
    file_text = list()
    while len(stack) > 0:
        element = stack.pop()
        if hasattr(element, 'children'):
            for child in element.children:
                stack.append(child)

        else:
            file_text.append(element)

    if soup.title:
        text = f"{soup.title.string}\n"
    else:
        text = ""

    for item in file_text[::-1]:
        text += item

    return text

# function to read the files
def reader(file_path):
    try:
        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif file_path.endswith(".html"):
            return html_reader(file_path)
        elif file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            data = "\n".join(doc.page_content for doc in documents)
            return data

        return "error not supported"
            
    except FileNotFoundError:
        return FileNotFoundError


# function to split the files data and save the metadata of every splitted data
def splitter(data, file_name):
    splitted_data = []
    metadata = []
    for part in text_splitter.split_text(data):
        splitted_data.append(part)
        metadata.append({"source": file_name})

    return splitted_data, metadata

# function to generate the answer
def generate_answer(query):
    if faiss_store is None:
        return {
            "result": "Answer: I dont know, can you upload some files so that I help you ?",
            "source_documents": []
        }
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=faiss_store.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    response = qa.invoke(query)
    return response

# function to initialize the models and the faiss index database
def initialize_model():
    global embedding_model_tokenizer, embedding_model, llm, QA_CHAIN_PROMPT, embedding, faiss_store, text_splitter
    embedding_model_name = "intfloat/multilingual-e5-large-instruct"
    embedding_model_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    embedding_model_tokenizer.pad_token = embedding_model_tokenizer.eos_token
    embedding_model = AutoModel.from_pretrained(embedding_model_name, trust_remote_code=True).to('cpu')
    # to run it on gpu
    # embedding_model.eval()
    # embedding_model.to('cuda')

    text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n", ".", "\n\n"],
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
    )

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        openai_api_key=OpenAI_key
    )

    template = """
        Use the following pieces of context to answer the question at the end.

        - Do not repeat information if multiple contexts say the same thing.
        - Ignore any context that is clearly irrelevant or unrelated to the question.
        - Provide a clear, unique, and concise answer based only on the relevant context.
        - If you don’t know the answer, just say you don’t know. Don’t make one up.
        - Keep the answer to a maximum of three sentences.
        - Always end your answer with: "Thanks for asking!"

        Contexts:
        {context}

        Question: {question}

        Answer:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    
    embedding = embedding_object()
    faiss_store = None
    if os.path.exists(index_folder):
        faiss_store = FAISS.load_local(index_folder, embeddings=embedding, allow_dangerous_deserialization=True)


# function to deal with the http get requests for the answer of the query
@server.route('/answer/<question>', methods=["GET"])
def get_answer(question):
    answer_data = generate_answer(question)
    files = list()
    files_data = list()
    for file in answer_data["source_documents"]:
        if file.metadata["source"] not in files:
            files.append(file.metadata["source"])
            files_data.append(file.page_content)

    important_data = {"answer":answer_data["result"], "source_files":files, "data":files_data}
    return jsonify(important_data)


# function to deal with the http get requests for getting files names
@server.route("/files", methods= ["GET"])
def get_all_files():
    if os.path.exists(mainFolder):
        files = os.listdir(mainFolder)
        response = {"files names": files}
        return jsonify(response)
    else:
        return jsonify({"error": "you have no saved files in the server"}), 404


# function to deal with the hhtp get requests for getting a specific file data to download or view the file
@server.route("/files/<file_name>", methods=["GET"])
def get_file(file_name):
    file_path = os.path.join(mainFolder, file_name)
    download = request.args.get("download").lower() == "true"
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=download, download_name=file_name)
    else:
        return jsonify({"error": "File not found"}), 404


# function to deal with the hhtp get requests for getting a source file content for the highlighted viewer 
@server.route("/file_content/<file_name>", methods=["GET"])
def get_file_content(file_name):
    file_path = os.path.join(mainFolder, file_name)
    if os.path.exists(file_path):
        try:
            content = reader(file_path)
            return jsonify({"content": content})
        except Exception as e:
            return jsonify({"error": f"Failed to read file: {str(e)}"}), 500
    else:
        return jsonify({"error": "File not found"}), 404


# function to deal with the hhtp get requests for saving number of files in the server and embedding their data
# then dave their hash value
@server.route("/saveFiles", methods=["POST"])
def save_files():
    global faiss_store
    if not os.path.exists(mainFolder):
        os.makedirs(mainFolder)
    
    if not os.path.exists(tempFolder):
        os.makedirs(tempFolder)
    
    saved_hashes = {}
    if os.path.exists(hash_json):
        with open(hash_json, "r") as file:
            saved_hashes = json.load(file)
     
    if "files" not in request.files:
        return jsonify({"error": "No files found in request"}), 400
    
    uploaded_files = request.files.getlist("files")
    response = ""
    for file in uploaded_files:
        if file.filename == "":
            continue

        if os.path.exists(index_folder):
            faiss_store = FAISS.load_local(index_folder, embeddings=embedding, allow_dangerous_deserialization=True)

        saving_path = os.path.join(mainFolder, file.filename)
        temp_saving_path = os.path.join(tempFolder, file.filename)
        file.save(temp_saving_path)
        file_data = reader(temp_saving_path)
        if file_data == FileNotFoundError:
            return jsonify({"error": "error while uploading the files"}), 501
            
        splitted_data, metadata = splitter(file_data, file.filename)
        hash_value = hash_function(splitted_data)
        if hash_value in saved_hashes.values():
            key = get_dict_key(saved_hashes, hash_value)
            response+=f"{file.filename} does exist and its name is {key}, "
            os.remove(temp_saving_path)
            continue

        if faiss_store == None:
            faiss_store = FAISS.from_texts(splitted_data, embedding=embedding, metadatas=metadata)
        else:
            faiss_store.add_texts(splitted_data, metadatas=metadata)

        faiss_store.save_local(index_folder)
        shutil.move(temp_saving_path, saving_path)
        saved_hashes[file.filename] = hash_value
        response+=f"{file.filename} uploaded successfully, "
    
    with open(hash_json, "w") as file:
        json.dump(saved_hashes, file, indent=4)

    return jsonify({"response": response})

# the main function to run the server
if __name__ == "__main__":
    initialize_model()
    server.run()
