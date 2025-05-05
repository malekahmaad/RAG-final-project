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


server = Flask(__name__)
CORS(server)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
persist_directory = 'chroma_db/'
index_folder = "faiss_index"
data_file = "data_file.txt"
mainFolder = "./filestest"
tempFolder = "./temp"
hash_json = f"{index_folder}/hash.json"


class embedding_object:
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                # print(outputs)
                hidden_states = outputs.hidden_states
                selected_states = hidden_states[-1]
                # print(f"{hidden_states.shape}\n{hidden_states}")
                embedding = selected_states.mean(dim=1)
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                embedding_np = np.vstack(embedding).astype("float32")
                embedding_list = [element for element in embedding_np[0]]
                # print(type(embedding_list))
                # print(type(embedding_list[0]))
                embeddings.append(embedding_list)
                # print(embeddings)

        return embeddings

    def embed_query(self, text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # print(outputs)
            hidden_states = outputs.hidden_states
            selected_states = hidden_states[-1]
            # print(f"{hidden_states.shape}\n{hidden_states}")
            embedding = selected_states.mean(dim=1)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            embedding_np = np.vstack(embedding).astype("float32")
            embedding_list = [element for element in embedding_np[0]]
            # print(type(embedding_list))
            # print(type(embedding_list[0]))
        
        return embedding_list
    
    def __call__(self, text):
        return self.embed_query(text)


def hash_function(data):
    hasher = hashlib.sha256()
    for chunk in data:
        hasher.update(chunk.encode("utf-8"))

    return hasher.hexdigest()


def get_dict_key(dict, v):
    for key, value in dict.items():
        if value == v:
            return key
    
    return None


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
            # print(element)
            file_text.append(element)

    if soup.title:
        text = f"{soup.title.string}\n"
    else:
        text = ""

    for item in file_text[::-1]:
        text += item

    return text


def reader(file_path):
    try:
        if file_path.endswith(".txt"):
            print("txt")
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif file_path.endswith(".html"):
            print("html")
            return html_reader(file_path)
        elif file_path.endswith(".csv"):
            ...
        elif file_path.endswith(".pdf"):
            print("pdf")
            loader = PyPDFLoader(file_path)
            # print(loader)
            documents = loader.load()
            data = "\n".join(doc.page_content for doc in documents)
            return data

        return "error not supported"
            
    except FileNotFoundError:
        print(f"{file_path} not found")
        return FileNotFoundError



def splitter(data, file_name):
    print("split")
    splitted_data = []
    metadata = []
    for part in text_splitter.split_text(data):
        splitted_data.append(part)
        metadata.append({"source": file_name})

    return splitted_data, metadata


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


def initialize_model():
    global tokenizer, model, llm, QA_CHAIN_PROMPT, embedding, faiss_store, text_splitter
    start_time = time.time()
    # initializing the llama3.2-1B model
    # model_name = "yam-peleg/Hebrew-Mistral-7B"
    model_name = "meta-llama/Llama-3.2-1B"
    # we dont need it anymore
    token = "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"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_name)
    text_generization_model = AutoModelForCausalLM.from_pretrained(model_name)

    text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n", ".", "\n\n"],
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
    )

    pipe = pipeline("text-generation", model=text_generization_model, truncation=True, tokenizer=tokenizer, max_length=4000, max_new_tokens=1000)

    llm = HuggingFacePipeline(pipeline=pipe)
    template = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know.
        Avoid repeating the same sentence or phrase. Be concise and informative.
        If the information is similar, combine it into a single sentence. 
        Provide a clear and unique answer. Always say "Thanks for asking!" at the end.

        {context}

        Answer:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # persist_directory = 'chroma_db/'
    # index_folder = "faiss_index"
    # data_file = "data_file.txt"
    embedding = embedding_object()
    faiss_store = None
    if os.path.exists(index_folder):
        faiss_store = FAISS.load_local(index_folder, embeddings=embedding, allow_dangerous_deserialization=True)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")


@server.route('/answer/<question>', methods=["GET"])
def get_answer(question):
    print(f"question is {question}")
    answer_data = generate_answer(question)
    files = list()
    for file in answer_data["source_documents"]:
        print(file)
        if file.metadata["source"] not in files:
            files.append(file.metadata["source"])
    # print(answer_data)
    # print(type(answer_data))
    answer_division = answer_data["result"].split("Answer:")
    important_data = {"answer":answer_division[1], "source_files":files}
    print(important_data)
    return jsonify(important_data)


@server.route("/files", methods= ["GET"])
def get_all_files():
    print("get files here")
    if os.path.exists(mainFolder):
        files = os.listdir(mainFolder)
        response = {"files names": files}
        return jsonify(response)
    else:
        return jsonify({"error": "you have no saved files in the server"}), 404


@server.route("/files/<file_name>", methods=["GET"])
def get_file(file_name):
    file_path = os.path.join(mainFolder, file_name)
    download = request.args.get("download").lower() == "true"
    if os.path.exists(file_path):
        print(f"{file_name} exists")
        return send_file(file_path, as_attachment=download, download_name=file_name)
    else:
        return jsonify({"error": "File not found"}), 404


@server.route("/saveFiles", methods=["POST"])
def save_files():
    global faiss_store
    if not os.path.exists(mainFolder):
        os.makedirs(mainFolder)
    
    if not os.path.exists(tempFolder):
        os.makedirs(tempFolder)
    
    # saved_files =  os.listdir(mainFolder)
    # print(saved_files)
    saved_hashes = {}
    if os.path.exists(hash_json):
        with open(hash_json, "r") as file:
            saved_hashes = json.load(file)
     
    print(saved_hashes)
    if "files" not in request.files:
        return jsonify({"error": "No files found in request"}), 400
    
    uploaded_files = request.files.getlist("files")
    # print(uploaded_files)
    response = ""
    for file in uploaded_files:
        if file.filename == "":
            continue
        # if file.filename not in saved_files:
        print(file.filename)
        if os.path.exists(index_folder):
            faiss_store = FAISS.load_local(index_folder, embeddings=embedding, allow_dangerous_deserialization=True)

        saving_path = os.path.join(mainFolder, file.filename)
        temp_saving_path = os.path.join(tempFolder, file.filename)
        file.save(temp_saving_path)
        # time.sleep(5)
        file_data = reader(temp_saving_path)
        if file_data == FileNotFoundError:
            return jsonify({"error": "error while uploading the files"}), 501
            # in splitter for every chunk give the source = filename
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

if __name__ == "__main__":
    initialize_model()
    server.run()
