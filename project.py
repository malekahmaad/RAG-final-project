from tkinter import *
import langchain
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModel
import torch
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import numpy as np
import time
from bs4 import BeautifulSoup

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

def find_similar_answer(question):
    results = faiss_store.similarity_search(question, k=2)
    for result in results:
        fileName = result.metadata
        answer = result.page_content
        print(f"Source: {fileName}")
        print(f"answer: {answer}", end="\n\n")


def on_enter_pressed(_, entry_widget):
    question = entry_widget.get()
    print(question)
    if(question != ""):
        answer = find_similar_answer(question)
        second = Toplevel()
        second.title("AI answer")
        second.geometry("250x200")
        second.configure(bg="black")
        l2 = Label(
                second,
                text="Answer:",
                bg="black",
                fg="white",
                font=("Times New Roman", 14, "underline")
            )
        l2.grid(row=0, sticky="w")
        l2 = Label(
                second,
                text=answer,
                bg="black",
                fg="white",
                font=("Times New Roman", 14)
            )
        l2.grid(row=1)
    entry_widget.delete(0, END)

def html_reader(file):
    with open(file, 'r', encoding='utf-8') as f:
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

def reader(path):
    text_files = list()
    pdfs = list()
    HTMLs = list()
    names = list()
    for file in os.listdir(path):
            _, extension = os.path.splitext(file)
            if extension.lower() == ".txt":
                with open(f"{path}/{file}", "r", encoding="utf-8") as f:
                    data = f.read()
                    text_files.append(data)
                    names.append(file)
            elif extension.lower() == ".pdf":
                loader = PyPDFLoader(f"{path}/{file}")
                pdfs.append(loader)
            elif extension.lower() == "html":
                html_text = html_reader(file)
                HTMLs.append(html_text)

    return names, text_files, pdfs

def splitter(pdfs, text_files, text_splitter, filesnames):
    splitted_docs = list()
    splitted_txt = list()
    names = list()
    for pdf in pdfs:
            pages = pdf.load()
            docs = text_splitter.split_documents(pages)
            for doc in docs:
                # print(doc)
                # print("\n\n")
                splitted_docs.append(doc)

    for file, txt in zip(filesnames, text_files):
        for part in text_splitter.split_text(txt):
            splitted_txt.append(part)
            names.append({"source":file})

    print(splitted_txt)
    print(names)
    return splitted_txt, splitted_docs, names

def main():
    global tokenizer, model, splitted_txt, index, faiss_store
    start_time = time.time()
    # initializing the llama3.2-1B model
    # model_name = "yam-peleg/Hebrew-Mistral-7B"
    model_name = "meta-llama/Llama-3.2-1B"
    # we dont need it anymore
    token = "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"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_name)

    persist_directory = 'chroma_db/'
    index_folder = "faiss_index"
    data_file = "data_file.txt"
    if not os.path.exists(index_folder):
# reading the data in the files folder
        path = "./filestest"
        names, text_files, pdfs = reader(path)
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )
        text_splitter2 = RecursiveCharacterTextSplitter(
            separators=["\n", ".", "\n\n"],
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )
        splitted_txt, splitted_docs, mete_data = splitter(pdfs, text_files, text_splitter2, names)
        embedding = embedding_object()
        faiss_store = FAISS.from_texts(splitted_txt, embedding=embedding, metadatas=mete_data)
        faiss_store.save_local(index_folder)
    else:
        embedding = embedding_object()
        faiss_store = FAISS.load_local(index_folder, embeddings=embedding, allow_dangerous_deserialization=True)
        print(type(faiss_store))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    while TRUE:
        query = input("question: ")
        find_similar_answer(query)

    # root =Tk()
    # root.title("AI")
    # root.geometry("400x300")
    # root.configure(bg="black")
    # l = Label(
    #         root,
    #         text="Welcome to our AI.\nPlease enter your question below.",
    #         bg="black",
    #         fg="white",
    #         font=("Times New Roman", 14)
    #     )
    # l.grid(row=0, column=0, columnspan=2, pady=20)
    # e = Entry(
    #         root,
    #         width=50,
    #         highlightbackground="grey",
    #         highlightthickness=5
    #     )
    # e.grid(row=1, column=0, columnspan=2, pady=50)
    # e.bind("<Return>", lambda event: on_enter_pressed(event, e))
    # root.grid_columnconfigure(0, weight=1)
    # root.grid_columnconfigure(1, weight=1)
    # root.mainloop()


if __name__ == "__main__":
    main()
