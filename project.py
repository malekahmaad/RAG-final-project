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
    ...


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


def main():
    global tokenizer, model, splitted_txt, index
    start_time = time.time()
    # initializing the llama3.2-1B model
    model_name = "meta-llama/Llama-3.2-1B"
    token = "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"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_name, token=token)

    persist_directory = 'chroma_db/'
    index_folder = "faiss_index"
    data_file = "data_file.txt"
    if not os.path.exists(index_folder):
# reading the data in the files folder
        path = "./files"
        text_files = list()
        pdfs = list()
        for file in os.listdir(path):
            _, extension = os.path.splitext(file)
            if extension.lower() == ".txt":
                with open(f"files/{file}", "r", encoding="utf-8") as f:
                    data = f.read()
                    text_files.append(data)
            elif extension.lower() == ".pdf":
                loader = PyPDFLoader(f"files/{file}")
                pdfs.append(loader)
                # pages = loader.load()
                # page = pages[0]
                # print(page.page_content)
            # print(f"{name} is a {extension} type")


        # for pdf in pdfs:
        #     pages = pdf.load()
        #     print(len(pages))

        # for txt in text_files:
        #     print(txt)

    # splitting the data we read in little chunks with chunk size of 1000 char
        splitted_docs = list()
        splitted_txt = list()
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
        for pdf in pdfs:
            pages = pdf.load()
            docs = text_splitter2.split_documents(pages)
            for doc in docs:
                # print(doc)
                # print("\n\n")
                splitted_docs.append(doc)

        for txt in text_files:
            for part in text_splitter2.split_text(txt):
                splitted_txt.append(part)

        print(splitted_txt)

    # embedding the data to numerical vectors
        # embedding_list = []
        # print(len(splitted_txt))
        # exit()
        # for txt in splitted_txt:
        #     # print(txt)
        #     inputs = tokenizer(txt, return_tensors="pt", truncation=True, padding=True, max_length=512)
        #     with torch.no_grad():
        #         outputs = model(**inputs, output_hidden_states=True)
        #         # print(outputs)
        #         hidden_states = outputs.hidden_states[-1]
        #         # print(f"{hidden_states.shape}\n{hidden_states}")
        #         embeddings = hidden_states.mean(dim=1)
        #         # print(embeddings)
        #         embedding_list.append(embeddings)

        # print(embedding_list[1].shape)
        # print(type(embedding_list))
        # print(embedding_list)

        # embeddings_np = np.vstack(embedding_list).astype("float32")
        # print(embeddings_np.shape)
        # print(type(embeddings_np))
        # print(embeddings_np)

        # index = faiss.IndexFlatL2(embeddings_np.shape[1])
        # index.add(embeddings_np)
        # # print(f"\nembeddings_np after faiss index:\n{embeddings_np.shape[1]}")
        # print(index.ntotal)
        embedding = embedding_object()
        faiss_store = FAISS.from_texts(splitted_txt, embedding=embedding)
        faiss_store.save_local(index_folder)

        # with open(data_file, "w", encoding="utf-8") as f:
        #     f.write("\n".join(splitted_txt))

    else:
        embedding = embedding_object()
        faiss_store = FAISS.load_local(index_folder, embeddings=embedding, allow_dangerous_deserialization=True)
        print(type(faiss_store))
    #     index = faiss.read_index(index_file)
    #     with open(data_file, "r", encoding="utf-8") as f:
    #         splitted_txt = [line.strip() for line in f.readlines()]

    # print(len(splitted_txt))
    # print(index.ntotal)

    query = "Who is anas?"
    results = faiss_store.similarity_search(query, k=2)
    for result in results:
        print(result)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    root =Tk()
    root.title("AI")
    root.geometry("400x300")
    root.configure(bg="black")
    l = Label(
            root,
            text="Welcome to our AI.\nPlease enter your question below.",
            bg="black",
            fg="white",
            font=("Times New Roman", 14)
        )
    l.grid(row=0, column=0, columnspan=2, pady=20)
    e = Entry(
            root,
            width=50,
            highlightbackground="grey",
            highlightthickness=5
        )
    e.grid(row=1, column=0, columnspan=2, pady=50)
    e.bind("<Return>", lambda event: on_enter_pressed(event, e))
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    root.mainloop()


if __name__ == "__main__":
    main()
