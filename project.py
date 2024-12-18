from tkinter import *
import langchain
from transformers import AutoTokenizer, AutoModel
import torch
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter


def on_enter_pressed(_, entry_widget):
    question = entry_widget.get()
    print(question)
    if(question != ""):
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
                text="my name is AI",
                bg="black",
                fg="white",
                font=("Times New Roman", 14)
            )
        l2.grid(row=1)
    entry_widget.delete(0, END)


def main():

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

# initializing the llama3.2-1B model
    model_name = "meta-llama/Llama-3.2-1B"
    token = "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"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_name, token=token)

# embedding the data to numerical vectors
    embedding_list = []
    # print(len(splitted_txt))
    # exit()
    for txt in splitted_txt:
        print(txt)
        inputs = tokenizer(txt, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # print(outputs)
            hidden_states = outputs.hidden_states[-1]
            print(f"{hidden_states.shape}\n{hidden_states}")
            embeddings = hidden_states.mean(dim=1)
            print(embeddings)
            embedding_list.append(embeddings)

    print(len(embedding_list))
    print(embedding_list)

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
