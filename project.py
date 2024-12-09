from tkinter import *
import langchain
import transformers
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter


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
    path = "./files"
    text_files = list()
    pdfs = list()
    for file in os.listdir(path):
        name, extension = os.path.splitext(file)
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
        print(f"{name} is a {extension} type")


    # for pdf in pdfs:
    #     pages = pdf.load()
    #     print(len(pages))

    # for txt in text_files:
    #     print(txt)

    splitted_docs = list()
    splitted_txt = list()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    for pdf in pdfs:
        pages = pdf.load()
        docs = text_splitter.split_documents(pages)
        for doc in docs:
            print(doc)
            print("\n\n")
            splitted_docs.append(doc)

    for txt in text_files:
        for part in text_splitter.split_text(txt):
            splitted_txt.append(part)

    print(splitted_txt)
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
