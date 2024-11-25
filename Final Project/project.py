from tkinter import *


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
    root =Tk()
    root.title("AI")
    root.geometry("400x300")
    root.configure(bg="black")
    l = Label(
            root,
            text="Welcome to out AI.\nPlease enter your question below.",
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
