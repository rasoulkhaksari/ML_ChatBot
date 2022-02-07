from tkinter import *

class GUI():
    def __init__(self,bot) -> None:
        self.bot=bot

    def send(self,ChatArea,EntryBox):
        msg = EntryBox.get("1.0", 'end-1c').strip()
        EntryBox.delete("0.0", END)
        if msg != "":
            ChatArea.config(state=NORMAL)
            ChatArea.insert(END, "You: " + msg + '\n\n')
            ChatArea.config(foreground="#442265", font=("Verdana", 12))
            res = self.bot.response(msg)
            ChatArea.insert(END, "Bot: " + res + '\n\n')
            ChatArea.config(state=DISABLED)
            ChatArea.yview(END)


    def create_ui(self) -> Tk:
        ui = Tk()
        ui.title("NLP Chatbot")
        ui.geometry("400x500")
        ui.resizable(width=FALSE, height=FALSE)
        ChatArea = Text(ui, bd=0, bg="white", height="8", width="50", font="Arial",)
        ChatArea.config(state=DISABLED)
        scrollbar = Scrollbar(ui, command=ChatArea.yview)
        ChatArea['yscrollcommand'] = scrollbar.set
        SendButton = Button(ui, font=("Verdana", 14, 'bold'), text="Send", bd=1, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff', command=lambda: self.send(ChatArea,EntryBox))
        EntryBox = Text(ui, bd=0, bg="white", width="29", height="5", font="Arial")
        scrollbar.place(x=376, y=6, height=386)
        ChatArea.place(x=6, y=6, height=386, width=370)
        EntryBox.place(x=6, y=401, height=90, width=265)
        SendButton.place(x=272, y=401, height=90, width=120)
        EntryBox.focus_set()
        return ui

    def run(self):
        app = self.create_ui()
        app.mainloop()

