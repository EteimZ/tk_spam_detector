#Tkinter Gui for Spam Detector.
import tkinter as tk
import joblib
from model import vect, text_process #Importing vect,text_process from model module



def check(event=None):
    """
    This function processes the user given text.
    """
    text = txt_edit.get('1.0', 'end')
    clean_text = text_process(text)
    in_text = [clean_text]
    text_dtm = vect.transform(in_text)
    result = loaded_model.predict(text_dtm)
    if result == 0:
        res = "This message is not a Spam."
    if result == 1:
        res = "This message is a Spam."
    popup = tk.Tk()
    popup.wm_title("prompt")
    label = tk.Label(popup, text=res)
    label.pack(side="top", fill="x", pady=10)
    B1 = tk.Button(popup, text="Okay", command = popup.destroy)
    B1.pack()
    popup.mainloop()
    

loaded_model = joblib.load("spam.sav")

window = tk.Tk()
window.title("Spam Detector")

window.rowconfigure(0, minsize=500, weight=1)
window.columnconfigure(1, minsize=500, weight=1)


txt_edit = tk.Text(window)
fr_buttons = tk.Frame(window)
btn_open = tk.Button(fr_buttons, text="check", command=check)
btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

fr_buttons.grid(row=0, column=0, sticky="ns")
txt_edit.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

window.mainloop()
