import tkinter as tk
from PIL import Image, ImageTk

class App:
    def __init__(self, root):
        self.root = root
        self.frame = tk.Frame(root)
        self.frame.pack(fill="both", expand=True)

        # Load image
        image = Image.open("tests/hogwarts_map_phenomenalia.jpg")
        image = image.resize((500, 500), Image.ANTIALIAS)  # Resize image
        self.photo = ImageTk.PhotoImage(image)

        # Add image to frame
        self.label = tk.Label(self.frame, image=self.photo)
        self.label.grid(row=0, column=0)

        # Add button to frame
        self.button = tk.Button(self.frame, text="Show Input", command=self.input_prompt)
        self.button.grid(row=1, column=0)
        
        self.location_name = tk.StringVar()

    def input_prompt(self):
        self.button.pack_forget()  # Hide button

        # Create input prompt
        self.input_frame = tk.Frame(self.frame)
        self.input_frame.place(relx=0.5, rely=0.5, anchor="center")

        label = tk.Label(self.input_frame, text="Input:")
        label.grid(
            row=0,
            column=0,
            padx=5,
            pady=5)
        name_entry = tk.Entry(self.input_frame, textvariable=self.location_name)
        name_entry.grid(
            row=0,
            column=1,
            padx=(0, 5),
            pady=5)
        confirm_button = tk.Button(self.input_frame, text="Confirm", command=self.confirm)
        confirm_button.grid(
            row=1,
            column=0,
            columnspan=2,
            padx=5,
            pady=(0, 5))
    
    def confirm(self):
        print(f"set location to {self.location_name.get()}")
        self.input_frame.destroy()  # Hide input prompt
        

root = tk.Tk()
app = App(root)
root.mainloop()
