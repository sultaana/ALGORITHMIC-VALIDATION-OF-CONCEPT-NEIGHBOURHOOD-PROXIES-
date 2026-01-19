import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import numpy as np

IMAGE_FOLDER = "dataset"          
SAVE_FILE = "human_labels.npy"   

# Load images
image_files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
labels_dict = {}

if os.path.exists(SAVE_FILE):
    labels_dict = np.load(SAVE_FILE, allow_pickle=True).item()

current_index = 0

root = tk.Tk()
root.title("Human Annotation Tool")

img_label = tk.Label(root)
img_label.pack()

entry_label = tk.Entry(root)
entry_label.pack()

def show_image(index):
    global current_index
    current_index = index
    if index >= len(image_files):
        messagebox.showinfo("Done", "All images labeled!")
        root.quit()
        return
    img_path = os.path.join(IMAGE_FOLDER, image_files[index])
    img = Image.open(img_path)
    img.thumbnail((500, 500)) 
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk
    entry_label.delete(0, tk.END)
    if image_files[index] in labels_dict:
        entry_label.insert(0, labels_dict[image_files[index]])

def save_label(event=None):
    label = entry_label.get().strip()
    if label == "":
        messagebox.showwarning("Warning", "Please enter a label!")
        return
    labels_dict[image_files[current_index]] = label
    np.save(SAVE_FILE, labels_dict)
    show_image(current_index + 1)

# Buttons
save_btn = tk.Button(root, text="Save & Next", command=save_label)
save_btn.pack()

# Bind Enter key to save label
root.bind('<Return>', save_label)

# Start GUI
show_image(0)
root.mainloop()

# After closing GUI, save all labels
np.save(SAVE_FILE, labels_dict)
print(f"Saved {len(labels_dict)} labels to {SAVE_FILE}")

