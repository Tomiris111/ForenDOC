# run_image.py

from slot_image_recovery import recover_images_from_dump
from tkinter import filedialog, messagebox, Tk

def gui():
    # Create hidden root window
    root = Tk()
    root.withdraw()

    # Ask user to select the raw dump file
    path = filedialog.askopenfilename(title="Select Dump File")
    if not path:
        return

    # Call image recovery function and report how many images were recovered
    count = recover_images_from_dump(path)
    messagebox.showinfo("Result", f"Recovered {count} image(s) from the dump.")

# Start the simple GUI
gui()
