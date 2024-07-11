# simple GUI for landmark tracing system

import tkinter as tk
from tkinter import PhotoImage, filedialog
import subprocess
from tqdm import tqdm
import openpyxl
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

def import_excel_file():
    file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx;*.xls")])
    if file_path:
        try:
            workbook = openpyxl.load_workbook(file_path)
            sheet = workbook.active
            data = []
            for row in sheet.iter_rows(values_only=True):
                data.append(row)
            display_data(data)
        except Exception as e:
            error_label.config(text=f"Error: {e}")

def display_data(data):
    # Create a new window
    new_window = tk.Toplevel(root)
    new_window.title("Excel Data")

    fig, ax = plt.subplots()

    # Assuming the first column is the x-axis and the remaining columns are y-axis data
    x_data = [row[0] for row in data]
    for i in range(1, len(data[0])):
        y_data = [row[i] for row in data]
        ax.plot(x_data, y_data, label=f"Line {i}")

    # Set labels and title
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title("Line Graph")
    ax.legend()

    # Create a canvas to display the line graph
    canvas = FigureCanvasTkAgg(fig, master=new_window)
    canvas.draw()
    canvas.get_tk_widget().pack()




# Create the main window
root = tk.Tk()
root.title("Hand Landmark Tracking System")
root.geometry("720x480")  # Set the window size to 720x480 pixels

# Create a label for the text
text_label = tk.Label(text="Hand Landmark Tracking System", font=("Arial", 24, "bold"), fg="blue")
text_label.pack(pady=20)  # Add some padding around the text label
text_label = tk.Label(text="MDE UX RESEARCH - Tim Yeh v1.0", font=("Arial", 8, "bold"), fg="blue")
text_label.pack(pady=20)  # Add some padding around the text label
text_label.place(x=510, y=65)

# Create a canvas for the background
canvas = tk.Canvas(root, width=720, height=480)
canvas.pack(fill="both", expand=True)

# Set a background image or color
background_image = PhotoImage(file="background.png")
canvas.create_image(0, 0, anchor="nw", image=background_image)
# Or, set a background color
# canvas.configure(bg="lightgray")


# Load the application icon
app_icon = PhotoImage(file="hx_icon.ico")
root.iconphoto(False, app_icon)

# Create a function to run the hand_landmark_excel_topview.py script
def run_handmark_topview():
    subprocess.run(["python", "hand_landmark_excel_topview.py"])

def run_handmark_frontview():
    subprocess.run(["python", "hand_landmark_excel_frontview.py"])

def run_handmark_sideview():
    subprocess.run(["python", "hand_landmark_excel_sideview.py"])

def exit_app():
    root.destroy()

# Create a function to run the hand_landmark_excel_topview.py script
def run_script():
    script_path = "hand_landmark_excel_frontview.py"
    with tqdm(total=100, unit="B", unit_scale=True, unit_divisor=1024, desc="Running script") as progress_bar:
        subprocess.run(["python", script_path], shell=True)
        progress_bar.update(100)


# Create a button to capture photos
capture_button = tk.Button(root, text="Capture TopView Handmark with MICE", command=run_handmark_topview)
capture_button.place(x=300, y=200)

capture_button = tk.Button(root, text="Capture FrontView Handmark with MICE", command=run_handmark_frontview)
capture_button.place(x=300, y=240)

capture_button = tk.Button(root, text="Capture SideView Handmark with MICE", command=run_handmark_sideview)
capture_button.place(x=300, y=280)

capture_button = tk.Button(root, text="Exit the app", command=exit_app)
capture_button.place(x=300, y=360)

import_button = tk.Button(root, text="Import Excel File", command=import_excel_file)
import_button.place(x=300, y=320)

error_label = tk.Label(root, text="")
error_label.place(x=10, y=360)

data_frame = tk.Frame(root)  # Create the data_frame instance
data_frame.place(x=10, y=380)

# Start the main event loop
root.mainloop()
