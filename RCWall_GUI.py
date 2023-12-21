import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Define the window size and title
root = tk.Tk()
root.geometry("800x500")
root.title("RC-SC Shear Wall Analysis")

# Create a container frame to hold all elements
container = tk.Frame(root)
container.pack(fill="both", expand=True)

# Define the input parameters frame
input_frame = tk.Frame(container, borderwidth=2, relief="groove")
input_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)

# Input parameter labels and entries
labels = [
    "N [kN]:",
    "Load Coefficient:",
    "T_w (mm):",
    "L_w (mm):",
    "H_w (mm):",
    "Lbe (mm):",
    "f_r (MPa):",
    "f_c (MPa):",
    "rou_Yb:",
    "rou_Yw:",
    "Load Coefficient:",
]

entries = []
for i, label in enumerate(labels):
    tk.Label(input_frame, text=label).grid(row=i, column=0, sticky="W")
    entry = tk.Entry(input_frame)
    entry.grid(row=i, column=1)
    entries.append(entry)

# Define the plot frame and output frame containers
plot_container = tk.Frame(container)
plot_container.pack(side="right", fill="both", expand=True)

output_container = tk.Frame(container)
output_container.pack(side="bottom", fill="both", expand=True)

# Create the plot frame inside its container
plot_frame = tk.Frame(plot_container, borderwidth=2, relief="groove")
plot_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Initialize the plot figure and canvas
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Create the output frame inside its container
output_frame = tk.Frame(output_container, borderwidth=2, relief="groove")
output_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Output labels and displays
output_labels = ["Base Shear (kN):", "Displacement (mm):"]
output_displays = []
for i, label in enumerate(output_labels):
    tk.Label(output_frame, text=label).grid(row=i, column=0, sticky="W")
    display = tk.Label(output_frame, text="")
    display.grid(row=i, column=1)
    output_displays.append(display)

# Define the button in its own frame
button_frame = tk.Frame(container)
button_frame.pack(side="bottom", anchor="w", padx=10, pady=10)

# Function to analyze input and update UI
def analyze_button_click():
    # Get input parameters from entries
    parameters = []
    for entry in entries:
        parameters.append(float(entry.get()))

    # Replace this with your actual analysis code using parameters
    # Example: base_shear, displacement = rcsc_shear_wall_analysis.analyze(parameters)
    base_shear = parameters[0]
    displacement = parameters[1]

    # Update output displays
    output_displays[0].configure(text=str(base_shear))
    output_displays[1].configure(text=str(displacement))

    # Generate and update the plot (replace with your actual plot generation code)
    # Example plot: force vs displacement
    forces = [0, base_shear]
    displacements = [0, displacement]
    ax.clear()
    ax.plot(forces, displacements)
    ax.set_xlabel("Force (kN)")
    ax.set_ylabel("Displacement (mm)")
    ax.set_title("Force vs. Displacement")

# Button to trigger analysis
analyze_button = tk.Button(button_frame, text="Analyze", command=analyze_button_click)
analyze_button.pack()


# Run the main event loop
root.mainloop()