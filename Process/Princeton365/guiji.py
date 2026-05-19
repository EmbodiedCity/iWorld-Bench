import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
from tkinter import messagebox


def read_camera_trajectory(file_path):
    """
    Reads the camera trajectory from the given file and extracts the translation components.
    """
    positions = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Split the line into columns
                columns = line.strip().split()
                if len(columns) >= 19:
                    # Extract the translation components (columns 10, 14, 18)
                    x = float(columns[10])  # Column 10 (x)
                    y = float(columns[14])  # Column 14 (y)
                    z = float(columns[18])  # Column 18 (z)
                    positions.append([x, y, z])
        return np.array(positions)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read file: {e}")
        return None


def plot_camera_trajectory(positions, ax):
    """
    Plots the camera trajectory in 3D space.
    """
    if positions is None or len(positions) == 0:
        messagebox.showerror("Error", "No valid positions to plot.")
        return

    # Clear the previous plot
    ax.clear()

    # Extract x, y, z coordinates
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    # Plot the trajectory
    ax.plot(x, y, z, label='Camera Trajectory', color='blue', marker='o')
    ax.scatter(x, y, z, color='red', s=10, label='Camera Positions')  # Optional: scatter points

    # Mark the first and last points with their indices
    ax.text(x[0], y[0], z[0], '0', color='green', fontsize=12)  # First point
    ax.text(x[-1], y[-1], z[-1], str(len(positions) - 1), color='purple', fontsize=12)  # Last point

    # Calculate the global min and max across all axes
    global_min = min(min(x), min(y), min(z))  # Global minimum across x, y, z
    global_max = max(max(x), max(y), max(z))  # Global maximum across x, y, z

    # Set the same range for all axes
    ax.set_xlim([global_min, global_max])  # X-axis range
    ax.set_ylim([global_min, global_max])  # Y-axis range
    ax.set_zlim([global_min, global_max])  # Z-axis range

    # Invert Z axis direction
    ax.invert_zaxis()

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Trajectory Visualization')
    ax.legend()

    # Redraw the canvas
    ax.figure.canvas.draw()




def update_view(ax, elev, azim):
    """
    Updates the view angle of the 3D plot.
    """
    ax.view_init(elev=elev, azim=azim)
    ax.figure.canvas.draw()


def handle_file_drop(event, ax, elev_slider, azim_slider):
    """
    Handles the drag-and-drop event for a file.
    """
    file_path = event.data.strip()  # Get the file path from the drag-and-drop event
    if file_path.endswith(".txt"):
        positions = read_camera_trajectory(file_path)
        plot_camera_trajectory(positions, ax)

        # Reset sliders to default values
        elev_slider.set(30)  # Default elevation
        azim_slider.set(45)  # Default azimuth

        # Display the file path
        file_label.config(text=f"Loaded file: {file_path}")
    else:
        messagebox.showerror("Error", "Please drop a valid .txt file.")


def save_plot():
    """
    Saves the current plot to a file.
    """
    save_path = "camera_trajectory.png"
    fig.savefig(save_path)
    messagebox.showinfo("Success", f"Plot saved as {save_path}")


# Create the main application window
root = TkinterDnD.Tk()  # Use TkinterDnD for drag-and-drop functionality
root.title("Camera Trajectory Visualization")
root.geometry("800x600")

# Create a matplotlib figure and axis
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Create a canvas to embed the matplotlib figure in the Tkinter window
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Add sliders for controlling elevation and azimuth
elev_slider = tk.Scale(root, from_=0, to=90, orient=tk.HORIZONTAL, label="Elevation (X-axis)", command=lambda val: update_view(ax, int(val), azim_slider.get()))
elev_slider.pack(side=tk.LEFT, padx=10, pady=10)
elev_slider.set(30)  # Default elevation

azim_slider = tk.Scale(root, from_=0, to=360, orient=tk.HORIZONTAL, label="Azimuth (Y-axis)", command=lambda val: update_view(ax, elev_slider.get(), int(val)))
azim_slider.pack(side=tk.RIGHT, padx=10, pady=10)
azim_slider.set(45)  # Default azimuth

# Add a label for drag-and-drop instructions
label = tk.Label(root, text="Drag and Drop a .txt File Here", font=("Arial", 14))
label.pack(pady=20)

# Add a label to display the loaded file path
file_label = tk.Label(root, text="No file loaded", font=("Arial", 12), fg="green")
file_label.pack(pady=10)

# Add a save button
save_button = tk.Button(root, text="Save Plot", command=save_plot)
save_button.pack(pady=10)

# Enable drag-and-drop functionality
root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', lambda event: handle_file_drop(event, ax, elev_slider, azim_slider))

# Run the application
root.mainloop()
