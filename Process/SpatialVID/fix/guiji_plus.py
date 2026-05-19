import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
from tkinter import messagebox

def read_camera_trajectory(file_path):
    """
    12columns/16columns，(x,y,z)
    - 16columns443、7、114/8/12columns，[:,3]columns
    - 12columns343、7、11columnscolumns
    - 19columns
    """
    positions = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                columns = line.split()
                cols_count = len(columns)

                # 16columns44，full_poses_m.txt
                if cols_count == 16:
                    x = float(columns[3])
                    y = float(columns[7])
                    z = float(columns[11])
                    positions.append([x, y, z])
                
                # 12columnslines34
                elif cols_count == 12:
                    x = float(columns[3])
                    y = float(columns[7])
                    z = float(columns[11])
                    positions.append([x, y, z])
                
                # 19columns
                elif cols_count >= 19:
                    x = float(columns[9])
                    y = float(columns[13])
                    z = float(columns[17])
                    positions.append([x, y, z])
                
                else:
                    messagebox.warning(
                        "",
                        f"{line_num}linescolumns{cols_count}，12/16/19columns，skipping"
                    )
        
        if not positions:
            messagebox.warning("", "！")
            return None
        return np.array(positions)
    
    except Exception as e:
        messagebox.showerror("Error", f"failed{str(e)}")
        return None

def plot_camera_trajectory(positions, ax):
    """3D，"""
    if positions is None or len(positions) == 0:
        messagebox.showerror("Error", "！")
        return

    ax.clear()
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    ax.plot(x, y, z, label='Camera Trajectory', color='blue', linewidth=1.5)
    ax.scatter(x, y, z, color='red', s=8, alpha=0.7, label='Camera Positions')

    # frames
    ax.text(x[0], y[0], z[0], 'Start (0)', color='green', fontsize=10, weight='bold')
    ax.text(x[-1], y[-1], z[-1], f'End ({len(positions)-1})', color='purple', fontsize=10, weight='bold')

    global_min = np.min(positions)
    global_max = np.max(positions)
    ax.set_xlim([global_min, global_max])
    ax.set_ylim([global_min, global_max])
    ax.set_zlim([global_min, global_max])
    ax.invert_zaxis()

    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_zlabel('Z (m)', fontsize=10)
    ax.set_title('3D Camera Trajectory Visualization', fontsize=12, pad=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.figure.canvas.draw()

def update_view(ax, elev, azim):
    """3D"""
    ax.view_init(elev=int(elev), azim=int(azim))
    ax.figure.canvas.draw()

def handle_file_drop(event, ax, elev_slider, azim_slider):
    """processing"""
    file_path = event.data.strip().strip('{}')
    if not file_path.endswith(".txt"):
        messagebox.showerror("Error", ".txt！")
        return

    positions = read_camera_trajectory(file_path)
    if positions is not None:
        plot_camera_trajectory(positions, ax)
        elev_slider.set(30)
        azim_slider.set(45)
        file_label.config(text=f"{file_path}", fg="darkgreen")

def save_plot(fig):
    """PNG"""
    try:
        save_path = "camera_trajectory_3d.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        messagebox.showinfo("", f"{os.path.abspath(save_path)}")
    except Exception as e:
        messagebox.showerror("failed", f"failed{str(e)}")

def select_file(ax, elev_slider, azim_slider):
    """，"""
    from tkinter import filedialog
    file_path = filedialog.askopenfilename(
        title="",
        filetypes=[("TXT", "*.txt"), ("", "*.*")]
    )
    if file_path:
        positions = read_camera_trajectory(file_path)
        if positions is not None:
            plot_camera_trajectory(positions, ax)
            elev_slider.set(30)
            azim_slider.set(45)
            file_label.config(text=f"{file_path}", fg="darkgreen")

# -------------------------- --------------------------
if __name__ == "__main__":
    import os

    root = TkinterDnD.Tk()
    root.title("3D12/16columns")
    root.geometry("900x700")
    root.resizable(True, True)

    # Matplotlib
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

    ctrl_frame = tk.Frame(root)
    ctrl_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

    elev_slider = tk.Scale(
        ctrl_frame, from_=0, to=90, orient=tk.HORIZONTAL,
        label=" (Elevation)", command=lambda val: update_view(ax, val, azim_slider.get()),
        font=("Arial", 9), length=300
    )
    elev_slider.grid(row=0, column=0, padx=10, pady=5)
    elev_slider.set(30)

    azim_slider = tk.Scale(
        ctrl_frame, from_=0, to=360, orient=tk.HORIZONTAL,
        label=" (Azimuth)", command=lambda val: update_view(ax, elev_slider.get(), val),
        font=("Arial", 9), length=300
    )
    azim_slider.grid(row=0, column=1, padx=10, pady=5)
    azim_slider.set(45)

    save_btn = tk.Button(
        ctrl_frame, text="", command=lambda: save_plot(fig),
        font=("Arial", 10), bg="#4CAF50", fg="white", padx=10, pady=5
    )
    save_btn.grid(row=0, column=2, padx=20, pady=5)

    select_btn = tk.Button(
        ctrl_frame, text="", command=lambda: select_file(ax, elev_slider, azim_slider),
        font=("Arial", 10), bg="#2196F3", fg="white", padx=10, pady=5
    )
    select_btn.grid(row=0, column=3, padx=10, pady=5)

    tip_label = tk.Label(
        root, text="💡 12columnslines/16columns44.txt",
        font=("Arial", 10), fg="#666666"
    )
    tip_label.pack(pady=5)

    file_label = tk.Label(
        root, text="", font=("Arial", 10), fg="red"
    )
    file_label.pack(pady=5)

    root.drop_target_register(DND_FILES)
    root.dnd_bind('<<Drop>>', lambda event: handle_file_drop(event, ax, elev_slider, azim_slider))

    # lines
    root.mainloop()