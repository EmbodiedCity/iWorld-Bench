import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import warnings
warnings.filterwarnings('ignore')

def read_translations_from_json(file_path, max_points=None):
    print(f"read {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    translations = []
    tokens = []
    sample_tokens = []
    
    for i, item in enumerate(data):
        if 'translation' in item and len(item['translation']) >= 3:
            translations.append(item['translation'])
            tokens.append(item.get('token', f'item_{i}'))
            sample_tokens.append(item.get('sample_token', f'sample_{i}'))
            
            if max_points and len(translations) >= max_points:
                break
    
    print(f"points={len(translations)}")
    
    return {
        'translations': np.array(translations),
        'tokens': tokens,
        'sample_tokens': sample_tokens,
        'total_points': len(translations)
    }

def downsample_data(translations, target_points=5000):
    if len(translations) <= target_points:
        return translations
    
    step = len(translations) // target_points
    indices = np.arange(0, len(translations), step)
    
    if indices[-1] != len(translations) - 1:
        indices = np.append(indices, len(translations) - 1)
    
    return translations[indices]

def plot_3d_path_with_direction(translations, save_path=None, title_suffix=""):
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    x = translations[:, 0]
    y = translations[:, 1]
    z = translations[:, 2]
    
    progress = np.linspace(0, 1, len(x))
    
    for i in range(len(x)-1):
        color = plt.cm.rainbow(progress[i])
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], 
                color=color, linewidth=0.5, alpha=0.8)
    
    key_indices = [0, 
                   len(x)//4, 
                   len(x)//2, 
                   3*len(x)//4, 
                   len(x)-1]
    
    for idx in key_indices:
        color = plt.cm.rainbow(progress[idx])
        ax.scatter(x[idx], y[idx], z[idx], 
                  color=color, s=100, alpha=0.9,
                  edgecolors='black', linewidth=1)
    
    ax.scatter(x[0], y[0], z[0], 
               c='lime', s=200, marker='*', 
               edgecolors='black', linewidth=2, 
               label='Start', zorder=10)
    
    ax.scatter(x[-1], y[-1], z[-1], 
               c='red', s=200, marker='X', 
               edgecolors='black', linewidth=2, 
               label='End', zorder=10)
    
    arrow_interval = max(1, len(x)//20)
    for i in range(0, len(x)-arrow_interval, arrow_interval):
        if i + arrow_interval < len(x):
            dx = x[i+arrow_interval] - x[i]
            dy = y[i+arrow_interval] - y[i]
            dz = z[i+arrow_interval] - z[i]
            
            color = plt.cm.rainbow(progress[i])
            ax.quiver(x[i], y[i], z[i], 
                     dx, dy, dz, 
                     length=np.sqrt(dx**2 + dy**2 + dz**2)*0.9,
                     arrow_length_ratio=0.1,
                     color=color, alpha=0.6,
                     linewidth=1)
    
    ax.set_xlabel('X Coordinate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=14, fontweight='bold')
    ax.set_zlabel('Z Coordinate', fontsize=14, fontweight='bold')
    
    title = f'3D Path with Direction{title_suffix}'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow, 
                              norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('Path Progress (0=Start, 1=End)', fontsize=12)
    
    ax.legend(loc='upper left', fontsize=11)
    
    ax.view_init(elev=30, azim=45)
    
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    z_range = z.max() - z.min()
    
    margin = 0.1
    ax.set_xlim(x.min() - x_range*margin, x.max() + x_range*margin)
    ax.set_ylim(y.min() - y_range*margin, y.max() + y_range*margin)
    ax.set_zlim(z.min() - z_range*margin, z.max() + z_range*margin)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"3D path saved to {save_path}")
    
    plt.show()
    plt.close(fig)
    
    return fig, ax

def plot_2d_projections_improved(translations, save_dir='./', title_suffix=""):
    x = translations[:, 0]
    y = translations[:, 1]
    z = translations[:, 2]
    
    progress = np.linspace(0, 1, len(x))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flat
    
    projections = [
        (x, y, 'X', 'Y', 'XY Plane (Top View)', axes[0]),
        (x, z, 'X', 'Z', 'XZ Plane (Side View)', axes[1]),
        (y, z, 'Y', 'Z', 'YZ Plane (Side View)', axes[2])
    ]
    
    for i, (data_x, data_y, xlabel, ylabel, title, ax) in enumerate(projections):
        for j in range(len(data_x)-1):
            color = plt.cm.rainbow(progress[j])
            ax.plot(data_x[j:j+2], data_y[j:j+2], 
                    color=color, linewidth=1, alpha=0.8)
        
        key_indices = [0, len(x)//2, len(x)-1]
        key_labels = ['Start', 'Mid', 'End']
        key_colors = ['lime', 'yellow', 'red']
        key_markers = ['*', 'o', 'X']
        
        for idx, label, color, marker in zip(key_indices, key_labels, key_colors, key_markers):
            ax.scatter(data_x[idx], data_y[idx], 
                      c=color, s=150, marker=marker,
                      edgecolors='black', linewidth=1.5,
                      label=label, zorder=10)
        
        ax.set_xlabel(f'{xlabel} Coordinate', fontsize=12)
        ax.set_ylabel(f'{ylabel} Coordinate', fontsize=12)
        ax.set_title(f'{title}{title_suffix}', fontsize=13, fontweight='bold')
        
        ax.grid(True, alpha=0.3, linestyle='--')
        if i == 0:
            ax.legend(fontsize=10)
        
        arrow_count = min(10, len(data_x)//100)
        if arrow_count > 0:
            interval = len(data_x) // (arrow_count + 1)
            for j in range(1, arrow_count + 1):
                idx = j * interval
                if idx < len(data_x) - 1:
                    dx = data_x[idx+1] - data_x[idx]
                    dy = data_y[idx+1] - data_y[idx]
                    color = plt.cm.rainbow(progress[idx])
                    ax.arrow(data_x[idx], data_y[idx], 
                            dx*0.8, dy*0.8,
                            head_width=np.sqrt(dx**2+dy**2)*0.05,
                            head_length=np.sqrt(dx**2+dy**2)*0.1,
                            fc=color, ec=color, alpha=0.7)
    
    ax_3d = axes[3]
    from mpl_toolkits.mplot3d import Axes3D
    ax_3d = fig.add_subplot(224, projection='3d')
    
    step = max(1, len(x)//1000)
    ax_3d.scatter(x[::step], y[::step], z[::step], 
                  c=progress[::step], cmap='rainbow', 
                  s=10, alpha=0.6)
    
    for i in range(0, len(x)-step, step*10):
        if i + step < len(x):
            color = plt.cm.rainbow(progress[i])
            ax_3d.plot(x[i:i+step+1:step], 
                       y[i:i+step+1:step], 
                       z[i:i+step+1:step], 
                       color=color, linewidth=0.5, alpha=0.6)
    
    ax_3d.scatter(x[0], y[0], z[0], c='lime', s=200, marker='*', label='Start')
    ax_3d.scatter(x[-1], y[-1], z[-1], c='red', s=200, marker='X', label='End')
    
    ax_3d.set_xlabel('X', fontsize=11)
    ax_3d.set_ylabel('Y', fontsize=11)
    ax_3d.set_zlabel('Z', fontsize=11)
    ax_3d.set_title('3D Overview', fontsize=13, fontweight='bold')
    ax_3d.view_init(elev=20, azim=45)
    ax_3d.legend(fontsize=9)
    
    plt.suptitle(f'Path Analysis - Multi-view Projections{title_suffix}', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    combined_path = f"{save_dir}multi_view_projections.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Multi-view projections saved to {combined_path}")
    
    plt.show()
    plt.close(fig)
    
    return fig, axes

def create_path_video(translations, save_path="path_animation.mp4", fps=30):
    print("anim...")
    
    if len(translations) > 2000:
        translations = downsample_data(translations, target_points=2000)
    
    x = translations[:, 0]
    y = translations[:, 1]
    z = translations[:, 2]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlabel('X Coordinate', fontsize=11)
    ax.set_ylabel('Y Coordinate', fontsize=11)
    ax.set_zlabel('Z Coordinate', fontsize=11)
    ax.set_title('3D Path Animation', fontsize=13, fontweight='bold')
    
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    z_range = z.max() - z.min()
    
    margin = 0.1
    ax.set_xlim(x.min() - x_range*margin, x.max() + x_range*margin)
    ax.set_ylim(y.min() - y_range*margin, y.max() + y_range*margin)
    ax.set_zlim(z.min() - z_range*margin, z.max() + z_range*margin)
    
    ax.grid(True, alpha=0.3)
    
    line, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.7)
    point, = ax.plot([], [], [], 'ro', markersize=8)
    progress_text = ax.text(0.02, 0.98, 0.98, '', 
                           transform=ax.transAxes, fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        progress_text.set_text('')
        return line, point, progress_text
    
    def update(frame):
        line.set_data(x[:frame+1], y[:frame+1])
        line.set_3d_properties(z[:frame+1])
        
        point.set_data([x[frame]], [y[frame]])
        point.set_3d_properties([z[frame]])
        
        progress = (frame + 1) / len(x) * 100
        progress_text.set_text(f'Progress: {progress:.1f}%\nPoint: {frame+1}/{len(x)}')
        
        ax.view_init(elev=20, azim=frame*360/len(x))
        
        return line, point, progress_text
    
    try:
        anim = animation.FuncAnimation(fig, update, frames=len(x),
                                      init_func=init, blit=True,
                                      interval=1000/fps)
        
        if os.path.exists('/usr/bin/ffmpeg'):
            writer = FFMpegWriter(fps=fps, metadata=dict(artist='Path Visualizer'), bitrate=1800)
            anim.save(save_path, writer=writer)
        else:
            gif_path = save_path.replace('.mp4', '.gif')
            anim.save(gif_path, writer='pillow', fps=fps)
            save_path = gif_path
        
        print(f"saved {save_path}")
        plt.close(fig)
        
    except Exception as e:
        print(f"anim err {e}; try gif")
        try:
            gif_path = save_path.replace('.mp4', '.gif')
            anim.save(gif_path, writer='pillow', fps=10)
            print(f"saved {gif_path}")
        except Exception as e2:
            print(f"gif err {e2}")
        finally:
            plt.close(fig)

def create_trajectory_heatmap(translations, save_dir='./'):
    print("heatmap...")
    
    x = translations[:, 0]
    y = translations[:, 1]
    z = translations[:, 2]
    
    fig = plt.figure(figsize=(15, 10))
    
    ax1 = fig.add_subplot(231)
    h1 = ax1.hist2d(x, y, bins=100, cmap='hot')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title('XY Plane Heatmap')
    plt.colorbar(h1[3], ax=ax1)
    
    ax2 = fig.add_subplot(232)
    h2 = ax2.hist2d(x, z, bins=100, cmap='hot')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Z Coordinate')
    ax2.set_title('XZ Plane Heatmap')
    plt.colorbar(h2[3], ax=ax2)
    
    ax3 = fig.add_subplot(233)
    h3 = ax3.hist2d(y, z, bins=100, cmap='hot')
    ax3.set_xlabel('Y Coordinate')
    ax3.set_ylabel('Z Coordinate')
    ax3.set_title('YZ Plane Heatmap')
    plt.colorbar(h3[3], ax=ax3)
    
    ax4 = fig.add_subplot(234)
    distances = np.sqrt(np.sum(np.diff(translations, axis=0)**2, axis=1))
    ax4.plot(distances, linewidth=1)
    ax4.set_xlabel('Point Index')
    ax4.set_ylabel('Step Distance')
    ax4.set_title('Step Distance Analysis')
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(235)
    cumulative_dist = np.cumsum(distances)
    ax5.plot(cumulative_dist, linewidth=2)
    ax5.set_xlabel('Point Index')
    ax5.set_ylabel('Cumulative Distance')
    ax5.set_title('Cumulative Distance')
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(236)
    ax6.hist(z, bins=50, alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Z Coordinate (Height)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Height Distribution')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Path Analysis - Heatmaps and Statistics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    heatmap_path = f"{save_dir}path_heatmaps.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"saved {heatmap_path}")
    
    plt.show()
    plt.close(fig)

def main():
    base_dir = "/path/to/local/data"
    json_file_path = os.path.join(base_dir, "sample_annotation.json")
    
    output_dir = "./path_visualization/"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("=" * 60)
        print("3D Path Visualization Tool")
        print("=" * 60)
        
        print(f"cwd={os.getcwd()} data={json_file_path}")
        
        if not os.path.exists(json_file_path):
            print("err missing json")
            return
        
        print("load...")
        data_dict = read_translations_from_json(json_file_path, max_points=50000)
        
        if data_dict['total_points'] == 0:
            print("err no translations")
            return
        
        print(f"n={data_dict['total_points']}")
        
        print("downsample...")
        sampled_data = downsample_data(data_dict['translations'], target_points=5000)
        print(f"n_vis={len(sampled_data)}")
        
        print("3d...")
        plot_3d_path_with_direction(
            sampled_data,
            save_path=os.path.join(output_dir, "3d_path_direction.png"),
            title_suffix=f" (Sampled: {len(sampled_data)} points)"
        )
        
        print("proj...")
        plot_2d_projections_improved(
            sampled_data,
            save_dir=output_dir,
            title_suffix=f" (Sampled: {len(sampled_data)} points)"
        )
        
        print("stats...")
        create_trajectory_heatmap(
            sampled_data,
            save_dir=output_dir
        )
        
        print("anim...")
        anim_data = downsample_data(sampled_data, target_points=1000)
        create_path_video(
            anim_data,
            save_path=os.path.join(output_dir, "path_animation.mp4"),
            fps=20
        )
        
        print("2d...")
        create_simple_trajectory_plots(sampled_data, output_dir)
        
        print("\n" + "=" * 60)
        print(f"done out={output_dir}")
        print("=" * 60)
        
        print("files:")
        for file in os.listdir(output_dir):
            if file.endswith(('.png', '.mp4', '.gif')):
                file_path = os.path.join(output_dir, file)
                size = os.path.getsize(file_path) / 1024
                print(f"  - {file} ({size:.1f} KB)")
        
    except Exception as e:
        print(f"err {e}")
        import traceback
        traceback.print_exc()

def create_simple_trajectory_plots(translations, save_dir):
    x = translations[:, 0]
    y = translations[:, 1]
    z = translations[:, 2]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(x, y, 'b-', linewidth=1, alpha=0.7)
    axes[0].scatter(x[0], y[0], c='green', s=100, marker='o', label='Start', zorder=5)
    axes[0].scatter(x[-1], y[-1], c='red', s=100, marker='s', label='End', zorder=5)
    axes[0].set_xlabel('X Coordinate')
    axes[0].set_ylabel('Y Coordinate')
    axes[0].set_title('2D Trajectory (XY Plane)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    axes[1].plot(z, 'g-', linewidth=2)
    axes[1].axhline(y=np.mean(z), color='r', linestyle='--', alpha=0.5, label=f'Mean: {np.mean(z):.2f}')
    axes[1].fill_between(range(len(z)), np.min(z), np.max(z), alpha=0.1, color='gray')
    axes[1].set_xlabel('Point Index')
    axes[1].set_ylabel('Z Coordinate (Height)')
    axes[1].set_title('Height Profile')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "simple_trajectory.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    required_libs = ['matplotlib', 'numpy']
    
    for lib in required_libs:
        try:
            if lib == 'matplotlib':
                import matplotlib
            elif lib == 'numpy':
                import numpy as np
        except ImportError:
            print(f"pip {lib}")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
    
    try:
        import subprocess
        subprocess.run(['ffmpeg', '-version'], capture_output=True)
        print("ffmpeg ok")
    except:
        print("no ffmpeg -> gif only")
    
    main()