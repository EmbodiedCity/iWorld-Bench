import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def load_pose_data(pose_file_path):
    poses = []
    with open(pose_file_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) == 7:
                pose = [float(v) for v in values]
                poses.append(pose)
    
    poses = np.array(poses)
    print(f"Loaded {len(poses)} poses")
    return poses

def quaternion_to_rotation_matrix(q):
    qw, qx, qy, qz = q
    
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

def plot_3d_trajectory(poses, save_path=None):
    positions = poses[:, :3]
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(x, y, z, 'b-', linewidth=2.0, alpha=0.8, label='Trajectory')
    
    ax.scatter(x[0], y[0], z[0], c='green', s=150, marker='o', label='Start', edgecolors='black', linewidth=2)
    ax.scatter(x[-1], y[-1], z[-1], c='red', s=150, marker='s', label='End', edgecolors='black', linewidth=2)
    
    step = max(1, len(poses) // 15)
    for i in range(0, len(poses), step):
        pos = positions[i]
        q = poses[i, 3:]
        R = quaternion_to_rotation_matrix([q[3], q[0], q[1], q[2]])
        
        axis_length = 0.3
        ax.quiver(pos[0], pos[1], pos[2], 
                 R[0, 0], R[1, 0], R[2, 0], 
                 length=axis_length, color='red', alpha=0.6, linewidth=1.5)
        ax.quiver(pos[0], pos[1], pos[2], 
                 R[0, 1], R[1, 1], R[2, 1], 
                 length=axis_length, color='green', alpha=0.6, linewidth=1.5)
        ax.quiver(pos[0], pos[1], pos[2], 
                 R[0, 2], R[1, 2], R[2, 2], 
                 length=axis_length, color='blue', alpha=0.6, linewidth=1.5)
    
    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
    ax.set_title('3D Trajectory Visualization', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 3D plot to: {save_path}")
    
    plt.show()
    
    return fig, ax

def plot_xy_plane(poses, save_path=None):
    positions = poses[:, :3]
    x, y = positions[:, 0], positions[:, 1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(x, y, 'b-', linewidth=2.0, alpha=0.8, label='Trajectory')
    
    ax.scatter(x[0], y[0], c='green', s=150, marker='o', label='Start', edgecolors='black', linewidth=2)
    ax.scatter(x[-1], y[-1], c='red', s=150, marker='s', label='End', edgecolors='black', linewidth=2)
    
    dx = np.diff(x)
    dy = np.diff(y)
    skip = max(1, len(dx) // 20)
    for i in range(0, len(dx), skip):
        ax.arrow(x[i], y[i], dx[i]*0.2, dy[i]*0.2, 
                head_width=0.2, head_length=0.2, fc='orange', ec='orange', alpha=0.6, linewidth=1.5)
    
    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax.set_title('XY Plane Projection', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved XY plot to: {save_path}")
    
    plt.show()
    
    return fig, ax

def plot_xz_plane(poses, save_path=None):
    positions = poses[:, :3]
    x, z = positions[:, 0], positions[:, 2]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(x, z, 'g-', linewidth=2.0, alpha=0.8, label='Trajectory')
    
    ax.scatter(x[0], z[0], c='blue', s=150, marker='o', label='Start', edgecolors='black', linewidth=2)
    ax.scatter(x[-1], z[-1], c='red', s=150, marker='s', label='End', edgecolors='black', linewidth=2)
    
    points = ax.scatter(x, z, c=z, cmap='viridis', s=20, alpha=0.6, edgecolors='none')
    
    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Z (m)', fontsize=12, fontweight='bold')
    ax.set_title('XZ Plane Projection (Color by Height)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(points, ax=ax)
    cbar.set_label('Height Z (m)', fontsize=11, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved XZ plot to: {save_path}")
    
    plt.show()
    
    return fig, ax

def plot_yz_plane(poses, save_path=None):
    positions = poses[:, :3]
    y, z = positions[:, 1], positions[:, 2]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(y, z, 'r-', linewidth=2.0, alpha=0.8, label='Trajectory')
    
    ax.scatter(y[0], z[0], c='blue', s=150, marker='o', label='Start', edgecolors='black', linewidth=2)
    ax.scatter(y[-1], z[-1], c='green', s=150, marker='s', label='End', edgecolors='black', linewidth=2)
    
    from scipy.ndimage import gaussian_filter
    
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()
    y_bins = np.linspace(y_min, y_max, 100)
    z_bins = np.linspace(z_min, z_max, 100)
    
    hist, y_edges, z_edges = np.histogram2d(y, z, bins=(y_bins, z_bins))
    hist = gaussian_filter(hist, sigma=1.0)
    
    extent = [y_edges[0], y_edges[-1], z_edges[0], z_edges[-1]]
    ax.imshow(hist.T, extent=extent, origin='lower', aspect='auto', 
             cmap='hot', alpha=0.2)
    
    ax.set_xlabel('Y (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Z (m)', fontsize=12, fontweight='bold')
    ax.set_title('YZ Plane Projection with Density Heatmap', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved YZ plot to: {save_path}")
    
    plt.show()
    
    return fig, ax

def plot_2d_comparison(poses, save_path=None):
    positions = poses[:, :3]
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].plot(x, y, 'b-', linewidth=1.5, alpha=0.8)
    axes[0].scatter(x[0], y[0], c='green', s=80, marker='o', edgecolors='black')
    axes[0].scatter(x[-1], y[-1], c='red', s=80, marker='s', edgecolors='black')
    axes[0].set_xlabel('X (m)', fontsize=11)
    axes[0].set_ylabel('Y (m)', fontsize=11)
    axes[0].set_title('XY Plane', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    sc = axes[1].scatter(x, z, c=z, cmap='viridis', s=30, alpha=0.8, edgecolors='none')
    axes[1].plot(x, z, 'k-', linewidth=0.5, alpha=0.5)
    axes[1].scatter(x[0], z[0], c='blue', s=80, marker='o', edgecolors='black')
    axes[1].scatter(x[-1], z[-1], c='red', s=80, marker='s', edgecolors='black')
    axes[1].set_xlabel('X (m)', fontsize=11)
    axes[1].set_ylabel('Z (m)', fontsize=11)
    axes[1].set_title('XZ Plane', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    cbar = plt.colorbar(sc, ax=axes[1])
    cbar.set_label('Height Z (m)', fontsize=10)
    
    axes[2].plot(y, z, 'r-', linewidth=1.5, alpha=0.8)
    axes[2].scatter(y[0], z[0], c='blue', s=80, marker='o', edgecolors='black')
    axes[2].scatter(y[-1], z[-1], c='green', s=80, marker='s', edgecolors='black')
    axes[2].set_xlabel('Y (m)', fontsize=11)
    axes[2].set_ylabel('Z (m)', fontsize=11)
    axes[2].set_title('YZ Plane', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 2D comparison plot to: {save_path}")
    
    plt.show()
    
    return fig, axes

def calculate_trajectory_stats(poses):
    positions = poses[:, :3]
    
    diffs = np.diff(positions, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    
    stats = {
        'num_poses': len(positions),
        'total_distance': np.sum(distances),
        'avg_distance_per_step': np.mean(distances),
        'max_distance_per_step': np.max(distances),
        'x_range': [np.min(positions[:, 0]), np.max(positions[:, 0])],
        'y_range': [np.min(positions[:, 1]), np.max(positions[:, 1])],
        'z_range': [np.min(positions[:, 2]), np.max(positions[:, 2])],
        'start_point': positions[0],
        'end_point': positions[-1],
        'displacement': positions[-1] - positions[0],
        'displacement_magnitude': np.linalg.norm(positions[-1] - positions[0])
    }
    
    print("="*60)
    print("TRAJECTORY STATISTICS:")
    print("="*60)
    print(f"Number of poses: {stats['num_poses']}")
    print(f"Total trajectory length: {stats['total_distance']:.2f} m")
    print(f"Average step length: {stats['avg_distance_per_step']:.3f} m")
    print(f"Maximum step length: {stats['max_distance_per_step']:.3f} m")
    print(f"X range: [{stats['x_range'][0]:.2f}, {stats['x_range'][1]:.2f}]")
    print(f"Y range: [{stats['y_range'][0]:.2f}, {stats['y_range'][1]:.2f}]")
    print(f"Z range: [{stats['z_range'][0]:.2f}, {stats['z_range'][1]:.2f}]")
    print(f"Start point: ({stats['start_point'][0]:.2f}, {stats['start_point'][1]:.2f}, {stats['start_point'][2]:.2f})")
    print(f"End point: ({stats['end_point'][0]:.2f}, {stats['end_point'][1]:.2f}, {stats['end_point'][2]:.2f})")
    print(f"Displacement vector: ({stats['displacement'][0]:.2f}, {stats['displacement'][1]:.2f}, {stats['displacement'][2]:.2f})")
    print(f"Displacement magnitude: {stats['displacement_magnitude']:.2f} m")
    print("="*60)
    
    return stats

def main():
    base_path = "/path/to/local/data"
    pose_file = os.path.join(base_path, "pose_left.txt")
    output_dir = os.path.join(base_path, "trajectory_plots")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(pose_file):
        print(f"File not found: {pose_file}")
        return
    
    print("Loading pose data...")
    poses = load_pose_data(pose_file)
    
    if len(poses) == 0:
        print("No valid pose data found")
        return
    
    stats = calculate_trajectory_stats(poses)
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS...")
    print("="*60)
    
    print("\n1. Generating 3D trajectory plot...")
    plot_3d_trajectory(poses, save_path=os.path.join(output_dir, "3d_trajectory.png"))
    
    print("\n2. Generating XY plane plot...")
    plot_xy_plane(poses, save_path=os.path.join(output_dir, "xy_plane.png"))
    
    print("\n3. Generating XZ plane plot...")
    plot_xz_plane(poses, save_path=os.path.join(output_dir, "xz_plane.png"))
    
    print("\n4. Generating YZ plane plot...")
    plot_yz_plane(poses, save_path=os.path.join(output_dir, "yz_plane.png"))
    
    print("\n5. Generating 2D comparison plot...")
    plot_2d_comparison(poses, save_path=os.path.join(output_dir, "2d_comparison.png"))
    
    np.savetxt(os.path.join(output_dir, "trajectory_data.csv"), 
               poses[:, :3], delimiter=',', 
               header='x,y,z', comments='')
    
    print("\n" + "="*60)
    print(f"All visualizations saved to: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()