import os
import numpy as np
from scipy.interpolate import interp1d

def process_single_folder(folder_path):
    """processing filesindexes.txtposes.npy，full_poses.txt"""
    indexes_path = os.path.join(folder_path, "indexes.txt")
    poses_path = os.path.join(folder_path, "poses.npy")
    output_path = os.path.join(folder_path, "full_poses.txt")

    if not os.path.exists(indexes_path):
        print(f" skipping {folder_path}：Not foundindexes.txt")
        return
    if not os.path.exists(poses_path):
        print(f" skipping {folder_path}：Not foundposes.npy")
        return

    try:
        # frames
        indexes_raw = np.loadtxt(indexes_path, dtype=int)
        keyframe_indices = indexes_raw[:, 1]
        
        poses = np.load(poses_path)
        
        if len(keyframe_indices) != poses.shape[0]:
            print(f" skipping {folder_path}frames({len(keyframe_indices)})({poses.shape[0]})")
            return
        
        poses_6d = poses[:, :6].astype(np.float64)

        # Videoframes
        total_frames = keyframe_indices.max() + 1

        full_poses = np.zeros((total_frames, 6), dtype=np.float64)
        for dim in range(6):
            interp_func = interp1d(
                x=keyframe_indices,
                y=poses_6d[:, dim],
                kind="linear",
                fill_value="extrapolate",
                assume_sorted=True,
                axis=0
            )
            all_frames = np.arange(total_frames, dtype=int)
            full_poses[:, dim] = interp_func(all_frames)

        # frames
        for i, frame_idx in enumerate(keyframe_indices):
            if not np.allclose(full_poses[frame_idx], poses_6d[i], atol=1e-8):
                print(f"⚠️ {folder_path} frames{frame_idx}")

        np.savetxt(
            output_path,
            full_poses,
            fmt="%.7e",
            delimiter=" ",
            newline="\n",
            encoding="utf-8"
        )
        print(f" processing {folder_path}full_poses.txt{total_frames}frames")

    except IndexError:
        print(f" skipping {folder_path}indexes.txt，lines2columns")
    except Exception as e:
        print(f" processing {folder_path} failed：{str(e)}")

def process_all_folders(root_dir):
    """processing"""
    if not os.path.exists(root_dir):
        print(f" does not exist{root_dir}")
        return

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            process_single_folder(folder_path)

if __name__ == "__main__":
    root_directory = r"/path/to/local/data"
    print(f"Processing{root_directory}")
    process_all_folders(root_directory)
    print("processing")