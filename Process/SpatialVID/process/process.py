import os
import shutil
import sys
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import cv2
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

"""
processing
==========================

1. group_0001group_0002+
2. groupprocessingpython process.py 6 group_00xx
3. ，groupprocessing
==========================
"""

# ===================== =====================
MODE = "PATH_ONLY"
RAW_BASE = r"/path/to/local/data"
TARGET_ROOT = r"/path/to/local/data"
VIEW_TYPE = r"Human_front"
IMG_WIDTH = 4096
IMG_HEIGHT = 2160
SUPPORTED_IMG_SUFFIX = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
SUPPORTED_VIDEO_SUFFIX = [".mp4", ".avi", ".mov", ".mkv"]
FRAME_SAVE_FMT = "frame_{:06d}.jpg"
FRAME_SKIP = 1
FRAME_QUALITY = [cv2.IMWRITE_JPEG_QUALITY, 95]
POSE_FILE_MAP = {
    6: ("full_poses_fixed.txt", "six_DoF.txt"),
    7: ("full_poses_fixed.txt", "seven_element.txt"),
    16: ("full_poses_fixed.txt", "extrinsics_matrix.txt")
}

# ===================== =====================
def load_intrinsics(intrinsics_npy_path):
    try:
        intrinsics_data = np.load(intrinsics_npy_path)
        fx_norm, fy_norm, cx_norm, cy_norm = intrinsics_data[0][:4]
        fx = fx_norm * IMG_WIDTH
        fy = fy_norm * IMG_HEIGHT
        cx = cx_norm * IMG_WIDTH
        cy = cy_norm * IMG_HEIGHT
        return fx, fy, cx, cy
    except Exception as e:
        print(f" failed {intrinsics_npy_path}{e}")
        return None

def ensure_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def find_nested_path(base_dir, target_name):
    """/，"""
    for root, dirs, files in os.walk(base_dir):
        if target_name in dirs:
            return os.path.join(root, target_name)
        for suffix in SUPPORTED_VIDEO_SUFFIX + SUPPORTED_IMG_SUFFIX:
            if f"{target_name}{suffix}" in files:
                return os.path.join(root, f"{target_name}{suffix}")
    return None  # Not found

# ===================== MOVE - processing =====================
def extract_video_frames(video_path, target_dir):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f" Cannot open video {video_path}")
            return []
        frame_paths = []
        frame_idx, saved_idx = 0, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % FRAME_SKIP == 0:
                frame_name = FRAME_SAVE_FMT.format(saved_idx)
                frame_path = os.path.join(target_dir, frame_name)
                cv2.imwrite(frame_path, frame, FRAME_QUALITY)
                frame_paths.append(frame_path)
                saved_idx += 1
            frame_idx += 1
        cap.release()
        print(f" Videoframes{video_path} -> {target_dir}{saved_idx}frames")
        return frame_paths
    except Exception as e:
        print(f" Videoframesfailed {video_path}{e}")
        return []

def copy_rgb_files(raw_rgb_dir, target_rgb_dir):
    copied_paths = []
    for file_name in os.listdir(raw_rgb_dir):
        if Path(file_name).suffix.lower() in SUPPORTED_IMG_SUFFIX:
            raw_path = os.path.join(raw_rgb_dir, file_name)
            target_path = os.path.join(target_rgb_dir, file_name)
            try:
                shutil.copy2(raw_path, target_path)
                copied_paths.append(target_path)
            except Exception as e:
                print(f" RGBfailed {raw_path}{e}")
    print(f" RGB{raw_rgb_dir} -> {target_rgb_dir}{len(copied_paths)} files")
    return copied_paths

# ===================== PATH_ONLY - =====================
def generate_rgb_path_txt(raw_rgb_path, target_rgb_dir):
    path_txt_path = os.path.join(target_rgb_dir, "path.txt")
    paths = []
    if os.path.isfile(raw_rgb_path) and Path(raw_rgb_path).suffix.lower() in SUPPORTED_VIDEO_SUFFIX:
        cap = cv2.VideoCapture(raw_rgb_path)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            for frame_idx in range(0, total_frames, FRAME_SKIP):
                paths.append(f"{raw_rgb_path},{frame_idx}")
            paths.append(f" Video{raw_rgb_path}")
    elif os.path.isdir(raw_rgb_path):
        img_files = sorted([f for f in os.listdir(raw_rgb_path) 
                           if Path(f).suffix.lower() in SUPPORTED_IMG_SUFFIX])
        for file_name in img_files:
            abs_path = os.path.abspath(os.path.join(raw_rgb_path, file_name))
            paths.append(abs_path)
        paths.append(f" RGB{raw_rgb_path}")
    with open(path_txt_path, "w", encoding="utf-8") as f:
        if paths:
            f.write("\n".join(paths))
            f.write("\n")
    print(f" RGB{path_txt_path}{len(paths)}")
    return paths

def generate_depth_path_txt(target_depth_dir):
    path_txt_path = os.path.join(target_depth_dir, "path.txt")
    with open(path_txt_path, "w", encoding="utf-8") as f:
        f.write("")
    print(f" Depth{path_txt_path}")

# ===================== =====================
def write_calibration_file(file_path, fx, fy, cx, cy):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("# Intrinsic Matrix (3x3)\n")
        f.write(f"{fx:.6f} 0.000000 {cx:.6f} 0.000000 {fy:.6f} {cy:.6f} 0.000000 0.000000 1.000000\n")
        f.write("# Image Size (length=x-axis, width=y-axis)\n")
        f.write(f"{IMG_WIDTH} {IMG_HEIGHT}\n")
        f.write("# Standardized Intrinsics (normalized by image size)\n")
        f.write(f"{fx/IMG_WIDTH:.6f} {fy/IMG_HEIGHT:.6f} {cx/IMG_WIDTH:.6f} {cy/IMG_HEIGHT:.6f}\n")

def write_empty_file(file_path, reason):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"MODE={MODE}\n")
        f.write(f"RAW_BASE={RAW_BASE}\n")
        f.write(f"TARGET_ROOT={TARGET_ROOT}\n")

# ===================== processing =====================
def copy_pose_file(raw_name, target_pose_dir, pose_type, group_info):
    group_dir = group_info["poses_dir"]
    group_name = group_info["name"]
    if pose_type not in POSE_FILE_MAP:
        print(f" {pose_type}，6/7/16")
        return False
    raw_file_name, target_file_name = POSE_FILE_MAP[pose_type]
    #raw_pose_path = find_nested_path(group_dir, raw_file_name)
    raw_pose_path =  os.path.join(group_dir, raw_name, raw_file_name)
    #if group_name == "group_0001":
        #raw_pose_path =  os.path.join(group_dir, raw_name, raw_file_name)
    #else:
        #raw_pose_path =  os.path.join(group_dir, raw_name, "SpatialVID", "annotations", group_name, raw_file_name)
    if not raw_pose_path:
        print(f"  does not exist，skipping{raw_file_name}{group_dir}")
        print(f"shit:{raw_name, target_pose_dir, group_dir}")
        return False
    target_pose_path = os.path.join(target_pose_dir, target_file_name)
    print(1)
    try:
        shutil.copy2(raw_pose_path, target_pose_path)
        print(f" {raw_pose_path} -> {target_pose_path}")
        return True
    except Exception as e:
        print(f" failed {raw_pose_path}{e}")
        print(f" failed {group_dir, raw_name}{e}")
        return False

# ===================== processing =====================
def process_single_data(raw_name, target_subdir, pose_type, group_info):
    """
    :param group_info: ，groupintrinsics/rgb/poses
    """
    target_rgb_dir = os.path.join(TARGET_ROOT, "rgb", VIEW_TYPE, target_subdir)
    target_depth_dir = os.path.join(TARGET_ROOT, "depth", VIEW_TYPE, target_subdir)
    target_pose_dir = os.path.join(TARGET_ROOT, "pose", VIEW_TYPE, target_subdir)
    group_name = group_info["name"]
    ensure_dir(target_rgb_dir)
    ensure_dir(target_depth_dir)
    ensure_dir(target_pose_dir)

    # 1. processingRGB
    base_rgb_path = os.path.join(group_info["rgb_dir"], raw_name)
    raw_rgb_path = None
    
    for suffix in SUPPORTED_VIDEO_SUFFIX:
        video_path = f"{base_rgb_path}{suffix}"
        if os.path.exists(video_path):
            raw_rgb_path = video_path
            break
    
    # Found video file，
    if not raw_rgb_path and os.path.isdir(base_rgb_path):
        raw_rgb_path = base_rgb_path

    if not raw_rgb_path:
        print(f"  Not foundRGB{raw_name}{group_info['rgb_dir']}")
        if MODE == "MOVE":
            if os.path.isfile(raw_rgb_path) and Path(raw_rgb_path).suffix.lower() in SUPPORTED_VIDEO_SUFFIX:
                extract_video_frames(raw_rgb_path, target_rgb_dir)
            elif os.path.isdir(raw_rgb_path):
                copy_rgb_files(raw_rgb_path, target_rgb_dir)
        elif MODE == "PATH_ONLY":
            generate_rgb_path_txt(raw_rgb_path, target_rgb_dir)
            generate_depth_path_txt(target_depth_dir)

    # 2. processing
    #raw_intrinsics_path = find_nested_path(os.path.join(group_info["intrinsics_dir"], raw_name), "intrinsics.npy")
    #print(group_info)
    raw_intrinsics_path =  os.path.join(group_info["intrinsics_dir"], raw_name, "intrinsics.npy")
    #if group_name == "group_0001":
        #raw_intrinsics_path =  os.path.join(group_info["intrinsics_dir"], raw_name, "intrinsics.npy")
    #SpatialVID\annotations\group_0002
    #else:
        #raw_intrinsics_path =  os.path.join(group_info["intrinsics_dir"], raw_name, "SpatialVID", "annotations", group_name, "intrinsics.npy")
    target_calibration_path = os.path.join(target_pose_dir, "calibration.txt")
    if raw_intrinsics_path and os.path.exists(raw_intrinsics_path):
        intrinsics = load_intrinsics(raw_intrinsics_path)
        if intrinsics:
            write_calibration_file(target_calibration_path, *intrinsics)
            print(f" {target_calibration_path}")
            write_empty_file(target_calibration_path, f"failed{raw_intrinsics_path}")
        write_empty_file(target_calibration_path, f"does not existintrinsics.npy{raw_name}")
        print(f"  does not exist{group_info}")
        print(f"shit", os.path.join(group_info["intrinsics_dir"]))
    # 3. processing
    copy_pose_file(raw_name, target_pose_dir, pose_type, group_info)

# ===================== group=====================
def main():
    # 【12pose_type + group_name】
    if len(sys.argv) != 3:
        print(f" Errorprocessinggroup")
        print(f"python process.py 6 group_0002")
        sys.exit(1)
    
    try:
        pose_type = int(sys.argv[1])
        if pose_type not in [6,7,16]:
            raise ValueError
        # groupgroup_0002
        target_group = sys.argv[2]
        if not target_group.startswith("group_"):
            print(f" group{target_group}，group_00xx")
            sys.exit(1)
    except ValueError:
        print(f" Args:{sys.argv[1]}，6/7/16")
        sys.exit(1)
    
    if MODE not in ["MOVE", "PATH_ONLY"]:
        print(f" {MODE}， MOVE / PATH_ONLY")
        return

    # group，processingtarget_group
    annotations_dir = os.path.join(RAW_BASE, "annotations", "SpatialVID", "annotations")
    video_dir = os.path.join(RAW_BASE, "videos", "SpatialVID", "videos")
    
    group_dirs = [target_group] if os.path.exists(os.path.join(annotations_dir, target_group)) else []
    if not group_dirs:
        print(f" Not foundgroup{target_group}{annotations_dir}")
        return

    total_items = 0
    all_raw_items = []
    group_info_list = []
    # group，processingraw_name
    for group_name in group_dirs:
        if group_name == "group_0001":
            group_intrinsics_dir = os.path.join(annotations_dir, group_name)
            group_poses_dir = os.path.join(annotations_dir, group_name)
            group_intrinsics_dir = os.path.join(annotations_dir, group_name, "SpatialVID", "annotations", group_name)
            group_poses_dir = os.path.join(annotations_dir, group_name, "SpatialVID", "annotations", group_name)
        if group_name <= "group_0011":
            group_rgb_dir = os.path.join(video_dir, group_name)
            group_rgb_dir = os.path.join(video_dir, group_name, "SpatialVID", "videos", group_name)
        #print(group_intrinsics_dir)
        #print(group_rgb_dir)
        #print(group_poses_dir)
        if not all([group_intrinsics_dir, group_rgb_dir, group_poses_dir]):
            print(f" skippinggroup{group_name}")
            continue
        
        # groupraw_name
        raw_items = [d for d in os.listdir(group_intrinsics_dir) if os.path.isdir(os.path.join(group_intrinsics_dir, d))]
        if not raw_items:
            print(f" group {group_name} ")
            continue
        
        # group
        group_info = {
            "name": group_name,
            "intrinsics_dir": group_intrinsics_dir,
            "rgb_dir": group_rgb_dir,
            "poses_dir": group_poses_dir
        }
        group_info_list.append((group_info, raw_items))
        total_items += len(raw_items)

    if total_items == 0:
        print(f" processinggroup{target_group}")
        return

    # processingprocessinggroup
    print(f" Processing {total_items} processinggroup{target_group}...")
    print("-" * 60)
    success_count = 0
    fail_count = 0

    if tqdm:
        pbar = tqdm(total=total_items, desc=f"processing{target_group}", unit="", file=sys.stdout)
        pbar = None

    for group_info, raw_items in group_info_list:
        DATASET_NAME = f"SpatialVID_hq_{group_info['name']}"
        for raw_name in raw_items:
            target_subdir = f"{DATASET_NAME}_{raw_name}"
            try:
                process_single_data(raw_name, target_subdir, pose_type, group_info)
                success_count += 1
            except Exception as e:
                fail_count += 1
                print(f"\n processingfailed {raw_name}{e}")
            if pbar:
                pbar.update(1)
                pbar.set_postfix({
                    "": success_count,
                    "failed": fail_count,
                    "group": group_info["name"]
                })

    if pbar:
        pbar.close()

    mode_file = os.path.join(TARGET_ROOT, "mode_info.txt")
    with open(mode_file, "w", encoding="utf-8") as f:
        f.write(f"MODE={MODE}\n")
        f.write(f"RAW_BASE={RAW_BASE}\n")
        f.write(f"TARGET_ROOT={TARGET_ROOT}\n")

    print("\n" + "-" * 60)
    print(" processing")
    print(f" processinggroup{target_group}")
    print(f" processing{total_items}")
    print(f" processing{success_count}")
    print(f" failedprocessing：{fail_count}")
    print(f" {TARGET_ROOT}")
    print(f" {mode_file}")
    print("\n processing！")

if __name__ == "__main__":
    main()

#/path/to/SpatialVID/annotations/group_0002
#/path/to/SpatialVID/annotations/group_0002
