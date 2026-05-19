#!/usr/bin/env python3
import os
import sys
import logging
import numpy as np

def _require_deps():
    missing = []
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    try:
        import scipy
    except ImportError:
        missing.append("scipy")
    if missing:
        print(f"pip install {' '.join(missing)}")
        return False
    return True

if not _require_deps():
    sys.exit(1)

import cv2
import numpy as np

DATASET_NAME = "7Scenes_local"
FRAME_COUNT_PER_SEG = 81
COMPRESSED_RESOLUTION = (832, 480)
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
PASSED_VIDEOS_LOG = os.path.join(LOG_DIR, "passed_videos_7scenes.txt")
PASSED_CAMERAS_LOG = os.path.join(LOG_DIR, "passed_cameras_7scenes.txt")
ORIGINAL_RESOLUTION = (640, 480)
SEVEN_SCENES_INTRINSICS = [0.0, 0.9140625, 1.21875, 0.5, 0.5, 0.0, 0.0]


def init_log_file(log_path):
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("")


def read_log(log_path):
    if not os.path.exists(log_path):
        return set()
    with open(log_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def write_log(log_path, content):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{content}\n")


def extract_base_name(folder_name):
    parts = folder_name.split("_")
    if len(parts) >= 4 and "-" in parts[-1] and len(parts[-1]) == 36:
        return "_".join(parts[:-1])
    return folder_name


def find_7scenes_data(cityworld_root):
    paired_data = []
    rgb_base = os.path.join(cityworld_root, "rgb", "Human_front")
    pose_base = os.path.join(cityworld_root, "pose", "Human_front")
    if not os.path.exists(rgb_base) or not os.path.exists(pose_base):
        logging.error("missing rgb or pose Human_front")
        return paired_data
    rgb_folders = [f for f in os.listdir(rgb_base) if f.startswith("7scenes_")]
    logging.info("rgb_folders=%d", len(rgb_folders))
    pose_folder_map = {}
    for folder in os.listdir(pose_base):
        if folder.startswith("7scenes_"):
            pose_folder_map[extract_base_name(folder)] = folder
    matched = 0
    for rgb_folder in rgb_folders:
        rgb_base_name = extract_base_name(rgb_folder)
        if rgb_base_name not in pose_folder_map:
            logging.warning("no pose for %s", rgb_base_name)
            continue
        pose_folder = pose_folder_map[rgb_base_name]
        rgb_folder_path = os.path.join(rgb_base, rgb_folder)
        pose_folder_path = os.path.join(pose_base, pose_folder)
        path_txt = os.path.join(rgb_folder_path, "path.txt")
        extrinsics_txt = os.path.join(pose_folder_path, "extrinsics_matrix.txt")
        if not (os.path.exists(path_txt) and os.path.exists(extrinsics_txt)):
            logging.warning("incomplete %s / %s", rgb_folder, pose_folder)
            continue
        try:
            with open(path_txt, "r", encoding="utf-8") as f:
                image_count = sum(1 for _ in f)
        except OSError:
            image_count = 0
        paired_data.append(
            {
                "rgb_folder": rgb_folder,
                "pose_folder": pose_folder,
                "rgb_base_name": rgb_base_name,
                "path_txt": path_txt,
                "extrinsics_txt": extrinsics_txt,
                "image_count": image_count,
            }
        )
        matched += 1
        logging.info("pair %s <-> %s", rgb_folder, pose_folder)
    logging.info("matched=%d", matched)
    return paired_data


def parse_extrinsics(extrinsic_path):
    matrices = []
    try:
        with open(extrinsic_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                nums = list(map(float, line.split()))
                if len(nums) == 16:
                    m4 = np.array(nums).reshape(4, 4)
                    matrices.append(np.hstack([m4[:3, :3], m4[:3, 3:4]]))
                else:
                    logging.warning("bad extrinsic line len=%d", len(nums))
    except Exception as e:
        logging.error("parse extrinsic %s: %s", extrinsic_path, e)
    logging.info("extrinsics n=%d %s", len(matrices), extrinsic_path)
    return matrices


def convert_windows_path_to_linux(windows_path):
    if windows_path.startswith("/path/to/local/data/"):
        return windows_path.replace("/path/to/local/data/", "/path/to/local/data/").replace("\\", "/")
    if windows_path.startswith("/path/to/local/data/"):
        return windows_path.replace("/path/to/local/data/", "/path/to/local/data/").replace("\\", "/")
    if ":\\" in windows_path:
        d = windows_path[0]
        return windows_path.replace(f"{d}:\\", f"/mnt/{d.lower()}/").replace("\\", "/")
    return windows_path


def create_video_from_images(image_paths, output_video_path, fps=30):
    if not image_paths:
        return False
    p0 = convert_windows_path_to_linux(image_paths[0])
    first = cv2.imread(p0) or cv2.imread(image_paths[0])
    if first is None:
        logging.error("read fail %s", image_paths[0])
        return False
    oh, ow = first.shape[:2]
    out = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, COMPRESSED_RESOLUTION
    )
    if not out.isOpened():
        logging.error("writer fail %s", output_video_path)
        return False
    ok = 0
    for img_path in image_paths:
        pl = convert_windows_path_to_linux(img_path)
        frame = cv2.imread(pl) or cv2.imread(img_path)
        if frame is None:
            frame = np.zeros((oh, ow, 3), dtype=np.uint8)
        if frame.shape[:2] != (oh, ow):
            frame = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_LINEAR)
        fr = cv2.resize(frame, ORIGINAL_RESOLUTION, interpolation=cv2.INTER_LINEAR)
        fc = cv2.resize(fr, COMPRESSED_RESOLUTION, interpolation=cv2.INTER_AREA)
        out.write(fc)
        ok += 1
    out.release()
    if ok == len(image_paths):
        logging.info("video ok %s frames=%d", output_video_path, ok)
        return True
    logging.warning("video incomplete %s", output_video_path)
    if os.path.exists(output_video_path):
        os.remove(output_video_path)
    return False


def write_camera_file(extrinsics, output_path, segment_idx):
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            start_idx = segment_idx * FRAME_COUNT_PER_SEG
            end_idx = start_idx + FRAME_COUNT_PER_SEG
            for i in range(start_idx, end_idx):
                ext_mat = extrinsics[i] if i < len(extrinsics) else (
                    extrinsics[-1] if extrinsics else np.hstack([np.eye(3), np.zeros((3, 1))])
                )
                line_data = [
                    0.0,
                    SEVEN_SCENES_INTRINSICS[1],
                    SEVEN_SCENES_INTRINSICS[2],
                    SEVEN_SCENES_INTRINSICS[3],
                    SEVEN_SCENES_INTRINSICS[4],
                    0.0,
                    0.0,
                    ext_mat[0, 0],
                    ext_mat[0, 1],
                    ext_mat[0, 2],
                    ext_mat[1, 0],
                    ext_mat[1, 1],
                    ext_mat[1, 2],
                    ext_mat[2, 0],
                    ext_mat[2, 1],
                    ext_mat[2, 2],
                    ext_mat[0, 3],
                    ext_mat[1, 3],
                    ext_mat[2, 3],
                ]
                f.write(" ".join(f"{x:.9f}" for x in line_data) + "\n")
        logging.info("camera ok %s", output_path)
        return True
    except Exception as e:
        logging.error("camera fail %s: %s", output_path, e)
        return False


def process_7scenes_data(data_info, video_dir, camera_dir, passed_videos, passed_cameras):
    rgb_base_name = data_info["rgb_base_name"]
    path_txt = data_info["path_txt"]
    extrinsics_txt = data_info["extrinsics_txt"]
    logging.info("seq %s", rgb_base_name)
    try:
        with open(path_txt, "r", encoding="utf-8") as f:
            image_paths = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logging.error("path.txt %s: %s", path_txt, e)
        return 0, 0
    extrinsics = parse_extrinsics(extrinsics_txt)
    if not extrinsics:
        return 0, 0
    min_frames = min(len(image_paths), len(extrinsics))
    seg_count = min_frames // FRAME_COUNT_PER_SEG
    if seg_count == 0:
        logging.warning("skip %s frames<81 img=%d ext=%d", rgb_base_name, len(image_paths), len(extrinsics))
        return 0, 0
    v_segs = c_segs = 0
    for seg_idx in range(seg_count):
        video_key = f"{rgb_base_name}_video_{seg_idx:03d}"
        if video_key not in passed_videos:
            s = seg_idx * FRAME_COUNT_PER_SEG
            e = s + FRAME_COUNT_PER_SEG
            video_name = f"{rgb_base_name}_video_{seg_idx + 1:03d}.mp4"
            video_path = os.path.join(video_dir, video_name)
            if create_video_from_images(image_paths[s:e], video_path):
                write_log(PASSED_VIDEOS_LOG, video_key)
                v_segs += 1
        camera_key = f"{rgb_base_name}_camera_{seg_idx:03d}"
        if camera_key not in passed_cameras:
            camera_name = f"{rgb_base_name}_camera_{seg_idx + 1:03d}.txt"
            camera_path = os.path.join(camera_dir, camera_name)
            if write_camera_file(extrinsics, camera_path, seg_idx):
                write_log(PASSED_CAMERAS_LOG, camera_key)
                c_segs += 1
    return v_segs, c_segs


def main(cityworld_root, output_root):
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(LOG_DIR, "process_7scenes.log")),
        ],
    )
    init_log_file(PASSED_VIDEOS_LOG)
    init_log_file(PASSED_CAMERAS_LOG)
    passed_videos = read_log(PASSED_VIDEOS_LOG)
    passed_cameras = read_log(PASSED_CAMERAS_LOG)
    out_base = os.path.join(output_root, DATASET_NAME)
    video_dir = os.path.join(out_base, "videos")
    camera_dir = os.path.join(out_base, "cameras")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(camera_dir, exist_ok=True)
    logging.info("in=%s out=%s", cityworld_root, output_root)
    paired = find_7scenes_data(cityworld_root)
    if not paired:
        logging.error("no pairs; check rgb/pose Human_front 7scenes_*")
        return
    tv = tc = 0
    for data_info in paired:
        a, b = process_7scenes_data(data_info, video_dir, camera_dir, passed_videos, passed_cameras)
        tv += a
        tc += b
    logging.info("done videos=%d cameras=%d -> %s", tv, tc, out_base)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: reflect.py <cityworld_root> <output_root>")
        sys.exit(1)
    cityworld_root, output_root = sys.argv[1], sys.argv[2]
    if not os.path.isdir(cityworld_root):
        print("error: cityworld_root missing")
        sys.exit(1)
    rgb_dir = os.path.join(cityworld_root, "rgb", "Human_front")
    pose_dir = os.path.join(cityworld_root, "pose", "Human_front")
    if not os.path.isdir(rgb_dir) or not os.path.isdir(pose_dir):
        print("error: need rgb/pose/Human_front")
        sys.exit(1)
    print(f"7scenes in={cityworld_root} out={output_root}")
    main(cityworld_root, output_root)
