import os
import sys
import shutil
import logging
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
import subprocess
import re

DATASET_NAME = "SpatialVID"
SUPPORTED_VIDEO_SUFFIX = [".mp4", ".avi", ".mov", ".mkv"]
FRAME_COUNT_PER_SEG = 81
COMPRESSED_RESOLUTION = (832, 480)
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
PASSED_VIDEOS_LOG = os.path.join(LOG_DIR, "passed_videos.txt")
PASSED_CAMERAS_LOG = os.path.join(LOG_DIR, "passed_cameras.txt")
UPLOADED_VIDEOS_LOG = os.path.join(LOG_DIR, "uploaded_videos.txt")
UPLOADED_CAMERAS_LOG = os.path.join(LOG_DIR, "uploaded_cameras.txt")

SERVER_CONFIG = {
    "host": "example.com",
    "user": os.environ.get("SSH_USER", "user"),
    "port": 50371,
    "remote_dir": "/path/to/local/data",
    "ssh_key_path": r"/path/to/local/data",
}

def init_log_file(log_path):
    """does not exist"""
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("")

def read_log(log_path):
    """，processing/"""
    if not os.path.exists(log_path):
        return set()
    with open(log_path, "r", encoding="utf-8") as f:
        return set([line.strip() for line in f if line.strip()])

def write_log(log_path, content):
    """"""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{content}\n")

def create_remote_dir(remote_dir):
    """passedssh，"""
    cmd = [
        "ssh",
        "-p", str(SERVER_CONFIG['port']),
        "-i", SERVER_CONFIG['ssh_key_path'],
        "-o", "StrictHostKeyChecking=no",
        "-o", "LogLevel=ERROR",
        f"{SERVER_CONFIG['user']}@{SERVER_CONFIG['host']}",
        f"mkdir -p {remote_dir}/videos {remote_dir}/cameras"
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', errors='ignore')
        logging.info(f"{remote_dir}/videos  {remote_dir}/cameras")
    except subprocess.CalledProcessError as e:
        logging.error(f"failed{e.stderr}，")
        raise

def get_paired_data(cityworld_root):
    """ CityWorld ，Video"""
    paired_data = []
    rgb_root = os.path.join(cityworld_root, "rgb")
    pose_root = os.path.join(cityworld_root, "pose")

    for view_dir in os.listdir(rgb_root):
        rgb_view_path = os.path.join(rgb_root, view_dir)
        pose_view_path = os.path.join(pose_root, view_dir)
        if not os.path.isdir(rgb_view_path) or not os.path.isdir(pose_view_path):
            continue

        for data_dir in os.listdir(rgb_view_path):
            rgb_data_path = os.path.join(rgb_view_path, data_dir)
            pose_data_path = os.path.join(pose_view_path, data_dir)
            if not os.path.isdir(rgb_data_path) or not os.path.isdir(pose_data_path):
                continue

            path_txt = os.path.join(rgb_data_path, "path.txt")
            calib_txt = os.path.join(pose_data_path, "calibration.txt")
            if not os.path.exists(path_txt) or not os.path.exists(calib_txt):
                continue

            extrinsic_files = {
                "extrinsics_matrix": os.path.join(pose_data_path, "extrinsics_matrix.txt"),
                "seven_element": os.path.join(pose_data_path, "seven_element.txt"),
                "six_DoF": os.path.join(pose_data_path, "six_DoF.txt")
            }
            extrinsic_path, extrinsic_type = None, None
            for ext_type, ext_path in extrinsic_files.items():
                if os.path.exists(ext_path):
                    extrinsic_path = ext_path
                    extrinsic_type = ext_type
                    break
            if not extrinsic_path:
                continue

            data_prefix = data_dir.replace("SpatialVID_hq_", "")
            paired_data.append((path_txt, calib_txt, extrinsic_path, extrinsic_type, data_prefix))

    return paired_data

def parse_image_resolution_from_calibration(calib_path):
    """
    calibration.txt
    1. # Image Sizelines 2. lines 3. (4096,2160)
    """
    try:
        with open(calib_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # 1Image Sizelines，
        size_markers = ["# Image Size", "# Resolution", "# Size"]
        for line in lines:
            for marker in size_markers:
                if marker in line:
                    clean_line = re.sub(r'[^0-9\s]', '', line.split(marker)[1].strip())
                    nums = clean_line.split()
                    if len(nums) >= 2:
                        try:
                            width, height = int(nums[0]), int(nums[1])
                            logging.info(f"{calib_path}: ({width}, {height})")
                            return (width, height)
                        except:
                            continue
        
        for line in lines:
            clean_line = re.sub(r'[^0-9\s]', '', line)
            nums = clean_line.split()
            if len(nums) == 2 and all(num.isdigit() for num in nums):
                width, height = map(int, nums)
                logging.info(f"{calib_path}lines: ({width}, {height})")
                return (width, height)
        
        logging.warning(f"{calib_path}，(4096, 2160)")
        return (4096, 2160)
    except Exception as e:
        logging.error(f"calibration.txtfailed: {e}，(4096, 2160)")
        return (4096, 2160)

def parse_intrinsics(calib_path):
    """， [fx, fy, cx, cy]"""
    with open(calib_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if not line.startswith("#") and line.strip()]
    
    intrinsics = []
    for line in lines:
        nums = list(map(float, line.split()))
        if len(nums) == 4:
            intrinsics = nums[:4]
            break
        elif len(nums) == 9:
            intrinsics = [nums[0], nums[4], nums[2], nums[5]]
            break
    if not intrinsics:
        raise ValueError("Not found")
    
    img_size = parse_image_resolution_from_calibration(calib_path)
    
    fx_norm = intrinsics[0] / img_size[0]
    fy_norm = intrinsics[1] / img_size[1]
    cx_norm = intrinsics[2] / img_size[0]
    cy_norm = intrinsics[3] / img_size[1]
    
    return [fx_norm, fy_norm, cx_norm, cy_norm]

def seven_element_to_extrinsic(seven_elements):
    """3x4 [rx, ry, rz, tx, ty, tz, s]"""
    rx, ry, rz, tx, ty, tz, s = seven_elements
    rot = R.from_euler('xyz', [rx, ry, rz]).as_matrix()
    trans = np.array([tx, ty, tz]).reshape(3, 1)
    extrinsic = np.hstack([rot * s, trans])
    return extrinsic

def six_dof_to_extrinsic(six_dof):
    """3x4 [rx, ry, rz, tx, ty, tz]"""
    rx, ry, rz, tx, ty, tz = six_dof
    rot = R.from_euler('xyz', [rx, ry, rz]).as_matrix()
    trans = np.array([tx, ty, tz]).reshape(3, 1)
    return np.hstack([rot, trans])

def parse_extrinsics(extrinsic_path, extrinsic_type):
    """，3x4columnsframes"""
    extrinsic_matrices = []
    with open(extrinsic_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if not line.startswith("#") and line.strip()]
    
    # lines
    if len(lines) < FRAME_COUNT_PER_SEG:
        logging.warning(f"lines{len(lines)}: {extrinsic_path}")
        return extrinsic_matrices
    
    for line in lines:
        nums = list(map(float, line.split()))
        if extrinsic_type == "extrinsics_matrix":
            mat = np.array(nums[:12]).reshape(3, 4)
        elif extrinsic_type == "seven_element":
            mat = seven_element_to_extrinsic(nums[:7])
        elif extrinsic_type == "six_DoF":
            mat = six_dof_to_extrinsic(nums[:6])
        else:
            raise ValueError(f": {extrinsic_type}")
        extrinsic_matrices.append(mat)
    
    return extrinsic_matrices

def process_video(raw_video_path, video_dir, data_prefix, passed_videos, total_ext_frames, calib_path):
    """
    processingVideo
    1. calibration.txtTARGET_RESOLUTION
    2. /，832*480
    """
    video_key = f"{data_prefix}_{os.path.basename(raw_video_path)}"
    if video_key in passed_videos:
        logging.info(f"Videoprocessing，skipping: {video_key}")
        return []
    
    TARGET_RESOLUTION = parse_image_resolution_from_calibration(calib_path)
    
    cap = cv2.VideoCapture(raw_video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video: {raw_video_path}")
        return []
    
    # Validating videoactualframes
    video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    valid_total_frames = min(total_ext_frames, video_total_frames)
    seg_count = valid_total_frames // FRAME_COUNT_PER_SEG
    
    if seg_count == 0:
        logging.warning(f"frames81{total_ext_frames}，Video{video_total_frames}，skippingVideo: {data_prefix}")
        cap.release()
        return []
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_segs = []

    for seg_idx in range(seg_count):
        seg_name = f"{data_prefix}_video_{seg_idx + 1:03d}.mp4"
        seg_path = os.path.join(video_dir, seg_name)
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(seg_path, fourcc, fps, COMPRESSED_RESOLUTION)
        if not out.isOpened():
            logging.error(f"Cannot create video: {seg_path}")
            continue
        
        # 81frames
        start_frame = seg_idx * FRAME_COUNT_PER_SEG
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_written = 0
        
        for _ in range(FRAME_COUNT_PER_SEG):
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Videoframes， {seg_idx+1} {frame_written}frames，skipping")
                out.release()
                if os.path.exists(seg_path):
                    os.remove(seg_path)
                break
            
            frame_original = cv2.resize(frame, TARGET_RESOLUTION, interpolation=cv2.INTER_LINEAR)
            frame_compressed = cv2.resize(frame_original, COMPRESSED_RESOLUTION, interpolation=cv2.INTER_AREA)
            out.write(frame_compressed)
            frame_written += 1
        
        out.release()
        
        if frame_written == FRAME_COUNT_PER_SEG and os.path.getsize(seg_path) > 0:
            output_segs.append(seg_path)
            logging.info(f"Video: {seg_path} ({TARGET_RESOLUTION}{COMPRESSED_RESOLUTION})")
        else:
            if os.path.exists(seg_path):
                os.remove(seg_path)
            logging.warning(f" {seg_path} ，")
    
    cap.release()
    write_log(PASSED_VIDEOS_LOG, video_key)
    return output_segs

def process_camera(intrinsics, extrinsics, output_dir, prefix, passed_cameras, seg_count):
    """processingArgs:，81，"""
    camera_key = f"{prefix}"
    if camera_key in passed_cameras:
        logging.info(f"processing，skipping: {camera_key}")
        return []
    
    output_segs = []
    for seg_idx in range(seg_count):
        seg_name = f"{prefix}_camera_{seg_idx + 1:03d}.txt"
        seg_path = os.path.join(output_dir, seg_name)
        
        with open(seg_path, "w", encoding="utf-8") as f:
            start_line = seg_idx * FRAME_COUNT_PER_SEG
            for line_idx in range(start_line, start_line + FRAME_COUNT_PER_SEG):
                ext_mat = extrinsics[line_idx]
                line_data = [
                    0.0,
                    intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3],
                    0.0, 0.0,
                    ext_mat[0,0], ext_mat[0,1], ext_mat[0,2],
                    ext_mat[1,0], ext_mat[1,1], ext_mat[1,2],
                    ext_mat[2,0], ext_mat[2,1], ext_mat[2,2],
                    ext_mat[0,3], ext_mat[1,3], ext_mat[2,3]
                ]
                f.write(" ".join([f"{x:.9f}" for x in line_data]) + "\n")
        
        if os.path.getsize(seg_path) > 0:
            output_segs.append(seg_path)
            logging.info(f": {seg_path}")
        else:
            os.remove(seg_path)
    
    write_log(PASSED_CAMERAS_LOG, camera_key)
    return output_segs

def upload_file(local_path, remote_dir):
    if not os.path.exists(local_path):
        logging.error(f"does not exist: {local_path}")
        return False
    
    remote_path = f"{SERVER_CONFIG['user']}@{SERVER_CONFIG['host']}:{os.path.join(remote_dir, os.path.basename(local_path))}"
    cmd = [
        "scp",
        "-P", str(SERVER_CONFIG['port']),
        "-i", SERVER_CONFIG['ssh_key_path'],
        "-o", "StrictHostKeyChecking=no",
        "-o", "LogLevel=ERROR",
        local_path,
        remote_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            errors='ignore'
        )
        logging.info(f": {local_path} -> {remote_path}")
        return True
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr if e.stderr else str(e)
        logging.error(f"failed: {local_path}, Error: {err_msg}")
        return False
    except Exception as e:
        logging.error(f": {local_path}, Error: {str(e)}")
        return False

def parallel_process(video_segs, camera_segs, video_dir, camera_dir):
    """lines"""
    def process_single(file_path, is_video):
        remote_subdir = "videos" if is_video else "cameras"
        remote_full_dir = os.path.join(SERVER_CONFIG['remote_dir'], remote_subdir)
        if upload_file(file_path, remote_full_dir):
            log_path = UPLOADED_VIDEOS_LOG if is_video else UPLOADED_CAMERAS_LOG
            write_log(log_path, os.path.basename(file_path))
            os.remove(file_path)
            logging.info(f": {file_path}")
        else:
            logging.error(f"failed，: {file_path}")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        for video_seg in video_segs:
            executor.submit(process_single, video_seg, True)
        for camera_seg in camera_segs:
            executor.submit(process_single, camera_seg, False)

def main(cityworld_root, output_root):
    init_log_file(PASSED_VIDEOS_LOG)
    init_log_file(PASSED_CAMERAS_LOG)
    init_log_file(UPLOADED_VIDEOS_LOG)
    init_log_file(UPLOADED_CAMERAS_LOG)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(LOG_DIR, "reflect.log"))]
    )

    # processing/
    passed_videos = read_log(PASSED_VIDEOS_LOG)
    passed_cameras = read_log(PASSED_CAMERAS_LOG)

    # Output directory
    remake_dir = os.path.join(output_root, f"{DATASET_NAME}_remake")
    create_remote_dir(SERVER_CONFIG['remote_dir'])
    video_dir = os.path.join(remake_dir, "videos")
    camera_dir = os.path.join(remake_dir, "cameras")
    os.makedirs(remake_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(camera_dir, exist_ok=True)

    create_remote_dir(SERVER_CONFIG['remote_dir'])

    logging.info("...")
    leftover_videos = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]
    leftover_cameras = [os.path.join(camera_dir, f) for f in os.listdir(camera_dir) if os.path.isfile(os.path.join(camera_dir, f))]
    
    if leftover_videos or leftover_cameras:
        logging.info(f"FoundVideo {len(leftover_videos)} ， {len(leftover_cameras)} 。...")
        parallel_process(leftover_videos, leftover_cameras, video_dir, camera_dir)
        logging.info("processing。")
    else:
        logging.info("。")

    process_dst = os.path.join(remake_dir, "process.py")
    shutil.copy2(os.path.abspath(__file__), process_dst)
    logging.info(f"processing: {process_dst}")

    paired_data = get_paired_data(cityworld_root)
    logging.info(f" {len(paired_data)} ")

    # processing
    for path_txt, calib_txt, ext_txt, ext_type, data_prefix in paired_data:
        try:
            # Video， process_video video_key
            # path.txtVideokey
            with open(path_txt, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if not line.startswith("#") and line.strip()]
            if not lines:
                logging.warning(f"path.txt: {path_txt}")
                continue

            # Videovideo_key
            raw_video_basename = None
            if "," in lines[0]:
                raw_video_basename = os.path.basename(lines[0].split(",")[0])
            else:
                img_dir = os.path.dirname(lines[0])
                for suffix in SUPPORTED_VIDEO_SUFFIX:
                    video_candidate = os.path.join(os.path.dirname(img_dir), f"{os.path.basename(img_dir)}{suffix}")
                    if os.path.exists(video_candidate):
                        raw_video_basename = os.path.basename(video_candidate)
                        break
            
            video_key = f"{data_prefix}_{raw_video_basename}" if raw_video_basename else None
            camera_key = f"{data_prefix}"

            # processing，processingskipping
            has_passed_video = video_key and (video_key in passed_videos)
            has_passed_camera = camera_key in passed_cameras

            if has_passed_video and has_passed_camera:
                logging.info(f" {data_prefix}Video{raw_video_basename}processing，skipping")
                continue
            elif has_passed_video:
                logging.info(f" {data_prefix} Videoprocessing，processing")
            elif has_passed_camera:
                logging.info(f" {data_prefix} processing，processingVideo")

            # Videoprocessing
            raw_video_path = None
            if "," in lines[0]:
                raw_video_path = lines[0].split(",")[0]
            else:
                img_dir = os.path.dirname(lines[0])
                for suffix in SUPPORTED_VIDEO_SUFFIX:
                    video_candidate = os.path.join(os.path.dirname(img_dir), f"{os.path.basename(img_dir)}{suffix}")
                    if os.path.exists(video_candidate):
                        raw_video_path = video_candidate
                        break
            if not raw_video_path or not os.path.exists(raw_video_path):
                logging.warning(f"Not foundVideo: {path_txt}")
                continue

            intrinsic_params = parse_intrinsics(calib_txt)
            extrinsic_matrices = parse_extrinsics(ext_txt, ext_type)
            total_ext_frames = len(extrinsic_matrices)
            seg_count = total_ext_frames // FRAME_COUNT_PER_SEG

            # linesprocessingVideopassedskipping
            with ProcessPoolExecutor(max_workers=2) as executor:
                video_future = executor.submit(
                    process_video, raw_video_path, video_dir, data_prefix, passed_videos, total_ext_frames, calib_txt
                ) if not has_passed_video else executor.submit(lambda: None)
                
                camera_future = executor.submit(
                    process_camera, intrinsic_params, extrinsic_matrices, camera_dir, data_prefix, passed_cameras, seg_count
                ) if not has_passed_camera else executor.submit(lambda: None)

                video_segs = video_future.result()
                camera_segs = camera_future.result()

            # linesprocessing
            if video_segs or camera_segs:
                parallel_process(video_segs, camera_segs, video_dir, camera_dir)

        except Exception as e:
            logging.error(f"processingfailed {data_prefix}: {str(e)}", exc_info=True)
            continue

    logging.info("processing！")
    logging.info(f"Output directory: {remake_dir}")
    logging.info(f": {LOG_DIR}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("python reflect.py <CityWorld> <Output root directory>")
        print("python reflect.py /path/to/CityWorld /path/to/output")
        sys.exit(1)
    
    cityworld_root = sys.argv[1]
    output_root = sys.argv[2]
    
    if not os.path.isdir(cityworld_root):
        print(f"ErrorCityWorlddoes not exist: {cityworld_root}")
        sys.exit(1)
    
    main(cityworld_root, output_root)