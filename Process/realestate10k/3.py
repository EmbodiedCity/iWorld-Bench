import os
import numpy as np
import cv2
import pandas as pd
from scipy.signal import savgol_filter
import logging
from multiprocessing import Pool, cpu_count
import platform

# ====================== ======================
RAW_VIDEO_DIR = "./videos"
RAW_POSE_DIR = "./cameras"
# 81frames
OUTPUT_VIDEO_DIR = "./videos"
OUTPUT_POSE_DIR = "./cameras"

UP_AXIS = 'y'
MIN_SEQ_LENGTH = 40
FRAME_YAW_THRESHOLD = 0.005
TOTAL_YAW_THRESHOLD = 0.8
VALID_FRAME_RATIO = 0.9
SMOOTH_WINDOW = 5

# Video
VIDEO_CODEC = cv2.VideoWriter_fourcc(*'mp4v')
VIDEO_FPS = 30.0
VIDEO_SIZE = None
PROCESS_NUM = cpu_count() - 1

# ====================== ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(process)d - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ====================== ======================
def extract_yaw_from_rot(R, up_axis='y'):
    """Extract yaw angle around vertical axis only (pure left/right rotation, excluding pitch/roll)"""
    try:
        if up_axis == 'y':
            # yy=，Yaw=
            cos_theta = R[0, 0]
            sin_theta = R[2, 0]
        else:
            cos_theta = R[0, 0]
            sin_theta = R[1, 0]
        yaw = np.arctan2(sin_theta, cos_theta)
        return yaw
    except Exception as e:
        logger.error(f"Yawfailed{e}")
        return 0.0

def read_pose_yaws(pose_file):
    """Pose file，Yawcolumns"""
    yaws = []
    poses_raw = []
    try:
        with open(pose_file, 'r', encoding='utf-8') as f:
            for line in f:
                line_strip = line.strip()
                if not line_strip:
                    continue
                poses_raw.append(line_strip)
                nums = list(map(float, line_strip.split()))
                if len(nums) < 12:
                    continue
                
                R_flat = nums[-12:-3]
                R = np.array(R_flat).reshape(3, 3)
                U, _, Vt = np.linalg.svd(R)
                R_ortho = U @ Vt
                if np.linalg.det(R_ortho) < 0:
                    Vt[-1, :] *= -1
                    R_ortho = U @ Vt
                
                yaw = extract_yaw_from_rot(R_ortho, UP_AXIS)
                yaws.append(yaw)
    except Exception as e:
        logger.error(f"Pose filefailed{e}")
    return poses_raw, np.array(yaws)

def correct_yaw_jump(yaws):
    """Correct yaw angle jumps across pi/-pi boundary"""
    corrected = [yaws[0]] if len(yaws) > 0 else []
    for i in range(1, len(yaws)):
        delta = yaws[i] - corrected[i-1]
        if delta > np.pi:
            corrected.append(yaws[i] - 2 * np.pi)
        elif delta < -np.pi:
            corrected.append(yaws[i] + 2 * np.pi)
        else:
            corrected.append(yaws[i])
    return np.array(corrected)

def find_pure_left_right_sequences(yaws_corrected):
    """
    columns/
    [(frames, frames, direction), ...]
    """
    sequences = []
    n_frames = len(yaws_corrected)
    if n_frames < MIN_SEQ_LENGTH:
        return sequences
    
    # framesYaw
    delta_yaws = np.diff(yaws_corrected)
    delta_yaws = np.where(delta_yaws > np.pi, delta_yaws - 2*np.pi, delta_yaws)
    delta_yaws = np.where(delta_yaws < -np.pi, delta_yaws + 2*np.pi, delta_yaws)
    
    # framesdirection
    directions = np.zeros_like(delta_yaws)
    directions[delta_yaws > FRAME_YAW_THRESHOLD] = 1
    directions[delta_yaws < -FRAME_YAW_THRESHOLD] = -1
    
    # columns
    start_idx = 0
    current_dir = 0
    for i in range(len(directions)):
        if directions[i] != current_dir:
            if current_dir != 0 and (i - start_idx) >= MIN_SEQ_LENGTH:
                total_yaw = abs(yaws_corrected[i-1] - yaws_corrected[start_idx])
                if total_yaw > np.pi:
                    total_yaw = 2*np.pi - total_yaw
                # 2. frames
                valid_frames = np.sum(directions[start_idx:i] == current_dir)
                valid_ratio = valid_frames / (i - start_idx)
                if total_yaw >= TOTAL_YAW_THRESHOLD and valid_ratio >= VALID_FRAME_RATIO:
                    dir_name = 'right' if current_dir == 1 else 'left'
                    logger.info(f"Found pure {dir_name}columnsframes[{start_idx},{i-1}]，total rotation={total_yaw:.3f}rad{total_yaw*180/np.pi:.1f}°")
                    sequences.append((start_idx, i-1, dir_name))
            start_idx = i
            current_dir = directions[i]
    
    # columns
    if current_dir != 0 and (n_frames - start_idx) >= MIN_SEQ_LENGTH:
        total_yaw = abs(yaws_corrected[-1] - yaws_corrected[start_idx])
        if total_yaw > np.pi:
            total_yaw = 2*np.pi - total_yaw
        valid_frames = np.sum(directions[start_idx:] == current_dir)
        valid_ratio = valid_frames / (n_frames - start_idx)
        if total_yaw >= TOTAL_YAW_THRESHOLD and valid_ratio >= VALID_FRAME_RATIO:
            dir_name = 'right' if current_dir == 1 else 'left'
            logger.info(f"Found pure {dir_name}columnsframes[{start_idx},{n_frames-1}]，total rotation={total_yaw:.3f}rad{total_yaw*180/np.pi:.1f}°")
            sequences.append((start_idx, n_frames-1, dir_name))
    
    return sequences

def extract_81frames_video(video_path, start, end, output_path):
    """81framesVideo40frames+1frames+40frames"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video：{video_path}")
        return False
    
    global VIDEO_SIZE, VIDEO_FPS
    VIDEO_SIZE = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    VIDEO_FPS = cap.get(cv2.CAP_PROP_FPS)
    
    # 81frames
    seq_start = start
    seq_end = start + 39
    if seq_end > end:
        seq_start = end - 39
        seq_end = end
    
    # Video
    out = cv2.VideoWriter(output_path, VIDEO_CODEC, VIDEO_FPS, VIDEO_SIZE)
    if not out.isOpened():
        logger.error(f"Cannot create video：{output_path}")
        cap.release()
        return False
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, seq_start)
    frames = []
    for _ in range(40):
        ret, frame = cap.read()
        if not ret:
            logger.warning("Insufficient video frames, padding with last frame")
            frame = frame if ret else np.zeros(VIDEO_SIZE + (3,), dtype=np.uint8)
        frames.append(frame)
    
    # frames
    frames.append(frames[-1])
    # frames
    frames.extend(frames[-2::-1])
    
    for frame in frames:
        out.write(frame)
    
    cap.release()
    out.release()
    logger.info(f"81-frame video generated：{output_path}")
    return True

def save_81frames_pose(poses_raw, start, end, output_path):
    """81frames"""
    seq_start = start
    seq_end = start + 39
    if seq_end > end:
        seq_start = end - 39
        seq_end = end
    
    new_poses = poses_raw[seq_start:seq_end+1] + [poses_raw[seq_end]] + poses_raw[seq_end:seq_start-1:-1]
    with open(output_path, 'w', encoding='utf-8') as f:
        for pose in new_poses:
            f.write(pose + '\n')
    logger.info(f"81-frame pose generated：{output_path}")
    return True

# ====================== processing ======================
def process_single_video(args):
    video_file, video_dir, pose_dir, output_video_dir, output_pose_dir = args
    video_name = os.path.splitext(video_file)[0]
    video_path = os.path.join(video_dir, video_file)
    logger.info(f"\nProcess {os.getpid()} processing：{video_name}")
    
    # Pose file
    pose_file = None
    for ext in ('.txt', '.csv', '.dat'):
        candidate = os.path.join(pose_dir, f"{video_name}{ext}")
        if os.path.exists(candidate):
            pose_file = candidate
            break
    if not pose_file:
        logger.warning("No matching pose file, skipping")
        return
    
    poses_raw, yaws = read_pose_yaws(pose_file)
    if len(yaws) < MIN_SEQ_LENGTH:
        logger.warning("Insufficient pose frames, skipping")
        return
    
    yaws_corrected = correct_yaw_jump(yaws)
    if SMOOTH_WINDOW > 0 and len(yaws_corrected) >= SMOOTH_WINDOW:
        yaws_corrected = savgol_filter(yaws_corrected, SMOOTH_WINDOW, 2)
    
    valid_sequences = find_pure_left_right_sequences(yaws_corrected)
    if not valid_sequences:
        logger.info("columns，skipping")
        return
    
    # 81framescolumns
    for seq_idx, (start, end, dir_name) in enumerate(valid_sequences):
        # 81framesVideo+
        video_out = os.path.join(output_video_dir, f"{video_name}_seq_{seq_idx:03d}.mp4")
        pose_out = os.path.join(output_pose_dir, f"{video_name}_seq_{seq_idx:03d}.txt")
        
        extract_81frames_video(video_path, start, end, video_out)
        save_81frames_pose(poses_raw, start, end, pose_out)

# ====================== ======================
def main():
    # Videocolumns
    video_exts = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in os.listdir(RAW_VIDEO_DIR) if f.lower().endswith(video_exts)]
    if not video_files:
        logger.error("Video directoryVideo")
        return
    
    # Processprocessing
    task_args = [(f, RAW_VIDEO_DIR, RAW_POSE_DIR, OUTPUT_VIDEO_DIR, OUTPUT_POSE_DIR) for f in video_files]
    logger.info(f"Process{len(task_args)} tasks，{PROCESS_NUM}Process")
    
    if platform.system() == "Windows":
        from multiprocessing import set_start_method
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass
    
    with Pool(PROCESS_NUM) as pool:
        pool.map(process_single_video, task_args)
    
    logger.info("81framescolumns！")

if __name__ == "__main__":
    main()
