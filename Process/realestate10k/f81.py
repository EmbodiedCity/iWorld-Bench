import os
import shutil
import numpy as np
import cv2
import logging
from pathlib import Path

# ====================== ======================
# 81framescolumns
INPUT_ROOT_DIR = "./combined_sequences"
INPUT_VIDEO_DIR = os.path.join(INPUT_ROOT_DIR, "videos")
INPUT_POSE_DIR = os.path.join(INPUT_ROOT_DIR, "poses")
OUTPUT_HIGH_ROT_DIR = "./high_rot_81frames"
OUTPUT_VIDEO_DIR = os.path.join(OUTPUT_HIGH_ROT_DIR, "videos")
OUTPUT_POSE_DIR = os.path.join(OUTPUT_HIGH_ROT_DIR, "poses")
MIN_SEGMENT_ROT = 0.5
UP_AXIS = 'y'

# ====================== ，main ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ====================== ======================
def ensure_dir(dir_path):
    """Output directory"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory：{dir_path}")

def read_pose_rotations(pose_path):
    """
    81linesPose file，frames，Yawcolumns
    """
    yaws = []
    try:
        with open(pose_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_strip = line.strip()
                if not line_strip:
                    continue
                nums = list(map(float, line_strip.split()))
                if len(nums) < 12:
                    logger.warning(f"Pose file {pose_path} linescolumns12，skippinglines")
                    continue
                R_flat = nums[-12:-3]
                R = np.array(R_flat).reshape(3, 3)
                
                U, _, Vt = np.linalg.svd(R)
                R_ortho = U @ Vt
                if np.linalg.det(R_ortho) < 0:
                    Vt[-1, :] *= -1
                    R_ortho = U @ Vt
                
                # Yaw
                if UP_AXIS == 'y':
                    cos_theta = R_ortho[0, 0]
                    sin_theta = R_ortho[0, 2]
                else:  # 'z'
                    cos_theta = R_ortho[0, 0]
                    sin_theta = -R_ortho[1, 0]
                yaw = np.arctan2(sin_theta, cos_theta)
                yaws.append(yaw)
        
        # Yawπ/-π
        corrected_yaws = [yaws[0]] if len(yaws) > 0 else []
        for i in range(1, len(yaws)):
            delta = yaws[i] - corrected_yaws[i-1]
            if delta > np.pi:
                corrected_yaws.append(yaws[i] - 2 * np.pi)
            elif delta < -np.pi:
                corrected_yaws.append(yaws[i] + 2 * np.pi)
            else:
                corrected_yaws.append(yaws[i])
        return np.array(corrected_yaws)
    except Exception as e:
        logger.error(f"Pose file {pose_path} failed{str(e)}")
        return None

def calculate_segment_rot(yaws, start_idx, end_idx):
    """total rotation"""
    if len(yaws) == 0 or start_idx < 0 or end_idx >= len(yaws) or start_idx > end_idx:
        logger.warning(f"Segment range[{start_idx},{end_idx}]Yawcolumns{len(yaws)}，returning 0")
        return 0.0
    # total rotation = Yaw - Yaw，processingπ
    total_rot = abs(yaws[end_idx] - yaws[start_idx])
    if total_rot > np.pi:
        total_rot = 2 * np.pi - total_rot
    return total_rot

def check_81frame_rot(yaws):
    """
    81framescolumns
    - 40frames0-39total rotation  MIN_SEGMENT_ROT
    - 40frames41-80total rotation  MIN_SEGMENT_ROT
    """
    if len(yaws) != 81:
        logger.warning(f"Yawcolumns={len(yaws)}≠81，cannot check")
        return False
    
    # 40frames/
    rot_front = calculate_segment_rot(yaws, 0, 39)
    # 40frames/
    rot_back = calculate_segment_rot(yaws, 41, 80)
    
    logger.info(f"40framestotal rotation{rot_front:.3f}rad{rot_front*180/np.pi:.1f}°")
    logger.info(f"40framestotal rotation{rot_back:.3f}rad{rot_back*180/np.pi:.1f}°")
    
    return (rot_front >= MIN_SEGMENT_ROT) and (rot_back >= MIN_SEGMENT_ROT)

# ====================== ======================
def main():
    # 1. Output directory
    ensure_dir(OUTPUT_HIGH_ROT_DIR)
    ensure_dir(OUTPUT_VIDEO_DIR)
    ensure_dir(OUTPUT_POSE_DIR)

    log_file_path = os.path.join(OUTPUT_HIGH_ROT_DIR, "high_rot_filter.log")
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    total_sequences = 0
    passed_sequences = 0

    logger.info("="*50)
    logger.info("81framescolumns")
    logger.info(f"40frames/40framestotal rotation{MIN_SEGMENT_ROT}rad{MIN_SEGMENT_ROT*180/np.pi:.1f}°")
    logger.info("="*50)

    # 4. Video，Pose file
    if not os.path.exists(INPUT_VIDEO_DIR):
        logger.error(f"Video directorydoes not exist{INPUT_VIDEO_DIR}")
        return
    
    # Video
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in os.listdir(INPUT_VIDEO_DIR) if f.lower().endswith(video_extensions)]

    if not video_files:
        logger.warning("Video directoryNot foundVideo")
        return

    # 5. processingcolumns
    for video_file in video_files:
        total_sequences += 1
        video_stem = Path(video_file).stem
        video_full_path = os.path.join(INPUT_VIDEO_DIR, video_file)
        pose_full_path = os.path.join(INPUT_POSE_DIR, f"{video_stem}.txt")

        logger.info(f"\n---------- processingcolumns{video_stem} ----------")
        
        # Pose file
        if not os.path.exists(pose_full_path):
            logger.warning(f"Not foundcorresponding pose file{pose_full_path}，skippingcolumns")
            continue

        yaw_sequence = read_pose_rotations(pose_full_path)
        if yaw_sequence is None:
            logger.warning(f"Pose filefailed，skippingcolumns")
            continue

        if check_81frame_rot(yaw_sequence):
            passed_sequences += 1
            logger.info(f"✅ columns {video_stem} Rotation amplitude meets threshold, keeping！")
            
            # VideoOutput directory
            dest_video = os.path.join(OUTPUT_VIDEO_DIR, video_file)
            shutil.copy2(video_full_path, dest_video)
            
            # Output directory
            dest_pose = os.path.join(OUTPUT_POSE_DIR, f"{video_stem}.txt")
            shutil.copy2(pose_full_path, dest_pose)
        else:
            logger.info(f"❌ columns {video_stem} Rotation amplitude below threshold, filtering！")

    logger.info("\n" + "="*50)
    logger.info("Filtering summary report")
    logger.info("="*50)
    logger.info(f"processingcolumns{total_sequences}")
    logger.info(f"columns{passed_sequences}")
    logger.info(f"columns{total_sequences - passed_sequences}")
    logger.info(f"\ncolumns{OUTPUT_HIGH_ROT_DIR}")
    logger.info(f"Filter log file path：{log_file_path}")

# ====================== lines ======================
if __name__ == "__main__":
    main()