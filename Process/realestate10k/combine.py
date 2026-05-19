import os
import cv2
import numpy as np
import logging

# ====================== ======================
LEFT_INPUT_DIR = "./left_rot_sequences"
RIGHT_INPUT_DIR = "./right_rot_sequences"
# processingVideo
OUTPUT_ROOT_DIR = "./combined_sequences"
OUTPUT_VIDEO_DIR = os.path.join(OUTPUT_ROOT_DIR, "videos")
OUTPUT_POSE_DIR = os.path.join(OUTPUT_ROOT_DIR, "poses")
# VideoVideo，
VIDEO_CODEC = cv2.VideoWriter_fourcc(*'mp4v')

# ====================== ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ====================== ======================
def ensure_dir(dir_path):
    """Ensure directory exists"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"Created directory：{dir_path}")

def read_video_frames(video_path):
    """Videoframes，framescolumns+Video、frames"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video：{video_path}")
        return None, None, None
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    # Video
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frames, frame_size, fps

def read_pose_lines(pose_path):
    """Pose filelines，linescolumns"""
    try:
        with open(pose_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines
    except Exception as e:
        logger.error(f"Pose file {pose_path} failed{e}")
        return None

def write_video(frames, output_path, frame_size, fps):
    """framescolumnsVideo"""
    out = cv2.VideoWriter(output_path, VIDEO_CODEC, fps, frame_size)
    if not out.isOpened():
        logger.error(f"Cannot create output video：{output_path}")
        return False
    
    for frame in frames:
        out.write(frame)
    out.release()
    logger.info(f"Video write completed：{output_path}")
    return True

def write_pose(lines, output_path):
    """linescolumns"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        logger.info(f"Pose write completed：{output_path}")
        return True
    except Exception as e:
        logger.error(f"Pose file {output_path} failed{e}")
        return False

# ====================== processingcolumns ======================
def process_left_sequence(seq_dir, output_video_dir, output_pose_dir):
    """
    processingcolumns40frames  41frames  42-81frames
    seq_dir: columns
    """
    seq_name = os.path.basename(seq_dir)
    logger.info(f"processingcolumns{seq_name}")

    # 1. columnsVideoPose file
    video_file = None
    pose_file = None
    for f in os.listdir(seq_dir):
        if f.lower().endswith('.mp4'):
            video_file = os.path.join(seq_dir, f)
        elif f.lower().endswith('.txt'):
            pose_file = os.path.join(seq_dir, f)
    if not video_file or not pose_file:
        logger.warning(f"columns {seq_name} Missing video/pose file, skipping")
        return

    # 2. Videoframes
    frames, frame_size, fps = read_video_frames(video_file)
    pose_lines = read_pose_lines(pose_file)
    if not frames or len(frames) != 40 or not pose_lines or len(pose_lines) != 40:
        logger.warning(f"columns {seq_name} Video/pose frame count mismatch (need 40), skipping")
        return

    # 3. 81framesVideoframes
    # 40frames | 41framesframes | 42-81frames
    new_frames = (
        frames[:40] +
        [frames[39]] +
        frames[39::-1]
    )
    if len(new_frames) != 81:
        logger.warning(f"columns {seq_name} Video frame concatenation failed, skipping")
        return

    # 4. 81lines
    new_poses = (
        pose_lines[:40] +
        [pose_lines[39]] +
        pose_lines[39::-1]
    )
    if len(new_poses) != 81:
        logger.warning(f"columns {seq_name} Pose concatenation failed, skipping")
        return

    output_video_path = os.path.join(output_video_dir, f"left_{seq_name}.mp4")
    output_pose_path = os.path.join(output_pose_dir, f"left_{seq_name}.txt")
    write_video(new_frames, output_video_path, frame_size, fps)
    write_pose(new_poses, output_pose_path)

# ====================== processingcolumns ======================
def process_right_sequence(seq_dir, output_video_dir, output_pose_dir):
    """
    processingcolumns40frames  41frames  42-81frames
    seq_dir: columns
    """
    seq_name = os.path.basename(seq_dir)
    logger.info(f"processingcolumns{seq_name}")

    # 1. columnsVideoPose file
    video_file = None
    pose_file = None
    for f in os.listdir(seq_dir):
        if f.lower().endswith('.mp4'):
            video_file = os.path.join(seq_dir, f)
        elif f.lower().endswith('.txt'):
            pose_file = os.path.join(seq_dir, f)
    if not video_file or not pose_file:
        logger.warning(f"columns {seq_name} Missing video/pose file, skipping")
        return

    # 2. Videoframes
    frames, frame_size, fps = read_video_frames(video_file)
    pose_lines = read_pose_lines(pose_file)
    if not frames or len(frames) != 40 or not pose_lines or len(pose_lines) != 40:
        logger.warning(f"columns {seq_name} Video/pose frame count mismatch (need 40), skipping")
        return

    # 3. 81framesVideoframes
    # 40frames | 41framesframes | 42-81frames
    reversed_frames = frames[39::-1]
    new_frames = (
        reversed_frames[:40] +
        [reversed_frames[39]] +
        frames[:40]
    )
    if len(new_frames) != 81:
        logger.warning(f"columns {seq_name} Video frame concatenation failed, skipping")
        return

    # 4. 81lines
    reversed_poses = pose_lines[39::-1]
    new_poses = (
        reversed_poses[:40] +
        [reversed_poses[39]] +
        pose_lines[:40]
    )
    if len(new_poses) != 81:
        logger.warning(f"columns {seq_name} Pose concatenation failed, skipping")
        return

    output_video_path = os.path.join(output_video_dir, f"right_{seq_name}.mp4")
    output_pose_path = os.path.join(output_pose_dir, f"right_{seq_name}.txt")
    write_video(new_frames, output_video_path, frame_size, fps)
    write_pose(new_poses, output_pose_path)

# ====================== processing ======================
def main():
    # Output directory
    ensure_dir(OUTPUT_ROOT_DIR)
    ensure_dir(OUTPUT_VIDEO_DIR)
    ensure_dir(OUTPUT_POSE_DIR)

    # 1. processingcolumns
    if os.path.exists(LEFT_INPUT_DIR):
        for seq_dir in os.listdir(LEFT_INPUT_DIR):
            full_seq_dir = os.path.join(LEFT_INPUT_DIR, seq_dir)
            if os.path.isdir(full_seq_dir):
                process_left_sequence(full_seq_dir, OUTPUT_VIDEO_DIR, OUTPUT_POSE_DIR)
    else:
        logger.warning(f"columns {LEFT_INPUT_DIR} does not exist，skipping")

    # 2. processingcolumns
    if os.path.exists(RIGHT_INPUT_DIR):
        for seq_dir in os.listdir(RIGHT_INPUT_DIR):
            full_seq_dir = os.path.join(RIGHT_INPUT_DIR, seq_dir)
            if os.path.isdir(full_seq_dir):
                process_right_sequence(full_seq_dir, OUTPUT_VIDEO_DIR, OUTPUT_POSE_DIR)
    else:
        logger.warning(f"columns {RIGHT_INPUT_DIR} does not exist，skipping")

    logger.info("columns！%s", OUTPUT_ROOT_DIR)

# ====================== lines ======================
if __name__ == "__main__":
    main()