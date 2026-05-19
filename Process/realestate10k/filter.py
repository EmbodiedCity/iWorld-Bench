import os
import shutil
import numpy as np
import cv2
from scipy.signal import savgol_filter
import logging
from multiprocessing import Pool, cpu_count
import platform

# ====================== ======================
VIDEO_DIR = "./videos"          
CAMERAS_DIR = "./cameras"      
LEFT_OUTPUT_DIR = "./left_rot_sequences"   
RIGHT_OUTPUT_DIR = "./right_rot_sequences" 
MIN_SEQ_LENGTH = 40            
ROT_THRESHOLD = 1e-3           
UP_AXIS = 'y'                  
SMOOTH_WINDOW = 5              
VIDEO_CODEC = cv2.VideoWriter_fourcc(*'mp4v')  
VIDEO_FPS = 30.0                              
VIDEO_SIZE = None                             
PROCESS_NUM = cpu_count() - 1  

# ====================== ======================
def setup_logger():
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(process)d - %(levelname)s - %(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger()

# ====================== read_camera_poses， ======================
def ensure_dir(dir_path):
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory：{dir_path}")
    except Exception as e:
        logger.error(f"Created directory {dir_path} failed：{e}")

def read_camera_poses(pose_file):
    """
    Pose file，SVD，
    - poses_raw: lineslist[str]
    - extrinsics_R: 33list[np.array]
    """
    poses_raw = []
    extrinsics_R = []
    try:
        with open(pose_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line_strip = line.strip()
                if not line_strip:
                    logger.warning(f"Pose file {pose_file} {line_idx+1}lines，skipping")
                    continue
                poses_raw.append(line_strip)
                
                nums = list(map(float, line_strip.split()))
                if len(nums) < 12:
                    logger.warning(f"Pose file {pose_file} {line_idx+1}columns12，skipping")
                    continue
                R_flat = nums[-12:-3]
                R = np.array(R_flat).reshape(3, 3)
                
                U, _, Vt = np.linalg.svd(R)
                R_ortho = U @ Vt
                # linescolumns1linescolumns-1，
                if np.linalg.det(R_ortho) < 0:
                    Vt[-1, :] *= -1
                    R_ortho = U @ Vt
                logger.debug(f"Pose file {pose_file} {line_idx+1}lines，")
                
                extrinsics_R.append(R_ortho)
    except Exception as e:
        logger.error(f"Pose file {pose_file} failed{e}")
    return poses_raw, extrinsics_R

# rotation_matrix_to_yaw、correct_yaw_jump
def rotation_matrix_to_yaw(R, up_axis='y'):
    try:
        if up_axis == 'y':
            cos_theta = R[0, 0]
            sin_theta = R[0, 2]
        elif up_axis == 'z':
            cos_theta = R[0, 0]
            sin_theta = -R[1, 0]
        else:
            raise ValueError(f"up_axis only supports 'y'/'z', got{up_axis}")
        return np.arctan2(sin_theta, cos_theta)
    except Exception as e:
        logger.error(f"Yawfailed{e}")
        return 0.0

def correct_yaw_jump(yaws):
    corrected = [yaws[0]] if len(yaws) > 0 else []
    try:
        for i in range(1, len(yaws)):
            delta = yaws[i] - corrected[i-1]
            if delta > np.pi:
                corrected.append(yaws[i] - 2 * np.pi)
            elif delta < -np.pi:
                corrected.append(yaws[i] + 2 * np.pi)
            else:
                corrected.append(yaws[i])
    except Exception as e:
        logger.error(f"Yawfailed{e}")
    return np.array(corrected)

def find_continuous_rot_sequences(directions, min_length=40):
    sequences = []
    n_frames = len(directions)
    if n_frames < min_length:
        return sequences
    
    try:
        start_idx = 0
        current_dir = 0
        
        for i in range(n_frames):
            if directions[i] != current_dir:
                if current_dir != 0 and (i - start_idx) >= min_length:
                    dir_name = 'right' if current_dir == 1 else 'left'
                    sequences.append((start_idx, i-1, dir_name))
                start_idx = i
                current_dir = directions[i]
        
        if current_dir != 0 and (n_frames - start_idx) >= min_length:
            dir_name = 'right' if current_dir == 1 else 'left'
            sequences.append((start_idx, n_frames-1, dir_name))
    except Exception as e:
        logger.error(f"columnsfailed{e}")
    return sequences

def extract_video_frames(video_path, start_frame, end_frame, output_video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video：{video_path}")
            return False
        
        global VIDEO_SIZE, VIDEO_FPS
        if VIDEO_SIZE is None:
            VIDEO_SIZE = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if VIDEO_FPS is None:
            VIDEO_FPS = cap.get(cv2.CAP_PROP_FPS)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame < 0 or end_frame >= total_frames or start_frame > end_frame:
            logger.error(f"Video {video_path} frame rangestart={start_frame}, end={end_frame}, total={total_frames}")
            cap.release()
            return False
        
        out = cv2.VideoWriter(output_video_path, VIDEO_CODEC, VIDEO_FPS, VIDEO_SIZE)
        if not out.isOpened():
            logger.error(f"Cannot create output video：{output_video_path}")
            cap.release()
            return False
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = 0
        while frame_count <= (end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Video {video_path} read to frame {start_frame + frame_count} interrupted")
                break
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        logger.info(f"Video cropping completed：{output_video_path}（frame range：{start_frame}-{end_frame}）")
        return True
    except Exception as e:
        logger.error(f"Videoframesfailed {video_path}{e}")
        return False

def save_pose_sequences(poses_raw, start_frame, end_frame, output_pose_path):
    try:
        with open(output_pose_path, 'w', encoding='utf-8') as f:
            for i in range(start_frame, end_frame + 1):
                if i < len(poses_raw):
                    f.write(poses_raw[i] + '\n')
                else:
                    logger.warning(f"Pose filelines，frames{i}lines")
        logger.info(f"Pose file{output_pose_path}frame range{start_frame}-{end_frame}")
        return True
    except Exception as e:
        logger.error(f"Pose filefailed {output_pose_path}{e}")
        return False

# ====================== processing ======================
def process_single_pair(args):
    video_path, pose_file, left_output_dir, right_output_dir = args
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    logger.info(f"\nProcess {os.getpid()} Processing：{video_name}")
    
    poses_raw, extrinsics_R = read_camera_poses(pose_file)
    if len(extrinsics_R) < MIN_SEQ_LENGTH:
        logger.warning(f"Process {os.getpid()}Pose file {pose_file} lines{MIN_SEQ_LENGTH}，skipping")
        return
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Process {os.getpid()}：Cannot open video {video_path}，skipping")
            return
        video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if abs(len(extrinsics_R) - video_total_frames) > 5:
            logger.warning(f"Process {os.getpid()}Video frame count ({video_total_frames}lines{len(extrinsics_R)}，skipping")
            return
    except Exception as e:
        logger.error(f"Process {os.getpid()}Videoframesfailed {video_path}{e}")
        return
    
    yaws = [rotation_matrix_to_yaw(R, up_axis=UP_AXIS) for R in extrinsics_R]
    yaws_corrected = correct_yaw_jump(yaws)
    
    if SMOOTH_WINDOW > 0 and len(yaws_corrected) >= SMOOTH_WINDOW:
        yaws_corrected = savgol_filter(yaws_corrected, window_length=SMOOTH_WINDOW, polyorder=2)
    
    directions = [0]
    for i in range(1, len(yaws_corrected)):
        delta_yaw = yaws_corrected[i] - yaws_corrected[i-1]
        if delta_yaw > ROT_THRESHOLD:
            directions.append(1)
        elif delta_yaw < -ROT_THRESHOLD:
            directions.append(-1)
        else:
            directions.append(0)
    
    valid_sequences = find_continuous_rot_sequences(directions, min_length=MIN_SEQ_LENGTH)
    if not valid_sequences:
        logger.info(f"Process {os.getpid()}{video_name} Not found{MIN_SEQ_LENGTH}frames/columns，skipping")
        return
    
    for seq_idx, (start, end, rot_dir) in enumerate(valid_sequences):
        seq_start = start
        seq_end = start + MIN_SEQ_LENGTH - 1
        if seq_end > end:
            seq_start = end - MIN_SEQ_LENGTH + 1
            seq_end = end
        
        output_root = left_output_dir if rot_dir == 'left' else right_output_dir
        seq_output_dir = os.path.join(output_root, f"{video_name}_seq_{seq_idx:03d}")
        ensure_dir(seq_output_dir)
        
        output_video_path = os.path.join(seq_output_dir, f"{video_name}_seq_{seq_idx:03d}.mp4")
        output_pose_path = os.path.join(seq_output_dir, f"{video_name}_seq_{seq_idx:03d}.txt")
        
        video_success = extract_video_frames(video_path, seq_start, seq_end, output_video_path)
        if not video_success:
            logger.error(f"Process {os.getpid()}columns {seq_idx} Videofailed，skipping")
            shutil.rmtree(seq_output_dir, ignore_errors=True)
            continue
        
        pose_success = save_pose_sequences(poses_raw, seq_start, seq_end, output_pose_path)
        if not pose_success:
            logger.error(f"Process {os.getpid()}columns {seq_idx} failed")
            shutil.rmtree(seq_output_dir, ignore_errors=True)
            continue
        
        logger.info(f"Process {os.getpid()}columns {seq_idx} processing{seq_output_dir}direction{rot_dir}")

# ====================== Processprocessing ======================
def batch_process_multiprocess(video_dir, cameras_dir, left_output_dir, right_output_dir):
    ensure_dir(left_output_dir)
    ensure_dir(right_output_dir)
    
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(video_extensions)]
    
    if not video_files:
        logger.error(f"Video directory {video_dir} Found video file")
        return
    
    task_args = []
    for video_file in video_files:
        video_name = os.path.splitext(video_file)[0]
        pose_file = None
        for ext in ('.txt', '.csv', '.dat'):
            candidate = os.path.join(cameras_dir, f"{video_name}{ext}")
            if os.path.exists(candidate):
                pose_file = candidate
                break
        
        if pose_file is None:
            logger.warning(f"Not found {video_name} corresponding pose file，skipping")
            continue
        
        video_path = os.path.join(video_dir, video_file)
        task_args.append((video_path, pose_file, left_output_dir, right_output_dir))
    
    if not task_args:
        logger.error("Video-Pose file，")
        return
    
    logger.info(f"\nProcessprocessing{len(task_args)} tasks，Process{PROCESS_NUM}")
    try:
        if platform.system() == "Windows":
            from multiprocessing import set_start_method
            try:
                set_start_method('spawn')
            except RuntimeError:
                pass
        
        with Pool(processes=PROCESS_NUM) as pool:
            pool.map(process_single_pair, task_args)
        logger.info("Processprocessing！")
    except Exception as e:
        logger.error(f"Processlinesfailed{e}")

# ====================== lines ======================
if __name__ == "__main__":
    batch_process_multiprocess(
        video_dir=VIDEO_DIR,
        cameras_dir=CAMERAS_DIR,
        left_output_dir=LEFT_OUTPUT_DIR,
        right_output_dir=RIGHT_OUTPUT_DIR
    )
    logger.info("processing！")