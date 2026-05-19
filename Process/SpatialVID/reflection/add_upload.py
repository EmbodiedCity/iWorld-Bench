import os
import sys
import logging
from concurrent.futures import ThreadPoolExecutor
import subprocess

# reflect.py
SUPPORTED_VIDEO_SUFFIX = [".mp4", ".avi", ".mov", ".mkv"]
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADED_VIDEOS_LOG = os.path.join(LOG_DIR, "uploaded_videos.txt")
UPLOADED_CAMERAS_LOG = os.path.join(LOG_DIR, "uploaded_cameras.txt")

ADD_VIDEOS_LOG = os.path.join(LOG_DIR, "add_videos.txt")
ADD_CAMERAS_LOG = os.path.join(LOG_DIR, "add_cameras.txt")
SUMMARY_LOG = os.path.join(LOG_DIR, "upload_summary.txt")

# reflect.py
SERVER_CONFIG = {
    "host": "example.com",
    "user": os.environ.get("SSH_USER", "user"),
    "port": 50371,
    "remote_dir": "/path/to/local/data",
    "ssh_key_path": r"/path/to/local/data",
}

def read_log(log_path):
    """，"""
    if not os.path.exists(log_path):
        return set()
    with open(log_path, "r", encoding="utf-8") as f:
        return set([line.strip() for line in f if line.strip()])

def write_log(log_path, content):
    """"""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{content}\n")

def upload_file(local_path, remote_dir):
    """，"""
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

def generate_summary():
    """，"""
    original_videos = read_log(UPLOADED_VIDEOS_LOG)
    original_cameras = read_log(UPLOADED_CAMERAS_LOG)
    add_videos = read_log(ADD_VIDEOS_LOG)
    add_cameras = read_log(ADD_CAMERAS_LOG)
    
    full_videos = original_videos.union(add_videos)
    full_cameras = original_cameras.union(add_cameras)
    
    with open(UPLOADED_VIDEOS_LOG, "w", encoding="utf-8") as f:
        for item in sorted(full_videos):
            f.write(f"{item}\n")
    with open(UPLOADED_CAMERAS_LOG, "w", encoding="utf-8") as f:
        for item in sorted(full_cameras):
            f.write(f"{item}\n")
    
    count_videos = len(full_videos)
    count_cameras = len(full_cameras)
    intersection = full_videos.intersection(full_cameras)
    count_intersection = len(intersection)
    count_union = len(full_videos.union(full_cameras))
    only_videos = full_videos - full_cameras
    only_cameras = full_cameras - full_videos
    
    with open(SUMMARY_LOG, "w", encoding="utf-8") as f:
        f.write("=====  =====\n\n")
        f.write(f" uploaded_videos.txt : {count_videos}\n")
        f.write(f" uploaded_cameras.txt : {count_cameras}\n\n")
        f.write(f" files: {count_intersection}\n")
        f.write(f" files: {count_union}\n\n")
        
        f.write("videoscameras:\n")
        for item in sorted(only_videos):
            f.write(f"  - {item}\n")
        
        f.write("\ncamerasvideos:\n")
        for item in sorted(only_cameras):
            f.write(f"  - {item}\n")
        
        f.write("\n=====  =====\n")
    
    logging.info(f": {SUMMARY_LOG}")
    logging.info(f"")

def process_remaining_files(remake_dir):
    """processing"""
    uploaded_videos = read_log(UPLOADED_VIDEOS_LOG)
    uploaded_cameras = read_log(UPLOADED_CAMERAS_LOG)
    
    video_dir = os.path.join(remake_dir, "videos")
    camera_dir = os.path.join(remake_dir, "cameras")
    
    if not os.path.exists(video_dir):
        logging.warning(f"Video directorydoes not exist: {video_dir}")
        video_files = []
    else:
        video_files = [f for f in os.listdir(video_dir) if any(f.endswith(suffix) for suffix in SUPPORTED_VIDEO_SUFFIX)]
    
    if not os.path.exists(camera_dir):
        logging.warning(f"does not exist: {camera_dir}")
        camera_files = []
    else:
        camera_files = [f for f in os.listdir(camera_dir) if f.endswith(".txt")]
    
    remaining_videos = [f for f in video_files if f not in uploaded_videos]
    remaining_cameras = [f for f in camera_files if f not in uploaded_cameras]
    
    logging.info(f"Found {len(remaining_videos)} Video，{len(remaining_cameras)} ")
    
    # lines
    def upload_worker(file_name, is_video):
        local_dir = video_dir if is_video else camera_dir
        local_path = os.path.join(local_dir, file_name)
        remote_subdir = "videos" if is_video else "cameras"
        remote_full_dir = os.path.join(SERVER_CONFIG['remote_dir'], remote_subdir)
        
        if upload_file(local_path, remote_full_dir):
            orig_log_path = UPLOADED_VIDEOS_LOG if is_video else UPLOADED_CAMERAS_LOG
            add_log_path = ADD_VIDEOS_LOG if is_video else ADD_CAMERAS_LOG
            
            write_log(orig_log_path, file_name)
            write_log(add_log_path, file_name)
            
            os.remove(local_path)
            logging.info(f": {local_path}")
    
    # linesprocessing
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Video
        for video_file in remaining_videos:
            executor.submit(upload_worker, video_file, True)
        for camera_file in remaining_cameras:
            executor.submit(upload_worker, camera_file, False)
    
    generate_summary()
    
    logging.info("processing")

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(LOG_DIR, "add_upload.log"))]
    )
    
    if len(sys.argv) != 2:
        print("python add_upload.py <SpatialVID_remake>")
        print("python add_upload.py /path/to/SpatialVID_remake")
        sys.exit(1)
    
    remake_dir = sys.argv[1]
    if not os.path.isdir(remake_dir):
        logging.error(f"does not exist: {remake_dir}")
        sys.exit(1)
    
    with open(ADD_VIDEOS_LOG, "w", encoding="utf-8") as f:
        pass
    with open(ADD_CAMERAS_LOG, "w", encoding="utf-8") as f:
        pass
    
    process_remaining_files(remake_dir)

if __name__ == "__main__":
    main()