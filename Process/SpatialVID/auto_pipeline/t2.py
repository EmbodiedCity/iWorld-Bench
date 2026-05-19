import os
import sys
import time
import subprocess
from pathlib import Path
import shutil

LOG_DIR = r"/path/to/local/data"
CHECK_INTERVAL = 180
VIEW_TYPE = "Human_front"

os.makedirs(LOG_DIR, exist_ok=True)

FULL_PROCESS_LOG = os.path.join(LOG_DIR, "full_process.txt")
FINISHED_FULL_LOG = os.path.join(LOG_DIR, "finished_full.txt")
FINISHED_FIXED_LOG = os.path.join(LOG_DIR, "finished_fixed.txt")
FINISHED_PROCESS_LOG = os.path.join(LOG_DIR, "finished_process.txt")
FINISHED_TRANSFER_LOG = os.path.join(LOG_DIR, "finished_transfer.txt")

def load_finished_groups(log_file):
    """groupcolumns"""
    if not os.path.exists(log_file):
        return set()
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        return set(line.strip() for line in f if line.strip())

def add_finished_group(log_file, group):
    """group"""
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{group}\n")

def get_all_groups(annotations_base, videos_base):
    """Videoannotation group"""
    valid_groups = []
    # annotation group
    annotation_groups = [g for g in os.listdir(annotations_base) 
                         if os.path.isdir(os.path.join(annotations_base, g))]
    
    for group in annotation_groups:
        # Video directory
        video_group_path = os.path.join(videos_base, group)
        if os.path.exists(video_group_path):
            valid_groups.append(group)
    return sorted(valid_groups)

def find_actual_group_path(base_path, group):
    """processinggroup，actual"""
    group_path = os.path.join(base_path, group)
    
    # (group/.../group)
    sub_dir = "annotations" if "annotations" in base_path else "videos"
    test_path = os.path.join(group_path, "SpatialVID", sub_dir, group)
    if os.path.exists(test_path):
        return test_path
    return group_path

def run_script(script_path, **kwargs):
    """linesprocessing"""
    try:
        cmd = [sys.executable, script_path]
        for key, value in kwargs.items():
            cmd.extend([f"--{key}", str(value)])
        
        # linesencoding，
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False
        )
        
        # utf-8，failedgbk
        def decode_output(output):
            for encoding in ["utf-8", "gbk", "latin-1"]:
                try:
                    return output.decode(encoding)
                except UnicodeDecodeError:
                    continue
            return f"{output[:100]}..."
        
        stdout = decode_output(result.stdout)
        stderr = decode_output(result.stderr)
        
        # lines
        print(f"lines {os.path.basename(script_path)} :")
        print(stdout)
        
        if result.returncode != 0:
            print(f"lines {os.path.basename(script_path)} :")
            print(stderr)
            return False
        return True
    except Exception as e:
        print(f"lines {script_path} failed: {str(e)}")
        return False

def modify_script_params(script_path, params):
    """"""
    try:
        encodings = ["utf-8", "gbk", "latin-1"]
        content = None
        for enc in encodings:
            try:
                with open(script_path, "r", encoding=enc) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            print(f" {script_path}")
            return False
        
        for key, value in params.items():
            if isinstance(value, str):
                pattern = f"{key} = "
                lines = content.split('\n')
                target_line = next((line for line in lines if line.strip().startswith(pattern)), None)
                if target_line:
                    indent = target_line[:len(target_line) - len(target_line.lstrip())]
                    quote = "'" if "'" in target_line else '"'
                    new_line = f"{indent}{key} = {quote}{value}{quote}"
                    content = content.replace(target_line, new_line)
        
        with open(script_path, "w", encoding=enc if encodings.index(enc) != -1 else "utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        print(f" {script_path} failed: {str(e)}")
        return False

def process_group(group, annotations_base, videos_base, cityworld_root):
    """processinggroupError"""
    print(f"\nProcessing group: {group}")
    
    # actual
    annotation_group_path = find_actual_group_path(annotations_base, group)
    video_group_path = find_actual_group_path(videos_base, group)
    
    print(f"actualannotation: {annotation_group_path}")
    print(f"actualvideo: {video_group_path}")
    
    finished_full = load_finished_groups(FINISHED_FULL_LOG)
    finished_fixed = load_finished_groups(FINISHED_FIXED_LOG)
    finished_process = load_finished_groups(FINISHED_PROCESS_LOG)
    finished_transfer = load_finished_groups(FINISHED_TRANSFER_LOG)
    
    # 1: linesmake_full_poses.py
    if group not in finished_full:
        print(f"lines make_full_poses.py processing group {group}")
        script_path = r"/path/to/make_full_poses.py"
        
        if not modify_script_params(script_path, {"root_directory": annotation_group_path}):
            return False
        
        success = False
        for _ in range(2):
            if run_script(script_path):
                success = True
                break
            time.sleep(2)
        
        if success:
            add_finished_group(FINISHED_FULL_LOG, group)
            print(f"make_full_poses.py processing group {group} ")
        else:
            print(f"make_full_poses.py processing group {group} failed")
            return False
    
    # 2: linesfixed_xz_6.py
    if group in finished_full and group not in finished_fixed:
        print(f"lines fixed_xz_6.py processing group {group}")
        script_path = r"/path/to/fixed_xz_6.py"
        
        if not modify_script_params(script_path, {"RAW_ROOT": annotation_group_path}):
            return False
        
        if run_script(script_path):
            add_finished_group(FINISHED_FIXED_LOG, group)
            print(f"fixed_xz_6.py processing group {group} ")
        else:
            print(f"fixed_xz_6.py processing group {group} failed")
            return False
    
    # 3: linesprocess.py
    if group in finished_fixed and group not in finished_process:
        print(f"lines process.py processing group {group}")
        script_path = r"/path/to/process.py"
        
        params = {
            "RAW_INTRINSICS_BASE": annotation_group_path,
            "RAW_POSES_BASE": annotation_group_path,
            "RAW_RGB_BASE": video_group_path,
            "TARGET_ROOT": cityworld_root
        }
        if not modify_script_params(script_path, params):
            return False
        
        if run_script(script_path, pose_type=6):
            add_finished_group(FINISHED_PROCESS_LOG, group)
            print(f"process.py processing group {group} ")
        else:
            print(f"process.py processing group {group} failed")
            return False
    
    # 4: linestransfer_6_pipeline.py
    if group in finished_process and group not in finished_transfer:
        print(f"lines transfer_6_pipeline.py processing group {group}")
        script_path = r"/path/to/transfer_6_pipeline.py"
        
        target_path = os.path.join(cityworld_root, "pose", VIEW_TYPE)
        
        if not modify_script_params(script_path, {"root_directory": target_path}):
            return False
        
        if run_script(script_path):
            add_finished_group(FINISHED_TRANSFER_LOG, group)
            print(f"transfer_6_pipeline.py processing group {group} ")
        else:
            print(f"transfer_6_pipeline.py processing group {group} failed")
            return False
    
    if group in finished_transfer:
        full_processed = load_finished_groups(FULL_PROCESS_LOG)
        if group not in full_processed:
            add_finished_group(FULL_PROCESS_LOG, group)
            print(f"group {group} processing")
        return True
    
    return False

def main():
    if len(sys.argv) != 4:
        print(": python auto_process.py <annotations_base> <videos_base> <cityworld_root>")
        print(': python t2.py /path/to/annotations /path/to/videos /path/to/cityworld')
        sys.exit(1)
    
    annotations_base = sys.argv[1]
    videos_base = sys.argv[2]
    cityworld_root = sys.argv[3]
    
    for path in [annotations_base, videos_base, cityworld_root]:
        if not os.path.exists(path):
            print(f"Error: does not exist - {path}")
            sys.exit(1)
    
    print("=== processing ===")
    print(f"Annotations: {annotations_base}")
    print(f"Videos: {videos_base}")
    print(f"CityWorld: {cityworld_root}")
    print(f": {CHECK_INTERVAL}")
    print(f": {LOG_DIR}")
    print("==========================")
    
    try:
        while True:
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ...")
            
            all_groups = get_all_groups(annotations_base, videos_base)
            full_processed = load_finished_groups(FULL_PROCESS_LOG)
            pending_groups = [g for g in all_groups if g not in full_processed]
            
            print(f"Found {len(all_groups)} group， {len(pending_groups)} processing")
            
            for group in pending_groups:
                process_group(group, annotations_base, videos_base, cityworld_root)
            
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ，...")
            time.sleep(CHECK_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n，")
    except Exception as e:
        print(f": {str(e)}")
        # Error
        with open(os.path.join(LOG_DIR, "error.log"), "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] : {str(e)}\n")

if __name__ == "__main__":
    main()
