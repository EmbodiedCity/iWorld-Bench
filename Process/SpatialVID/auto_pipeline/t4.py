import os
import sys
import time
import subprocess
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

LOG_DIR = r"/path/to/local/data"
CHECK_INTERVAL = 180
VIEW_TYPE = "Human_front"
MAX_WORKERS = 4

os.makedirs(LOG_DIR, exist_ok=True)

FULL_PROCESS_LOG = os.path.join(LOG_DIR, "full_process.txt")
FINISHED_FULL_LOG = os.path.join(LOG_DIR, "finished_full.txt")
FINISHED_FIXED_LOG = os.path.join(LOG_DIR, "finished_fixed.txt")
FINISHED_PROCESS_LOG = os.path.join(LOG_DIR, "finished_process.txt")
FINISHED_TRANSFER_LOG = os.path.join(LOG_DIR, "finished_transfer.txt")

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "auto_process.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

def load_finished_groups(log_file):
    """groupcolumns"""
    if not os.path.exists(log_file):
        return set()
    with open(log_file, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def add_finished_group(log_file, group):
    """group"""
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{group}\n")
    logging.info(f"Group {group} ， {os.path.basename(log_file)}")

def get_all_groups(annotations_base, videos_base):
    """Videoannotation group"""
    valid_groups = []
    annotation_groups = [g for g in os.listdir(annotations_base) 
                         if os.path.isdir(os.path.join(annotations_base, g))]
    
    for group in annotation_groups:
        video_group_path = os.path.join(videos_base, group)
        if os.path.exists(video_group_path):
            valid_groups.append(group)
    return sorted(valid_groups)

def find_actual_group_path(base_path, group):
    """processinggroup，actual"""
    group_path = os.path.join(base_path, group)
    subdir = "annotations" if "annotations" in base_path else "videos"
    test_path = os.path.join(group_path, "SpatialVID", subdir, group)
    return test_path if os.path.exists(test_path) else group_path

def run_script(script_path, **kwargs):
    """lines，lines"""
    try:
        cmd = [sys.executable, script_path]
        for key, value in kwargs.items():
            cmd.extend([f"--{key}", str(value)])
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            timeout=3600
        )
        
        script_name = os.path.basename(script_path)
        logging.info(f"{script_name} : {result.stdout[:500]}...")  # 
        if result.returncode != 0:
            logging.error(f"{script_name} Error: {result.stderr[:500]}...")
            return False, result.stderr
        return True, result.stdout
    except Exception as e:
        logging.error(f"lines {script_path} failed: {str(e)}")
        return False, str(e)

def modify_script_params(script_path, params):
    """，"""
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        for key, value in params.items():
            if isinstance(value, str):
                pattern = f"{key} = "
                lines = content.split('\n')
                target_line = next((line for line in lines if line.strip().startswith(pattern)), None)
                if target_line:
                    indent = target_line[:len(target_line) - len(target_line.lstrip())]
                    quote = "'" if "'" in target_line else '"'
                    new_line = f"{indent}{key} = r{quote}{value}{quote}"
                    content = content.replace(target_line, new_line)
        
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        logging.error(f" {script_path} failed: {str(e)}")
        return False

def process_group(group, annotations_base, videos_base, cityworld_root):
    """processinggroup，processing"""
    try:
        logging.info(f"Processing group: {group}")
        annotation_group_path = find_actual_group_path(annotations_base, group)
        video_group_path = find_actual_group_path(videos_base, group)
        
        finished_full = load_finished_groups(FINISHED_FULL_LOG)
        finished_fixed = load_finished_groups(FINISHED_FIXED_LOG)
        finished_process = load_finished_groups(FINISHED_PROCESS_LOG)
        finished_transfer = load_finished_groups(FINISHED_TRANSFER_LOG)
        
        # 1: linesmake_full_poses.py
        if group not in finished_full:
            logging.info(f"lines make_full_poses.py processing group {group}")
            script_path = r"/path/to/make_full_poses.py"
            if not modify_script_params(script_path, {"root_directory": annotation_group_path}):
                return False, f"make_full_posesfailed"
            
            success, _ = run_script(script_path)
            if not success:
                return False, f"make_full_posesProcessing failed"
            add_finished_group(FINISHED_FULL_LOG, group)
        
        # 2: linesfixed_xz_6.py
        if group in finished_full and group not in finished_fixed:
            logging.info(f"lines fixed_xz_6.py processing group {group}")
            script_path = r"/path/to/fixed_xz_6.py"
            if not modify_script_params(script_path, {"RAW_ROOT": annotation_group_path}):
                return False, f"fixed_xz_6failed"
            
            success, _ = run_script(script_path)
            if not success:
                return False, f"fixed_xz_6Processing failed"
            add_finished_group(FINISHED_FIXED_LOG, group)
        
        # 3: linesprocess.py
        if group in finished_fixed and group not in finished_process:
            logging.info(f"lines process.py processing group {group}")
            script_path = r"/path/to/process.py"
            params = {
                "RAW_INTRINSICS_BASE": annotation_group_path,
                "RAW_POSES_BASE": annotation_group_path,
                "RAW_RGB_BASE": video_group_path,
                "TARGET_ROOT": cityworld_root
            }
            if not modify_script_params(script_path, params):
                return False, f"processfailed"
            
            success, _ = run_script(script_path, pose_type=6)
            if not success:
                return False, f"processProcessing failed"
            add_finished_group(FINISHED_PROCESS_LOG, group)
        
        # 4: linestransfer_6_pipeline.py
        if group in finished_process and group not in finished_transfer:
            logging.info(f"lines transfer_6_pipeline.py processing group {group}")
            script_path = r"/path/to/transfer_6_pipeline.py"
            target_path = os.path.join(cityworld_root, "pose", VIEW_TYPE)
            if not modify_script_params(script_path, {"root_directory": target_path}):
                return False, f"transferfailed"
            
            success, _ = run_script(script_path)
            if not success:
                return False, f"transferProcessing failed"
            add_finished_group(FINISHED_TRANSFER_LOG, group)
        
        if group in finished_transfer:
            full_processed = load_finished_groups(FULL_PROCESS_LOG)
            if group not in full_processed:
                add_finished_group(FULL_PROCESS_LOG, group)
                logging.info(f"group {group} processing")
            return True, ""
        
        return False, ""
    except Exception as e:
        logging.error(f"processinggroup {group} : {str(e)}")
        return False, str(e)

def main():
    if len(sys.argv) != 4:
        print(": python auto_process.py <annotations_base> <videos_base> <cityworld_root>")
        print(': python t4.py /path/to/annotations /path/to/videos /path/to/cityworld')
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
    print(f": {CHECK_INTERVAL}，lines: {MAX_WORKERS}")
    print(f": {LOG_DIR}")
    print("==========================")
    logging.info("processing")
    
    try:
        while True:
            current_time = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n[{current_time}] ...")
            logging.info("")
            
            # processinggroup
            all_groups = get_all_groups(annotations_base, videos_base)
            full_processed = load_finished_groups(FULL_PROCESS_LOG)
            pending_groups = [g for g in all_groups if g not in full_processed]
            
            print(f"Found {len(all_groups)} group， {len(pending_groups)} processing")
            logging.info(f"Found {len(all_groups)} group，{len(pending_groups)} processing")
            
            # linesprocessingprocessinggroup
            if pending_groups:
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = {
                        executor.submit(
                            process_group, 
                            group, 
                            annotations_base, 
                            videos_base, 
                            cityworld_root
                        ): group for group in pending_groups
                    }
                    
                    # processing
                    for future in as_completed(futures):
                        group = futures[future]
                        try:
                            success, msg = future.result()
                            if success:
                                print(f"✅ group {group} processing: {msg}")
                            else:
                                print(f"❌ group {group} Processing failed: {msg}")
                        except Exception as e:
                            print(f"❌ group {group} : {str(e)}")
                            logging.error(f"group {group} linesprocessing: {str(e)}")
            
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ，...")
            logging.info("，")
            time.sleep(CHECK_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n，")
        logging.info("，")
    except Exception as e:
        print(f": {str(e)}")
        logging.error(f": {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
