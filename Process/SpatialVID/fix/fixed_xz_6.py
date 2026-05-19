import os
import sys
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

"""
6
==========================

- 
  Xdirection = ，Zdirection = ，Ydirection = 
- 
  Xdirection = ，Zdirection = ，Ydirection = 
- 6lines6tx, ty, tz, rx, ry, rz
  1. tx' = -txX、ty' = tyY、tz' = -tzZ
  2. rx' = -rxX、ry' = ryY、rz' = -rzZ
==========================

1. RAW_ROOT = ""
2. pip install tqdm
3. linespython fixed_6dof.py
"""

# ===================== =====================
RAW_ROOT = r"/path/to/local/data"
INPUT_FILE = "full_poses.txt"
OUTPUT_FILE = "full_poses_fixed.txt"
RECURSIVE = True

# ===================== =====================
def correct_6dof_poses(original_values):
    """
    6
    :param original_values: 6columns [tx, ty, tz, rx, ry, rz]
    :return: 6columns [tx', ty', tz', rx', ry', rz']
    """
    if len(original_values) != 6:
        return None
    
    tx, ty, tz, rx, ry, rz = original_values
    tx_fixed = -tx
    ty_fixed = ty
    tz_fixed = -tz
    rx_fixed = -rx
    ry_fixed = ry
    rz_fixed = -rz
    
    return [tx_fixed, ty_fixed, tz_fixed, rx_fixed, ry_fixed, rz_fixed]

def process_single_file(file_path, output_path):
    """
    processingfull_poses.txt
    :param file_path: 
    :param output_path: 
    :return: (processing, lines, failedlines)
    """
    success_count = 0
    fail_count = 0
    fixed_lines = []
    
    try:
        # UTF-8/GBK
        encodings = ["utf-8", "gbk", "gb2312"]
        lines = None
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    lines = f.readlines()
                break
            except:
                continue
        
        if lines is None:
            print(f"\n {file_path}UTF-8/GBK/GB2312")
            return False, 0, 0
        
        # linesprocessing
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                fixed_lines.append("")
                continue
            
            try:
                original_values = [float(v) for v in line.split()]
                fixed_values = correct_6dof_poses(original_values)
                
                if fixed_values is None:
                    fail_count += 1
                    fixed_lines.append(line)
                    print(f"\n  {file_path} {line_idx+1}linesError6，")
                    continue
                
                fixed_line = " ".join([f"{v}" for v in fixed_values])
                fixed_lines.append(fixed_line)
                success_count += 1
                
            except Exception as e:
                fail_count += 1
                fixed_lines.append(line)
                print(f"\n  {file_path} {line_idx+1}linesfailed {str(e)[:50]}...，")
        
        with open(output_path, "w", encoding=enc) as f:
            f.write("\n".join(fixed_lines))
        
        return True, success_count, fail_count
    
    except Exception as e:
        print(f"\n {file_path}failed {str(e)}")
        return False, success_count, fail_count

# ===================== + =====================
def find_all_target_files(root_dir, target_file, recursive=True):
    """"""
    target_files = []
    for root, dirs, files in os.walk(root_dir):
        if target_file in files:
            target_files.append(os.path.join(root, target_file))
        if not recursive:
            break
    return target_files

def batch_process():
    """processing"""
    # 1. processing
    print(f"  {RAW_ROOT}  {INPUT_FILE} ...")
    target_files = find_all_target_files(RAW_ROOT, INPUT_FILE, RECURSIVE)
    
    if not target_files:
        print(f" Not found {INPUT_FILE} ")
        return
    
    print(f"  {len(target_files)} processing")
    print("-" * 60)
    
    total_success = 0
    total_fail = 0
    fail_files = []
    
    if tqdm:
        pbar = tqdm(target_files, desc="processing", unit="", file=sys.stdout)
    else:
        pbar = target_files
        print("  tqdm，lines pip install tqdm ")
    
    # 3. processing
    for file_path in pbar:
        output_path = os.path.join(os.path.dirname(file_path), OUTPUT_FILE)
        status, success, fail = process_single_file(file_path, output_path)
        
        total_success += success
        total_fail += fail
        if not status:
            fail_files.append(file_path)
        
        if tqdm:
            pbar.set_postfix({
                "": len(target_files) - len(fail_files),
                "failed": len(fail_files),
                "lines": total_success,
                "failedlines": total_fail
            })
    
    if tqdm:
        pbar.close()
    
    print("\n" + "-" * 60)
    print(" 6")
    print(f" processing{len(target_files)}")
    print(f" processing{len(target_files) - len(fail_files)}")
    print(f" failedprocessing{len(fail_files)}")
    print(f" processinglines{total_success}")
    print(f"  failedprocessinglines{total_fail}")
    
    if fail_files:
        print("\n failedcolumns")
        for f in fail_files:
            print(f"   - {f}")
    
    print("\n ！{}".format(OUTPUT_FILE))

if __name__ == "__main__":
    batch_process()