import numpy as np
import os
import sys

"""

==========================

- 
  Xdirection = 
  Ydirection = 
  Zdirection = 
- 
  Xdirection = 
  Ydirection = 
  Zdirection = 
- 
  1. X  X' = -X
  2. Z  Z' = -Z
  3. YY' = Y
  4. ，
==========================

1. pip install numpy tqdm
2. RAW_ROOT = ""
3. linespython fixed_batch.py
"""

# ===================== =====================
RAW_ROOT = r"/path/to/local/data"
# processing
INPUT_FILE_NAME = "extrinsics_matrix.txt"
OUTPUT_FILE_NAME = "extrinsics_matrix.txt"
RECURSIVE = True

# ===================== =====================
def correct_extrinsic_matrix(original_matrix):
    """
    X/Z，Y
    :param original_matrix: 4x4numpy
    :return: 4x4
    """
    R_correction = np.array([
        [-1,  0,  0,  0],
        [ 0,  1,  0,  0],
        [ 0,  0, -1,  0],
        [ 0,  0,  0,  1]
    ], dtype=np.float64)
    
    # T_fixed = R_correction @ T_original @ R_correction
    corrected_matrix = R_correction @ original_matrix @ R_correction
    return corrected_matrix

def process_single_file(file_path, output_path):
    """
    processingfull_poses_m.txt
    :param file_path: 
    :param output_path: 
    :return: (processing, lines, failedlines)
    """
    success_count = 0
    fail_count = 0
    corrected_lines = []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            try:
                values = np.array(line.split(), dtype=np.float64)
                if len(values) != 16:
                    fail_count += 1
                    print(f"\n⚠️  {file_path} {line_idx+1}linesError16，skipping")
                    continue
                
                original_matrix = values.reshape(4, 4)
                corrected_matrix = correct_extrinsic_matrix(original_matrix)
                corrected_values = corrected_matrix.flatten()
                
                corrected_line = " ".join([f"{v}" for v in corrected_values])
                corrected_lines.append(corrected_line)
                success_count += 1
                
            except Exception as e:
                fail_count += 1
                print(f"\n⚠️  {file_path} {line_idx+1}linesProcessing failed {str(e)[:50]}...，skipping")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(corrected_lines))
        
        return True, success_count, fail_count
    
    except Exception as e:
        print(f"\n❌ {file_path}failed {str(e)}")
        return False, success_count, fail_count

# ===================== + =====================
def find_all_target_files(root_dir, target_file, recursive=True):
    """
    
    :param root_dir: 
    :param target_file: 
    :param recursive: 
    :return: columns
    """
    target_files = []
    for root, dirs, files in os.walk(root_dir):
        if target_file in files:
            file_path = os.path.join(root, target_file)
            target_files.append(file_path)
        if not recursive:
            break
    return target_files

def batch_process():
    """processing"""
    # 1. processing
    print(f"🔍  {RAW_ROOT}  {INPUT_FILE_NAME} ...")
    target_files = find_all_target_files(RAW_ROOT, INPUT_FILE_NAME, RECURSIVE)
    
    if not target_files:
        print(f"❌ Not found {INPUT_FILE_NAME} ")
        return
    
    print(f"✅  {len(target_files)} processing")
    print("-" * 50)
    
    # 2. processing
    total_success = 0
    total_fail = 0
    fail_files = []
    
    try:
        from tqdm import tqdm
        pbar = tqdm(target_files, desc="processing", unit="", file=sys.stdout)
    except ImportError:
        print("⚠️  tqdm，lines pip install tqdm ")
        pbar = target_files
    
    for file_path in pbar:
        output_path = os.path.join(os.path.dirname(file_path), OUTPUT_FILE_NAME)
        
        # processing files
        status, success, fail = process_single_file(file_path, output_path)
        
        total_success += success
        total_fail += fail
        if not status:
            fail_files.append(file_path)
        
        if 'tqdm' in str(type(pbar)):
            pbar.set_postfix({
                "": len(target_files) - len(fail_files),
                "failed": len(fail_files),
                "lines": total_success,
                "failedlines": total_fail
            })
    
    if 'tqdm' in str(type(pbar)):
        pbar.close()
    
    print("\n" + "-" * 50)
    print("📊 processing")
    print(f"📁 processing{len(target_files)}")
    print(f"✅ processing{len(target_files) - len(fail_files)}")
    print(f"❌ failedprocessing{len(fail_files)}")
    print(f"📝 processinglines{total_success}")
    print(f"⚠️  failedprocessinglines{total_fail}")
    
    if fail_files:
        print("\n❌ failedcolumns")
        for f in fail_files:
            print(f"   - {f}")
    
    print("\n🎉 processing！")

if __name__ == "__main__":
    # processing
    batch_process()