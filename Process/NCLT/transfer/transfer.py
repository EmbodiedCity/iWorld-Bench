#!/usr/bin/env python3

import os
import numpy as np
from tqdm import tqdm

def euler_to_rotation_matrix(roll, pitch, yaw):
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def euler_to_quaternion(roll, pitch, yaw):
    half_roll = roll / 2.0
    half_pitch = pitch / 2.0
    half_yaw = yaw / 2.0
    
    cr = np.cos(half_roll)
    sr = np.sin(half_roll)
    cp = np.cos(half_pitch)
    sp = np.sin(half_pitch)
    cy = np.cos(half_yaw)
    sy = np.sin(half_yaw)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if norm > 0:
        qw /= norm
        qx /= norm
        qy /= norm
        qz /= norm
    
    return qw, qx, qy, qz

def six_dof_to_extrinsic_matrix(x, y, z, roll, pitch, yaw):
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    
    C = np.array([x, y, z]).reshape(3, 1)
    
    t = -np.dot(R, C)
    
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = R
    extrinsic_matrix[:3, 3] = t.flatten()
    
    return extrinsic_matrix

def six_dof_to_seven_params(x, y, z, roll, pitch, yaw):
    extrinsic_matrix = six_dof_to_extrinsic_matrix(x, y, z, roll, pitch, yaw)
    
    tx, ty, tz = extrinsic_matrix[:3, 3]
    
    R = extrinsic_matrix[:3, :3]
    
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    
    norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if norm > 0:
        qw /= norm
        qx /= norm
        qy /= norm
        qz /= norm
    
    return [tx, ty, tz, qw, qx, qy, qz]

def process_single_folder(folder_path):
    input_file = os.path.join(folder_path, "six_Dof.txt")
    
    if not os.path.exists(input_file):
        print(f"warn no six_Dof.txt {folder_path}")
        return False
    
    seven_output = os.path.join(folder_path, "seven_element.txt")
    extrinsic_output = os.path.join(folder_path, "extrinsics_matrix.txt")
    
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"err read {input_file}: {e}")
        return False
    
    seven_elements = []
    extrinsics_matrices = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        try:
            parts = list(map(float, line.split()))
            if len(parts) != 6:
                continue
            
            x, y, z, roll, pitch, yaw = parts
            
            seven_param = six_dof_to_seven_params(x, y, z, roll, pitch, yaw)
            seven_elements.append(seven_param)
            
            extrinsic_matrix = six_dof_to_extrinsic_matrix(x, y, z, roll, pitch, yaw)
            extrinsics_matrices.append(extrinsic_matrix.flatten())
            
        except Exception:
            continue
    
    try:
        with open(seven_output, 'w') as f:
            for params in seven_elements:
                line_str = ' '.join([f"{val:.10f}" for val in params])
                f.write(line_str + '\n')
    except Exception as e:
        print(f"err write {seven_output}: {e}")
        return False
    
    try:
        with open(extrinsic_output, 'w') as f:
            for matrix in extrinsics_matrices:
                line_str = ' '.join([f"{val:.10f}" for val in matrix])
                f.write(line_str + '\n')
    except Exception as e:
        print(f"err write {extrinsic_output}: {e}")
        return False
    
    return True

def find_six_dof_folders(root_dir):
    six_dof_folders = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "six_Dof.txt" in filenames:
            six_dof_folders.append(dirpath)
    
    return six_dof_folders

def main():
    root_directory = r"/path/to/local/data"
    
    if not os.path.exists(root_directory):
        print(f"error: root {root_directory}")
        return
    folders = find_six_dof_folders(root_directory)
    if not folders:
        print("error: no six_Dof.txt under root")
        return
    print(f"folders={len(folders)}")
    ok = fail = 0
    for folder in tqdm(folders, desc="convert", unit="dir"):
        if process_single_folder(folder):
            ok += 1
        else:
            fail += 1
    print(f"done ok={ok} fail={fail}")

if __name__ == "__main__":
    main()