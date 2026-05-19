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

def six_dof_to_extrinsic_matrix(x, y, z, roll, pitch, yaw):
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    C = np.array([x, y, z]).reshape(3, 1)
    t = -np.dot(R, C)
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = R
    extrinsic_matrix[:3, 3] = t.flatten()
    return extrinsic_matrix

def euler_to_quaternion(roll, pitch, yaw):
    half_roll = roll / 2
    half_pitch = pitch / 2
    half_yaw = yaw / 2
    
    q_x = [np.cos(half_roll), np.sin(half_roll), 0, 0]
    q_y = [np.cos(half_pitch), 0, np.sin(half_pitch), 0]
    q_z = [np.cos(half_yaw), 0, 0, np.sin(half_yaw)]
    
    def quat_mult(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return [
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ]
    
    q = quat_mult(q_x, quat_mult(q_y, q_z))
    q_norm = np.linalg.norm(q)
    return [x / q_norm for x in q]

def six_dof_to_seven_params(x, y, z, roll, pitch, yaw):
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    C = np.array([x, y, z]).reshape(3, 1)
    t = -np.dot(R, C).flatten()
    q_w, q_x, q_y, q_z = euler_to_quaternion(roll, pitch, yaw)
    return [t[0], t[1], t[2], q_w, q_x, q_y, q_z]

def process_all_files(root_dir):
    six_dof_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == 'six_DoF.txt':
                six_dof_files.append(os.path.join(dirpath, filename))
    
    for file_path in tqdm(six_dof_files, desc="Processing files"):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        seven_elements = []
        extrinsics_matrices = []
        for line in lines:
            parts = list(map(float, line.strip().split()))
            x, y, z, roll, pitch, yaw = parts
            extrinsic = six_dof_to_extrinsic_matrix(x, y, z, roll, pitch, yaw)
            seven_param = six_dof_to_seven_params(x, y, z, roll, pitch, yaw)
            extrinsics_matrices.append(extrinsic.flatten())
            seven_elements.append(seven_param)
        
        dir_path = os.path.dirname(file_path)
        seven_output = os.path.join(dir_path, 'seven_element.txt')
        with open(seven_output, 'w') as f:
            for params in seven_elements:
                f.write(' '.join(map(str, params)) + '\n')
        
        extrinsic_output = os.path.join(dir_path, 'extrinsics_matrix.txt')
        with open(extrinsic_output, 'w') as f:
            for mat in extrinsics_matrices:
                f.write(' '.join(map(str, mat)) + '\n')

if __name__ == "__main__":
    root_directory = r"/path/to/local/data"
    process_all_files(root_directory)