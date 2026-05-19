import numpy as np
import pandas as pd

def euler_to_rotation_matrix(roll, pitch, yaw):
    """rollpitchyaw，rad"""
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
    """6DOF to 4x4 extrinsic matrix"""
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    C = np.array([x, y, z]).reshape(3, 1)
    t = -np.dot(R, C)
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = R
    extrinsic_matrix[:3, 3] = t.flatten()
    return extrinsic_matrix

def process_xlsx_to_extrinsic(xlsx_input_path, output_path, is_angle_degree=False):
    """
    processingXLSX6DOF44
    Args:
        xlsx_input_path: XLSX
        output_path: CSV/XLSX
        is_angle_degree: rad，False
    """
    # XLSXopenpyxl
    df = pd.read_excel(xlsx_input_path, engine='openpyxl')
    results = []
    
    for _, row in df.iterrows():
        id_val = row['id']
        timestamp = row['timestamp']
        x, y, z = row['x'], row['y'], row['z']
        roll = np.radians(row['roll']) if is_angle_degree else row['roll']
        pitch = np.radians(row['pitch']) if is_angle_degree else row['pitch']
        yaw = np.radians(row['yaw']) if is_angle_degree else row['yaw']
        
        extrinsic_mat = six_dof_to_extrinsic_matrix(x, y, z, roll, pitch, yaw)
        result_row = [id_val, timestamp] + extrinsic_mat.flatten().tolist()
        results.append(result_row)
    
    # columns
    col_names = ['id', 'timestamp'] + [f'mat_{i}_{j}' for i in range(4) for j in range(4)]
    result_df = pd.DataFrame(results, columns=col_names)
    # CSVto_excel(output_path, engine='openpyxl')XLSX
    result_df.to_csv(output_path, index=False)
    print(f"Extrinsic matrix saved to：{output_path}")

# lines
if __name__ == "__main__":
    INPUT_XLSX = "/path/to/locations_raw.xlsx"
    OUTPUT_FILE = "output_extrinsic_matrix.csv"
    process_xlsx_to_extrinsic(INPUT_XLSX, OUTPUT_FILE)