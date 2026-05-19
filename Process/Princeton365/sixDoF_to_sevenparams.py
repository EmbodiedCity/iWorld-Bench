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

def euler_to_quaternion(roll, pitch, yaw):
    """rollpitchyaw，rad"""
    half_roll = roll / 2
    half_pitch = pitch / 2
    half_yaw = yaw / 2
    
    q_x = [np.cos(half_roll), np.sin(half_roll), 0, 0]
    q_y = [np.cos(half_pitch), 0, np.sin(half_pitch), 0]
    q_z = [np.cos(half_yaw), 0, 0, np.sin(half_yaw)]
    
    # Quaternion multiplication
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
    """6DOF to 7 params (quaternion + translation, translation consistent with extrinsic matrix t)"""
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    C = np.array([x, y, z]).reshape(3, 1)
    t = -np.dot(R, C).flatten()
    q_w, q_x, q_y, q_z = euler_to_quaternion(roll, pitch, yaw)
    # 4. 7Args: + tC
    return [q_w, q_x, q_y, q_z, t[0], t[1], t[2]]

def process_xlsx_to_seven_params(xlsx_input_path, output_path, is_angle_degree=False):
    """
    processingXLSX6DOF7
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
        
        seven_params = six_dof_to_seven_params(x, y, z, roll, pitch, yaw)
        result_row = [id_val, timestamp] + seven_params
        results.append(result_row)
    
    # columns
    col_names = ['id', 'timestamp', 'q_w', 'q_x', 'q_y', 'q_z', 'tx', 'ty', 'tz']
    result_df = pd.DataFrame(results, columns=col_names)
    # CSVto_excel(output_path, engine='openpyxl')XLSX
    result_df.to_csv(output_path, index=False)
    print(f"7-params saved to：{output_path}")

# lines
if __name__ == "__main__":
    INPUT_XLSX = "/path/to/locations_raw.xlsx"
    OUTPUT_FILE = "output_seven_params.csv"
    process_xlsx_to_seven_params(INPUT_XLSX, OUTPUT_FILE)