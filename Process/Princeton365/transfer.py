import os
import numpy as np
import pandas as pd

def euler_to_rotation_matrix(roll, pitch, yaw):
    """rad33rollpitchyaw"""
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
    return np.dot(R_z, np.dot(R_y, R_x))

def euler_to_quaternion(roll, pitch, yaw):
    """rad[qw, qx, qy, qz]"""
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
    return [x / np.linalg.norm(q) for x in q]

def process_xlsx_to_calibration_files(input_xlsx, output_folder="calibration_files", is_angle_degree=False):
    """
    XLSX6DOF，4TXToutput_folder
    Args:
        input_xlsx: XLSX
        output_folder: 
        is_angle_degree: rad，False
    """
    # 1. does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    file_paths = {
        "six_DoF": os.path.join(output_folder, "six_DoF.txt"),
        "extrinsics_matrix": os.path.join(output_folder, "extrinsics_matrix.txt"),
        "seven_element": os.path.join(output_folder, "seven_element.txt"),
        "calibration": os.path.join(output_folder, "calibration.txt")
    }

    # 3. XLSX
    df = pd.read_excel(input_xlsx, engine='openpyxl')

    # 4. linesprocessing，TXT
    with open(file_paths["six_DoF"], 'w') as f_six, \
         open(file_paths["extrinsics_matrix"], 'w') as f_ext, \
         open(file_paths["seven_element"], 'w') as f_seven:
        
        for _, row in df.iterrows():
            # 6DOFsix_DoF.txt
            x_raw = row['x']
            y_raw = row['y']
            z_raw = row['z']
            roll_raw = row['roll']
            pitch_raw = row['pitch']
            yaw_raw = row['yaw']

            roll = np.radians(roll_raw) if is_angle_degree else roll_raw
            pitch = np.radians(pitch_raw) if is_angle_degree else pitch_raw
            yaw = np.radians(yaw_raw) if is_angle_degree else yaw_raw


            # ---------------------- six_DoF.txt ----------------------
            # x y z roll pitch yaw
            f_six.write(f"{x_raw} {y_raw} {z_raw} {roll_raw} {pitch_raw} {yaw_raw}\n")


            # ---------------------- extrinsics_matrix.txt ----------------------
            # R + t = -R·CCx/y/z
            R = euler_to_rotation_matrix(roll, pitch, yaw)
            C = np.array([x_raw, y_raw, z_raw]).reshape(3, 1)
            t = (-np.dot(R, C)).flatten()  # t = [t0, t1, t2]
            
            # r11 r12 r13 t0 r21 r22 r23 t1 r31 r32 r33 t2 0 0 0 1
            r11, r12, r13 = R[0]
            r21, r22, r23 = R[1]
            r31, r32, r33 = R[2]
            t0, t1, t2 = t
            f_ext.write(f"{r11} {r12} {r13} {t0} {r21} {r22} {r23} {t1} {r31} {r32} {r33} {t2} 0 0 0 1\n")


            # ---------------------- seven_element.txt ----------------------
            qw, qx, qy, qz = euler_to_quaternion(roll, pitch, yaw)
            
            # tx ty tz qw qx qy qztx=t0, ty=t1, tz=t2
            f_seven.write(f"{t0} {t1} {t2} {qw} {qx} {qy} {qz}\n")


    # ---------------------- calibration.txt ----------------------
    with open(file_paths["calibration"], 'w') as f_calib:
        f_calib.write("# Intrinsic Matrix (3x3)\n")
        f_calib.write("1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0\n")


    print(f"✅ All files generated to directory：{os.path.abspath(output_folder)}")
    print(f"📂 Contains files：{os.listdir(output_folder)}")


# ------------------------------
# ------------------------------
if __name__ == "__main__":
    INPUT_XLSX = "/path/to/locations_raw.xlsx"
    process_xlsx_to_calibration_files(INPUT_XLSX, is_angle_degree=False)
