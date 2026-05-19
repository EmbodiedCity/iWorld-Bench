import numpy as np
import os
import glob
from tqdm import tqdm


def quaternion_to_rotation_matrix(q):
    qw, qx, qy, qz = q
    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ]
    )


def rotation_matrix_to_euler_angles(R):
    if abs(R[2, 0]) > 0.9999:
        yaw = 0
        if R[2, 0] < 0:
            pitch = np.pi / 2
            roll = np.arctan2(R[0, 1], R[0, 2])
        else:
            pitch = -np.pi / 2
            roll = np.arctan2(-R[0, 1], -R[0, 2])
    else:
        pitch = -np.arcsin(R[2, 0])
        roll = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
        yaw = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))
    return roll, pitch, yaw


def transform_coordinate_system(tx, ty, tz, qw, qx, qy, qz):
    tx_new, ty_new, tz_new = tx, -tz, ty
    q_orig = np.array([qw, qx, qy, qz])
    angle = -np.pi / 2
    q_rot = np.array([np.cos(angle / 2), np.sin(angle / 2), 0, 0])

    def quaternion_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([w, x, y, z])

    q_new = quaternion_multiply(q_rot, q_orig)
    q_new = q_new / np.linalg.norm(q_new)
    return tx_new, ty_new, tz_new, q_new[0], q_new[1], q_new[2], q_new[3]


def process_pose_file(input_file, output_dir):
    data = []
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                values = list(map(float, line.split()))
                if len(values) >= 7:
                    data.append(values[:7])
    if not data:
        print(f"warn empty {input_file}")
        return
    output_extrinsics = os.path.join(output_dir, "extrinsics_matrix.txt")
    output_seven = os.path.join(output_dir, "seven_element.txt")
    output_six = os.path.join(output_dir, "six_DoF.txt")
    with open(output_extrinsics, "w") as f_ext, open(output_seven, "w") as f_seven, open(
        output_six, "w"
    ) as f_six:
        for row in data:
            tx, ty, tz, qx, qy, qz, qw = row
            tx_new, ty_new, tz_new, qw_new, qx_new, qy_new, qz_new = transform_coordinate_system(
                tx, ty, tz, qw, qx, qy, qz
            )
            f_seven.write(
                f"{tx_new:.10e} {ty_new:.10e} {tz_new:.10e} "
                f"{qw_new:.10e} {qx_new:.10e} {qy_new:.10e} {qz_new:.10e}\n"
            )
            R = quaternion_to_rotation_matrix([qw_new, qx_new, qy_new, qz_new])
            extrinsic_matrix = np.eye(4)
            extrinsic_matrix[:3, :3] = R
            extrinsic_matrix[:3, 3] = [tx_new, ty_new, tz_new]
            flat_matrix = extrinsic_matrix.flatten()
            f_ext.write(" ".join(f"{flat_matrix[j]:.10e}" for j in range(16)) + "\n")
            roll, pitch, yaw = rotation_matrix_to_euler_angles(R)
            f_six.write(
                f"{tx_new:.10e} {ty_new:.10e} {tz_new:.10e} {roll:.10e} {pitch:.10e} {yaw:.10e}\n"
            )


def find_pose_files(base_dir):
    pose_files = []
    scenes = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith(".")
    ]
    print(f"scenes={len(scenes)}")
    for scene in scenes:
        scene_path = os.path.join(base_dir, scene)
        easy_path = os.path.join(scene_path, "Easy")
        if not os.path.exists(easy_path):
            print(f"warn no Easy {scene}")
            continue
        p_folders = []
        for item in os.listdir(easy_path):
            item_path = os.path.join(easy_path, item)
            if os.path.isdir(item_path) and item.startswith("P"):
                p_folders.append(item)
        if not p_folders:
            print(f"warn no P* {scene}")
            continue
        p_folders.sort()
        for p_folder in p_folders:
            p_path = os.path.join(easy_path, p_folder)
            pose_path = os.path.join(p_path, "pose_left.txt")
            if os.path.exists(pose_path):
                pose_files.append((scene, p_folder, pose_path))
                print(f"pose {scene}/{p_folder}")
            else:
                print(f"warn no pose_left {scene}/{p_folder}")
    return pose_files


def main():
    tartanair_base = "/path/to/local/data"
    output_base = "/path/to/local/data"
    os.makedirs(output_base, exist_ok=True)
    print(f"scan {tartanair_base}")
    pose_files = find_pose_files(tartanair_base)
    if not pose_files:
        pattern = os.path.join(tartanair_base, "**", "pose_left.txt")
        all_files = glob.glob(pattern, recursive=True)
        if not all_files:
            print("error: no pose_left.txt")
            return
        print(f"glob n={len(all_files)}")
        for file_path in all_files:
            parts = file_path.replace("\\", "/").split("/")
            scene_name = p_folder = None
            for i, part in enumerate(parts):
                if part in ("Easy", "Hard"):
                    if i > 0:
                        scene_name = parts[i - 1]
                    if i + 1 < len(parts):
                        p_folder = parts[i + 1]
                    break
            if scene_name and p_folder:
                pose_files.append((scene_name, p_folder, file_path))
    print(f"poses={len(pose_files)}")
    for scene_name, p_folder, pose_file in tqdm(pose_files, desc="convert"):
        output_dir = os.path.join(output_base, f"TartanAir_{scene_name}_{p_folder}")
        os.makedirs(output_dir, exist_ok=True)
        try:
            process_pose_file(pose_file, output_dir)
            print(f"ok {scene_name}/{p_folder}")
        except Exception as e:
            print(f"fail {scene_name}/{p_folder}: {e}")
    print(f"done -> {output_base}")


if __name__ == "__main__":
    main()
