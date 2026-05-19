import os
import json
import numpy as np

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    rad33rollpitchyaw，R_z·R_y·R_x
    roll(X)、pitch(Y)、yaw(Z)rad
    Output: 3x3 rotation matrix R
    """
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
    # R_z · R_y · R_x
    return np.dot(R_z, np.dot(R_y, R_x))

def process_single_json(json_path, output_txt_path):
    """
    processingJSON6DOF19/linesTXT
    Args:
        json_path: input JSON file path
        output_txt_path: output TXT file path
    """
    # 1. JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    camera_frames = json_data.get("CineCameraActor", {})

    # 2. frames076007601lines
    sorted_frame_keys = sorted(camera_frames.keys(), key=lambda k: int(k))

    fixed_prefix = "0.0 0.532139961 0.946026558 0.5 0.5 0.0 0.0"

    # 4. framesprocessingTXT
    with open(output_txt_path, 'w', encoding='utf-8') as f_out:
        for frame_key in sorted_frame_keys:
            frame_data = camera_frames[frame_key]
            # position: [x, y, z]
            x, y, z = frame_data["position"]
            # rotation: [roll, pitch, yaw]，rad
            roll_deg, pitch_deg, yaw_deg = frame_data["rotation"]
            roll_rad = np.radians(roll_deg)
            pitch_rad = np.radians(pitch_deg)
            yaw_rad = np.radians(yaw_deg)

            R = euler_to_rotation_matrix(roll_rad, pitch_rad, yaw_rad)
            # t = -R · [x,y,z]^T
            C = np.array([x, y, z]).reshape(3, 1)
            t = (-np.dot(R, C)).flatten()  # t = [t0, t1, t2]

            # 123lines，lines[0,0,0,1]
            extrinsic_12 = [
                R[0,0], R[0,1], R[0,2], t[0],
                R[1,0], R[1,1], R[1,2], t[1],
                R[2,0], R[2,1], R[2,2], t[2]
            ]
            line = fixed_prefix + " " + " ".join(map(str, extrinsic_12))
            f_out.write(line + "\n")

    print(f"✅ Processing completed：{os.path.basename(json_path)} → {os.path.basename(output_txt_path)}")

def main():
    """Process 3 JSON files in same directory, output to output_txts folder"""
    # 1. processing3JSON
    target_json_files = [
        "AnimeCitySuburbs_0.json",
        "AncientTowns_0.json",
        "AncientTempleEnv_0.json"
    ]
    output_folder = "output_txts"
    os.makedirs(output_folder, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 4. processingJSON
    for json_filename in target_json_files:
        json_path = os.path.join(script_dir, json_filename)
        if not os.path.exists(json_path):
            print(f"⚠️  File not found：{json_path}，skippingprocessing")
            continue
        txt_filename = os.path.splitext(json_filename)[0] + ".txt"
        output_txt_path = os.path.join(output_folder, txt_filename)
        # processingJSON
        process_single_json(json_path, output_txt_path)

    print(f"\n🎉 All files output to directory：{os.path.abspath(output_folder)}")

if __name__ == "__main__":
    main()