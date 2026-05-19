import os
import sys
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R
import re

DATASET_NAME = "SpatialVID_NCLT"
FRAME_COUNT_PER_SEG = 81
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
PASSED_CAMERAS_LOG = os.path.join(LOG_DIR, "passed_cameras_only.txt")


def init_log_file(log_path):
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("")


def read_log(log_path):
    if not os.path.exists(log_path):
        return set()
    with open(log_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def write_log(log_path, content):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{content}\n")


def get_paired_data_for_camera_only(cityworld_root):
    paired_data = []
    if not os.path.exists(cityworld_root):
        logging.error("root missing %s", cityworld_root)
        return paired_data
    view = "UGV_front"
    rgb_view_path = os.path.join(cityworld_root, "rgb", view)
    pose_view_path = os.path.join(cityworld_root, "pose", view)
    if not os.path.exists(rgb_view_path):
        logging.error("no rgb %s", rgb_view_path)
        return paired_data
    if not os.path.exists(pose_view_path):
        logging.error("no pose %s", pose_view_path)
        return paired_data
    rgb_sequences = []
    for item in os.listdir(rgb_view_path):
        p = os.path.join(rgb_view_path, item)
        if os.path.isdir(p):
            rgb_sequences.append(item)
    logging.info("rgb seqs=%d", len(rgb_sequences))
    for seq_name in rgb_sequences:
        rgb_seq_path = os.path.join(rgb_view_path, seq_name)
        pose_seq_path = os.path.join(pose_view_path, seq_name)
        if not os.path.exists(pose_seq_path):
            continue
        path_txt = os.path.join(rgb_seq_path, "path.txt")
        if not os.path.exists(path_txt):
            continue
        extrinsic_files = {
            "extrinsics_matrix": os.path.join(pose_seq_path, "extrinsics_matrix.txt"),
            "seven_element": os.path.join(pose_seq_path, "seven_element.txt"),
            "six_DoF": os.path.join(pose_seq_path, "six_DoF.txt"),
        }
        extrinsic_path, extrinsic_type = None, None
        for ext_type, ext_path in extrinsic_files.items():
            if os.path.exists(ext_path):
                extrinsic_path, extrinsic_type = ext_path, ext_type
                break
        if not extrinsic_path:
            continue
        calib_txt = os.path.join(pose_seq_path, "calibration.txt")
        if not os.path.exists(calib_txt):
            continue
        try:
            with open(path_txt, "r", encoding="utf-8") as f:
                image_count = sum(1 for line in f if line.strip())
            with open(extrinsic_path, "r", encoding="utf-8") as f:
                pose_count = sum(1 for line in f if line.strip())
            if image_count != pose_count:
                logging.warning("skip mismatch %s img=%d pose=%d", seq_name, image_count, pose_count)
                continue
        except Exception as e:
            logging.error("skip %s: %s", seq_name, e)
            continue
        data_prefix = f"{view}_{seq_name}"
        paired_data.append((path_txt, calib_txt, extrinsic_path, extrinsic_type, data_prefix))
        logging.info("pair %s", data_prefix)
    return paired_data


def parse_image_resolution_from_calibration(calib_path):
    try:
        with open(calib_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        for line in lines:
            if "# Image Size" in line:
                clean_line = re.sub(r"[^0-9\s]", "", line.split("# Image Size")[1].strip())
                nums = clean_line.split()
                if len(nums) >= 2:
                    try:
                        return int(nums[0]), int(nums[1])
                    except ValueError:
                        continue
        return 1280, 1024
    except Exception:
        return 1280, 1024


def parse_intrinsics(calib_path):
    with open(calib_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if not line.startswith("#") and line.strip()]
    intrinsics = []
    for line in lines:
        nums = list(map(float, line.split()))
        if len(nums) == 4:
            intrinsics = nums[:4]
            break
        if len(nums) == 9:
            intrinsics = [nums[0], nums[4], nums[2], nums[5]]
            break
    if not intrinsics:
        intrinsics = [640.0, 640.0, 640.0, 512.0]
    img_size = parse_image_resolution_from_calibration(calib_path)
    return [
        intrinsics[0] / img_size[0],
        intrinsics[1] / img_size[1],
        intrinsics[2] / img_size[0],
        intrinsics[3] / img_size[1],
    ]


def seven_element_to_extrinsic(seven_elements):
    rx, ry, rz, tx, ty, tz, s = seven_elements
    rot = R.from_euler("xyz", [rx, ry, rz]).as_matrix()
    trans = np.array([tx, ty, tz]).reshape(3, 1)
    return np.hstack([rot * s, trans])


def six_dof_to_extrinsic(six_dof):
    rx, ry, rz, tx, ty, tz = six_dof
    rot = R.from_euler("xyz", [rx, ry, rz]).as_matrix()
    trans = np.array([tx, ty, tz]).reshape(3, 1)
    return np.hstack([rot, trans])


def parse_extrinsics(extrinsic_path, extrinsic_type):
    extrinsic_matrices = []
    with open(extrinsic_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if not line.startswith("#") and line.strip()]
    if len(lines) < FRAME_COUNT_PER_SEG:
        return extrinsic_matrices
    for line in lines:
        nums = list(map(float, line.split()))
        if extrinsic_type == "extrinsics_matrix":
            if len(nums) >= 16:
                mat = np.array(nums[:16]).reshape(4, 4)[:3, :]
            else:
                continue
        elif extrinsic_type == "seven_element":
            if len(nums) >= 7:
                mat = seven_element_to_extrinsic(nums[:7])
            else:
                continue
        elif extrinsic_type == "six_DoF":
            if len(nums) >= 6:
                mat = six_dof_to_extrinsic(nums[:6])
            else:
                continue
        else:
            raise ValueError(extrinsic_type)
        extrinsic_matrices.append(mat)
    return extrinsic_matrices


def process_camera_only(intrinsics, extrinsics, output_dir, prefix, passed_cameras):
    if prefix in passed_cameras:
        return []
    n = len(extrinsics)
    seg_count = n // FRAME_COUNT_PER_SEG
    if seg_count == 0:
        return []
    output_segs = []
    for seg_idx in range(seg_count):
        seg_path = os.path.join(output_dir, f"{prefix}_camera_{seg_idx + 1:03d}.txt")
        with open(seg_path, "w", encoding="utf-8") as f:
            start_line = seg_idx * FRAME_COUNT_PER_SEG
            for line_idx in range(start_line, start_line + FRAME_COUNT_PER_SEG):
                ext_mat = extrinsics[line_idx]
                line_data = [
                    0.0,
                    intrinsics[0],
                    intrinsics[1],
                    intrinsics[2],
                    intrinsics[3],
                    0.0,
                    0.0,
                    ext_mat[0, 0],
                    ext_mat[0, 1],
                    ext_mat[0, 2],
                    ext_mat[1, 0],
                    ext_mat[1, 1],
                    ext_mat[1, 2],
                    ext_mat[2, 0],
                    ext_mat[2, 1],
                    ext_mat[2, 2],
                    ext_mat[0, 3],
                    ext_mat[1, 3],
                    ext_mat[2, 3],
                ]
                f.write(" ".join(f"{x:.9f}" for x in line_data) + "\n")
        if os.path.getsize(seg_path) > 0:
            output_segs.append(seg_path)
        elif os.path.exists(seg_path):
            os.remove(seg_path)
    write_log(PASSED_CAMERAS_LOG, prefix)
    return output_segs


def main(cityworld_root, output_root):
    init_log_file(PASSED_CAMERAS_LOG)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(LOG_DIR, "reflect_nclt_camera_only.log")),
        ],
    )
    passed_cameras = read_log(PASSED_CAMERAS_LOG)
    remake_dir = os.path.join(output_root, DATASET_NAME)
    camera_dir = os.path.join(remake_dir, "cameras")
    os.makedirs(camera_dir, exist_ok=True)
    logging.info("out %s", camera_dir)
    paired_data = get_paired_data_for_camera_only(cityworld_root)
    if not paired_data:
        logging.error("no data")
        return
    total = 0
    for _, calib_txt, ext_txt, ext_type, data_prefix in paired_data:
        try:
            if data_prefix in passed_cameras:
                continue
            intr = parse_intrinsics(calib_txt)
            extm = parse_extrinsics(ext_txt, ext_type)
            if not extm:
                continue
            segs = process_camera_only(intr, extm, camera_dir, data_prefix, passed_cameras)
            total += len(segs)
            logging.info("%s files=%d", data_prefix, len(segs))
        except Exception as e:
            logging.error("%s %s", data_prefix, e, exc_info=True)
    logging.info("total camera files=%d", total)


def batch_process_all():
    cityworld_root = "/path/to/local/data"
    output_root = "/path/to/local/data"
    print(f"nclt cameras-only in={cityworld_root} out={output_root}")
    if not os.path.isdir(cityworld_root):
        print("error: missing root")
        sys.exit(1)
    main(cityworld_root, output_root)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        cr, oroot = sys.argv[1], sys.argv[2]
        if not os.path.isdir(cr):
            print("error: missing root")
            sys.exit(1)
        main(cr, oroot)
    else:
        batch_process_all()
