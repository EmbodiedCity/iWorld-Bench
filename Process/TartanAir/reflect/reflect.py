import os
import sys
import logging
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from concurrent.futures import ProcessPoolExecutor
import re

DATASET_NAME = "SpatialVID_TartanAir"
FRAME_COUNT_PER_SEG = 81
COMPRESSED_RESOLUTION = (832, 480)
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
PASSED_VIDEOS_LOG = os.path.join(LOG_DIR, "passed_videos.txt")
PASSED_CAMERAS_LOG = os.path.join(LOG_DIR, "passed_cameras.txt")


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


def get_paired_data(cityworld_root):
    paired_data = []
    rgb_root = os.path.join(cityworld_root, "rgb", "UAV_front_new")
    pose_root = os.path.join(cityworld_root, "pose", "UAV_front_new")
    if not os.path.exists(rgb_root):
        logging.error("no rgb root %s", rgb_root)
        return paired_data
    if not os.path.exists(pose_root):
        logging.error("no pose root %s", pose_root)
        return paired_data
    rgb_dirs = [
        d
        for d in os.listdir(rgb_root)
        if d.startswith("TartanAir_") and os.path.isdir(os.path.join(rgb_root, d))
    ]
    logging.info("rgb_dirs=%d", len(rgb_dirs))
    for rgb_dir in rgb_dirs:
        rgb_data_path = os.path.join(rgb_root, rgb_dir)
        pose_data_path = os.path.join(pose_root, rgb_dir)
        if not os.path.exists(pose_data_path):
            logging.warning("skip no pose %s", rgb_dir)
            continue
        path_txt = os.path.join(rgb_data_path, "path.txt")
        if not os.path.exists(path_txt):
            logging.warning("skip no path.txt %s", rgb_dir)
            continue
        extrinsic_files = {
            "extrinsics_matrix": os.path.join(pose_data_path, "extrinsics_matrix.txt"),
            "seven_element": os.path.join(pose_data_path, "seven_element.txt"),
            "six_DoF": os.path.join(pose_data_path, "six_DoF.txt"),
        }
        extrinsic_path, extrinsic_type = None, None
        for ext_type, ext_path in extrinsic_files.items():
            if os.path.exists(ext_path):
                extrinsic_path, extrinsic_type = ext_path, ext_type
                break
        if not extrinsic_path:
            logging.warning("skip no extrinsic %s", rgb_dir)
            continue
        calib_txt = os.path.join(pose_data_path, "calibration.txt")
        if not os.path.exists(calib_txt):
            logging.warning("default calib %s", rgb_dir)
            create_default_calibration(calib_txt)
        paired_data.append((path_txt, calib_txt, extrinsic_path, extrinsic_type, rgb_dir))
        logging.info("pair %s", rgb_dir)
    return paired_data


def create_default_calibration(calib_path):
    default_intrinsics = [320.0, 320.0, 320.0, 240.0]
    with open(calib_path, "w", encoding="utf-8") as f:
        f.write("# default intrinsics TartanAir 640x480\n")
        f.write(
            f"{default_intrinsics[0]} {default_intrinsics[1]} "
            f"{default_intrinsics[2]} {default_intrinsics[3]}\n"
        )
        f.write("# Image Size: 640 480\n")
    logging.info("wrote default calib %s", calib_path)


def parse_image_resolution_from_calibration(calib_path):
    try:
        with open(calib_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        for marker in ("# Image Size", "# Resolution", "# Size"):
            for line in lines:
                if marker in line:
                    clean_line = re.sub(r"[^0-9\s]", "", line.split(marker)[1].strip())
                    nums = clean_line.split()
                    if len(nums) >= 2:
                        try:
                            return int(nums[0]), int(nums[1])
                        except ValueError:
                            continue
        for line in lines:
            clean_line = re.sub(r"[^0-9\s]", "", line)
            nums = clean_line.split()
            if len(nums) == 2 and all(n.isdigit() for n in nums):
                return int(nums[0]), int(nums[1])
        logging.warning("calib size fallback 640x480 %s", calib_path)
        return 640, 480
    except Exception as e:
        logging.error("calib parse %s: %s", calib_path, e)
        return 640, 480


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
        intrinsics = [320.0, 320.0, 320.0, 240.0]
    w, h = parse_image_resolution_from_calibration(calib_path)
    return [intrinsics[0] / w, intrinsics[1] / h, intrinsics[2] / w, intrinsics[3] / h]


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
        logging.warning("extrinsic short %d %s", len(lines), extrinsic_path)
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
    logging.info("parsed %d %s", len(extrinsic_matrices), extrinsic_path)
    return extrinsic_matrices


def convert_windows_to_wsl_path(windows_path):
    if windows_path.startswith("/path/to/local/data/"):
        return "/path/to/local/data/" + windows_path[3:].replace("\\", "/")
    if windows_path.startswith("/path/to/local/data/"):
        return "/path/to/local/data/" + windows_path[3:].replace("\\", "/")
    if windows_path.startswith("/path/to/local/data/"):
        return "/path/to/local/data/" + windows_path[3:].replace("\\", "/")
    if windows_path.startswith("/path/to/local/data/"):
        return "/path/to/local/data/" + windows_path[3:].replace("\\", "/")
    return windows_path.replace("\\", "/")


def process_image_sequence(
    image_paths, video_dir, data_prefix, passed_videos, total_ext_frames, calib_path
):
    video_key = f"{data_prefix}_images"
    if video_key in passed_videos:
        return []
    target_res = parse_image_resolution_from_calibration(calib_path)
    nimg = len(image_paths)
    nfr = min(total_ext_frames, nimg)
    seg_count = nfr // FRAME_COUNT_PER_SEG
    if seg_count == 0:
        logging.warning("skip %s frames<81", data_prefix)
        return []
    fps = 30
    output_segs = []
    for seg_idx in range(seg_count):
        seg_path = os.path.join(video_dir, f"{data_prefix}_video_{seg_idx + 1:03d}.mp4")
        out = cv2.VideoWriter(
            seg_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, COMPRESSED_RESOLUTION
        )
        if not out.isOpened():
            logging.error("writer fail %s", seg_path)
            continue
        start = seg_idx * FRAME_COUNT_PER_SEG
        written = 0
        for i in range(start, start + FRAME_COUNT_PER_SEG):
            if i >= len(image_paths):
                break
            img = cv2.imread(image_paths[i])
            if img is None:
                continue
            fo = cv2.resize(img, target_res, interpolation=cv2.INTER_LINEAR)
            fc = cv2.resize(fo, COMPRESSED_RESOLUTION, interpolation=cv2.INTER_AREA)
            out.write(fc)
            written += 1
        out.release()
        if written == FRAME_COUNT_PER_SEG and os.path.getsize(seg_path) > 0:
            output_segs.append(seg_path)
            logging.info("video %s", seg_path)
        elif os.path.exists(seg_path):
            os.remove(seg_path)
    write_log(PASSED_VIDEOS_LOG, video_key)
    return output_segs


def process_camera(intrinsics, extrinsics, output_dir, prefix, passed_cameras, seg_count):
    if prefix in passed_cameras:
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
        else:
            os.remove(seg_path)
    write_log(PASSED_CAMERAS_LOG, prefix)
    return output_segs


def main(cityworld_root, output_root):
    init_log_file(PASSED_VIDEOS_LOG)
    init_log_file(PASSED_CAMERAS_LOG)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(LOG_DIR, "reflect_tartanair.log")),
        ],
    )
    passed_videos = read_log(PASSED_VIDEOS_LOG)
    passed_cameras = read_log(PASSED_CAMERAS_LOG)
    remake_dir = os.path.join(output_root, DATASET_NAME)
    video_dir = os.path.join(remake_dir, "videos")
    camera_dir = os.path.join(remake_dir, "cameras")
    os.makedirs(remake_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(camera_dir, exist_ok=True)
    logging.info("out %s", remake_dir)
    paired_data = get_paired_data(cityworld_root)
    total_v = total_c = 0
    for path_txt, calib_txt, ext_txt, ext_type, data_prefix in paired_data:
        try:
            vk, ck = f"{data_prefix}_images", data_prefix
            hv, hc = vk in passed_videos, ck in passed_cameras
            if hv and hc:
                continue
            with open(path_txt, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if not ln.startswith("#") and ln.strip()]
            if not lines:
                continue
            valid = [p for p in (convert_windows_to_wsl_path(ln) for ln in lines) if os.path.exists(p)]
            if not valid:
                continue
            intr = parse_intrinsics(calib_txt)
            extm = parse_extrinsics(ext_txt, ext_type)
            nseg = len(extm) // FRAME_COUNT_PER_SEG
            with ProcessPoolExecutor(max_workers=2) as ex:
                vf = ex.submit(
                    process_image_sequence,
                    valid,
                    video_dir,
                    data_prefix,
                    passed_videos,
                    len(extm),
                    calib_txt,
                ) if not hv else ex.submit(list)
                cf = ex.submit(
                    process_camera, intr, extm, camera_dir, data_prefix, passed_cameras, nseg
                ) if not hc else ex.submit(list)
                vs, cs = vf.result(), cf.result()
            total_v += len(vs)
            total_c += len(cs)
            logging.info("%s v=%d c=%d", data_prefix, len(vs), len(cs))
        except Exception as e:
            logging.error("%s %s", data_prefix, e, exc_info=True)
    logging.info("done videos=%d cameras=%d", total_v, total_c)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        cityworld_root, output_root = "/path/to/local/data", "/path/to/local/data"
    elif len(sys.argv) == 3:
        cityworld_root, output_root = sys.argv[1], sys.argv[2]
    else:
        print("usage: reflect.py [cityworld_root] [output_root]")
        sys.exit(1)
    if not os.path.isdir(cityworld_root):
        print("error: cityworld_root missing")
        sys.exit(1)
    main(cityworld_root, output_root)
