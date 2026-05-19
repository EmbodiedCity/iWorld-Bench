import os
import sys
import logging
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from concurrent.futures import ProcessPoolExecutor
import re


def count_lines_in_file(file_path):
    if not os.path.exists(file_path):
        return 0
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def downsample_pose_file_strict(pose_file_path, image_count, pose_count):
    if pose_count <= image_count:
        print(f"skip downsample: poses={pose_count} images={image_count}")
        return pose_count
    with open(pose_file_path, "r", encoding="utf-8") as f:
        all_lines = [line.strip() for line in f if line.strip()]
    sampled_lines = []
    if image_count == 1:
        sampled_lines.append(all_lines[pose_count // 2])
    elif image_count == 2:
        sampled_lines.append(all_lines[0])
        sampled_lines.append(all_lines[-1])
    else:
        step = (pose_count - 1) / (image_count - 1)
        sampled_lines.append(all_lines[0])
        for i in range(1, image_count - 1):
            index = round(i * step)
            index = max(1, min(index, pose_count - 2))
            sampled_lines.append(all_lines[index])
        sampled_lines.append(all_lines[-1])
    with open(pose_file_path, "w", encoding="utf-8") as f:
        for line in sampled_lines:
            f.write(f"{line}\n")
    return len(sampled_lines)


def create_backup(pose_file_path, backup_dir):
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = os.path.join(backup_dir, os.path.basename(pose_file_path))
    if os.path.exists(pose_file_path):
        with open(pose_file_path, "r", encoding="utf-8") as src:
            content = src.read()
        with open(backup_path, "w", encoding="utf-8") as dst:
            dst.write(content)
        return True
    return False


def check_pose_alignment(rgb_seq_path, pose_seq_path):
    if not os.path.exists(rgb_seq_path):
        return False, f"missing rgb: {rgb_seq_path}"
    if not os.path.exists(pose_seq_path):
        return False, f"missing pose: {pose_seq_path}"
    path_txt = os.path.join(rgb_seq_path, "path.txt")
    if not os.path.exists(path_txt):
        return False, f"missing path.txt: {path_txt}"
    image_count = count_lines_in_file(path_txt)
    if image_count == 0:
        return False, "empty path.txt"
    pose_files = ["extrinsics_matrix.txt", "seven_element.txt", "six_DoF.txt"]
    backup_dir = os.path.join(pose_seq_path, "original_backup")
    pose_counts = []
    for file_name in pose_files:
        pose_file_path = os.path.join(pose_seq_path, file_name)
        if not os.path.exists(pose_file_path):
            return False, f"missing {file_name}"
        create_backup(pose_file_path, backup_dir)
        pose_counts.append(count_lines_in_file(pose_file_path))
    if len(set(pose_counts)) != 1:
        return False, f"pose line count mismatch {pose_counts}"
    pose_count = pose_counts[0]
    if pose_count <= image_count:
        return False, f"poses<=images ({pose_count}<={image_count}) nothing to do"
    processed_counts = []
    for file_name in pose_files:
        pose_file_path = os.path.join(pose_seq_path, file_name)
        processed_counts.append(
            downsample_pose_file_strict(pose_file_path, image_count, pose_count)
        )
    if len(set(processed_counts)) != 1:
        return False, f"post-downsample mismatch {processed_counts}"
    final_count = processed_counts[0]
    if final_count != image_count:
        return False, f"count {final_count} != images {image_count}"
    ratio = pose_count / image_count
    return True, f"ok images={image_count} poses {pose_count}->{final_count} ratio={ratio:.2f}"


def preprocess_pose_data(base_path, view_type):
    print("stage1: pose downsample")
    rgb_base_path = os.path.join(base_path, "rgb", view_type)
    pose_base_path = os.path.join(base_path, "pose", view_type)
    if not os.path.exists(rgb_base_path):
        print(f"fail: no rgb dir {rgb_base_path}")
        return False, []
    if not os.path.exists(pose_base_path):
        print(f"fail: no pose dir {pose_base_path}")
        return False, []
    rgb_sequences = [
        d for d in os.listdir(rgb_base_path)
        if os.path.isdir(os.path.join(rgb_base_path, d))
    ]
    common_sequences = [
        s for s in rgb_sequences
        if os.path.exists(os.path.join(pose_base_path, s))
    ]
    print(f"sequences: {len(common_sequences)}")
    if not common_sequences:
        return False, []
    success_count = 0
    fail_count = 0
    results = []
    for i, seq in enumerate(common_sequences):
        print(f"[{i+1}/{len(common_sequences)}] {seq}")
        rgb_seq_path = os.path.join(rgb_base_path, seq)
        pose_seq_path = os.path.join(pose_base_path, seq)
        try:
            ok, msg = check_pose_alignment(rgb_seq_path, pose_seq_path)
            print(f"  {'ok' if ok else 'fail'}: {msg}")
            results.append((seq, ok, msg))
            if ok:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"  fail: {e}")
            fail_count += 1
            results.append((seq, False, str(e)))
    print(f"stage1 done ok={success_count} fail={fail_count}")
    return success_count > 0, results


DATASET_NAME = "SpatialVID_NCLT"
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


def get_paired_data(cityworld_root, view_type, processed_sequences=None):
    paired_data = []
    if not os.path.exists(cityworld_root):
        logging.error("root missing %s", cityworld_root)
        return paired_data
    rgb_view_path = os.path.join(cityworld_root, "rgb", view_type)
    pose_view_path = os.path.join(cityworld_root, "pose", view_type)
    if not os.path.exists(rgb_view_path):
        logging.info("skip no rgb %s", rgb_view_path)
        return paired_data
    if not os.path.exists(pose_view_path):
        logging.info("skip no pose %s", pose_view_path)
        return paired_data
    rgb_sequences = []
    for item in os.listdir(rgb_view_path):
        item_path = os.path.join(rgb_view_path, item)
        if os.path.isdir(item_path):
            if processed_sequences is not None and item not in processed_sequences:
                continue
            rgb_sequences.append(item)
    logging.info("view=%s seqs=%d", view_type, len(rgb_sequences))
    for seq_name in rgb_sequences:
        rgb_seq_path = os.path.join(rgb_view_path, seq_name)
        pose_seq_path = os.path.join(pose_view_path, seq_name)
        if not os.path.exists(pose_seq_path):
            logging.warning("skip no pose dir %s", seq_name)
            continue
        path_txt = os.path.join(rgb_seq_path, "path.txt")
        if not os.path.exists(path_txt):
            logging.warning("skip no path.txt %s", seq_name)
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
            logging.warning("skip no extrinsic %s", seq_name)
            continue
        calib_txt = os.path.join(pose_seq_path, "calibration.txt")
        if not os.path.exists(calib_txt):
            logging.warning("default calib %s", seq_name)
            create_default_calibration(calib_txt)
        try:
            ic = count_lines_in_file(path_txt)
            pc = count_lines_in_file(extrinsic_path)
            if ic != pc:
                logging.warning("skip count mismatch %s img=%d pose=%d", seq_name, ic, pc)
                continue
        except Exception as e:
            logging.error("skip %s: %s", seq_name, e)
            continue
        data_prefix = f"{view_type}_{seq_name}"
        paired_data.append((path_txt, calib_txt, extrinsic_path, extrinsic_type, data_prefix))
        logging.info("pair %s", data_prefix)
    return paired_data


def create_default_calibration(calib_path):
    default_intrinsics = [640.0, 640.0, 640.0, 512.0]
    with open(calib_path, "w", encoding="utf-8") as f:
        f.write("# default intrinsics NCLT 1280x1024\n")
        f.write(
            f"{default_intrinsics[0]} {default_intrinsics[1]} "
            f"{default_intrinsics[2]} {default_intrinsics[3]}\n"
        )
        f.write("# Image Size: 1280 1024\n")
    logging.info("wrote default calib %s", calib_path)


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
        logging.warning("calib size fallback 1280x1024 %s", calib_path)
        return 1280, 1024
    except Exception as e:
        logging.error("calib parse %s: %s", calib_path, e)
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
        logging.warning("intrinsic fallback %s", calib_path)
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
        logging.warning("extrinsic too short %d %s", len(lines), extrinsic_path)
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
    logging.info("parsed %d extrinsics %s", len(extrinsic_matrices), extrinsic_path)
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
        logging.info("skip video %s", video_key)
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
        else:
            if os.path.exists(seg_path):
                os.remove(seg_path)
    write_log(PASSED_VIDEOS_LOG, video_key)
    return output_segs


def process_camera(intrinsics, extrinsics, output_dir, prefix, passed_cameras, seg_count):
    if prefix in passed_cameras:
        logging.info("skip camera %s", prefix)
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


def process_segments(cityworld_root, output_root, view_type, processed_sequences):
    print("stage2: segments")
    init_log_file(PASSED_VIDEOS_LOG)
    init_log_file(PASSED_CAMERAS_LOG)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(LOG_DIR, "reflect_nclt.log")),
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
    paired_data = get_paired_data(cityworld_root, view_type, processed_sequences)
    if not paired_data:
        logging.error("no pairs")
        return False
    total_v, total_c = 0, 0
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
    return True


def wait_for_user_confirmation():
    print("review stage1; backups in original_backup/")
    while True:
        c = input("continue stage2? [y/n]: ").strip().lower()
        if c == "y":
            return True
        if c == "n":
            return False
        print("y or n")


def display_check_results(results):
    ok = sum(1 for _, s, _ in results if s)
    bad = len(results) - ok
    print(f"stage1 summary ok={ok} fail={bad}")


def main():
    base_path = "/path/to/local/data"
    view_type = "UGV_front"
    output_root = "/path/to/local/data"
    print(f"nclt pipeline base={base_path} view={view_type} out={output_root}")
    if not os.path.exists(base_path):
        print("error: base missing")
        sys.exit(1)
    ok1, res1 = preprocess_pose_data(base_path, view_type)
    if not ok1:
        print("stage1 failed")
        sys.exit(1)
    display_check_results(res1)
    if not wait_for_user_confirmation():
        print("cancelled")
        sys.exit(0)
    processed = [s for s, ok, _ in res1 if ok]
    print(f"stage2 seqs={len(processed)}")
    if process_segments(base_path, output_root, view_type, processed):
        d = os.path.join(output_root, DATASET_NAME)
        print(f"done out={d}")
    else:
        print("stage2 failed")


if __name__ == "__main__":
    main()
