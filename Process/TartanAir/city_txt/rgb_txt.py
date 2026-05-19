import os
import re
from tqdm import tqdm


def get_scene_folders(tartanair_base):
    out = []
    for item in os.listdir(tartanair_base):
        p = os.path.join(tartanair_base, item)
        if os.path.isdir(p) and not item.startswith("."):
            if os.path.exists(os.path.join(p, "Easy")):
                out.append((item, p))
                print(f"scene {item}")
    return out


def find_image_directories(scene_path, scene_name):
    info = []
    easy_path = os.path.join(scene_path, "Easy")
    if not os.path.exists(easy_path):
        print(f"warn no Easy {scene_name}")
        return info
    p_folders = [
        x
        for x in os.listdir(easy_path)
        if os.path.isdir(os.path.join(easy_path, x)) and x.startswith("P")
    ]
    if not p_folders:
        print(f"warn no P* {scene_name}")
        return info
    p_folders.sort()
    for pf in p_folders:
        il = os.path.join(easy_path, pf, "image_left")
        if os.path.isdir(il):
            info.append((scene_name, pf, il))
        else:
            print(f"warn no image_left {scene_name}/{pf}")
    return info


def find_image_files(image_dir):
    exts = (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif")
    files = []
    for fn in os.listdir(image_dir):
        if fn.lower().endswith(exts):
            files.append(os.path.join(image_dir, fn))
    return sorted(files, key=os.path.basename)


def to_win_path(wsl_path):
    if wsl_path.startswith("/path/to/local/data/"):
        return "G:" + wsl_path[6:].replace("/", "\\")
    if wsl_path.startswith("/path/to/local/data/"):
        return "G:" + wsl_path[6:].replace("/", "\\")
    if wsl_path.startswith("/path/to/local/data/"):
        return "I:" + wsl_path[6:].replace("/", "\\")
    if wsl_path.startswith("/path/to/local/data/"):
        return "I:" + wsl_path[6:].replace("/", "\\")
    return wsl_path.replace("/", "\\")


def sort_nicely(paths):
    def key_fn(p):
        return [
            int(t) if t.isdigit() else t.lower()
            for t in re.split(r"([0-9]+)", os.path.basename(p))
        ]

    return sorted(paths, key=key_fn)


def generate_rgb_list(scene_name, p_folder, image_dir, output_base):
    image_files = sort_nicely(find_image_files(image_dir))
    if not image_files:
        print(f"warn no images {image_dir}")
        return False
    out_dir = os.path.join(output_base, f"TartanAir_{scene_name}_{p_folder}")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "path.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# RGB paths, one per line, aligned with extrinsics\n")
        f.write(f"# scene={scene_name} traj={p_folder}\n")
        for wsl_path in image_files:
            f.write(f"{to_win_path(wsl_path)}\n")
    print(f"  n={len(image_files)} -> {out_file}")
    return True


def main():
    tartanair_base = "/path/to/local/data"
    output_base = "/path/to/local/data"
    os.makedirs(output_base, exist_ok=True)
    scene_info = get_scene_folders(tartanair_base)
    if not scene_info:
        print("error: no scenes")
        return
    all_dirs = []
    for sn, sp in scene_info:
        all_dirs.extend(find_image_directories(sp, sn))
    if not all_dirs:
        print("error: no image_left")
        return
    print(f"image_left dirs={len(all_dirs)}")
    ok = bad = 0
    for sn, pf, imd in tqdm(all_dirs, desc="rgb_list"):
        try:
            if generate_rgb_list(sn, pf, imd, output_base):
                ok += 1
            else:
                bad += 1
        except Exception as e:
            print(f"fail {sn}/{pf}: {e}")
            bad += 1
    print(f"done ok={ok} fail={bad} -> {output_base}")


def verify_file_counts():
    pose_base = "/path/to/local/data"
    rgb_base = "/path/to/local/data"
    if not os.path.isdir(rgb_base):
        return
    rgb_dirs = [
        d
        for d in os.listdir(rgb_base)
        if d.startswith("TartanAir_") and os.path.isdir(os.path.join(rgb_base, d))
    ]
    bad = 0
    for name in rgb_dirs:
        pose_f = os.path.join(pose_base, name, "extrinsics_matrix.txt")
        rgb_f = os.path.join(rgb_base, name, "path.txt")
        pc = sum(1 for _ in open(pose_f)) if os.path.isfile(pose_f) else 0
        rc = 0
        if os.path.isfile(rgb_f):
            with open(rgb_f, encoding="utf-8") as f:
                rc = sum(1 for ln in f if not ln.startswith("#") and ln.strip())
        if pc and rc and pc != rc:
            print(f"mismatch {name} pose={pc} rgb={rc}")
            bad += 1
    print(f"verify mismatches={bad}")


if __name__ == "__main__":
    main()
    print("verify")
    verify_file_counts()
