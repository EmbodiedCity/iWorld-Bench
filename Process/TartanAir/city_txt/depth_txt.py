import os
import glob
import re
from tqdm import tqdm

SCENES = [
    "abandonedfactory_night_sample_P002",
    "abandonedfactory_sample_P001",
    "amusement_sample_P008",
    "carwelding_sample_P007",
    "endofworld_sample_P001",
    "gascola_sample_P001",
    "hospital_sample_P000",
    "japanesealley_sample_P007",
    "neighborhood_sample_P002",
    "ocean_sample_P006",
    "office2_sample_P003",
    "seasidetown_sample_P003",
    "seasonsforest_sample_P002",
    "seasonsforest_winter_sample_P006",
    "soulcity_sample_P003",
    "westerndesert_sample_P002",
]


def get_scene_folders(tartanair_base):
    out = []
    for scene in SCENES:
        p = os.path.join(tartanair_base, scene)
        if os.path.exists(p):
            out.append((scene, p))
        else:
            print(f"warn missing {scene}")
    return out


def find_depth_directories(scene_path, scene_name):
    tail = scene_name.split("_")[-1]
    candidates = [
        os.path.join(scene_path, scene_name, tail, "depth_left"),
        os.path.join(scene_path, scene_name, "depth_left"),
        os.path.join(scene_path, "depth_left"),
        os.path.join(scene_path, tail, "depth_left"),
        os.path.join(scene_path, scene_name, tail, "depth"),
        os.path.join(scene_path, scene_name, "depth"),
    ]
    for d in candidates:
        if os.path.isdir(d):
            return d
    for pattern in (
        os.path.join(scene_path, "**", "*depth*"),
        os.path.join(scene_path, "**", "*Depth*"),
    ):
        for dir_path in glob.glob(pattern, recursive=True):
            if os.path.isdir(dir_path) and find_depth_files(dir_path):
                print(f"depth dir {dir_path}")
                return dir_path
    return None


def find_depth_files(depth_dir):
    exts = (".png", ".tiff", ".tif", ".exr", ".pfm", ".dpt", ".npy", ".npz")
    files = [
        os.path.join(depth_dir, fn)
        for fn in os.listdir(depth_dir)
        if fn.lower().endswith(exts)
    ]
    return sorted(files, key=os.path.basename)


def to_win(wsl_path):
    if wsl_path.startswith("/path/to/local/data/"):
        return "G:" + wsl_path[6:].replace("/", "\\")
    if wsl_path.startswith("/path/to/local/data/"):
        return "G:" + wsl_path[6:].replace("/", "\\")
    return wsl_path.replace("/", "\\")


def sort_nicely(paths):
    def k(p):
        return [
            int(t) if t.isdigit() else t.lower()
            for t in re.split(r"([0-9]+)", os.path.basename(p))
        ]

    return sorted(paths, key=k)


def generate_depth_list(scene_name, scene_path, output_base):
    depth_dir = find_depth_directories(scene_path, scene_name)
    if not depth_dir:
        print(f"warn no depth dir {scene_name}")
        return False
    depth_files = sort_nicely(find_depth_files(depth_dir))
    if not depth_files:
        print(f"warn no depth files {depth_dir}")
        return False
    out_dir = os.path.join(output_base, f"TartanAir_{scene_name}")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "path.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# depth paths, one per line\n")
        for w in depth_files:
            f.write(f"{to_win(w)}\n")
    print(f"  n={len(depth_files)} -> {out_file}")
    return True


def main():
    tartanair_base = "/path/to/local/data"
    output_base = "/path/to/local/data"
    os.makedirs(output_base, exist_ok=True)
    scene_info = get_scene_folders(tartanair_base)
    if not scene_info:
        print("error: no scenes")
        return
    ok = bad = 0
    for sn, sp in tqdm(scene_info, desc="depth_list"):
        try:
            if generate_depth_list(sn, sp, output_base):
                ok += 1
            else:
                bad += 1
        except Exception as e:
            print(f"fail {sn}: {e}")
            bad += 1
    print(f"done ok={ok} fail={bad} -> {output_base}")


def verify_file_counts():
    pose_base = "/path/to/local/data"
    data_base = "/path/to/local/data"
    bad = 0
    for scene in SCENES:
        pose_f = os.path.join(pose_base, f"TartanAir_{scene}", "extrinsics_matrix.txt")
        data_f = os.path.join(data_base, f"TartanAir_{scene}", "path.txt")
        if not os.path.isfile(pose_f):
            print(f"warn no pose {scene}")
            continue
        if not os.path.isfile(data_f):
            print(f"warn no depth list {scene}")
            continue
        pc = sum(1 for _ in open(pose_f))
        with open(data_f, encoding="utf-8") as f:
            dc = sum(1 for ln in f if not ln.startswith("#") and ln.strip())
        if pc != dc:
            print(f"mismatch {scene} pose={pc} depth={dc}")
            bad += 1
    print(f"verify mismatches={bad}")


def compare_rgb_depth_counts():
    rgb_base = "/path/to/local/data"
    depth_base = "/path/to/local/data"
    bad = 0
    for scene in SCENES:
        rf = os.path.join(rgb_base, f"TartanAir_{scene}", "path.txt")
        df = os.path.join(depth_base, f"TartanAir_{scene}", "path.txt")
        rc = dc = 0
        if os.path.isfile(rf):
            with open(rf, encoding="utf-8") as f:
                rc = sum(1 for ln in f if not ln.startswith("#") and ln.strip())
        if os.path.isfile(df):
            with open(df, encoding="utf-8") as f:
                dc = sum(1 for ln in f if not ln.startswith("#") and ln.strip())
        if rc and dc and rc != dc:
            print(f"rgb/depth mismatch {scene} {rc} {dc}")
            bad += 1
    print(f"rgb_depth mismatches={bad}")


if __name__ == "__main__":
    main()
    print("verify pose vs depth")
    verify_file_counts()
    print("compare rgb vs depth")
    compare_rgb_depth_counts()
