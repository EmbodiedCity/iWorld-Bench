import os

base_dir = "/path/to/local/data"
output_base_dir = "/path/to/local/data"

if not os.path.exists(base_dir):
    print("error: base missing")
    raise SystemExit(1)
date_folders = []
for item in sorted(os.listdir(base_dir)):
    p = os.path.join(base_dir, item)
    if os.path.isdir(p) and "_lb3" in item:
        date_folders.append(item)
if not date_folders:
    print("error: no date folders")
    raise SystemExit(1)
print(f"dates={len(date_folders)}")
for date_folder in date_folders:
    date_str = date_folder.replace("_lb3", "")
    wsl_dir = os.path.join(base_dir, date_folder, date_folder, date_str, "lb3", "Cam5")
    output_dir = os.path.join(output_base_dir, f"NCLT_{date_folder}_cam5")
    output_file = os.path.join(output_dir, "path.txt")
    if not os.path.exists(wsl_dir):
        print(f"skip no cam5 {date_folder}")
        continue
    tiff_files = []
    try:
        for fn in sorted(os.listdir(wsl_dir)):
            if fn.lower().endswith((".tiff", ".tif")):
                fp = os.path.join(wsl_dir, fn)
                if os.path.isfile(fp):
                    tiff_files.append(fp)
    except OSError as e:
        print(f"skip listdir {date_folder}: {e}")
        continue
    if not tiff_files:
        print(f"skip no tiff {date_folder}")
        continue
    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for wsl_path in tiff_files:
                if wsl_path.startswith("/path/to/local/data/"):
                    win = "G:" + wsl_path[6:].replace("/", "\\")
                elif wsl_path.startswith("/path/to/local/data/"):
                    win = "G:" + wsl_path[6:].replace("/", "\\")
                else:
                    win = wsl_path.replace("/", "\\")
                f.write(f"{win}\n")
        print(f"{date_folder} paths={len(tiff_files)} -> {output_file}")
    except OSError as e:
        print(f"skip write {date_folder}: {e}")
print("done")
