#!/usr/bin/env python3
import os
import time
import numpy as np

try:
    import imageio.v2 as imageio
except ImportError:
    print("install: pip install imageio")
    raise SystemExit(1)

BASE_IMAGE_DIR = "/path/to/local/data"
SPECIFIED_DATES = [
    "2012-05-11",
    "2012-03-17",
    "2012-05-26",
    "2012-03-25",
    "2012-08-04",
    "2012-03-31",
    "2012-04-29",
]


def rotate_image_90_clockwise(image_array):
    return np.rot90(image_array, 3)


def process_cam5_folder(date_folder):
    cam5_dir = os.path.join(BASE_IMAGE_DIR, date_folder, "lb3", "lb3", "Cam5")
    print(f"dir {cam5_dir}")
    if not os.path.exists(cam5_dir):
        print("missing cam5")
        return False, 0, 0, 0
    try:
        files = os.listdir(cam5_dir)
    except OSError as e:
        print(f"listdir err {e}")
        return False, 0, 0, 0
    tiff_files = [f for f in files if f.lower().endswith((".tif", ".tiff"))]
    if not tiff_files:
        print("no tiff")
        return False, 0, 0, 0
    tiff_files.sort()
    success = failed = 0
    t0 = time.time()
    for i, filename in enumerate(tiff_files, 1):
        file_path = os.path.join(cam5_dir, filename)
        try:
            arr = imageio.imread(file_path)
            h, w = arr.shape[:2]
            rot = rotate_image_90_clockwise(arr)
            tmp = file_path + ".temp"
            imageio.imwrite(tmp, rot, format="tiff")
            os.remove(file_path)
            os.rename(tmp, file_path)
            success += 1
            if i % 50 == 0 or i == len(tiff_files):
                dt = time.time() - t0
                print(f"  {i}/{len(tiff_files)} {100 * i / len(tiff_files):.0f}% {dt:.0f}s")
        except Exception as e:
            print(f"  fail {filename}: {str(e)[:48]}")
            failed += 1
    return True, success, 0, failed


def main():
    print(f"tiff rotate base={BASE_IMAGE_DIR}")
    if not os.path.exists(BASE_IMAGE_DIR):
        print("error: base missing")
        return
    for d in SPECIFIED_DATES:
        print(f"date {d} exists={os.path.exists(os.path.join(BASE_IMAGE_DIR, d))}")
    if input("overwrite in-place? [yes/no]: ").lower() not in ("yes", "y"):
        print("cancelled")
        return
    tot_ok = tot_fail = 0
    t0 = time.time()
    for date_folder in SPECIFIED_DATES:
        p = os.path.join(BASE_IMAGE_DIR, date_folder)
        if not os.path.exists(p):
            print(f"skip missing {date_folder}")
            continue
        _, ok, _, fail = process_cam5_folder(date_folder)
        tot_ok += ok
        tot_fail += fail
    print(f"done ok={tot_ok} fail={tot_fail} time={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
