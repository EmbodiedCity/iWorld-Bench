#!/usr/bin/env python3
import os
import csv
import glob
import sys


def extract_date_from_filename(filename):
    base_name = os.path.basename(filename)
    if base_name.startswith("groundtruth_"):
        return base_name[12:-4]
    return None


def transform_coordinates(row):
    if len(row) != 7:
        return None
    try:
        _ts, x, y, z, roll, yaw, heading = map(float, row)
        x_new, y_new, z_new = y, -z, x
        roll_new, yaw_new, heading_new = yaw, -heading, roll
        return [x_new, y_new, z_new, roll_new, yaw_new, heading_new]
    except ValueError:
        return None


def process_single_file(input_file, output_root):
    try:
        date_str = extract_date_from_filename(input_file)
        if not date_str:
            print(f"skip bad name {input_file}")
            return False
        output_dir = os.path.join(output_root, "UGV_front", f"NCLT_{date_str}_lb3_cam5")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "six_Dof.txt")
        lines_out = []
        n_ok = n_bad = 0
        with open(input_file, "r", newline="") as csvfile:
            for row in csv.reader(csvfile):
                if not row:
                    continue
                n_ok += 1
                t = transform_coordinates(row)
                if t:
                    lines_out.append(" ".join(str(v) for v in t))
                else:
                    n_bad += 1
                if n_ok % 10000 == 0:
                    print(f"  {os.path.basename(input_file)} lines={n_ok}")
        with open(output_file, "w") as txtfile:
            txtfile.write("\n".join(lines_out))
        print(f"{os.path.basename(input_file)} -> {output_file} ok={len(lines_out)} bad={n_bad}")
        return True
    except Exception as e:
        print(f"fail {input_file}: {e}")
        return False


def main():
    input_dir = r"/path/to/local/data"
    output_root = r"G:/CityWorld/pose"
    if sys.platform != "win32":
        output_root = output_root.replace("G:", "/mnt/g")
    if not os.path.exists(input_dir):
        print(f"error: missing {input_dir}")
        return
    csv_files = glob.glob(os.path.join(input_dir, "groundtruth_*.csv"))
    if not csv_files:
        print("error: no groundtruth_*.csv")
        return
    ok = sum(process_single_file(f, output_root) for f in csv_files)
    print(f"done {ok}/{len(csv_files)}")


if __name__ == "__main__":
    main()
