#!/usr/bin/env python3
import os
import re
import cv2
from pathlib import Path
from natsort import natsorted

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
_CAMERA_NAME_PATTERN = re.compile(
    r"__(?:CAM_FRONT|CAM_BACK|CAM_BACK_LEFT|CAM_BACK_RIGHT|CAM_FRONT_LEFT|CAM_FRONT_RIGHT)__",
    re.I,
)


def scene_key_from_filename(name: str) -> str:
    stem = Path(name).stem
    m = _CAMERA_NAME_PATTERN.search(stem)
    if m:
        return stem[: m.start()]
    if "__" in stem:
        return stem.split("__", 1)[0]
    return stem


def check_and_fix_path(folder_path):
    if os.path.exists(folder_path):
        return folder_path
    original_path = folder_path
    folder_path = folder_path.lower() if folder_path.startswith("/mnt/") else folder_path
    path_parts = original_path.split("/")
    if len(path_parts) >= 4 and path_parts[2].lower() == "nusence":
        for path in (
            original_path,
            original_path.replace("NuSence", "nusence"),
            original_path.replace("NuSence", "Nusence"),
            original_path.replace("NuSence", "NUSENCE"),
        ):
            if os.path.exists(path):
                return path
    for mount in ("/path/to/local/data/", "/path/to/local/data/", "/path/to/local/data/", "/path/to/local/data/"):
        test_path = (
            mount + "/".join(original_path.split("/")[3:])
            if original_path.startswith("/mnt/")
            else mount + original_path.lstrip("/")
        )
        if os.path.exists(test_path):
            return test_path
    return original_path


def gather_image_files(folder_path):
    folder_path = check_and_fix_path(folder_path)
    if not os.path.exists(folder_path):
        return []
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        for pattern in (f"*{ext}", f"*{ext.upper()}"):
            try:
                image_files.extend(Path(folder_path).glob(pattern))
            except OSError:
                pass
    if not image_files:
        try:
            image_files = [
                file
                for file in Path(folder_path).iterdir()
                if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
            ]
        except OSError:
            pass
    seen = set()
    unique = []
    for p in image_files:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return natsorted(unique, key=lambda p: p.name)


def split_into_contiguous_scene_runs(sorted_paths):
    if not sorted_paths:
        return []
    runs = []
    prev_key = scene_key_from_filename(sorted_paths[0].name)
    cur = [sorted_paths[0]]
    for p in sorted_paths[1:]:
        key = scene_key_from_filename(p.name)
        if key == prev_key:
            cur.append(p)
        else:
            runs.append((prev_key, cur))
            prev_key, cur = key, [p]
    runs.append((prev_key, cur))
    return runs


def _open_video_writer(output_path, fps, width, height):
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if writer.isOpened():
        return writer
    writer.release()
    for codec in ("mp4v", "avc1", "x264", "XVID"):
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*codec), fps, (width, height)
        )
        if writer.isOpened():
            return writer
        writer.release()
    return None


def write_video_from_image_paths(image_paths, output_path, fps):
    if not image_paths:
        return False
    first = cv2.imread(str(image_paths[0]))
    if first is None:
        print(f"err read {image_paths[0]}")
        return False
    height, width = first.shape[:2]
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    vw = _open_video_writer(output_path, fps, width, height)
    if vw is None:
        print(f"err writer {output_path}")
        return False
    n = 0
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        vw.write(img)
        n += 1
    vw.release()
    if n > 0:
        print(f"wrote {n}f -> {output_path}")
        return True
    print(f"err no frames {output_path}")
    return False


def create_batched_videos_by_scene(
    input_folder,
    output_dir,
    frames_per_clip=81,
    fps=15,
    max_clips=None,
    write_short_last_clip=True,
):
    print(f"in={input_folder} out={output_dir} clip={frames_per_clip} fps={fps}")
    images = gather_image_files(input_folder)
    if not images:
        print("err no images")
        return
    runs = split_into_contiguous_scene_runs(images)
    print(f"images={len(images)} runs={len(runs)}")
    os.makedirs(output_dir, exist_ok=True)
    clip_index = 0
    for run_idx, (scene_key, run_paths) in enumerate(runs):
        if max_clips is not None and clip_index >= max_clips:
            break
        n = len(run_paths)
        start_clip = clip_index
        pos = 0
        while pos < n:
            if max_clips is not None and clip_index >= max_clips:
                break
            batch = run_paths[pos : pos + frames_per_clip]
            pos += frames_per_clip
            if not batch:
                break
            if len(batch) < frames_per_clip and not write_short_last_clip:
                continue
            clip_index += 1
            out_path = os.path.join(output_dir, f"front_{clip_index:02d}.mp4")
            write_video_from_image_paths(batch, out_path, fps)
        if clip_index > start_clip:
            print(f"run {run_idx + 1}/{len(runs)} clips+={clip_index - start_clip} frames={n}")
    print("done")


def create_video_from_folder(folder_path, output_path, fps=30):
    folder_path = check_and_fix_path(folder_path)
    if not os.path.exists(folder_path):
        for search_path in (
            "/path/to/local/data",
            "/path/to/local/data",
            "/path/to/local/data",
        ):
            if os.path.exists(search_path):
                folder_path = search_path
                break
    if not os.path.exists(folder_path):
        print("err folder")
        return
    image_files = gather_image_files(folder_path)
    if not image_files:
        print("err no images")
        return
    write_video_from_image_paths(image_files, output_path, fps)


def main():
    create_batched_videos_by_scene(
        r"/path/to/local/data",
        r"/path/to/local/data",
        frames_per_clip=81,
        fps=15,
        max_clips=None,
        write_short_last_clip=True,
    )


if __name__ == "__main__":
    main()
