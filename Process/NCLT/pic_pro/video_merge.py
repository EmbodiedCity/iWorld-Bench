#!/usr/bin/env python3
import os
import sys
import glob
import time
import subprocess

try:
    import cv2
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
    import cv2

input_dir = "/path/to/local/data"
output_file = "/path/to/cam5_output.avi"

print(f"tiff2video in={input_dir}")
tiff_files = sorted(glob.glob(os.path.join(input_dir, "*.tif*")))
if not tiff_files:
    print("error: no tiff")
    sys.exit(1)
print(f"frames={len(tiff_files)}")
img = cv2.imread(tiff_files[0])
if img is None:
    print("error: read fail")
    sys.exit(1)
height, width = img.shape[:2]
codec, fps = "XVID", 10
print(f"size={width}x{height} codec={codec} fps={fps}")
fourcc = cv2.VideoWriter_fourcc(*codec)
video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
if not video.isOpened():
    for alt_codec in ("MJPG", "mp4v", "DIVX", "I420"):
        alt_out = output_file.replace(".avi", ".mp4") if alt_codec == "mp4v" else output_file
        video = cv2.VideoWriter(alt_out, cv2.VideoWriter_fourcc(*alt_codec), fps, (width, height))
        if video.isOpened():
            output_file, codec = alt_out, alt_codec
            print(f"codec={alt_codec}")
            break
    if not video.isOpened():
        print("error: no codec")
        sys.exit(1)
t0 = time.time()
n = 0
for i, tiff_file in enumerate(tiff_files, 1):
    im = cv2.imread(tiff_file)
    if im is not None:
        video.write(im)
        n += 1
    if i % 500 == 0 or i == len(tiff_files):
        dt = time.time() - t0
        print(f"  {i}/{len(tiff_files)} {100 * i / len(tiff_files):.0f}% {dt:.0f}s")
video.release()
print(f"done frames={n}/{len(tiff_files)} time={time.time() - t0:.1f}s -> {output_file}")
if os.path.exists(output_file):
    print(f"size_mb={os.path.getsize(output_file) / (1024 * 1024):.1f}")
    print(f"ffmpeg -i {output_file!r} -c:v libx264 -crf 23 out.mp4")
