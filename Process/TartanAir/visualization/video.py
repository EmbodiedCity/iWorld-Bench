import cv2
import os
import glob


def create_video_from_pngs(png_folder, output_video_path, fps=30):
    png_files = sorted(glob.glob(os.path.join(png_folder, "*.png")))
    if not png_files:
        print(f"no png in {png_folder}")
        return
    first = cv2.imread(png_files[0])
    if first is None:
        print("err read first")
        return
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    if not vw.isOpened():
        print("err writer")
        return
    for i, p in enumerate(png_files):
        im = cv2.imread(p)
        if im is None:
            continue
        vw.write(im)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(png_files)}")
    vw.release()
    print(f"ok -> {output_video_path}")


def main():
    base = r"/path/to/local/data"
    png_folder = os.path.join(base, "image_left")
    out = os.path.join(base, "output_video.mp4")
    if not os.path.exists(png_folder):
        print("err no folder")
        return
    create_video_from_pngs(png_folder, out, fps=30)


if __name__ == "__main__":
    main()
