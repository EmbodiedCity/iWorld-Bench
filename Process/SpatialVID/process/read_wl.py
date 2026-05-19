import os
import cv2
from pathlib import Path

def get_media_size(file_or_dir_path):
    """
    Video/columns
    :param file_or_dir_path: Video  RGBcolumns
    :return: (width, height)  Nonefailed
    """
    # Windows
    path = Path(file_or_dir_path).resolve()
    
    if path.is_file():
        # Videomp4/avi/mov
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return (width, height)
        # Videofailed，jpg/png/bmp
        else:
            img = cv2.imread(str(path))
            if img is not None:
                height, width = img.shape[:2]
                return (width, height)
            else:
                print(f"Error {path}Video/")
                return None
    
    # 2RGBcolumns
    elif path.is_dir():
        img_suffixes = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        for file in os.listdir(path):
            file_path = path / file
            if file_path.suffix.lower() in img_suffixes:
                img = cv2.imread(str(file_path))
                if img is not None:
                    height, width = img.shape[:2]
                    print(f"RGBcolumns{file}")
                    return (width, height)
        print(f"Error {path} Not found{img_suffixes}")
        return None
    
    # does not exist
    else:
        print(f"Error {path} does not exist")
        return None

# ==================== RGBcolumns ====================
if __name__ == "__main__":
    # RGBcolumns
    rgb_seq_dir = r"/path/to/local/data"
    
    media_size = get_media_size(rgb_seq_dir)
    
    if media_size:
        width, height = media_size
        print(f"\nRGBcolumns={width}px，={height}px")
    else:
        print("\nfailed！")