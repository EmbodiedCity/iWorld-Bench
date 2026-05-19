import os
import cv2
import logging
from pathlib import Path

# ====================== ======================
# Output directory
OUTPUT_ROOT_DIR = "./combined_sequences"
OUTPUT_VIDEO_DIR = os.path.join(OUTPUT_ROOT_DIR, "videos")
OUTPUT_POSE_DIR = os.path.join(OUTPUT_ROOT_DIR, "poses")
EXPECTED_VIDEO_FRAMES = 81
EXPECTED_POSE_LINES = 81

# ====================== ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUTPUT_ROOT_DIR, "check_report.log"), encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ====================== ======================
def check_video_frame_count(video_path):
    """
    Videoframes
    (passed, actualframes, Error)
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, 0, f"Cannot open video"
        
        # actualframesVideoframes，
        actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if actual_frames == EXPECTED_VIDEO_FRAMES:
            return True, actual_frames, ""
        else:
            return False, actual_frames, f"Frame count mismatch (expected{EXPECTED_VIDEO_FRAMES}，actual{actual_frames}）"
    except Exception as e:
        return False, 0, f"failed{str(e)}"

def check_pose_line_count(pose_path):
    """
    Pose filelineslines
    (passed, actuallines, Error)
    """
    try:
        with open(pose_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        actual_lines = len(lines)
        
        if actual_lines == EXPECTED_POSE_LINES:
            return True, actual_lines, ""
        else:
            return False, actual_lines, f"lines{EXPECTED_POSE_LINES}，actual{actual_lines}"
    except FileNotFoundError:
        return False, 0, "does not exist"
    except Exception as e:
        return False, 0, f"failed{str(e)}"

def get_matched_pose_file(video_name, pose_dir):
    """
    Videocorresponding pose fileleft_xxx.mp4  left_xxx.txt
    Pose fileNoneNot found
    """
    pose_filename = Path(video_name).stem + ".txt"
    pose_path = os.path.join(pose_dir, pose_filename)
    return pose_path if os.path.exists(pose_path) else None

# ====================== ======================
def main():
    total_videos = 0
    pass_videos = 0
    fail_videos = 0
    total_poses = 0
    pass_poses = 0
    fail_poses = 0
    error_details = []

    logger.info("="*50)
    logger.info("Starting validation of concatenated results")
    logger.info(f"Validation standard: video{EXPECTED_VIDEO_FRAMES}frames, pose{EXPECTED_POSE_LINES}lines")
    logger.info("="*50)

    # 1. Video directory
    if not os.path.exists(OUTPUT_VIDEO_DIR):
        logger.error(f"Video directorydoes not exist：{OUTPUT_VIDEO_DIR}")
    else:
        # Videomp4/avi/mov
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        video_files = [f for f in os.listdir(OUTPUT_VIDEO_DIR) if f.lower().endswith(video_extensions)]
        
        if not video_files:
            logger.warning("Video directoryVideo")
        else:
            total_videos = len(video_files)
            logger.info(f"\nFound{total_videos}video files, starting per-file validation：")
            
            for video_file in video_files:
                video_path = os.path.join(OUTPUT_VIDEO_DIR, video_file)
                logger.info(f"\nValidating video：{video_file}")
                
                # Validating videoframes
                video_pass, actual_frames, video_err = check_video_frame_count(video_path)
                if video_pass:
                    pass_videos += 1
                    logger.info(f"  ✅ Video frame count validation passed（{actual_frames}frames）")
                else:
                    fail_videos += 1
                    logger.error(f"  ❌ Video frame count validation failed：{video_err}")
                    error_details.append(f"Video {video_file}：{video_err}")
                
                # Pose file
                pose_path = get_matched_pose_file(video_file, OUTPUT_POSE_DIR)
                if pose_path:
                    total_poses += 1
                    logger.info(f"Validating pose：{os.path.basename(pose_path)}")
                    
                    pose_pass, actual_lines, pose_err = check_pose_line_count(pose_path)
                    if pose_pass:
                        pass_poses += 1
                        logger.info(f"  ✅ linespassed{actual_lines}lines")
                    else:
                        fail_poses += 1
                        logger.error(f"  ❌ linesfailed{pose_err}")
                        error_details.append(f" {os.path.basename(pose_path)}{pose_err}")
                else:
                    logger.warning(f"  ⚠️ Not found{video_file}corresponding pose file")
                    error_details.append(f"Video {video_file}Pose file")

    # 2. Pose fileVideo
    if os.path.exists(OUTPUT_POSE_DIR):
        pose_files = [f for f in os.listdir(OUTPUT_POSE_DIR) if f.lower().endswith('.txt')]
        for pose_file in pose_files:
            video_filename = Path(pose_file).stem + ".mp4"
            video_path = os.path.join(OUTPUT_VIDEO_DIR, video_filename)
            if not os.path.exists(video_path):
                logger.warning(f"\n⚠️ FoundPose file{pose_file}Video")
                error_details.append(f" {pose_file}no corresponding video file")

    logger.info("\n" + "="*50)
    logger.info("Validation summary report")
    logger.info("="*50)
    logger.info(f"Total videos：{total_videos} | passed：{pass_videos} | failed：{fail_videos}")
    logger.info(f"Total poses：{total_poses} | passed：{pass_poses} | failed：{fail_poses}")
    
    if error_details:
        logger.error("\nError")
        for idx, err in enumerate(error_details, 1):
            logger.error(f"  {idx}. {err}")
    else:
        logger.info("\n✅ passed！")

    logger.info(f"\nValidation log saved to：{os.path.join(OUTPUT_ROOT_DIR, 'check_report.log')}")

# ====================== lines ======================
if __name__ == "__main__":
    main()