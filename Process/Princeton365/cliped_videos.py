import os
import re
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Set

# Windows PowerShell
try:
    from colorama import init, Fore, Style

    init(autoreset=True)
except ImportError:
    print("⚠️ colorama not installed, auto-installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "colorama"])
    from colorama import init, Fore, Style

    init(autoreset=True)

# ffmpeg
try:
    import ffmpeg
except ImportError:
    print("⚠️ ffmpeg-python not installed, auto-installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ffmpeg-python"])
    import ffmpeg

# -------------------------- --------------------------
TEST_DIR = "test"
VIDEO_INPUT_DIR = "test_videos"
CLIP_OUTPUT_DIR = "cliped_videos"
SUCCESS_RECORD = "cliped_videos.txt"
FAILED_RECORD = "failed_cliped_videos.txt"

MICROSECOND_TO_SECOND = 1e-6

# Video
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".mpeg", ".mpg"}


# -------------------------- --------------------------
class Color:
    GREEN = Fore.GREEN + Style.BRIGHT
    YELLOW = Fore.YELLOW + Style.BRIGHT
    RED = Fore.RED + Style.BRIGHT
    BLUE = Fore.BLUE + Style.BRIGHT
    RESET = Style.RESET_ALL


# -------------------------- --------------------------
def check_ffmpeg() -> bool:
    """Check if ffmpeg is available"""
    try:
        subprocess.check_call(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"{Color.RED}❌ ffmpeg not found, please install and add to PATH{Color.RESET}")
        print(f"{Color.RED}Download：https://ffmpeg.org/download.html{Color.RESET}")
        return False


def init_env() -> None:
    """Initialize directories and record files"""
    # Output directory
    Path(CLIP_OUTPUT_DIR).mkdir(exist_ok=True)
    # does not exist
    Path(SUCCESS_RECORD).touch(exist_ok=True)
    Path(FAILED_RECORD).touch(exist_ok=True)
    print(f"{Color.YELLOW}⚠️  Initialized directory：{CLIP_OUTPUT_DIR}")
    print(f"{Color.YELLOW}⚠️  Initialized record files：{SUCCESS_RECORD}、{FAILED_RECORD}{Color.RESET}")


def get_video_file_by_prefix(video_dir: Path, file_prefix: str) -> Path | None:
    """
    Video.
    Video，None
    """
    if not video_dir.exists():
        return None

    for ext in SUPPORTED_VIDEO_EXTENSIONS:
        video_path = video_dir / f"{file_prefix}{ext}"
        if video_path.exists() and video_path.is_file():
            return video_path

    for file in video_dir.iterdir():
        if file.is_file() and file.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
            file_name_parts = file.name.split('.', 1)
            if len(file_name_parts) >= 1 and file_name_parts[0] == file_prefix:
                return file

    return None


def collect_valid_files() -> List[Path]:
    """Collect valid txt files (with matching videos)"""
    valid_files = []
    test_dir = Path(TEST_DIR)
    video_dir = Path(VIDEO_INPUT_DIR)

    if not test_dir.exists():
        print(f"{Color.RED}❌ Source directory {TEST_DIR} does not exist！{Color.RESET}")
        return []

    if not video_dir.exists():
        print(f"{Color.RED}❌ Video directory {VIDEO_INPUT_DIR} does not exist！{Color.RESET}")
        return []

    for txt_file in test_dir.glob("*.txt"):
        file_prefix = txt_file.stem
        video_path = get_video_file_by_prefix(video_dir, file_prefix)

        if video_path:
            valid_files.append(txt_file)
            print(f"{Color.BLUE}🔍 Found matching video：{txt_file.name} -> {video_path.name}{Color.RESET}")
        else:
            print(f"{Color.YELLOW}⚠️ No matching video found{file_prefix}，skippingprocessing{Color.RESET}")

    valid_files.sort(key=lambda x: x.name)
    return valid_files


def load_processed_files(file_path: str) -> Set[str]:
    """processingcolumns"""
    file = Path(file_path)
    if not file.exists() or file.stat().st_size == 0:
        return set()

    with open(file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return set(lines)


def save_to_record(file_path: str, content: str) -> None:
    """Append content to record file"""
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(content + "\n")


def filter_invalid_filename_chars(name: str) -> str:
    """Filter invalid Windows filename characters"""
    invalid_chars = r'\/:*?"<>|'
    for char in invalid_chars:
        name = name.replace(char, "_")
    return name


def parse_timestamp_data(txt_file: Path) -> List[int]:
    """Parse timestamp data (microseconds) from txt file"""
    try:
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        # linesURL，lines
        if len(lines) < 2:
            raise ValueError("Insufficient file content, no timestamp data")

        timestamps = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) != 19:
                raise ValueError(f"Line format mismatch（expected 19 columns, got{len(parts)}columns）：{line}")

            # columns
            try:
                timestamp = int(parts[0])
                timestamps.append(timestamp)
            except ValueError:
                raise ValueError(f"Timestamp is not a valid integer：{parts[0]}")

        if not timestamps:
            raise ValueError("No valid timestamp data found")

        return timestamps

    except Exception as e:
        print(f"{Color.RED}❌ Failed to parse timestamps：{str(e)}{Color.RESET}")
        raise


def get_video_duration(video_path: Path) -> float:
    """Get video duration (seconds)"""
    try:
        probe = ffmpeg.probe(str(video_path))
        duration = float(probe['format']['duration'])
        return duration
    except Exception as e:
        print(f"{Color.RED}❌ Failed to get video duration：{str(e)}{Color.RESET}")
        raise


def clip_video(
        video_path: Path,
        output_path: Path,
        start_time: float,
        end_time: float
) -> bool:
    """Clip video segment"""
    try:
        # Output directory
        output_path.parent.mkdir(exist_ok=True, parents=True)

        # ffmpegVideo
        (
            ffmpeg
            .input(str(video_path), ss=start_time, to=end_time)
            .output(str(output_path), codec="copy", loglevel="error")
            .run(overwrite_output=True)
        )
        return True
    except Exception as e:
        print(f"{Color.RED}❌ Video clipping failed：{str(e)}{Color.RESET}")
        return False


def process_single_video(
        txt_file: Path,
        success_set: Set[str],
        failed_set: Set[str]
) -> bool:
    """Process single video clip"""
    file_prefix = txt_file.stem
    print(f"\n{Color.BLUE}🔍 Processing：{file_prefix}{Color.RESET}")

    # processing
    if file_prefix in success_set:
        print(f"{Color.YELLOW}⚠️  Already processed, skipping：{file_prefix}{Color.RESET}")
        return True
    if file_prefix in failed_set:
        print(f"{Color.YELLOW}⚠️ Processing failed，skipping{file_prefix}{Color.RESET}")
        return False

    try:
        timestamps = parse_timestamp_data(txt_file)
        start_ts = timestamps[0]
        end_ts = timestamps[-1]

        start_time = start_ts * MICROSECOND_TO_SECOND
        end_time = end_ts * MICROSECOND_TO_SECOND

        print(f"{Color.BLUE}⏱️  Clip time range：{start_time:.2f}s - {end_time:.2f}s{Color.RESET}")

        # 2. Video
        video_path = get_video_file_by_prefix(Path(VIDEO_INPUT_DIR), file_prefix)
        if not video_path:
            raise FileNotFoundError(f"No matching video found{file_prefix}")

        print(f"{Color.BLUE}📽️  Found video file：{video_path}{Color.RESET}")

        # 3. Video
        video_duration = get_video_duration(video_path)
        if end_time > video_duration:
            print(f"{Color.YELLOW}⚠️  Adjusted end time (exceeds video duration)：{video_duration:.2f}s{Color.RESET}")
            end_time = video_duration

        if start_time >= end_time:
            raise ValueError(f"Invalid time range: start time({start_time}) >= end time({end_time})")

        # 4. Execute clippingVideo
        output_path = Path(CLIP_OUTPUT_DIR) / f"{file_prefix}{video_path.suffix}"
        if clip_video(video_path, output_path, start_time, end_time):
            # Clip succeeded
            save_to_record(SUCCESS_RECORD, file_prefix)
            success_set.add(file_prefix)
            print(f"{Color.GREEN}✅ Clip succeeded：{output_path}{Color.RESET}")
            return True
        else:
            raise Exception("ffmpeg clipping returned error")

    except Exception as e:
        # Clip failed
        if file_prefix not in failed_set:
            save_to_record(FAILED_RECORD, file_prefix)
            failed_set.add(file_prefix)
        print(f"{Color.RED}❌ Processing failed：{file_prefix} - {str(e)}{Color.RESET}")
        return False


def run_clip_batch() -> None:
    """Execute batch clipping task"""
    # 1. Check dependencies
    if not check_ffmpeg():
        sys.exit(1)

    # 2. Initialize environment
    init_env()

    # 3. Collect files and statistics
    valid_files = collect_valid_files()
    total_count = len(valid_files)
    success_set = load_processed_files(SUCCESS_RECORD)
    success_count = len(success_set)
    failed_set = load_processed_files(FAILED_RECORD)
    failed_count = len(failed_set)

    # Count remaining files
    remaining_files = []
    for file in valid_files:
        if file.stem not in success_set and file.stem not in failed_set:
            remaining_files.append(file)
    remaining_count = len(remaining_files)

    # 4. Display statistics
    print(f"\n{Color.BLUE}======================================={Color.RESET}")
    print(f"{Color.BLUE}📊 Task statistics{Color.RESET}")
    print(f"{Color.BLUE}Total valid files：{total_count}{Color.RESET}")
    print(f"{Color.GREEN}✅ Successfully clipped：{success_count}{Color.RESET}")
    print(f"{Color.RED}❌ Clip failed：{failed_count}{Color.RESET}")
    print(f"{Color.YELLOW}🔄 Files to process：{remaining_count}{Color.RESET}")
    print(f"{Color.BLUE}📁 Output directory：{CLIP_OUTPUT_DIR}{Color.RESET}")
    print(f"{Color.BLUE}======================================={Color.RESET}")

    # 5. Exit if no remaining files
    if remaining_count == 0:
        print(f"{Color.GREEN}🎉 All files processed！{Color.RESET}")
        return

    # 6. Execute clipping
    print(f"\n{Color.BLUE}🚀 Processing {remaining_count}  files...{Color.RESET}")
    print(f"{Color.BLUE}======================================={Color.RESET}")

    current = 1
    success = 0
    fail = 0

    for txt_file in remaining_files:
        print(f"\n{Color.BLUE}[{current}/{remaining_count}]{Color.RESET}")
        if process_single_video(txt_file, success_set, failed_set):
            success += 1
        else:
            fail += 1
        current += 1

    # 7. Task summary
    print(f"\n{Color.BLUE}======================================={Color.RESET}")
    print(f"{Color.GREEN}🎉 ！{Color.RESET}")
    print(f"{Color.BLUE}📋 processing{remaining_count} {Color.RESET}")
    print(f"{Color.GREEN}✅ {success} {Color.RESET}")
    print(f"{Color.RED}❌ failed{fail} {Color.RESET}")
    print(f"{Color.BLUE}======================================={Color.RESET}")


# -------------------------- --------------------------
def main():
    """Main: execute video clipping task"""
    print(f"{Color.BLUE}🚀 Video clipping tool started...{Color.RESET}")
    print("=======================================")

    try:
        run_clip_batch()
    except Exception as e:
        print(f"{Color.RED}❌ Runtime error：{str(e)}{Color.RESET}")
        sys.exit(1)

    print(f"\n{Color.GREEN}🎉 All tasks completed！{Color.RESET}")


if __name__ == "__main__":
    main()
