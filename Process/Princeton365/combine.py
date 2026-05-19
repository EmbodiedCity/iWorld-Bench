import os
import shutil
import argparse

def copy_cliped_videos(work_parent_dir, output_root_dir):
    """
    work1~work30cliped_videosVideoOutput directory
    :param work_parent_dir: Parent directory of work1~work30
    :param output_root_dir: Output root directory
    """
    VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v')
    
    # Output root directorycliped_videos
    output_cliped_dir = os.path.join(output_root_dir, "cliped_videos")
    os.makedirs(output_cliped_dir, exist_ok=True)
    print(f"✅ Created output directory{output_cliped_dir}")

    total_work = 30
    found_cliped_count = 0
    copied_video_count = 0
    skip_work_list = []

    # work1~work30
    for work_num in range(1, total_work + 1):
        work_dir_name = f"work{work_num}"
        work_dir = os.path.join(work_parent_dir, work_dir_name)
        cliped_src_dir = os.path.join(work_dir, "cliped_videos")

        if not os.path.exists(work_dir):
            print(f"⚠️ [{work_num}/{total_work}] {work_dir_name} does not exist，skipping")
            skip_work_list.append(work_dir_name)
            continue

        # 2. cliped_videos
        if not os.path.exists(cliped_src_dir):
            print(f"⚠️ [{work_num}/{total_work}] {work_dir_name} has no cliped_videos folder, skipping")
            skip_work_list.append(work_dir_name)
            continue

        found_cliped_count += 1
        print(f"\n📌 [{work_num}/{total_work}] Processing {work_dir_name} cliped_videos...")

        # 3. cliped_videosVideo
        for root, dirs, files in os.walk(cliped_src_dir):
            for file in files:
                # Video
                if file.lower().endswith(VIDEO_EXTENSIONS):
                    src_video_path = os.path.join(root, file)
                    # work，os.path.join(output_cliped_dir, work_dir_name, file)
                    dst_video_path = os.path.join(output_cliped_dir, file)

                    # processingwork
                    if os.path.exists(dst_video_path):
                        file_name, file_ext = os.path.splitext(file)
                        dst_video_path = os.path.join(output_cliped_dir, f"{file_name}_{work_dir_name}{file_ext}")
                        print(f"ℹ️  Duplicate filename, renamed to：{os.path.basename(dst_video_path)}")

                    try:
                        shutil.copy2(src_video_path, dst_video_path)
                        copied_video_count += 1
                        print(f"✅ Copy succeeded：{file} → {os.path.basename(dst_video_path)}")
                    except Exception as e:
                        print(f"❌ Copy failed：{file} → Error：{str(e)}")

    print("\n" + "="*60)
    print(f"📊 Batch copy completed！")
    print(f"Total work count：{total_work}")
    print(f"Work dirs with cliped_videos：{found_cliped_count}")
    print(f"Videos copied successfully：{copied_video_count}")
    print(f"Skipped work count：{len(skip_work_list)}")
    if skip_work_list:
        print(f"skippingworkcolumns{', '.join(skip_work_list)}")
    print(f"VideoOutput directory：{output_cliped_dir}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Batch copy videos from work1~work30 cliped_videos')
    parser.add_argument('--work_parent', required=True, help='Parent directory of work1~work30')
    parser.add_argument('--output', required=True, help='Output root directory')
    args = parser.parse_args()

    copy_cliped_videos(args.work_parent, args.output)

if __name__ == "__main__":
    main()