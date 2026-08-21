import os
import subprocess
from pathlib import Path
import re

def main():
    input_dir = Path("input/Seq1/imageSequence/Segments")
    output_dir = Path("output/gt_segments")
    
    if not input_dir.exists():
        print(f"Lỗi: Không tìm thấy thư mục {input_dir}")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Tìm tất cả các file .mp4
    mp4_files = list(input_dir.glob("*.mp4"))
    
    print(f"Found {len(mp4_files)} segment videos. Starting process...")
    
    for mp4_file in mp4_files:
        json_file = mp4_file.with_suffix(".json")
        
        if not json_file.exists():
            print(f"Skipping {mp4_file.name} because json file not found.")
            continue
            
        # Lấy id camera từ tên file (ví dụ: video_0_seg_4 -> camera0)
        match = re.search(r"video_(\d+)", mp4_file.name)
        if not match:
            print(f"Cannot determine camera key from {mp4_file.name}")
            continue
            
        camera_id = match.group(1)
        camera_key = f"camera{camera_id}"
        
        output_file = output_dir / mp4_file.name
        
        print(f"Processing {mp4_file.name} (Camera: {camera_key})...")
        
        # Gọi script draw_gt_on_video.py
        cmd = [
            "python", "draw_gt_on_video.py",
            "--video", str(mp4_file),
            "--gt", str(json_file),
            "--camera_key", camera_key,
            "--output", str(output_file)
        ]
        
        try:
            # Chạy lệnh
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            print(f"  -> Done! Saved to: {output_file}")
        except subprocess.CalledProcessError:
            print(f"  -> Error processing {mp4_file.name}")

    print(f"Finished processing all {len(mp4_files)} segments!")
    print(f"All videos saved to: {output_dir}")

if __name__ == "__main__":
    main()
