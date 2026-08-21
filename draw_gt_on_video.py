import argparse
import cv2
import json
import numpy as np
from pathlib import Path

# Cấu trúc các đoạn xương để vẽ
FULL_SKELETON = [
    ("head", "neck"),
    ("neck", "right_shoulder"),
    ("neck", "left_shoulder"),
    ("neck", "pelvis"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_hand"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_hand"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_toe"),
    ("right_ankle", "right_foot"),
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_toe"),
    ("left_ankle", "left_foot"),
]

def estimate_intrinsics(w, h):
    """Ước lượng ma trận nội tại của camera từ kích thước video."""
    fx = fy = float((w * w + h * h) ** 0.5)
    cx = float(w / 2.0)
    cy = float(h / 2.0)
    return fx, fy, cx, cy

def project_point(xyz, fx, fy, cx, cy):
    """Chiếu điểm 3D (meters) trong toạ độ camera về 2D pixel."""
    arr = np.asarray(xyz, dtype=float).reshape(-1)
    if arr.size < 3:
        return None
    x, y, z = arr[:3]
    if not np.isfinite([x, y, z]).all() or z <= 1e-6:
        return None
    u = int(round(fx * x / z + cx))
    v = int(round(fy * y / z + cy))
    return (u, v)


# MPI-INF-3DHP 28-joint index mapping
MPI_28_JOINT_MAP = {
    "head": 6,
    "neck": 5,
    "pelvis": 4,
    "left_shoulder": 9,   "right_shoulder": 14,
    "left_elbow": 10,     "right_elbow": 15,
    "left_wrist": 11,     "right_wrist": 16,
    "left_hand": 12,      "right_hand": 17,
    "left_hip": 18,       "right_hip": 23,
    "left_knee": 19,      "right_knee": 24,
    "left_ankle": 20,     "right_ankle": 25,
    "left_foot": 21,      "right_foot": 26,
    "left_toe": 22,       "right_toe": 27,
}

def parse_gt_joints(frame_data: dict, camera_key: str = None) -> dict:
    """
    Parse GT frame thành dict {tên_khớp: xyz_meters}.
    Hỗ trợ 2 định dạng:
      1. Mảng pose3d (MPI-INF-3DHP 28-joint, mm) → dùng MPI_28_JOINT_MAP
      2. Dict camera_key → {tên: xyz} (meters) → đọc trực tiếp
    """
    # Định dạng 1: có trường pose3d (GT segment chuẩn)
    if "pose3d" in frame_data:
        pose3d = np.array(frame_data["pose3d"], dtype=float) / 1000.0  # mm → m
        joints = {}
        for name, idx in MPI_28_JOINT_MAP.items():
            if idx < len(pose3d):
                joints[name] = pose3d[idx]
        return joints
    
    # Định dạng 2: dict theo camera_key
    if camera_key and camera_key in frame_data:
        return {k: np.asarray(v, dtype=float) for k, v in frame_data[camera_key].items()}
    
    return {}

def main():
    parser = argparse.ArgumentParser(description="Vẽ Ground Truth (GT) pose 3D lên video thật")
    parser.add_argument("--video", type=str, required=True, help="Đường dẫn đến file video (VD: input/Seq1/imageSequence/Segments/video_0_seg_4.mp4)")
    parser.add_argument("--gt", type=str, required=True, help="Đường dẫn đến file JSON của segment HOẶC thư mục GT (VD: input/Seq1/imageSequence/Segments/video_0_seg_4.json)")
    parser.add_argument("--camera_key", type=str, default="camera0", help="Key của camera trong file json GT (VD: camera0, camera1...)")
    parser.add_argument("--output", type=str, default="output_gt_video.mp4", help="Đường dẫn file video đầu ra")
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    gt_path = Path(args.gt)
    
    if not video_path.exists():
        print(f"Lỗi: Không tìm thấy video: {video_path}")
        return
    if not gt_path.exists():
        print(f"Lỗi: Không tìm thấy GT: {gt_path}")
        return
        
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if w <= 0 or h <= 0:
        print(f"Lỗi: Không đọc được thông tin kích thước video từ {video_path}")
        return
        
    fx, fy, cx, cy = estimate_intrinsics(w, h)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
    
    frame_idx = 0
    drawn_frames = 0
    
    print(f"Đang xử lý video {w}x{h} ({fps} FPS), tổng cộng {total_frames} frames...")
    
    # Đọc trước file JSON nếu truyền vào là 1 file (dành cho segment)
    segment_data = None
    if gt_path.is_file() and gt_path.suffix == '.json':
        with open(gt_path, "r") as f:
            segment_data = json.load(f)
            if not isinstance(segment_data, list):
                # Nếu chỉ là 1 dict duy nhất
                segment_data = [segment_data]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        joints_dict = None
        current_frame_id = frame_idx + 1
        
        # Nếu GT là file JSON của segment
        if segment_data is not None:
            if frame_idx < len(segment_data):
                frame_data = segment_data[frame_idx]
                joints_dict = parse_gt_joints(frame_data, args.camera_key)
                if "frame_id" in frame_data:
                    current_frame_id = frame_data["frame_id"]
        # Nếu GT là thư mục chứa nhiều file JSON
        else:
            gt_file = gt_path / f"frame_{current_frame_id:06d}.json"
            if gt_file.exists():
                with open(gt_file, "r") as f:
                    data = json.load(f)
                joints_dict = parse_gt_joints(data, args.camera_key)
                
        if joints_dict:
            # Vẽ pose
            proj = {}
            for name, xyz in joints_dict.items():
                p = project_point(xyz, fx, fy, cx, cy)
                if p is not None:
                    proj[name] = p
            
            # Vẽ các đường nối (xương)
            for a, b in FULL_SKELETON:
                if a in proj and b in proj:
                    cv2.line(frame, proj[a], proj[b], (200, 200, 200), 2, cv2.LINE_AA)
            
            # Vẽ các điểm khớp (joint)
            for p in proj.values():
                cv2.circle(frame, p, 4, (0, 255, 255), -1, cv2.LINE_AA)
                
            drawn_frames += 1

        
        cv2.putText(frame, f"Frame {current_frame_id}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"Đã xử lý {frame_idx}/{total_frames} frames...")
            
    cap.release()
    out.release()
    print(f"Hoàn thành! Đã vẽ GT lên {drawn_frames} frames.")
    print(f"Video đã được lưu tại: {args.output}")

if __name__ == "__main__":
    main()
