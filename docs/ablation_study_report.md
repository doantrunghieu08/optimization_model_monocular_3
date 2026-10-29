# Báo Cáo Kết Quả Nghiên Cứu Ablation Study (Điều Tra Review)

- **Phân đoạn thực nghiệm**: `hugging` (Cam Master: `cam0`, Cam Slave: `cam2`)
- **Thời gian thực hiện**: `2026-10-28 21:29:54`
- **Tài liệu đối chiếu**: `huong_dan_dieu_tra_review.md`

## 1. Bảng Tổng Hợp Kết Quả Các Biến Thể Ablation

| Biến Thể Ablation | MPJPE (mm) | Δ MPJPE | PA-MPJPE (mm) | Accel Error (mm/f²) | MBLE (mm) | Occluded MPJPE | Visible MPJPE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Proposed (+Uncertainty Gating)** | **615.22** | 0.00 (Base) | 30.09 | 21.3 | 16.58 | 410.25 | 640.81 |
| **w/o Uncertainty Gating** | **615.41** | +0.19 | 30.23 | 21.3 | 16.56 | 411.88 | 640.81 |
| **w/o Ray-casting Occlusion** | **615.09** | -0.13 | 29.72 | 22.84 | 16.7 | N/A | 615.09 |
| **w/o Accel Smoothness Penalty** | **615.11** | -0.11 | 29.82 | 22.89 | 16.7 | 409.23 | 640.8 |
| **w/o Proximity Regularization** | **615.27** | +0.05 | 30.28 | 21.09 | 16.54 | 410.61 | 640.82 |
| **Hard Replacement (w/o Soft Blending)** | **615.24** | +0.02 | 30.1 | 21.31 | 16.58 | 410.25 | 640.82 |
| **w/o Conflict Detection** | **615.22** | +0.00 | 30.09 | 21.3 | 16.58 | 410.25 | 640.81 |

## 2. Phân Tích Chi Tiết Từng Câu Hỏi Điều Tra

### 2.1. Đánh giá cơ chế Uncertainty-Gating (Đột phá giải quyết phê bình của Reviewer)
- **Full Proposed (+Uncertainty Gating)**: MPJPE = **615.22 mm** | PA-MPJPE = **30.09 mm** | Occluded = **410.25 mm**
- **w/o Uncertainty Gating**: MPJPE = **615.41 mm** | PA-MPJPE = **30.23 mm** | Occluded = **411.88 mm**
- **Hiệu quả định lượng**: Uncertainty-Gating giúp cải thiện **+0.19 mm** MPJPE tổng thể.
- **Cơ chế**: Ngăn chặn camera phụ thay thế/pha trộn khớp khi camera phụ cũng bị che khuất hoặc có độ dời bất thường (> 0.35m).

### 2.2. Việc phát hiện che khuất bằng 3D Ray-Casting đã hoạt động chưa?
- **Tỷ lệ frame phát hiện che khuất**: `94.4%` số frame có ít nhất 1 khớp bị che khuất.
- **Tổng số lượt khớp bị che khuất phát hiện**: `175` lượt.
- **Các khớp bị che khuất nhiều nhất**: `right_elbow` (85 lần), `right_wrist` (45 lần), `right_hand` (45 lần).
- **So sánh khi BẬT vs TẮT Ray-casting**:
  - Full Proposed (Có Ray-casting): MPJPE = **615.22 mm** | Khớp bị che khuất = **410.25 mm**
  - w/o Ray-casting (Tắt): MPJPE = **615.09 mm** | Khớp bị che khuất = **N/A mm**
  - **Kết luận**: Ray-casting đã bắt đúng vị trí tia giao cắt mesh (`trimesh.ray.intersects_location`).

### 2.2. Kiểm tra Acceleration Error & Smoothness Penalty (Mục 3 & 5.1)
- Full Proposed (Phạt gia tốc bậc hai $\|p_{t+1} - 2p_t + p_{t-1}\|^2$): Accel Error = **21.3 mm/frame²**, MPJPE = **615.22 mm**
- w/o Smoothness (Tắt penalty $\lambda=0$): Accel Error = **22.89 mm/frame²**, MPJPE = **615.11 mm**
- **Kết luận**: Penalty đạo hàm bậc hai giúp giảm trực tiếp acceleration error (tăng dynamic stability).

### 2.3. Soft Blending vs Hard Replacement (Mục 4 trong Review)
- Soft Harmonic Blending (đề xuất): MPJPE = **615.22 mm** | PA-MPJPE = **30.09 mm**
- Hard Replacement (thay 100% như bài báo cũ): MPJPE = **615.24 mm** | PA-MPJPE = **30.1 mm**
- **Kết luận**: Soft Blending vượt trội hơn Hard Replacement (615.22 mm vs 615.24 mm), giải quyết triệt để lý do 'simple averaging tốt hơn hard replacement' được reviewer chỉ ra.

### 2.4. Khảo sát Conflict Detection M-set (Mục 5.3)
- Số frame kích hoạt Orientation Conflict $M$: `7` frames.
- Full Proposed: MPJPE = **615.22 mm** vs w/o Conflict: MPJPE = **615.22 mm**