# Kế hoạch tối ưu fusion pose từ 2 camera (đầu ra WHAM)

Ngày: 2026-10-21 · Phạm vi: 1 người, đoạn `video_0_seg_2` (43 frame, 25 fps ≈ 1.72 s)

---

## 1. Kiểm tra dữ liệu: cái gì dùng được?

### 1.1 Tổng quan từng file

| File | Nội dung (đã kiểm tra) | Dùng cho tối ưu? |
|---|---|---|
| `video_0_seg_2.pkl` | 1 người (id `0`), 43 frame liên tục (`frame_ids` 0–42). Có `pose`, `trans`, `pose_world`, `trans_world`, `betas`, `verts`. | **Có – nguồn chính** (khởi tạo + prior) |
| `tracking_results.pth` | `keypoints` COCO-17 (43×17×3: x, y, conf), `bbox` (cx, cy, scale), `features` ViT 1024-d, `init_*`, và bản `flipped_*`. | **Có – `keypoints` là dữ liệu quan trọng nhất thiếu trong pkl** |
| `calib.txt` | `fx = fy = 2896.31`, `cx = cy = 1024` (ảnh vuông 2048×2048). Chỉ có nội tham số, không có méo, không có ngoại tham số. | **Có** (nội tham số). Thiếu ngoại tham số giữa 2 camera |
| `slam_results.pth` | Shape (43, 7), toàn bộ là `[0,0,0,0,0,0,1]` (tịnh tiến 0, quaternion đơn vị). | **Không có thông tin**, nhưng xác nhận camera **đứng yên** |
| `output.mp4` | 4096×2048, 25 fps, 43 frame. Video render của WHAM (rộng gấp đôi ảnh gốc 2048 nên có thể là ảnh gốc + render cạnh nhau; chưa xem nội dung). | Chỉ để kiểm tra bằng mắt |

### 1.2 Chi tiết từng trường

| Trường | Đánh giá | Ghi chú |
|---|---|---|
| `keypoints` (tracking) | **Tốt** | Confidence trung bình theo khớp 0.71–0.96; **không có điểm nào < 0.3**. Tọa độ pixel nằm trong ảnh 2048×2048. Dải x/y của keypoint (1144–1775 / 928–1658) khớp với mesh WHAM chiếu bằng K trong calib (1114–1791 / 857–1686) ⇒ **K trong calib nhất quán với dữ liệu** |
| `pose[:,3:]` (23 khớp cục bộ) | Dùng được | Giống hệt `pose_world[:,3:]` (chênh lệch tối đa 0) ⇒ chỉ cần fuse một bản |
| `pose[:,:3]` (global orient hệ camera) | Dùng được | Cần xoay sang hệ chung khi fuse |
| `pose_world[:,:3]`, `trans_world` | Dùng cẩn thận | Mượt (jitter gia tốc ≈ 0.007) nhưng **hệ world của mỗi camera là độc lập** (chỉ chung hướng trọng lực), không thể trộn trực tiếp |
| `trans` (hệ camera) | **Nhiễu, cần tối ưu** | Jitter gia tốc ≈ 0.121 (gấp ~16 lần bản world). Độ sâu z: 6.86 → 9.75 m |
| `betas` | Dùng được sau khi thống nhất | Độ lệch chuẩn qua frame 0.07–0.27 trên từng chiều, trong khi dáng người là cố định ⇒ nên thay bằng 1 bộ chung |
| `verts` (43×6890×3) | Dùng được | Dùng để kiểm tra chiếu và tính khớp; cần `J_regressor` để lấy khớp 3D |
| `features`, `init_*`, `flipped_*` | Không cần | Đặc trưng nội bộ của WHAM / chỉ khởi tạo frame đầu |

### 1.3 Phát hiện quan trọng: độ sâu và quỹ đạo không nhất quán

Camera đứng yên (SLAM = đơn vị), nên quãng đường người đi phải **như nhau** ở hệ camera và hệ world. Thực tế:

| | Dịch chuyển đầu → cuối | Tổng quãng đường |
|---|---|---|
| `trans_world` | 1.35 m | 1.38 m |
| `trans` (hệ camera) | 2.50 m | 4.44 m (do jitter) |

Ước lượng độ sâu thô từ chiều cao keypoint 2D (giả sử người cao 1.75 m, chỉ mang tính tham khảo) cho 7.4 → 11.6 m, trong khi `trans` cho 7.3 → 9.8 m.
**Kết luận:** độ sâu từ một camera là thành phần kém tin cậy nhất; chưa thể nói bản nào đúng. Đây chính là chỗ camera thứ 2 mang lại giá trị lớn nhất, nên bước tối ưu cần dựa vào **reprojection 2D ở cả hai view**, không dựa vào `trans`.

### 1.4 Còn thiếu (cần để làm đủ pipeline)

1. `pkl` + `tracking_results.pth` của **camera thứ 2** (cùng đoạn thời gian).
2. **Ngoại tham số** giữa 2 camera (R, T). Nếu không có, ước lượng ở Bước 1.
3. **Model SMPL** (`SMPL_NEUTRAL.pkl`) và regressor khớp COCO-17 của WHAM (để lấy khớp 3D khớp với keypoint 2D).
4. Nội tham số + ảnh kích thước của camera thứ 2 (nếu khác camera 1).
5. Mốc thời gian của hai camera (để kiểm tra đồng bộ; 43 frame ở camera 1 đã liên tục).

---

## 2. Kế hoạch tối ưu

### Bước 0 – Chuẩn bị và kiểm tra đầu vào
- Đọc 2 bộ (pkl + tracking), kiểm tra `frame_ids` trùng nhau; nếu lệch thì ghép theo timestamp, nội suy slerp (pose) và tuyến tính (trans).
- Dựng SMPL, tính `J = J_regressor @ verts`; tính `J_coco` để so với keypoint 2D.
- **Kiểm tra sanity:** chiếu `J_coco` bằng K, tính sai số so với `keypoints` (baseline reprojection error ở từng camera). Đây là số "trước tối ưu".

### Bước 1 – Đưa hai kết quả về cùng hệ tọa độ
- **Có ngoại tham số:** dùng `pose`/`trans` (hệ camera 1 và 2), biến đổi sang hệ chung (có thể chọn camera 1 làm hệ chung).
- **Chưa có:** ước lượng ngoại tham số từ chính dữ liệu:
  1. Dùng bản world (đã căn trọng lực): chỉ còn yaw + tịnh tiến (4 bậc tự do), giải Kabsch trên quỹ đạo khớp 3D của hai camera.
  2. Tinh chỉnh bằng reprojection 2D ở cả hai view (xem Bước 3), để có thể cho `R, T` tối ưu cùng.
  3. Nếu có thể, hiệu chuẩn thật bằng ChArUco rồi dùng làm ground truth cho bước này.
- **Lưu ý:** `trans` là tham số tịnh tiến SMPL, không phải vị trí pelvis. Biến đổi vị trí `p = J_pelvis(betas) + trans`, rồi tính lại `trans`.

### Bước 2 – Gộp thô (khởi tạo)
- `betas`: median qua mọi frame của cả hai camera → **một bộ chung**.
- Pose 23 khớp cục bộ: trung bình trọng số trên SO(3) (quaternion / chordal mean), trọng số theo confidence keypoint tương ứng; không trung bình axis-angle từng thành phần.
- Global orient: xoay về hệ chung trước, rồi trung bình như trên.
- Tịnh tiến: tạm lấy trung bình có trọng số, sau đó chỉnh bằng Bước 3.

### Bước 3 – Tối ưu chung một SMPL cho cả hai view (bước chính)
**Biến:** `global_orient[t]`, `trans[t]`, `body_pose[t]` (23×3), `betas` chung (1 bộ), tùy chọn `R, T` giữa camera.

**Hàm mục tiêu:**

```
L = λ_reproj · Σ_cam Σ_t Σ_j  w_{cam,t,j} · ρ( ‖ π_cam(J3D_j(t)) − kp2D_{cam,t,j} ‖ )
  + λ_prior · Σ_t ‖ θ_t − θ_WHAM,fused,t ‖²        # bám pose WHAM đã fuse
  + λ_smooth · Σ_t ‖ J3D(t+1) − 2·J3D(t) + J3D(t−1) ‖²   # mượt gia tốc
  + λ_betas · ‖ β − β_median ‖²
  + λ_pose · pose_prior(θ_t)                      # prior (GMM/VPoser) tùy chọn
```

- `w = confidence` keypoint (đã đủ cao, không có điểm < 0.3); `ρ` là Huber/Cauchy.
- `π_cam` dùng K trong `calib.txt` (ảnh 2048×2048).
- Khởi tạo từ Bước 2; tối ưu 2 pha: (a) chỉ `trans` + `global_orient` (cố định pose), (b) mở khóa toàn bộ `body_pose` với `λ_prior` lớn rồi giảm dần.
- Công cụ: PyTorch + L-BFGS hoặc Adam; chạy theo cửa sổ trượt nếu độ dài đoạn lớn.
- **Lưu ý đa đoạn (`seg_2`):** các đoạn khác nhau cần ràng buộc liên tục tại biên (pose, trans, betas) khi ghép.

### Bước 4 – Làm mượt và hậu xử lý
- One Euro Filter (real-time) hoặc Savitzky–Golay (offline) trên quaternion và `trans`.
- Loại các điểm nhảy bất thường (vận tốc khớp vượt ngưỡng) và nội suy lại.

### Bước 5 – Xuất kết quả
- pkl mới cùng định dạng WHAM (`pose`, `trans`, `betas`, `verts`, `frame_ids`) để dùng thay cho đầu vào cũ, kèm `R, T` đã ước lượng.
- Video render so sánh trước/sau trên cả hai view.

---

## 3. Đánh giá

| Chỉ số | Mục đích | Mục tiêu |
|---|---|---|
| Sai số reprojection (px) mỗi camera, trước/sau | Độ khớp với 2D | Giảm so với baseline Bước 0 |
| Khoảng cách khớp 3D giữa hai camera sau khi về hệ chung (mm) | Mức bất đồng giữa 2 nguồn | Giảm sau tối ưu |
| Độ lệch chuẩn chiều dài xương qua thời gian | Ổn định dáng người | Gần 0 (do `betas` chung) |
| Jitter (gia tốc khớp trung bình) của `trans` và khớp | Độ mượt | `trans` hệ camera hiện 0.121 → tiến gần mức bản world (~0.007) mà không làm mất chuyển động thật |
| Nhất quán quãng đường camera-vs-world | Phát hiện lỗi độ sâu | Quãng đường 2 hệ khớp nhau (hiện 4.44 m so với 1.38 m) |
| MPJPE / PA-MPJPE (nếu có ground truth/mocap) | Độ chính xác tuyệt đối | Tùy dữ liệu |

**Ablation:** bật từng thành phần (betas chung → trung bình SO(3) → reprojection 2 view → smoothing) để biết phần nào thực sự cải thiện.

---

## 4. Rủi ro và cách xử lý

| Rủi ro | Cách xử lý |
|---|---|
| Ngoại tham số ước lượng từ WHAM bị lệch do độ sâu đơn mắt sai | Tinh chỉnh `R, T` chung với reprojection 2 view; ưu tiên hiệu chuẩn ChArUco |
| Hai camera không đồng bộ | Kiểm tra `frame_ids`/timestamp; nội suy slerp; nếu lệch lớn thì ước lượng offset thời gian bằng cross-correlation của quỹ đạo khớp |
| Keypoint COCO-17 ít khớp, dễ ambiguity ở cổ tay/bàn chân | Giữ `λ_prior` và `λ_pose` đủ lớn; dùng prior pose |
| Che khuất ở một view | Giảm trọng số view đó theo confidence; ràng buộc xương + prior giữ độ sâu |
| Lỗi nhầm trái/phải (flip) | Kiểm tra bằng `flipped_keypoints` và tính nhất quán 2 view/theo thời gian |
| Đoạn quá ngắn (43 frame) → smoothing/prior ảnh hưởng mạnh | Chạy trên các đoạn dài hơn, ghép đoạn với ràng buộc liên tục ở biên |

---

## 5. Thứ tự thực hiện đề xuất

1. Bước 0 (đo baseline) – xong khi có dữ liệu camera 2 + SMPL.
2. Bước 1–2 (căn hệ tọa độ + gộp betas/pose) – cho kết quả đầu tiên, ít rủi ro.
3. Bước 3 (tối ưu chung reprojection 2 view) – bước mang lại cải thiện chính, đặc biệt cho độ sâu.
4. Bước 4–5 (làm mượt, xuất, đánh giá) và ablation.

## 6. Việc cần bạn cung cấp để bắt đầu

- [ ] pkl + `tracking_results.pth` của camera thứ 2.
- [ ] Ngoại tham số 2 camera (nếu có) và nội tham số camera 2.
- [ ] Model SMPL và regressor COCO-17 của WHAM.
- [ ] Ground truth 3D/mocap (nếu có) để tính MPJPE.
- [ ] Yêu cầu: real-time hay offline, và độ dài đoạn video thực tế.
