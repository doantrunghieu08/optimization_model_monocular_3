# Báo Cáo Khắc Phục Các Vấn Đề Kỹ Thuật Trong Review

Tài liệu này ghi lại chi tiết các lỗi kỹ thuật gốc rễ được phát hiện từ tài liệu điều tra [`huong_dan_dieu_tra_review.md`](../huong_dan_dieu_tra_review.md), cùng với các thay đổi code đã thực hiện và kết quả kiểm thử qua `code-review-graph`.

---

## 1. Bảng Tổng Hợp Vấn Đề và Giải Pháp Khắc Phục

| Vấn đề Reviewer chỉ ra | Nguyên nhân kỹ thuật gốc rễ | Giải pháp đã sửa | File & Dòng sửa |
| :--- | :--- | :--- | :--- |
| **Acceleration Error cao nhất Table 1 / Bỏ smoothness lại giảm jitter** (Mục 3 & 5.1) | Code tính sai phân bậc 1 (phạt vận tốc $\|p_t - p_{t-1}\|^2$) thay vì sai phân bậc 2 (gia tốc $\|p_{t+1} - 2p_t + p_{t-1}\|^2$). Phạt vận tốc níu khớp lại, ép frame sau phải nhảy vọt theo dữ liệu mới $\rightarrow$ giật cục (jerk tăng). | Đổi toàn bộ sang sai phân bậc 2 đo gia tốc khớp 100% với metric đánh giá gia tốc `_compute_acceleration_error`. | [`learnable_pipeline/post_opt.py:L80-87`](../learnable_pipeline/post_opt.py)<br>[`fusion_pipeline/optimization.py:L80-109`](../fusion_pipeline/optimization.py) |
| **Simple averaging tốt hơn phương pháp đề xuất** (Mục 4) | Thay thế cứng 100% (hard replacement) bằng biến đổi RANSAC similarity toàn cục. Sai số affine và biến dạng phi tuyến làm khớp lệch nhiều hơn trung bình cộng. | Thay hard replacement bằng **Soft Blending** theo tỷ lệ trọng số tin cậy harmonic: $\alpha = \frac{H_1}{H_1 + H_2}$. | [`fusion_pipeline/correction.py:L76-123`](../fusion_pipeline/correction.py) |
| **Conflict detection không có tác dụng đo được** (Mục 5.3) | Khớp `m_set` sau khi sửa rotation mismatch không được đưa vào `a_new` (anchors), mà bị ném vào `f_list` để optimizer SLSQP giải lại từ đầu $\rightarrow$ kết quả sửa bị optimizer đè bẹp hoàn toàn. | Đưa `m_set` vào `a_new` (`a_new = sorted(set(a_list) \| k1_set \| k2_set \| m_set)`), bảo toàn các khớp đã hiệu chỉnh. | [`fusion_pipeline/executor.py:L264`](../fusion_pipeline/executor.py) |
| **Occlusion tính từ predicted mesh / không dùng ray-casting** (Mục 8 & 8.1) | Code cũ chỉ là heuristic chiếu 2D + so sánh độ sâu Z gần đúng; đồng thời bị tắt cứng trong config (`enabled: false`) do WHAM PKL thiếu `verts_cam`. | 1) Chuyển sang **3D Ray-Casting thật sự** dùng `trimesh`.<br>2) Bật `occlusion.enabled: true` trong config.<br>3) Hỗ trợ fallback tự nhận diện `verts` nếu không có `verts_cam`.<br>4) `load_torso_mask` fallback an toàn không gây crash. | [`fusion_pipeline/detector.py:L58-135`](../fusion_pipeline/detector.py)<br>[`fusion_pipeline/executor.py:L106-110`](../fusion_pipeline/executor.py)<br>[`configs/pipeline.yml:L40-42`](../configs/pipeline.yml) |
| **14 tham số cứng / Overfitting 1 subject** (Mục 6 & 7) | `HEIGHT = 1.63m` và tỷ lệ xương bị hardcode; biến `prev_result` trong fusion pipeline bị bỏ quên không gán lại ở cuối vòng lặp $\rightarrow$ `temporal_lambda` trong fusion hoàn toàn không có tác dụng qua các frame. | 1) Lưu và cập nhật `prev_result` và `prev_prev_result` qua từng frame.<br>2) `compute_dynamic_scale` tự ước tính theo tỷ lệ khoảng cách 2 vai (biacromial ratio) trước khi fallback `HEIGHT`. | [`fusion_pipeline/executor.py:L353-432`](../fusion_pipeline/executor.py)<br>[`fusion_pipeline/optimization.py:L46-58`](../fusion_pipeline/optimization.py) |

---

## 2. Chi Tiết Các Thay Đổi Mã Nguồn

### 2.1. Đổi Smoothness Term sang Sai Phân Bậc 2 (Acceleration Penalty)

- **File**: `learnable_pipeline/post_opt.py`
```python
# Temporal acceleration smoothness on pose and trans (2nd-order difference matching evaluation metric)
if n_frames > 2:
    accel_pose = pose_var[2:] - 2.0 * pose_var[1:-1] + pose_var[:-2]
    accel_trans = trans_var[2:] - 2.0 * trans_var[1:-1] + trans_var[:-2]
    loss_temp = torch.mean(torch.sum(accel_pose ** 2, dim=-1)) + torch.mean(torch.sum(accel_trans ** 2, dim=-1))
elif n_frames > 1:
    loss_temp = torch.mean(torch.sum((pose_var[1:] - pose_var[:-1]) ** 2, dim=-1)) + torch.mean(torch.sum((trans_var[1:] - trans_var[:-1]) ** 2, dim=-1))
else:
    loss_temp = torch.tensor(0.0, device=device)
```

- **File**: `fusion_pipeline/optimization.py`
```python
if name in prev_prev_cam1:
    p1_pp = np.asarray(prev_prev_cam1[name], dtype=float)
    penalty += float(np.sum((p1_curr - 2.0 * p1_prev + p1_pp) ** 2))
else:
    penalty += float(np.sum((p1_curr - p1_prev) ** 2))
```

---

### 2.2. Triển Khai 3D Ray-Casting & Fallback Dữ Liệu Lưới

- **File**: `fusion_pipeline/detector.py`
  Bắn tia 3D từ tâm camera $(0, 0, 0)$ hướng tới từng khớp trong `OCCLUSION_CHECK_JOINTS`. Nếu tia cắt bề mặt lưới thân 3D tại khoảng cách $t < d_{\text{joint}} - \tau$, khớp được đánh dấu là bị che khuất (`visibility = False`).

- **File**: `fusion_pipeline/executor.py`
```python
verts_cam1 = person_1.get("verts_cam", person_1.get("verts")) if person_1 else None
verts_cam2 = person_2.get("verts_cam", person_2.get("verts")) if person_2 else None
if person_1 is None or person_2 is None or verts_cam1 is None or verts_cam2 is None:
    raise ValueError("Neither 'verts_cam' nor 'verts' found in WHAM PKL inputs; fusion.occlusion.enabled requires mesh vertices")
```

---

### 2.3. Bảo Toàn Kết Quả Conflict Detection (`m_set`)

- **File**: `fusion_pipeline/executor.py`
```python
# Preserve conflict-corrected joints in anchors so SLSQP optimizer does not overwrite them
a_new = sorted(set(a_list) | k1_set | k2_set | m_set)
f_list = [n for n in names if n not in set(a_new)]
```

---

### 2.4. Soft Blending Cho Confidence & Rotation Corrections

- **File**: `fusion_pipeline/correction.py`
  Khi camera 1 có độ tin cậy vượt trội camera 2:
  $$p_{\text{blended}} = \alpha \cdot \text{Similarity}(p_1) + (1 - \alpha) \cdot p_2, \quad \text{với } \alpha = \frac{H_1}{H_1 + H_2}$$
  Tránh hiện tượng giật cục tọa độ và hạn chế sai số biến dạng toàn cục.

---

### 2.5. Sửa Lỗi Mất Chuỗi Thời Gian `prev_result` & Dynamic Scale

- **File**: `fusion_pipeline/executor.py`
  Cập nhật trạng thái `prev_prev_result = prev_result` và `prev_result = result` ở cuối mỗi vòng lặp xử lý frame (cả trong khối `try` và `except`).
- **File**: `fusion_pipeline/optimization.py`
  Hàm `compute_dynamic_scale` tự động ước lượng kích thước cơ thể từ khoảng cách hai vai nếu các tỷ lệ chi phụ bị thiếu, không mặc định gán cứng `1.63m`.

---

## 3. Kết Quả Kiểm Tra và Đánh Giá

### 3.1. Phân Tích Kiến Trúc qua `code-review-graph`
- **Số file thay đổi**: 6 file (`configs/pipeline.yml`, `fusion_pipeline/correction.py`, `fusion_pipeline/detector.py`, `fusion_pipeline/executor.py`, `fusion_pipeline/optimization.py`, `learnable_pipeline/post_opt.py`).
- **Số node thay đổi**: 51 nodes.
- **Node chịu ảnh hưởng gián tiếp**: 9 nodes trong 4 file (`learnable_pipeline/executor.py`, `brute_force_runner.py`, `pipeline.py`, `main.py`).
- **Tương thích ngược**: Tất cả tham số mới đều có giá trị mặc định (`blend=True`, `prev_prev_data=None`, `intrinsics=None`), không làm vỡ các module gọi ngoài.

### 3.2. Kết Quả Kiểm Thử (Unit / Self-Test)
- Tất cả 5 file Python biên dịch thành công (`py_compile.compile`).
- Tự động hóa kiểm thử tích hợp (3D Ray-Casting, Soft Blending, Dynamic Scale, SLSQP với 2nd-order acceleration penalty) đạt **100% PASS**:
  ```text
  Visibility: {'left_wrist': False, 'right_wrist': True, 'left_ankle': True, 'neck': True}
  ALL SELF-TESTS PASSED!
  ```
