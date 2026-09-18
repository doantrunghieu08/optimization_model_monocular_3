# Hướng Dẫn Chi Tiết Ý Nghĩa Các Cột Trong Báo Cáo CSV

> [!NOTE]
> Tài liệu này tổng hợp và giải thích chi tiết ý nghĩa của tất cả các cột dữ liệu trong các file báo cáo CSV được sinh ra bởi hệ thống Fusion & Evaluation 3D Human Pose.

---

## 1. Báo Cáo Thử Nghiệm Tổng Hợp (`brute_force_local_report.csv` / Google Sheets)

File báo cáo này tổng hợp kết quả chạy thực nghiệm vét cạn (brute-force) hoặc kiểm chứng (ablation study) trên các phân đoạn video và cặp camera khác nhau.

### 1.1. Định danh Dữ liệu & Cặp Camera (Metadata)

| Tên Cột | Mô Tả Ý Nghĩa | Ví Dụ Dữ Liệu |
| :--- | :--- | :--- |
| **`Set`** | Đường dẫn hoặc phân đoạn tập dữ liệu gốc | `Seq1/wham_output_video_0_seg_1` |
| **`Segment`** | Phân đoạn video đang thực hiện thử nghiệm | `seg_1`, `seg_2`, `seg_5` |
| **`Cam Master`** | ID camera chính được dùng làm góc nhìn tham chiếu (Camera 1) | `cam2`, `cam0` |
| **`Cam Slave`** | ID camera phụ bổ trợ quan sát (Camera 2) | `cam5`, `cam8` |

### 1.2. Tham Số Cấu Hình Thử Nghiệm (Pipeline Configurations)

| Tên Cột | Cấu Hình Tương Ứng | Mô Tả Chi Tiết |
| :--- | :--- | :--- |
| **`Alpha`** | `fusion.belief.alpha` | Hệ số phạt theo khoảng cách 3D trong hàm tin cậy Belief score. |
| **`Beta`** | `fusion.belief.beta` | Hệ số lan truyền độ tin cậy qua cấu trúc xương (skeleton). |
| **`Global Belief`** | `fusion.belief.global` | Bật (`true`) / Tắt (`false`) hòa trộn belief với các khớp láng giềng kề nhau. |
| **`Local Method`** | `fusion.belief.local_method` | Phương pháp tính belief cục bộ (`naive_distance_belief` hoặc `optical_aware_belief`). |
| **`Kinematic Constraints`** | `optimization.use_kinematic_constraints` | Bật/tắt ràng buộc bảo toàn độ dài xương trong bộ giải SLSQP. |
| **`Loss Type`** | `optimization.loss_type` | Hàm mất mát tối ưu hóa trong SLSQP (`huber` kháng outlier hoặc `mse`). |
| **`Optimization Enabled`** | `optimization.enabled` | Trạng thái bật/tắt bộ tối ưu hóa SLSQP. |
| **`Learnable`** | `learnable.enabled` | Trạng thái bật/tắt module học máy hậu xử lý SMPLify. |
| **`Learnable Extra`** | `learnable_extra.enabled` | Trạng thái bật/tắt module Learnable Extra. |

### 1.3. Kết Quả Các Chỉ Số Đánh Giá (Main Evaluation Metrics)

| Tên Cột | Đơn Vị | Mô Tả Chi Tiết |
| :--- | :---: | :--- |
| **`MPJPE`** | `mm` | Sai số vị trí 3D trung bình các khớp của mô hình đề xuất **Fused**. *(Càng nhỏ càng tốt)* |
| **`PA-MPJPE`** | `mm` | Sai số MPJPE sau khi đã xoay chỉnh hình dạng Procrustes. *(Càng nhỏ càng tốt)* |
| **`MBLE`** | `mm` | Sai số độ dài xương trung bình so với ground truth (Mean Bone Length Error). |
| **`Accel Error (mm/frame^2)`** | `mm/f²` | Sai số gia tốc khung hình của mô hình Fused (đo độ mượt chuyển động). |
| **`Fusion MBLE`** | `mm` | Sai số độ dài xương tính riêng cho giai đoạn Fused. |
| **`LE MBLE`** | `mm` | Sai số độ dài xương tính riêng cho giai đoạn Learnable Extra. |
| **`Old MBLE`** | `mm` | Sai số độ dài xương ban đầu của dữ liệu thô (Raw Pose baseline). |
| **`GT Accel Error`** | `mm/f²` | Sai số gia tốc của Ground Truth (mặc định = 0.0). |
| **`Fusion Accel Error`** | `mm/f²` | Sai số gia tốc chuyển động của giai đoạn Fused. |
| **`LE Accel Error`** | `mm/f²` | Sai số gia tốc chuyển động của giai đoạn Learnable Extra. |
| **`Old Accel Error`** | `mm/f²` | Sai số gia tốc chuyển động của dữ liệu thô (Raw Pose baseline). |
| **`LE MPJPE Master`** | `mm` | Sai số MPJPE của mô hình Learnable Extra trên camera chính. |
| **`LE PA-MPJPE Master`** | `mm` | Sai số PA-MPJPE của mô hình Learnable Extra trên camera chính. |

### 1.4. Thống Kê Che Khuất & Độ Tin Cậy (Diagnostics & Belief)

| Tên Cột | Mô Tả Chi Tiết |
| :--- | :--- |
| **`belief Master`** | Mảng giá trị điểm tin cậy trung bình của từng khớp trên camera chính. |
| **`belief Slave`** | Mảng giá trị điểm tin cậy trung bình của từng khớp trên camera phụ. |
| **`Occluded Joint-Frames Master`** | Tổng số lượt khớp-frame bị che khuất (`vis = False`) trên camera chính. |
| **`Occluded Joint-Frames Slave`** | Tổng số lượt khớp-frame bị che khuất (`vis = False`) trên camera phụ. |

### 1.5. Mức Độ Cải Thiện & Môi Trường Thực Thi (Improvement & System Info)

| Tên Cột | Mô Tả Chi Tiết |
| :--- | :--- |
| **`Old MPJPE`** | Sai số MPJPE ban đầu của dữ liệu thô (Raw Pose Baseline). |
| **`Old PA-MPJPE`** | Sai số PA-MPJPE ban đầu của dữ liệu thô (Raw Pose Baseline). |
| **`% Δ_MPJPE`** | Tỷ lệ % cải thiện MPJPE so với thô ($\Delta > 0$ là giảm sai số/tốt hơn). |
| **`% Δ_PA-MPJPE`** | Tỷ lệ % cải thiện PA-MPJPE so với thô ($\Delta > 0$ là giảm sai số/tốt hơn). |
| **`OS Version`** | Phiên bản hệ điều hành của máy thực thi thí nghiệm. |
| **`Username`** | Tên tài khoản người dùng/máy chạy thí nghiệm. |
| **`Timestamp`** | Thời gian bắt đầu hoặc hoàn thành lượt chạy thí nghiệm. |

---

## 2. Các File Báo Cáo Chi Tiết Theo Frame (`output/evaluation_results/`)

Các file này lưu dữ liệu chi tiết theo từng khung hình (frame-by-frame) để phân tích biến động chuyển động.

### 2.1. Các file `MPJPE_cam1.csv`, `PA-MPJPE_cam1.csv`, `PCK_cam1.csv`

| Tên Cột | Mô Tả Ý Nghĩa |
| :--- | :--- |
| **`Frame`** | Số thứ tự khung hình trong video (`0`, `1`, `2`...). |
| **`Evaluated_Camera`** | Camera đang được đánh giá (`camera1` hoặc `camera2`). |
| **`Ground_Truth_Camera`** | Tên camera tương ứng trong tập dữ liệu chuẩn Ground Truth. |
| **`<module>_priority1_mm`** | Sai số trung bình (mm hoặc %) của nhóm khớp ưu tiên 1 thuộc `<module>` (`posed`, `fused`, `fusion-learnable`, `only_learnable`). |
| **`<module>_priority2_mm`** | Sai số trung bình (mm hoặc %) của nhóm khớp ưu tiên 2 thuộc `<module>`. |
| **`<module>_<joint>_mm`** | Sai số mm chi tiết của khớp `<joint>` tại frame đó. |
| **`AVERAGE`** | Dòng cuối cùng lưu giá trị trung bình cộng của tất cả các frame. |

### 2.2. File `MBLE_cam1.csv` (Sai Số Độ Dài Xương)

| Tên Cột | Mô Tả Ý Nghĩa |
| :--- | :--- |
| **`Frame`** | Số thứ tự khung hình (`0`, `1`, `2`...). |
| **`Evaluated_Camera`** | Camera đang đánh giá (`camera1` / `camera2`). |
| **`Ground_Truth_Camera`** | Tên camera ground truth tương ứng. |
| **`Module`** | Tên module xuất kết quả (`posed`, `fused`, `only_learnable`...). |
| **`Bone`** | Tên đoạn xương đang đánh giá (ví dụ: `left_knee-left_ankle`). |
| **`Pred_Length_mm`** | Độ dài đoạn xương dự đoán (mm). |
| **`GT_Length_mm`** | Độ dài đoạn xương thực tế từ Ground Truth (mm). |
| **`Bone_Error_mm`** | Sai số chênh lệch độ dài $|Pred - GT|$ (mm). |
| **`Frame_MBLE_mm`** | Sai số MBLE trung bình của toàn bộ khung xương tại frame đó. |

### 2.3. File `Accel_cam1.csv` (Sai Số Gia Tốc & Độ Mượt)

| Tên Cột | Mô Tả Ý Nghĩa |
| :--- | :--- |
| **`Frame`** | Số thứ tự khung hình (`0`, `1`, `2`...). |
| **`Evaluated_Camera`** | Camera đang đánh giá (`camera1` / `camera2`). |
| **`Ground_Truth_Camera`** | Tên camera ground truth tương ứng. |
| **`Module`** | Tên module xuất kết quả (`posed`, `fused`, `only_learnable`...). |
| **`Joint`** | Tên khớp đang đánh giá (ví dụ: `right_wrist`). |
| **`Pred_Accel_mm_frame2`**| Gia tốc chuyển động dự đoán ($\text{mm/f}^2$). |
| **`GT_Accel_mm_frame2`**  | Gia tốc chuyển động thực tế từ Ground Truth ($\text{mm/f}^2$). |
| **`Accel_Error_mm_frame2`**| Sai số gia tốc $|Pred\_Accel - GT\_Accel|$ ($\text{mm/f}^2$). |
| **`Frame_Accel_Error_mm_frame2`** | Sai số gia tốc trung bình của tất cả các khớp tại frame đó. |
