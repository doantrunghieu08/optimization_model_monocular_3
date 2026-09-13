# Báo cáo Cải tiến & Nâng cấp Hệ thống Optimization Model Monocular (v3.0)

## Tổng quan

Tài liệu này tổng hợp toàn bộ các điểm cải tiến cốt lõi, nâng cấp kiến trúc và tối ưu thuật toán của phiên bản **Optimization Model Monocular v3.0** so với phiên bản gốc trước đó (monolithic notebook / v1-v2). Phiên bản này chuyển đổi hệ thống từ các kịch bản chạy đơn lẻ (standalone scripts) sang một **Framework Modular chuyên nghiệp**, có khả năng mở rộng, kiểm thử tự động và đánh giá đa chiều.

---

## 1. Kiến trúc Hệ thống & Tái cấu trúc Modular (Modular Refactoring)

- **Phân tách các Phase độc lập**:
  - Hệ thống được tách thành các module riêng biệt trong từng thư mục: `preprocess_pipeline`, `pose_pipeline`, `fusion_pipeline`, `learnable_pipeline`, `evaluation_pipeline`, và `visualization_pipeline`.
  - Mỗi phase chịu trách nhiệm cho một công đoạn duy nhất, giao tiếp qua dữ liệu định dạng chuẩn JSON (`keypoints3d` và `metadata`), giúp dễ dàng bảo trì, phát triển độc lập và viết Unit Test.
- **Hệ thống Quản lý Cấu hình Tập trung (`config_loader.py`)**:
  - Sử dụng YAML configuration (`configs/pipeline.yml` và `configs/brute_force.yml`) với cơ chế nạp biến môi trường động (`${ALPHA}`, `${BETA}`).
  - Tự động chuẩn hóa đường dẫn tuyệt đối (absolute path resolution), kiểm tra loại dữ liệu (type validation) và các điều kiện ràng buộc giữa các module trước khi thực thi (`validate_config`).
- **Lớp Tương thích Đa nền tảng (`compat.py`)**:
  - Giải quyết triệt để các lỗi tương thích deserialization/pickle từ các thư viện cũ (như `Joblib`, `NumPy` versions khác nhau) khi chạy trên Windows, Linux hoặc Google Colab.

---

## 2. Nâng cấp Thuật toán Kết hợp 3D (3D Fusion Pipeline Improvements)

- **Phát hiện Che khuất nâng cao bằng Ray-Casting (Torso Mesh Ray-Casting Occlusion)**:
  - Tích hợp mô hình lưới thân (torso mesh) trong không gian tọa độ camera để phát hiện hiện tượng khớp bị che khuất bởi chính cơ thể.
  - Tính toán điểm giao giữa tia chiếu (ray-casting) từ camera tới khớp với lưới torso; nếu khoảng cách nhỏ hơn ngưỡng $\tau$, khớp đó sẽ được gắn nhãn bị occlusion và điều chỉnh giảm điểm tin cậy (belief score).
- **Véc-tơ hóa và Tối ưu hóa Niềm tin Cục bộ (Belief Score Vectorization)**:
  - Thuật toán tính toán độ tin cậy kết hợp khoảng cách từ camera ($\alpha$) và mối liên kết skeleton lân cận ($\beta$) được tối ưu hóa bằng các phép toán ma trận NumPy/Torch, tăng tốc độ xử lý hàng loạt khung hình.
- **RANSAC & SLSQP Stabilization**:
  - Bổ sung cơ chế fallback linh hoạt và giới hạn tỷ lệ khớp lỗi (`max_fallback_ratio`), đảm bảo quá trình tìm ma trận biến đổi Similarity Transform giữa các camera không bị phân kỳ khi dữ liệu đầu vào nhiễu nặng.

---

## 3. Kiến trúc Đa nhánh Learnable (Dual-Branch Learnable Execution)

- **Hỗ trợ 2 nhánh thử nghiệm (Ablation Study Support)**:
  - **`learnable` (`Fusion + Learnable`)**: Lấy đầu vào từ kết quả 3D Fusion (`fused_output_dir`), cho phép mô hình Neural Network `NetBody25` học mượt hóa và tinh chỉnh dựa trên dữ liệu đã tối ưu không gian 3D đa camera.
  - **`learnable_extra` (`Pose + Learnable`)**: Lấy đầu vào trực tiếp từ kết quả 3D Pose đơn camera (`pose_output_dir`), giúp so sánh đối chứng trực tiếp hiệu quả của bước Fusion đối với mô hình Learnable.
- **Cơ chế Bảo vệ & Fallback (Guarded Prediction Fallback)**:
  - Nếu dự đoán của mạng Learnable gây ra sai số lớn hơn dữ liệu ban đầu, hệ thống sẽ tự động khôi phục (fallback) về pose gốc, ngăn chặn hiện tượng méo mó khung xương hoặc suy giảm chất lượng sau khi đi qua mạng.

---

## 4. Mở rộng Bộ chỉ số Đánh giá (Expanded Evaluation Metrics)

Ngoài các chỉ số truyền thống, phiên bản này bổ sung thêm các metric đánh giá chất lượng động học (kinematics) và độ ổn định thời gian:

1. **PA-MPJPE (Procrustes Aligned MPJPE)**: Sai số vị trí khớp trung bình (mm) sau khi căn chỉnh hình học Procrustes.
2. **MPJPE (Mean Per Joint Position Error)**: Sai số vị trí 3D tuyệt đối tính bằng mm.
3. **PCK (Percentage of Correct Keypoints)**: Tỷ lệ khớp được ước lượng đúng trong bán kính ngưỡng cho phép.
4. **MBLE (Mean Bone Length Error)**: Chỉ số mới đánh giá độ biến dạng chiều dài xương qua các khung hình, giúp đo lường sự co giãn bất thường của khung xương.
5. **Accel (Acceleration Error)**: Sai số gia tốc thời gian giữa các khung hình (đơn vị $\text{mm/s}^2$), phản ánh độ giật (jitter) và độ mượt mà của chuyển động 3D.

---

## 5. Công cụ Vét cạn Đa Camera & Báo cáo HTML (Brute-Force Runner & HTML Benchmark)

- **Chạy tự động 56+ cặp Camera (`brute_force_runner.py`)**:
  - Hỗ trợ quét toàn bộ các tổ hợp ghép cặp từ hệ thống đa camera (ví dụ: 8 camera).
  - Cho phép truyền tham số `alpha`, `beta` tùy biến qua lệnh CLI hoặc file cấu hình.
- **Báo cáo HTML Trực quan (`brute_force_report.html`)**:
  - Tự động tổng hợp và hiển thị bảng xếp hạng chi tiết (Rank 1 - Top performance được highlight màu nổi bật).
  - Xuất báo cáo đẹp mắt, đầy đủ biểu đồ so sánh các chỉ số PA-MPJPE, MPJPE, MBLE, Accel cho từng cặp camera.

---

## 6. Trực quan hóa & Tự động hóa Pipeline (Visualization & Automation)

- **Video Render 3 Cột (3-Column Comparison Animation)**:
  - Render so sánh trực quan đồng thời giữa **Video gốc**, **Khung xương 3D Fusion**, và **Khung xương Learnable** theo từng view camera.
- **Tự động Mã hóa Video & Lưu trữ Kết quả**:
  - Tích hợp tự động công cụ `ffmpeg` để mã hóa video đầu ra sang chuẩn `H.264/yuv420p` tương thích hoàn hảo trên Web/Colab.
  - Tự động đóng gói nén zip các báo cáo đánh giá (`evaluation_results.zip`).

---

## Bảng So sánh Tóm tắt

| Tính năng                     | Phiên bản Gốc (v1-v2 / Monolith)         | Phiên bản Mới (v3.0 Refactored)                                  |
| :------------------------------ | :------------------------------------------ | :------------------------------------------------------------------ |
| **Cấu trúc Mã nguồn** | Script/Notebook đơn khối, khó bảo trì | Tách module hóa chuẩn mực, dễ mở rộng                        |
| **Bắt Occlusion**        | Đơn giản hoặc chưa có                 | **Ray-Casting trên Torso Mesh 3D**                           |
| **Nhánh Learnable**      | Chỉ hỗ trợ 1 luồng cố định           | **Hỗ trợ Đa nhánh (`learnable` & `learnable_extra`)** |
| **Chỉ số Đánh giá**  | MPJPE, PA-MPJPE                             | **Bổ sung MBLE (Bone Error) & Accel (Jitter)**          |
| **Quét Đa Camera**      | Chạy thủ công từng cặp                 | **Vét cạn tự động 56+ cặp & Xuất HTML Report**         |
| **Kiểm tra Cấu hình**  | Không có                                  | **Validation tập trung (`config_loader.py`)**              |
| **Xuất Video**           | Phụ thuộc script ngoài                   | **Tự động mã hóa ffmpeg & nén Zip kết quả**           |
