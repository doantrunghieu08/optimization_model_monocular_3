# Optimization Model Monocular

Dự án này cung cấp một đường ống (pipeline) hoàn chỉnh để tối ưu hóa dáng điệu cơ thể người 3D (3D Human Pose Estimation) từ nhiều camera đơn (monocular cameras). Hệ thống tự động tiền xử lý, ước lượng độ trễ (offset), kết hợp (fusion) dữ liệu từ các camera, chạy mô hình tối ưu hóa (Learnable SMPLify) và đánh giá độ chính xác so với Ground Truth.

## Cấu trúc Pipeline

Pipeline bao gồm các module chính:
1. **Preprocess (Tiền xử lý)**: 
   - Trích xuất frames từ video (thông qua `ffmpeg`).
   - Ước lượng độ trễ thời gian (Offset Estimation) giữa các camera bằng thuật toán Dynamic Time Warping (DTW) dựa trên dữ liệu khớp xương.
2. **Pose**: Xử lý và xuất dữ liệu keypoints 2D.
3. **Fusion**: Kết hợp dữ liệu từ 2 camera (dùng RANSAC và SLSQP) để ước lượng vị trí 3D tối ưu nhất (giảm thiểu nhiễu và che khuất).
4. **Learnable (SMPLify)**: Sử dụng mô hình `NetBody25` và SMPL để tối ưu hóa và xuất ra lưới cơ thể (mesh) 3D mượt mà, bám sát bộ xương dự đoán.
5. **Evaluation**: Đánh giá độ chính xác bằng các số đo MPJPE, PA-MPJPE, PCK.
6. **Visualization**: Trực quan hóa kết quả 3D lên video. (Có thể bật/tắt).

## Yêu cầu Hệ thống

- **Python 3.10+**
- **Card đồ họa (GPU) NVIDIA** có hỗ trợ CUDA (để đảm bảo tốc độ chạy model Learnable và SMPL). Khuyến nghị cài đặt thư viện `torch` phiên bản CUDA. Cài đặt môi trường:
```bash
pip install -r requirements.txt
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Hướng dẫn sử dụng

### 1. Chạy Pipeline Chuẩn (2 Camera)
Nếu bạn đã biết trước chính xác 2 camera cần ghép nối, bạn có thể thiết lập đường dẫn trong `configs/pipeline.yml` và chạy trực tiếp script chính:

```bash
python main.py
```
- **Cấu hình**: Chỉnh sửa file `configs/pipeline.yml` (bật/tắt các bước, đổi đường dẫn input video/pkl, cài đặt device chạy GPU/CPU).

### 2. Chạy Vét Cạn Tìm Cặp Camera Tốt Nhất (Brute-Force)
Nếu bạn có một mảng nhiều camera (ví dụ 8 camera) và không biết cặp nào ghép với nhau sẽ cho kết quả 3D tốt nhất, bạn có thể dùng công cụ quét vét cạn. Công cụ này sẽ sinh ra tất cả các hoán vị (ví dụ 56 cặp), chạy toàn bộ quá trình tính toán và xuất báo cáo xếp hạng bằng HTML.

- **Cấu hình**: Mở `configs/brute_force.yml`, khai báo đường dẫn Ground Truth và danh sách các camera của từng segment.
- **Lệnh chạy**:
```bash
python brute_force_runner.py
```
- **Kết quả**: Sau khi chạy xong, file `brute_force_report.html` sẽ được tạo ra tại thư mục gốc. Bạn mở bằng trình duyệt để xem cặp camera nào (Rank 1 - được bôi nền vàng) cho chỉ số MPJPE / PA-MPJPE thấp nhất.

## Cấu trúc thư mục

```text
optimization_model_monocular_3/
├── configs/                   # Thư mục chứa cấu hình (pipeline.yml, brute_force.yml, keypoints map)
├── input/                     # Dữ liệu đầu vào (Video, file PKL của các camera, thư mục Ground Truth)
├── models/                    # Trọng số (weights) của SMPL model và các checkpoint
├── output/                    # Kết quả sinh ra từ các bước (Pose, Fusion, Learnable, Evaluation, Visualization)
├── _learnable_backend/        # Chứa mã nguồn kiến trúc mạng Neural NetBody25
├── *_pipeline/                # Các thư mục chứa mã nguồn riêng của từng bước (preprocess, fusion, pose, learnable,...)
├── brute_force_runner.py      # Script chạy vét cạn đa camera
├── main.py                    # Script chạy chính gốc (2 camera)
└── compat.py                  # Module vá lỗi tương thích thư viện cũ (numpy)
```

## Các chỉ số Đánh giá (Evaluation Metrics)
- **MPJPE (Mean Per Joint Position Error)**: Khoảng cách trung bình giữa khớp dự đoán và khớp thực tế (tính bằng mm).
- **PA-MPJPE (Procrustes Aligned MPJPE)**: Sai số sau khi đã dùng thuật toán Procrustes để loại bỏ sự sai lệch do xoay, tịnh tiến và tỷ lệ khung hình.
