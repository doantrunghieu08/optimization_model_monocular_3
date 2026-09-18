# Optimization Model Monocular

Dự án này cung cấp một đường ống (pipeline) hoàn chỉnh để tối ưu hóa dáng điệu cơ thể người 3D (3D Human Pose Estimation) từ nhiều camera đơn (monocular cameras). Hệ thống tự động tiền xử lý, ước lượng độ trễ (offset), kết hợp (fusion) dữ liệu từ các camera, chạy mô hình tối ưu hóa (Learnable SMPLify) và đánh giá độ chính xác so với Ground Truth.

Nếu bạn muốn chạy trên Colab Notebook thì có thể truy cập đường link sau (phiên bản 260821): https://colab.research.google.com/drive/1aV1j6aKP3EiMsqS5OKXRd-mLE8Nkpair

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
Nếu bạn có một mảng nhiều camera (ví dụ 8 camera) và không biết cặp nào ghép với nhau sẽ cho kết quả 3D tốt nhất, bạn có thể dùng công cụ quét vét cạn. Công cụ này sẽ sinh ra tất cả các hoán vị (ví dụ 56 cặp), chạy toàn bộ quá trình tính toán và xuất báo cáo xếp hạng.

- **Cấu hình**: Mở `configs/brute_force.yml`, khai báo đường dẫn Ground Truth và danh sách các camera của từng segment.
- **Lệnh chạy trên Local**:
```bash
python brute_force_runner_local.py
```
- **Lệnh chạy trên Colab / Cloud**:
```bash
python brute_force_runner.py
```
- **Kết quả**: Báo cáo CSV được lưu tự động vào thư mục `output/reports/brute_force_local_report.csv` và `output/reports/brute_force_full_report.csv`.

## Cấu trúc thư mục

```text
optimization_model_monocular_3/
├── configs/                   # Thư mục chứa các file cấu hình YAML (pipeline.yml, brute_force.yml,...)
├── docs/                      # Tài liệu phân tích kỹ thuật và báo cáo Markdown
├── input/                     # Dữ liệu đầu vào (Video, file PKL của các camera, Ground Truth)
├── models/                    # Trọng số (weights) của mô hình SMPL và các checkpoint
├── output/                    # Kết quả sinh ra từ các pipeline và thư mục reports/ chứa CSV/JSON
├── notebooks/                 # Thư mục chứa các Jupyter Notebooks phục vụ thử nghiệm & Ablation
├── scripts/                   # Thư mục chứa các script chạy vét cạn thực nghiệm và tiện ích phụ
├── src/                       # MÃ NGUỒN CHÍNH CỦA DỰ ÁN (Python Package)
│   ├── core/                  # Các module tiện ích lõi (config_loader, json_io, keypoints_map, compat)
│   └── pipelines/             # Các module pipeline xử lý theo từng giai đoạn
│       ├── preprocess/        # Tiền xử lý (DTW offset, extract 2D, calib)
│       ├── pose/              # Trích xuất dáng điệu 3D
│       ├── fusion/            # Hợp nhất đa camera (RANSAC, SLSQP, Correction)
│       ├── learnable/         # Learnable SMPLify & Neural Backend (NetBody25)
│       ├── evaluation/        # Đánh giá chỉ số (MPJPE, PA-MPJPE, PCK)
│       ├── visualization/     # Vẽ và xuất video 3D
│       └── orchestrator.py    # Điều phối toàn bộ luồng pipeline
├── tests/                     # Bộ kiểm thử Unit Tests tự động
├── brute_force_runner.py      # Script chạy vét cạn đa camera (Google Sheets / Colab)
├── brute_force_runner_local.py # Script chạy vét cạn đa camera dành riêng cho môi trường Local
├── main.py                    # Script chạy pipeline 2 camera chính
└── requirements.txt           # Danh sách các thư viện phụ thuộc
```

## Các chỉ số Đánh giá (Evaluation Metrics)
- **MPJPE (Mean Per Joint Position Error)**: Khoảng cách trung bình giữa khớp dự đoán và khớp thực tế (tính bằng mm).
- **PA-MPJPE (Procrustes Aligned MPJPE)**: Sai số sau khi đã dùng thuật toán Procrustes để loại bỏ sự sai lệch do xoay, tịnh tiến và tỷ lệ khung hình.
