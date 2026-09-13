# Báo cáo review lỗi toàn bộ repository

Ngày review: 2026-09-13
Phạm vi: toàn bộ mã Python, cấu hình YAML, notebook Colab, test và trạng thái output hiện có trong repository.

## Kết luận nhanh

- Không phát hiện lỗi cú pháp: toàn bộ file Python compile thành công.
- Có **5 lỗi mức cao**, **5 lỗi mức trung bình** và **2 vấn đề môi trường/trạng thái**.
- Rủi ro lớn nhất không phải crash trực tiếp mà là pipeline vẫn báo hoàn tất trong khi dùng dữ liệu fallback hoặc output cũ, làm kết quả evaluation có vẻ hợp lệ nhưng không thuộc lần chạy hiện tại.
- Bộ test hiện có quá hẹp để bắt các lỗi liên kết giữa preprocess → pose → fusion → learnable → evaluation → visualization.

## Lỗi mức cao

### H-01 — Tắt fusion vẫn có thể evaluation bằng fused output cũ

**Vị trí:** `fusion_pipeline/executor.py:256-269`, `pipeline.py:26-36`, `evaluation_pipeline/executor.py:319-328`.

`run_fusion()` trả về ngay khi `fusion.enabled=false`, trước bước dọn output. Tuy nhiên full pipeline vẫn chạy evaluation, và evaluation luôn đưa module `fused` vào danh sách bắt buộc, không phụ thuộc `fusion.enabled`. Nếu `output/fused_results` còn dữ liệu từ lần chạy trước, dữ liệu đó được ghép theo số frame với pose/GT mới và được tính như kết quả hiện tại.

**Tác động:** metric sai nhưng pipeline không nhất thiết báo lỗi. Đây là lỗi nhiễm dữ liệu giữa các lần chạy.

**Cách tái hiện tối thiểu:** chạy một cấu hình A để tạo fused output; đổi input sang cấu hình B, đặt `fusion.enabled=false`, sau đó chạy full pipeline. Evaluation vẫn đọc `fused_results` của A.

**Sửa tối thiểu:** khi fusion bị tắt, không đưa `fused` vào downstream/evaluation; hoặc fail sớm nếu stage yêu cầu fused output. Không dùng output cũ như fallback ngầm.

### H-02 — Fusion nuốt mọi exception và xuất raw pose dưới nhãn fused

**Vị trí:** `fusion_pipeline/executor.py:310-353`.

Khối `except Exception` bắt cả lỗi dữ liệu lẫn lỗi lập trình, gọi `make_raw_judgement_fallback()` rồi vẫn ghi `fused_data_*.json`. Pipeline tiếp tục và evaluation coi file đó là kết quả fusion bình thường.

**Tác động:** một lỗi hệ thống có thể làm toàn bộ hoặc nhiều frame không được fusion, nhưng tiến trình vẫn kết thúc với `[Fusion] Done` và sinh metric “fused”. Console có log `FAILED`, metadata có `fallback_reason`, nhưng không có tổng hợp/fail condition để ngăn kết quả sai đi tiếp.

**Sửa tối thiểu:** chỉ fallback cho lỗi dữ liệu đã biết; lỗi lập trình phải raise. Cuối stage phải báo tổng số fallback và fail nếu vượt ngưỡng cấu hình (mặc định nên là 0).

### H-03 — Evaluation có thể hoàn tất với 0 frame được đánh giá

**Vị trí:** `evaluation_pipeline/executor.py:382-397`, `evaluation_pipeline/executor.py:496-614`.

Frame bị bỏ qua khi thiếu metadata hoặc không tìm thấy source frame trong GT. Sau vòng lặp không có kiểm tra `evaluated_frames` rỗng; code vẫn tạo CSV và in `[Evaluation] Done`.

**Tác động:** người dùng nhận file báo cáo không có dữ liệu thật; brute-force có thể chuyển trạng thái này thành `inf`/`N/A` thay vì chỉ ra lỗi nguồn dữ liệu.

**Sửa tối thiểu:** raise `ValueError` khi không có frame hợp lệ; đồng thời in số frame bị bỏ và lý do.

### H-04 — Visualization ghép sai ảnh/video khi offset khác 0

**Vị trí:** `pose_pipeline/executor.py:101-145`, `visualization_pipeline/executor.py:389-405`, `visualization_pipeline/executor.py:501-522`.

Pose đã lưu đúng `source_frame_indices` cho từng camera, nhưng visualization tra ảnh bằng `frame_id` của output đồng bộ (`image_map.get(frame_id)`). Với offset khác 0, `pose_data_1` có thể tương ứng frame nguồn 1 ở camera 1 nhưng frame nguồn 6 ở camera 2; visualization vẫn hiển thị `images_frame_1.jpg` cho cả hai.

**Tác động:** skeleton được overlay lên sai frame và video so sánh gây hiểu nhầm về chất lượng mô hình. Đây là nhánh cốt lõi vì pipeline có bước ước lượng offset.

**Sửa tối thiểu:** tạo mapping output frame → `source_frame_indices[camera] + 1` từ pose metadata và dùng mapping này cho cả comparison lẫn project2d.

### H-05 — Brute-force CLI hỏng ngoài Google Colab

**Vị trí:** `brute_force_runner.py:18-36`, `requirements.txt`.

Import `google.colab`, `google.auth` và `gspread` được bọc trong `try/except ImportError`, nhưng sau khi cảnh báo code vẫn gọi các tên `auth`, `default`, `gspread`. Các dependency này cũng không có trong `requirements.txt`.

**Bằng chứng tái hiện:** import module ngoài Colab rồi gọi `get_gspread_client()` trả về:

```text
Cảnh báo: Không tìm thấy thư viện google colab/gspread.
NameError: name 'auth' is not defined
```

README hướng dẫn chạy `python brute_force_runner.py` như CLI thông thường nên đây không chỉ là giới hạn notebook.

**Sửa tối thiểu:** hoặc khai báo công cụ này chỉ chạy trên Colab và fail sớm với thông báo rõ ràng, hoặc tách xác thực Colab khỏi `gspread` và thêm dependency cần thiết.

## Lỗi mức trung bình

### M-01 — Visualization luôn yêu cầu nhánh learnable dù đã tắt

**Vị trí:** `visualization_pipeline/executor.py:565-582`, `visualization_pipeline/executor.py:367-385`.

`run_visualization()` luôn load `learnable_output_dir`, và project2d luôn thêm module `learnable`. Trong khi đó pipeline cho phép `learnable.enabled=false` (đúng như `configs/pipeline.yml` hiện tại).

**Tác động:** khi bật `visualization.enabled=true`, cấu hình hợp lệ có thể crash vì thiếu learnable output; nếu thư mục cũ tồn tại, visualization có thể dùng nhầm output cũ.

**Sửa tối thiểu:** chỉ load/render các module đang bật; lấy giao theo frame ID và kiểm tra provenance.

### M-02 — Pose không hỗ trợ `betas` tĩnh dạng `(1, 10)`

**Vị trí:** `pose_pipeline/smpl_runner.py:29-35`, `pose_pipeline/executor.py:41-50`.

`slice_person_frames()` cố ý giữ nguyên tensor không có số frame bằng `total_frames`, nhưng `get_3d_joints_for_frame()` luôn lấy `betas[frame_idx:frame_idx+1]`. Từ frame thứ hai, `betas` dạng `(1, 10)` trở thành mảng `(0, 10)`.

**Bằng chứng:** phép slice hiện tại ở frame 2 cho kết quả `frame_2_betas_shape=(0, 10)`. Các module `calib` và `learnable` đã coi `(1, 10)` là shape hợp lệ, nên hành vi giữa các stage không nhất quán.

**Sửa tối thiểu:** nếu `betas.shape[0] == 1` thì luôn dùng hàng 0; nếu không thì mới index theo frame.

### M-03 — CLI bỏ qua checkpoint trong YAML

**Vị trí:** `main.py:31-32`.

Sau khi load và validate YAML, `main.py` luôn gán lại `learnable.checkpoint` thành `models/best_ckpt.pth.tar`. Mọi checkpoint tùy chỉnh trong file `--config` bị bỏ qua.

**Tác động:** chạy nhầm model mà không có cảnh báo; kết quả khó tái lập giữa CLI và `brute_force_runner.py` vì brute-force không đi qua đoạn override này.

**Sửa tối thiểu:** chỉ đặt giá trị mặc định bằng `setdefault`, không ghi đè giá trị người dùng.

### M-04 — Import module có side effect ghi source code xuống đĩa

**Vị trí:** `fusion_pipeline/executor.py:357-382`.

File fusion chứa một bản sao lớn của toàn bộ learnable backend trong `LEARNABLE_VENDOR_FILES` và ghi các file còn thiếu ngay lúc import. Chỉ cần `import pipeline` cũng có thể thay đổi filesystem hoặc fail trong môi trường read-only.

**Tác động:** test/import không còn thuần đọc; source bị nhân đôi và có nguy cơ lệch phiên bản với `_learnable_backend` đã commit.

**Sửa tối thiểu:** xóa khối sinh file; dùng trực tiếp `_learnable_backend` đã có trong repository.

### M-05 — Chuyển đổi video báo thành công dù ffmpeg thất bại

**Vị trí:** `main.py:45-58`.

`subprocess.run()` không dùng `check=True`, nhưng code luôn in `Created ...` ngay sau đó.

**Tác động:** thiếu codec, ffmpeg lỗi hoặc ổ đĩa đầy vẫn được báo là tạo file thành công.

**Sửa tối thiểu:** thêm `check=True`; chỉ in thành công sau khi return code bằng 0 và output tồn tại.

## Vấn đề môi trường và trạng thái hiện tại

### E-01 — Môi trường chạy mặc định không tái tạo được test

- `venv/Scripts/python.exe` trỏ tới `C:\Users\Trung\AppData\Local\Programs\Python\Python311\python.exe`, nhưng interpreter đó không còn tồn tại.
- `python` mặc định hiện tại không có `numpy`.
- Conda env `gvhmr` có phần lớn thư viện khoa học nhưng thiếu `ruamel.yaml`, khiến 3 test module không import được.
- Chạy `unittest` theo cách thông thường trong `gvhmr`: 3 import errors vì thiếu `ruamel`.
- Khi chỉ shim phần import `ruamel` để cô lập lỗi môi trường, cả **12/12 test hiện có đều pass**.

**Khuyến nghị:** xóa/tạo lại `venv` từ interpreter đang tồn tại rồi cài `requirements.txt`; không commit hoặc chia sẻ virtualenv theo máy.

### E-02 — Output hiện có không khớp cấu hình đang mở

- `configs/pipeline.yml` cấu hình camera 1 là `video_0_seg_1.pkl`.
- `output/preprocess_results/data_cam1.json` hiện ghi source là `video_8_seg_1.pkl`.
- `output/pose_results/metadata/pose_data_1.json` hiện không có `source_pkl_stems`, trong khi code mới yêu cầu trường này để bảo vệ evaluation.
- Các CSV evaluation hiện chỉ có 2 dòng (header và `AVERAGE`), không còn chi tiết từng frame để audit lại kết quả.

Output này là artifact cũ, không nên dùng cho `runtime.stage=evaluation`. Cần chạy lại full pipeline với `clean_output=true` sau khi sửa các lỗi mức cao.

## Kiểm tra đã thực hiện

| Kiểm tra | Kết quả |
|---|---|
| Compile toàn bộ Python (`compileall`) | Pass |
| Unit test trong env hiện tại | Fail import: thiếu `ruamel.yaml` |
| Unit test với import môi trường được cô lập | 12/12 pass |
| Đối chiếu config với input/model | Các file input và model chính tồn tại |
| Đối chiếu config với output hiện có | Không khớp camera 1; metadata/output cũ |
| Rà luồng end-to-end | Phát hiện các lỗi H-01 đến H-05 |

## Thứ tự sửa đề xuất

1. Sửa H-01, H-02, H-03 trước khi tin bất kỳ metric mới nào.
2. Sửa H-04 và M-01 trước khi dùng video visualization để đánh giá định tính.
3. Sửa H-05 nếu cần chạy brute-force ngoài notebook Colab.
4. Sửa M-02 đến M-05 và dựng lại môi trường.
5. Xóa output cũ, chạy full pipeline trên một segment ngắn, rồi mới chạy toàn bộ 482 frame/brute-force.

## Khoảng trống test cần bổ sung

Chỉ cần ba integration test nhỏ để khóa các lỗi quan trọng:

1. `fusion.enabled=false` không được phép đọc fused output cũ.
2. Offset khác 0 phải chọn đúng source image cho từng camera.
3. Evaluation phải fail khi số frame đánh giá bằng 0 hoặc khi có fallback fusion ngoài ngưỡng.
