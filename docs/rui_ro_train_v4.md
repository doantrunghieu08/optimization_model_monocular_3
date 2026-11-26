# Rủi ro khi train `weight_model` trên v4 và các sửa đổi đã xác minh

Cập nhật: 26/11/2026. Đối chiếu mã tại nhánh `experiment/observability-aware-trust-region`, nền `fe237c7`, cùng các sửa đổi trong working tree.

Mục tiêu: giảm PA-MPJPE và MPJPE khi chuyển từ v4 sang dữ liệu sử dụng thực tế.

**Phạm vi bằng chứng:** đã đọc trainer, fusion, evaluator, cấu hình, notebook và `MODEL_CHANGES.md`; đã chạy kiểm thử tổng hợp. Chưa train lại toàn bộ v4 hay benchmark Hugging trong lần sửa này. Các con số từ Sheet và kết quả cũ bên dưới là thông tin của bản tài liệu trước, chưa được kiểm đếm lại từ nguồn dữ liệu.

## 1. Nhận xét chính

Bản cũ xác định đúng vấn đề chuyển miền, chất lượng ghép track/thời gian và loss chỉ dùng PA. Tuy nhiên, cần sửa các kết luận sau:

- Trainer đã lưu `history.json`, checkpoint từng epoch và notebook đã vẽ đường học. Thiếu dừng sớm, không phải thiếu log.
- Validation đã gọi `fuse_partwise`, bao gồm `sequence_stability`. Điểm lệch là trainer ép tắt `multi_person` và thiếu visibility/confidence từ dữ liệu chuẩn bị.
- Nhánh trọng số học cũ đã có căn xoay theo visibility và cổng chống tráo người khi bật multi-person; các bảo vệ bị bỏ qua là bias che khuất theo xương và phát hiện độ sâu bất thường.
- Một phép xoay camera cố định không bảo đảm hướng trung bình sau fusion bằng Master, cũng không bảo đảm MPJPE giảm. Đây phải là giả thuyết kiểm tra bằng thực nghiệm.
- Chỉ có 8 trọng số vẫn có thể overfit khi số sequence độc lập ít; số frame hoặc cặp camera lớn không thay thế số cảnh độc lập.
- `MODEL_CHANGES.md`, mục 6, ghi rõ held-out Hugging đã được dùng để điều chỉnh thiết kế căn theo phần thân. Có thể dùng lại để kiểm tra hồi quy, nhưng không gọi đây là test độc lập hoàn toàn.

## 2. Dữ liệu và kết quả cũ: cần xác nhận lại

Nguồn ghi trong bản trước: sheet `ThanhNT_harmony4d_test_index`, tab `v4 22/11/26`.

| Nội dung bản trước ghi nhận | Cách diễn giải hiện tại |
|---|---|
| 148 dòng, 30 sequence: 24 train, 6 val; 2–6 camera/sequence | Số liệu đầu vào lịch sử; số thực dùng phải đọc từ `protocol.json` sau prepare |
| `colmap`, `image_size`, `path_2d` trống; 7 dòng có `tracking_result` | Trainer hiện chỉ chuẩn bị đặc trưng từ khớp 3D, không nạp các cột này thành visibility/confidence |
| Không trùng sequence hoặc GT giữa train/val | Có kiểm tra bằng `validate_rows`; chưa chứng minh độc lập về người, hành động hoặc camera |
| Có Group, Single và WithDress | Cần báo cáo theo nhóm; không tự động loại nhóm khó chỉ vì lỗi cao |

Comment cấu hình cũ ghi val PA-MPJPE: learned 85.82 mm, 50–50 86.49 mm, mono 92.73 mm. Learned hơn 50–50 **0.67 mm**, chưa đủ để kết luận tổng quát hóa. Đây là PA trên **12 khớp thân** của trainer; không được coi là kết quả `coco17` của evaluator.

## 3. Quy ước MPJPE và giới hạn so sánh

Trong `evaluation_pipeline/executor.py`, chế độ `sequence` ưu tiên phép xoay camera có sẵn. Khi thiếu phép xoay camera, evaluator ước lượng **một phép xoay từ baseline `posed` cho mỗi camera/sequence**, rồi dùng chung cho các module. Khi bật extrinsics thực, quy ước chấm còn phụ thuộc chế độ extrinsics.

MPJPE đo cả sai hướng còn lại của fusion trong hệ quy chiếu cố định. Nó không bắt buộc fusion giữ hướng trung bình của Master: thay đổi hướng có thể tốt hoặc xấu tùy GT. PA-MPJPE khử xoay, tịnh tiến và scale theo frame, nên không thể dùng riêng để chọn mô hình nếu mục tiêu có cả MPJPE.

Trainer mới căn GT về hệ Master bằng một phép xoay trên **toàn bộ frame đã associate của mỗi action/camera**, dùng chung giữa các cặp và đoạn liên tục. Phép xoay này không được fit lại theo kết quả fusion. Tuy nhiên, trainer dùng 12 khớp thân và tập frame của bước chuẩn bị; evaluator có thể dùng `coco17`, identity mask và tập frame khác. Hai quy trình cùng nguyên tắc căn nhưng **chưa đồng nhất hoàn toàn về số đo**.

## 4. Rủi ro và xử lý

### R1. Chuyển miền và đảo dấu độ sâu — cao

Trọng số cũ có hệ số hiệu dụng `d_depth = -0.47934 / 2.04826 ≈ -0.234 /m`, trong khi `depth_logit` có `b/sd ≈ +1.225 /m`. `MODEL_CHANGES.md` ghi nhận đảo dấu độ sâu làm Hugging xấu đi; đây là bằng chứng lịch sử trên miền đó, không chứng minh dấu dương đúng trên mọi cảnh.

**Đã sửa:** khởi tạo hệ số độ sâu bằng `(b/sd) × std_train`; hệ số khác bằng 0. Phạt L2 kéo về vector khởi tạo và chiếu hệ số độ sâu về miền không âm sau mỗi bước optimizer. Đặc trưng hằng số trên train được đặt `std = 1`, trọng số 0 và xuất `active_features` để vẫn bị bỏ qua khi suy luận có thêm tín hiệu.

Đây là prior có chủ đích cho miền sử dụng. Khởi tạo chỉ khớp **độ dốc** của `depth_logit`: đặc trưng độ sâu bị clip ở ±3 m và mô hình không có intercept để giữ phản đối xứng. Nó không tái tạo chính xác toàn bộ hàm `depth_logit` cũ. L2 cũng không bảo đảm hiệu quả trên miền mới; vẫn phải so với baseline độ sâu.

### R2. Validation nhỏ và đã dùng để chọn mô hình — cao

Tách sequence/GT không đồng nghĩa tách người hoặc nhóm hành động. Chưa đủ bằng chứng để khẳng định train/val cùng người hay cùng camera nếu chưa xem metadata.

**Đã sửa:** giữ log từng epoch, bổ sung MPJPE và objective kết hợp, dừng sớm theo objective validation. Checkpoint lưu trạng thái dừng sớm để resume không làm mất lịch sử.

**Còn cần dữ liệu/protocol:** dành tập test theo nhóm hành động hoặc cảnh mới trước khi chỉnh tiếp; kiểm tra cả GT ID lẫn họ hành động để tránh chia các take gần nhau sang nhiều split. Có thể chạy leave-one-family-out để xem độ nhạy giữa nhóm. Dừng sớm không làm validation trở thành test độc lập. Chưa tự đổi split trong Sheet.

### R3. Nhiễu ghép người và thời gian — cao

`associate` dùng GT để tìm timeline và nối track, có ngưỡng PA 150 mm. PA lớn có thể do WHAM kém hoặc ghép sai; riêng ngưỡng đó không phân biệt được hai nguyên nhân. `switches > 0` cũng có thể chỉ là nối các track bị phân mảnh của cùng người, không phải luôn tráo ID.

**Đã sửa:** báo cáo camera có `split`; `protocol.json` thống kê action, cặp camera có hướng, đoạn và tổng frame của đoạn theo split. Notebook đánh dấu cả chuyển track để xem lại, cùng các cờ rejection, coverage và PA đã có. Tổng frame của đoạn có lặp giữa cặp camera, không phải số frame độc lập.

**Chưa tự động làm:** siết ngưỡng, loại hoặc giảm trọng số sequence dựa riêng vào PA/switches. Cần xem phân bố, timeline và danh tính trước; lọc chỉ dựa vào lỗi GT có thể làm tập đánh giá dễ hơn và đổi câu hỏi nghiên cứu. Cần công bố số bị loại và lý do theo split.

### R4. Khác biệt giữa train và suy luận — trung bình đến cao

**Đã sửa:** không còn ép `multi_person.enabled = false` trong trainer. Validation giữ cấu hình multi-person và `sequence_stability` đã chọn. Nhánh learned dùng chung cổng chống tráo người, căn xoay theo visibility, bias che khuất và bảo vệ độ sâu bất thường với nhánh độ sâu. Khi độ sâu bất thường, trọng số learned của frame đó được thay bằng trọng số độ sâu đã qua bảo vệ.

**Giới hạn còn lại:** dữ liệu chuẩn bị chưa có visibility/confidence; gọi đầy đủ hàm fusion không tái tạo được tín hiệu đầu vào còn thiếu. Bước gradient dùng bản fusion khả vi, chưa bao gồm các cổng rời rạc và hậu xử lý; validation mới là điểm chấm đầy đủ để chọn epoch.

Chưa suy ra visibility từ ray-casting. Cần kiểm tra hệ tọa độ, mesh của mọi người và cách đưa kết quả vào pipeline trước; camera nằm tại gốc không tự giải quyết che khuất do người khác.

### R5. Loss chỉ tối ưu PA và cách căn target theo đoạn — trung bình

**Đã sửa:** loss dữ liệu là `PA + λ·MPJPE`, cộng L2 về prior khi tối ưu. Target dùng chung phép xoay mono trên action/camera thay vì fit riêng từng run. Validation và chọn best dùng `PA_mm + λ·MPJPE_mm` sau `fuse_partwise`.

Mặc định mới trong `fusion.partwise.training`:

```yaml
mpjpe_weight: 0.25
prior_l2: 0.001
patience: 10
min_delta_mm: 0.01
```

Loss gradient dùng mét; điểm validation/objective báo bằng mm. L2 tính trên trọng số đã chuẩn hóa, một lần mỗi bước cập nhật action. Đây là giá trị khởi đầu chưa được xác nhận bằng benchmark; phải cố định hoặc chọn trên validation, không chỉnh bằng test. Objective tổng có thể giảm khi một thành phần tăng, nên tiêu chí chấp nhận vẫn phải xét riêng từng chỉ số.

`history.json` giữ tên cũ `train_mpjpe_mm` và `val_mpjpe_mm` cho **PA** để tương thích; MPJPE mới nằm trong `*_sequence_mpjpe_mm`. `*_objective_mm` là điểm kết hợp, không gồm L2. Log validation có thêm baseline độ sâu, 50–50 và mono; 50–50 vẫn qua các cổng/hậu xử lý cấu hình nên không có nghĩa mọi trọng số cuối cùng đều bằng 0.5.

Nhánh `_apply_static_global_orientation` đã tồn tại. Giữ `static_camera_orientation: false`; thử riêng khi camera thực sự cố định. Ước lượng `R_cam` bền vững có thể hấp thụ sai hướng cố định của một view, không bảo đảm hướng trung bình của fusion bằng Master. Phép xoay cứng từng frame giữ PA và chiều dài xương về toán học, nhưng MPJPE và Accel vẫn cần đo. Không thêm cơ chế SLERP mới trước khi đánh giá nhánh có sẵn.

### R6. Dữ liệu không đồng nhất — trung bình

Báo kết quả Group/Single, Dress/không Dress và từng nhóm camera. Không loại cảnh ngồi/đứng hoặc nhóm ít camera chỉ vì khác phần còn lại; trước hết xác định chúng có thuộc miền sử dụng hay không. Nguồn thật/tổng hợp và metadata về người chưa được xác minh trong lần sửa này.

## 5. Tiêu chí kiểm chứng sau sửa

Hugging aria02, cam03/05/17/20 là bài kiểm tra hồi quy miền đích đã dùng trước đây. Cần thêm cảnh/người chưa dùng để phát triển cho kết luận tổng quát hóa.

Chạy mono, aotr, partwise độ sâu và partwise learned với cùng cặp camera có hướng, frame hợp lệ, `joint_set: coco17`, quy ước `mpjpe_alignment`, extrinsics, identity mask và hậu xử lý. Ghi cả PA, MPJPE, MBLE và Accel. So learned với **partwise độ sâu cùng các bảo vệ**, không chỉ so với 50–50.

| Chỉ số | Tiêu chí dự kiến, cố định trước khi chạy |
|---|---|
| PA-MPJPE | Không tăng quá 1 mm so với partwise độ sâu |
| MPJPE | Giảm; báo hiệu theo cặp và độ bất định, không chỉ trung bình |
| MBLE | Bằng baseline Master trong sai số số học khi tắt anatomical calibration |
| Accel | Không tăng quá 2 mm/frame²; giữ cùng tốc độ lấy mẫu và cách tính |
| cam20 làm Master | Báo riêng và kiểm tra trường hợp xấu đi |

Nếu dùng khoảng tin cậy 95%, không coi frame hay cặp đảo chiều là quan sát độc lập. Nên resample theo sequence/cảnh độc lập, tính hiệu learned − baseline trên cùng dữ liệu. Chỉ một cảnh Hugging không đủ để ước lượng chắc chắn độ bất định qua cảnh; khi đó báo phân bố cặp như mô tả, không khẳng định ý nghĩa thống kê về tổng quát hóa.

Trong cấu hình chính, `weight_model.enabled` đã đặt **false** để chưa dùng trọng số cũ. `fusion.method` vẫn là `aotr`; muốn benchmark partwise phải chọn phương pháp tương ứng. `best_partwise.yml` và `pipeline_best.yml` xuất mô hình learned bật sẵn để thử nghiệm, không phải bằng chứng đã đạt tiêu chí. Nếu không đạt, tiếp tục dùng baseline đã kiểm chứng.

## 6. Cách dùng bản sửa và việc còn lại

- [x] Tắt trọng số learned cũ trong cấu hình chính.
- [x] Khởi tạo độ dốc từ prior, phạt L2, ràng buộc dấu và vô hiệu hóa đặc trưng hằng số.
- [x] Kết hợp learned với các bảo vệ multi-person đã có.
- [x] Thêm MPJPE, dùng chung phép xoay target, chọn epoch bằng objective kết hợp và dừng sớm.
- [x] Thêm số liệu thực dùng theo split và đồng bộ notebook từ `build_colab_v4.py`.
- [x] Bổ sung kiểm thử trainer và fusion trên dữ liệu tổng hợp.
- [ ] Kiểm tra dữ liệu/alignment thật, định danh nhóm hành động và chốt test độc lập.
- [ ] Train lại v4, so sánh baseline và báo kết quả riêng theo nhóm.
- [ ] Benchmark miền đích theo mục 5; thử hướng toàn thân như một ablation riêng.

Mở lại `colab_train_v4.ipynb` đã tạo từ builder; thư mục mặc định mới là `run_003`. Dùng output mới vì signature và dữ liệu chuẩn bị đã thay đổi; không resume checkpoint của loss/cách căn cũ. Notebook lưu toàn bộ cấu hình partwise dùng để validation vào `pipeline_best.yml`, gồm mặt nạ `active_features`.

Không đổi split hoặc tự sửa dữ liệu trên Drive trong lần cập nhật này. Hiệu quả thực tế của mô hình mới vẫn chờ các thí nghiệm còn lại ở trên.
