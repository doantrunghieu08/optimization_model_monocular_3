# Nhật Ký Thay Đổi Mô Hình (Model Changes)

Tài liệu ghi lại quá trình phát triển của `optimization_model_monocular_3`, từ pipeline hai tầng `legacy` đến phương pháp mặc định hiện tại: **dung hợp anisotropic MLE không cần thông số camera** (`fusion.method: aniso`). Mọi con số đều được đo bằng `evaluation_pipeline` chính thức. Kết quả được báo cáo theo nguyên tắc **tách tập thiết kế và tập held-out**.

---

## Hiện Trạng (Tóm Tắt)

| Thành phần | Trạng thái |
| :--- | :--- |
| Dung hợp mặc định | `fusion.method: aniso` ([fusion_pipeline/aniso.py](fusion_pipeline/aniso.py)) |
| Stage 1 (ray-casting + belief) | Vẫn chạy để cung cấp trọng số belief và nhãn che khuất cho `aniso` |
| Stage 2 (GPU Adam), Savitzky-Golay | Chỉ dùng khi `method: legacy` |
| `selective_joint_fusion` (V3), `learnable` (NetBody25) | Tắt (thử nghiệm). Bật V3 thì code tự rơi về nhánh `legacy` |
| Khóa chiều dài xương | Bật (`enforce_bone_lengths: true`) cho cả hai nhánh |
| Kiểm thử | `python -m pytest tests -q` → **67 passed** |

---

## Phần 1: Dung Hợp Anisotropic Không Cần Thông Số Camera (`aniso`, mặc định)

### 1.1. Tích hợp

* **Mã nguồn**: [fusion_pipeline/aniso.py](fusion_pipeline/aniso.py). Nhánh `fusion_method == "aniso"` nằm trong [run_fusion()](fusion_pipeline/executor.py#L871); cấu hình ở `fusion.method` và `fusion.aniso` trong [configs/pipeline.yml](configs/pipeline.yml); test ở [tests/test_aniso_fusion.py](tests/test_aniso_fusion.py) (12 test).
* Stage 1 chạy như cũ. Sau đó [run_aniso_fusion()](fusion_pipeline/aniso.py#L266) **tính lại toàn bộ keypoint từ pose mono gốc**, dùng `joint_confidence` × hệ số che khuất (`vis1`, `vis2`) trong metadata Stage 1 làm belief. Thông tin chẩn đoán (nhiễu ước lượng, góc nhìn trung vị, λ) được ghi vào `fused_results/aniso_fusion_info.json`.

### 1.2. Cơ chế (ước lượng hợp lý cực đại, nghiệm dạng đóng)

1. Đưa khớp của mỗi camera về tọa độ tương đối root (trung điểm hai hông).
2. Ước lượng phép xoay camera2 → camera1 **riêng cho từng frame** bằng Kabsch có trọng số $\min(w_1, w_2)$ trên 21 khớp, kèm IRLS Cauchy 5 vòng. Không dùng được một phép xoay toàn cục, vì hướng toàn thân của GVHMR dao động 7–11° (trung vị) giữa các frame.
3. Hiệp phương sai kéo dài theo tia nhìn:
   $$\Sigma_i = \frac{a_i^2(I - dd^\top) + b_i^2\, dd^\top}{w_i}, \qquad d = P/\|P\|$$
   Không cần intrinsics, vì khớp đã nằm trong hệ tọa độ camera. Belief $w$ được kẹp vào $[0.05, 1]$.
4. Tự hiệu chỉnh nhiễu ngang $a_i$ và nhiễu sâu $b_i$ của từng camera bằng MLE trên phần lệch giữa hai view. Residual lớn hơn `max_residual_m = 0.5 m` bị loại. Bước này không dùng GT và không chỉnh tay.
5. Dung hợp: $X^* = (\Lambda_1 + \Lambda_2')^{-1}(\Lambda_1 X_1 + \Lambda_2' R_t X_2)$. Kết quả được trả về hệ tọa độ của từng camera rồi cộng lại root của camera đó.
6. Làm mượt gia tốc bậc 2 trên quỹ đạo tuyệt đối ($\lambda = 0.5$, giải hệ banded), sau đó khóa chiều dài xương theo SMPL mono như nhánh cũ.

### 1.3. Kết quả trên tập thiết kế

Tập thiết kế: MMA, 741 frame, GVHMR, 4 kịch bản `cam01+02` / `cam19+20` × `aria01` / `aria02`. Số liệu là trung bình 8 ca (kịch bản × camera); đơn vị mm và mm/frame².

| Phương pháp | PA-MPJPE | PA (che) | MPJPE | MPJPE (che) | MBLE | Accel |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mono | 77.93 | 98.31 | 103.52 | 131.78 | 37.08 | 97.64 |
| `legacy` | 73.71 | 90.73 | 99.89 | 120.39 | 37.08 | 92.87 |
| **`aniso`** | **68.27** | **73.20** | **94.71** | **103.08** | 37.08 | **45.13** |

* `aniso` tốt hơn `legacy` về PA-MPJPE, MPJPE và Accel ở **8/8** ca. MBLE không đổi vì xương bị khóa.
* **Ablation**: phần lớn mức cải thiện đến từ căn chỉnh robust từng frame kết hợp trọng số nghịch đảo phương sai. Tách riêng hướng độ sâu đóng góp khoảng 1.5 mm PA. Belief đóng góp khoảng 1.4 mm PA (khoảng 3.4 mm ở khớp bị che). Áp cùng bộ làm mượt cho `legacy` chỉ giảm Accel, không giảm PA.

### 1.4. Các tùy chọn đã thử trên tập thiết kế nhưng không được chấp nhận

Script: `scratch/design_mma.py`. Các tùy chọn vẫn nằm trong code (mặc định tắt) để tái lập ablation.

* `noise_model: per_joint` (tách nhiễu theo từng camera/khớp dựa trên độ rung): không cải thiện. Sai số của GVHMR chủ yếu ở tần số thấp; tín hiệu rung chỉ chiếm khoảng 1% tổng sai số, nên với hai view không phân định được sai số thuộc camera nào.
* `bone_lengths: shared` (một chiều dài chung, không đổi theo thời gian cho mỗi xương): MBLE tăng từ 37.08 lên 37.33.
* `smoothing_lambda: gcv`: PA giảm 0.8 mm, nhưng Accel tăng từ 45 lên 70 mm/f². GCV chọn λ quá nhỏ vì nhiễu tương quan theo thời gian, nên giữ λ = 0.5.

---

## Phần 2: Kiểm Chứng Tổng Quát Hóa (Held-Out, Cấu Hình Đóng Băng)

Quy trình: đóng băng `configs/pipeline.yml`, chạy `scratch/generalization_runner.py <suite>`, rồi tổng hợp bằng `scratch/generalization_summary.py`. Các tập dưới đây không được dùng trong lúc thiết kế.

| Tập held-out | Số ca | PA-MPJPE legacy → aniso | MPJPE legacy → aniso | Accel legacy → aniso | aniso thắng PA |
| :--- | ---: | :---: | :---: | :---: | :---: |
| MMA, 8 cặp camera chưa dùng (`cam03`–`cam18`) × 2 người, GVHMR | 32 | 71.90 → **68.53** | 98.94 → **95.63** | 97.45 → **46.97** | **32/32** |
| Harmony4D Hugging, 3 cặp × 2 người, WHAM, chuỗi ngắn | 12 | 49.60 → **47.72** | 64.45 → **59.51** | 24.97 → **11.59** | 5/12 |
| MPI-INF-3DHP (GT đồng bộ lại bằng DTW, seg 1/6/11), WHAM | 36 | 83.23 → **82.46** | 129.09 → 129.67 | 50.15 → **35.23** | 27/36 |

### Nhận xét

* **Có lạc quan do thiết kế**: trên MMA held-out, PA giảm 3.4 mm, ít hơn mức 5.4 mm trên tập thiết kế.
* **Độ nhạy tham số**: trên MMA held-out, mọi biến thể đều cho PA 67.6–69.7 mm và đều tốt hơn `legacy` (71.9). Các biến thể gồm $\lambda \in [0, 2]$, IRLS 1–10 vòng, ngưỡng residual 0.3–1.0 m, cửa sổ xoay ±2 và covariance đẳng hướng. Như vậy kết quả không phụ thuộc vào giá trị tinh chỉnh; $\lambda$ chỉ đánh đổi đơn điệu giữa PA và Accel.
* **Hugging không ổn định theo từng ca**: trung bình tốt hơn nhưng chỉ thắng 5/12 ca về PA. Nguyên nhân mang tính cấu trúc: `aniso` xuất **một bộ xương đồng thuận cho cả hai camera**, nên khi một view tốt hơn hẳn, kết quả kéo view tốt xuống. Với hai view, residual chéo chỉ cho biết tổng sai số, không cho biết camera nào sai. Thành phần anisotropic cũng không tổng quát hóa ổn định: trên Hugging, bản đẳng hướng đạt 46.83, tốt hơn `aniso` (47.72).
* **MPI**:
  * Segment WHAM trong `input/MPI` **không cùng frame** với file GT cùng tên. Ví dụ seg_1 có 482 frame WHAM nhưng 2036 frame GT. Đánh giá ngây thơ cho PA khoảng 189 mm, kể cả với mono.
  * Đánh giá được sửa bằng DTW 2D, lấy ánh xạ từ các camera ngoài cặp đang xét (`scratch/mpi_sync_build.py`, `scratch/mpi_sync_gt.py`). Ánh xạ chỉ tin cậy ở seg 1/6/11.
  * Trần cải thiện thấp: hai view WHAM chỉ lệch nhau khoảng 28 mm PA, và trung bình oracle của hai view chỉ giảm khoảng 2 mm. Phần lớn sai số là phần chung của cả hai view: lệch quy ước khớp hông khoảng 70–90 mm, sai số đồng bộ GT còn sót và prior của WHAM.

### Thí nghiệm chiều cao oracle (`scratch/height_oracle.py`)

Thí nghiệm này co giãn output sao cho chiều cao dự đoán khớp chiều cao GT, rồi chấm lại bằng `run_evaluation`.

* PA-MPJPE không đổi, vì Procrustes đã loại bỏ tỷ lệ.
* MMA (GVHMR): GVHMR đánh giá thấp chiều cao khoảng 5%. Với chiều cao thật, MPJPE giảm từ 95.6 xuống 94.8 và MBLE từ 36.4 xuống 27.0. Dùng chiều cao trung bình quần thể cho kết quả tương đương (MPJPE 94.3, MBLE 27.7), tức đây là một sai lệch tỷ lệ toàn cục chứ không phải thông tin riêng của từng người.
* Hugging và MPI (WHAM): tỷ lệ đã nằm trong khoảng ±2%, mức cải thiện ≤ 0.5 mm.
* Sai số MBLE chủ yếu nằm ở **tỷ lệ từng xương**: đùi và cánh tay trên ngắn hơn GT 4–29%, cẳng chân và cẳng tay dài hơn. Đây là lệch quy ước khớp so với GT, một hệ số chiều cao duy nhất không sửa được. **Kết luận**: không đưa chiều cao vào pipeline.

### Hạn chế còn lại

* $\lambda$ làm mượt là giá trị chọn tay.
* Với hai view, tỷ lệ chia sai số ngang giữa hai camera chỉ xác định được yếu. Sai số độ sâu và tổng sai số ngang thì xác định tốt.
* `brute_force_runner.py` chọn cặp camera tốt nhất dựa trên GT, tức là chọn oracle. Không dùng con số đó làm kết quả mô hình.

---

## Phần 3: Thay Đổi Đánh Giá & Nhập Liệu

1. **Accel trong cùng hệ tọa độ** ([_estimate_sequence_rotation()](evaluation_pipeline/executor.py#L158)): dự đoán nằm trong hệ camera, còn GT nằm trong hệ thế giới. Với mỗi chuỗi (camera, module), hệ thống fit một phép xoay Kabsch trên các khớp đã trừ root, gộp qua mọi frame, rồi xoay gia tốc dự đoán trước khi so sánh. Phép tịnh tiến hằng không ảnh hưởng đến gia tốc. **Các số Accel đo trước thay đổi này không so sánh được** với bảng ở Phần 1 và 2.
2. **Tự chọn file theo người** ([resolve_target_pt_path()](compat.py#L139)): với input GVHMR `.pt`, `inputs.target_person` (`aria01`, `aria2`, `aria_1`, …) được chuẩn hóa và tự đổi sang file anh em tương ứng, ví dụ `hmr4d_results_aria01.pt` → `..._aria02.pt`. Việc này được thực hiện ngay khi nạp config ([config_loader.py](config_loader.py)), nên log, kiểm tra provenance khi `runtime.stage: evaluation` ([pipeline.py](pipeline.py)) và bước nạp dữ liệu đều dùng cùng một đường dẫn.
3. [main.py](main.py) gọi `compat.patch_numpy_and_inspect()` trước mọi import khác, để `chumpy` và SMPL chạy được với NumPy/Python mới.
4. `learnable_pipeline` đọc offset đồng bộ từ metadata `pose_data_1.json` (dự phòng bằng hàm cũ), tránh lệch frame với Stage 1.
5. `brute_force_runner.py`: bỏ hậu tố `EVALUATION_PROTOCOL` cũ (`no_gt_v10_master_only_postopt`).

---

## Phần 4: Lịch Sử Pipeline `legacy` (Vẫn Dùng Được Qua `method: legacy`)

### 4.1. Nền tảng Stage 1

* **Che khuất bằng ray-casting 3D** ([compute_visibility_from_mesh_vertices()](fusion_pipeline/detector.py#L121)): bắn tia từ quang tâm tới các khớp chi và giao với `ConvexHull` của lưới thân SMPL. Khớp bị coi là che khuất khi $t_{hit} < \|P\| - \tau$ ($\tau = 0.05$ m) hoặc khi confidence 2D < 0.25. Phần này vẫn là đầu vào của `aniso`.
* **Belief** ([compute_harmonic_precision()](fusion_pipeline/detector.py#L183)): $P = C / (1 + \alpha L^2)$ với $C = 1$ (nhìn rõ) hoặc $0.1$ (bị che), $\alpha = 0.001$, harmonic mean với các khớp lân cận ($\beta = 0.8$), trộn thêm confidence 2D.
* **Sửa lỗi parse biến môi trường âm**: regex cũ đọc `${ALPHA:-0.001}` thành $\alpha = -0.001$, khiến camera xa được tin hơn camera gần. Đã sửa regex và kẹp $\alpha \ge 0$.
* **Xoay xương động học** thay cho biến đổi cứng toàn thân: $\hat P(c) = P_{tgt}(p) + R\,(P_{src}(c) - P_{src}(p))$, kết hợp RANSAC-Umeyama đẳng cự ($s = 1$) trên các điểm mốc thân.
* **Đối xứng hóa** ([_fuse_single_view()](fusion_pipeline/correction.py#L384)): hai camera dùng chung một quy tắc. *Occluded rescue* áp dụng khi khớp đích bị che và độ lệch ≤ 0.75 m; *visible consensus* trộn theo tỷ trọng belief. Mọi ứng viên phải qua kiểm tra góc bản lề (30°–175°) và độ lệch chiều dài xương ±20%.
* Đã loại bỏ hoàn toàn tối ưu tái chiếu 2D SMPL (`two_view_smpl.py`), vì nó gây trôi tỷ lệ và sập chiều sâu trên camera không đồng bộ phần cứng.

### 4.2. Stage 2: tối ưu chuỗi trên GPU

* [optimize_sequence_gpu()](fusion_pipeline/optimization.py#L205): Adam (`lr = 0.01`, 35 vòng) tối ưu phần dư $\Delta_{1,2} \in \mathbb{R}^{T \times 21 \times 3}$ với hàm mục tiêu:
  $$\mathcal{L} = \lambda_{reg}\,\mathcal{L}_{prox}(c) + 0.03\,\mathcal{L}_{accel} + 5.0\,\mathcal{L}_{bone} + 0.5\,\mathcal{L}_{mv}$$
  Belief liên tục $c \in [0.05, 1]$ thay cho khóa cứng `mask = 0` của các phiên bản trước.
* Tắt Savitzky-Golay, vì bộ lọc gây trễ pha và làm mất phần cải thiện PA; nhiệm vụ khử rung chuyển sang thành phần gia tốc bậc 2. Sau tối ưu, [_enforce_bone_lengths()](fusion_pipeline/executor.py#L624) khóa chiều dài xương trên cả hai camera.

### 4.3. Nhánh đã tắt

* **NetBody25** (`learnable.enabled: false`): trên MMA Cam 1, PA-MPJPE tăng từ 75.85 lên 311.69 mm do lệch chuẩn hóa tọa độ Body25 khi đổi tập dữ liệu.
* **Selective joint fusion V3** (`selective_joint_fusion.enabled: false`): heuristic Zero-GT này không phải ước lượng hiệu chuẩn của mức cải thiện MPJPE và đã gây hồi quy, nên chỉ dùng cho A/B test trên tập held-out.

### 4.4. Ghi chú về các bảng benchmark cũ

Các bảng `legacy` trong phiên bản tài liệu trước được đo **trước khi sửa Accel** (Phần 3.1) và theo một cấu hình đánh giá khác: ví dụ MBLE khi đó là 18.40 mm, còn hiện nay là 37.08 mm. Vì vậy chúng không được so sánh trực tiếp với các bảng ở Phần 1 và 2. Muốn so sánh `legacy` với `aniso`, hãy dùng các số ở Phần 1 và 2, nơi hai phương pháp được chạy trên cùng output Stage 1 và chấm bằng cùng một `run_evaluation`.

---

## Phần 5: Kiểm Thử

```bash
python -m pytest tests -q
```

Kết quả hiện tại: **67 passed**, gồm:

| File | Số test | Nội dung chính |
| :--- | ---: | :--- |
| [tests/test_aniso_fusion.py](tests/test_aniso_fusion.py) | 12 | Kabsch/IRLS, tự hiệu chỉnh nhiễu, dung hợp, làm mượt banded, GCV, per-joint/shared, không dùng thông số camera/GT, config mặc định `aniso` |
| [tests/test_regressions.py](tests/test_regressions.py) | 44 | Hồi quy pipeline, nạp GT, bảo vệ Zero-GT ([test L184](tests/test_regressions.py#L184)) |
| [tests/test_selective_fusion.py](tests/test_selective_fusion.py) | 8 | Selective fusion V3 |
| [tests/test_selective_regression_guard.py](tests/test_selective_regression_guard.py) | 3 | Chặn hồi quy khi bật V3 |
