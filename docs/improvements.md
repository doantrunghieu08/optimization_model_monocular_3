# Các Cải Tiến Của Phương Pháp Đề Xuất So Với Mô Hình Gốc

> [!NOTE]
> Tài liệu này tổng hợp chi tiết các cải tiến về mặt kiến trúc, thuật toán nhận biết che khuất (occlusion), mô hình hóa độ tin cậy (belief score), hiệu chỉnh liên camera (cross-view correction), và quy trình đánh giá thực nghiệm của pipeline Fusion 3D Pose.

---

## 1. Phạm Vi So Sánh

Mô hình gốc đã xây dựng một pipeline hoàn chỉnh gồm:
* Đồng bộ dữ liệu đa góc nhìn.
* Ước lượng Pose 3D đơn quan sát (monocular) cho từng camera.
* Căn chỉnh hai hệ tọa độ bằng thuật toán RANSAC–Umeyama.
* Tối ưu hóa vị trí khớp bằng bộ giải SLSQP.
* Hậu xử lý Learnable (SMPLify) và đánh giá qua các chỉ số MPJPE, PA-MPJPE, PCK.

Đóng góp của phiên bản cải tiến **không nằm ở việc thay thế backbone** (như WHAM, SMPL hay `NetBody25`), mà tập trung giải quyết 3 vấn đề cốt lõi trong giai đoạn Fusion:
1. **Nhận biết che khuất góc nhìn (Occlusion Detection)** bằng Ray-Casting 3D.
2. **Mô hình hóa độ tin cậy thích nghi (Adaptive Belief Score)** kết hợp hình học và belief 2D.
3. **Kiểm soát sai số & hiệu chỉnh an toàn (Robust Cross-View Correction)** giữa 2 camera.

---

## 2. Phát Hiện Tự Che Khuất Bằng Ray-Casting Trên Mesh SMPL

### Hạn chế của mô hình gốc
Ở bản gốc, che khuất được ước lượng bằng cách chiếu các đỉnh phần thân lên ảnh 2D, tạo bao lồi (convex hull) và so sánh độ sâu của khớp với nhóm đỉnh lân cận.
* Phụ thuộc vào camera intrinsics và file phân vùng SMPL bên ngoài.
* Phụ thuộc trường `verts_cam` trong dữ liệu đầu vào. Do đầu vào thực tế không luôn chứa `verts_cam`, chức năng này mặc định bị tắt.
* Bao lồi 2D không phản ánh đúng hình dạng bề mặt 3D thực tế và khoảng trống tự nhiên của cơ thể.

### Giải pháp cải tiến
Phiên bản cải tiến tái tạo trực tiếp mesh SMPL trong hệ tọa độ camera ở bước Pose và lưu đồng bộ với các frame keypoint.
* Các mặt thuộc vùng thân (`torso`) được xác định từ trọng số skinning (`lbs_weights`) của mô hình SMPL.
* Với mỗi khớp cần kiểm tra, hệ thống dựng tia (ray) từ tâm camera đến khớp và tính giao điểm với tam giác torso bằng thuật toán **Ray–Triangle Intersection (Möller–Trumbore)**.
* Khớp được xác định bị che (`vis = False`) nếu tồn tại giao điểm nằm trước khớp một khoảng lớn hơn ngưỡng an toàn $\tau$ (`occlusion_tau`):
  $$d_{\text{hit}} < d_{\text{target}} - \tau$$

> [!TIP]
> **Ưu điểm vượt trội:**
> 1. Kiểm tra che khuất trực tiếp trong không gian 3D, chính xác hơn xấp xỉ bao lồi 2D.
> 2. Sử dụng chính xác mesh và frame đã tạo ra pose của từng camera.
> 3. Độc lập hoàn toàn với thông số intrinsics camera hay file phân vùng ngoài.

Kết quả `visibility` được đưa thẳng vào Belief Score: khớp bị che khuất sẽ có độ tin cậy bằng $0$ trước khi hòa trộn dữ liệu với camera còn lại.

---

## 3. Belief Score Thích Nghi Theo Khoảng Cách, Che Khuất Và Cấu Trúc Xương

### Baseline cũ (`naive_distance_belief`)
Mô hình gốc sử dụng hàm belief cục bộ cố định giảm dần theo khoảng cách từ camera tới khớp:

$$P_j = \frac{C_j}{1 + \alpha L_j^2}$$

Trong đó $C_j \in \{0, 1\}$ là cờ visibility và $L_j$ là khoảng cách 3D từ camera đến khớp $j$.

### Hàm mới (`optical_aware_belief`)
Phiên bản cải tiến bổ sung mô hình suy giảm chất lượng quang học (`optical_aware_belief`), chia không gian quan sát thành 3 vùng rõ rệt:
* **Vùng quá gần camera ($L_j < L_{\min}$):** Belief suy giảm theo hàm Gaussian do hiệu ứng méo ống kính/out-of-focus.
* **Vùng quan sát tối ưu ($L_{\min} \le L_j \le L_{\max}$):** Belief giảm từ từ theo bình phương khoảng cách.
* **Vùng quá xa camera ($L_j > L_{\max}$):** Belief suy giảm mạnh theo hàm Gaussian do giảm độ phân giải điểm ảnh.

### Lan truyền belief toàn cục (Global Harmonic Belief)
Khi `global = true`, belief cục bộ $P_j$ được hòa trộn với trung bình belief của các khớp láng giềng kề trên khung xương:

$$H_j = \frac{2 P_j B_j}{P_j + B_j + \varepsilon}$$

$$B_j = (1 - \beta) P_j + \beta \cdot \mathrm{mean}_{k \in \mathcal{N}(j)}(P_k)$$

> [!NOTE]
> Các khớp láng giềng bị che khuất ($P_k = 0$) sẽ tự động bị loại khỏi phép tính trung bình $B_j$, giúp tránh hiện tượng một khớp visible bị giảm tin cậy oan do láng giềng bị che.

Belief hình học sau đó tiếp tục được kết hợp với belief 2D (từ OpenPose/YOLO/AlphaPose nếu có), đảm bảo việc chọn khớp từ camera nào được cân nhắc toàn diện: khoảng cách 3D, che khuất, belief 2D và tính nhất quán của chuỗi xương.

---

## 4. Hiệu Chỉnh Liên Camera Và Tối Ưu Có Cơ Chế Bảo Vệ

Khung RANSAC–Umeyama và bộ giải SLSQP được nâng cấp thêm các cơ chế bảo vệ an toàn nghiêm ngặt:

1. **Điều kiện RANSAC tối thiểu:** Phép biến đổi đồng dạng (similarity transform) chỉ được tính toán khi có **ít nhất 3 điểm anchor không thẳng hàng** và tọa độ hợp lệ (không chứa `NaN`/`Inf`).
2. **Ngưỡng dịch chuyển tối đa (Belief Correction Guard):** Một khớp chỉ được trộn với vị trí dự đoán từ camera đối diện khi khoảng cách dịch chuyển không vượt quá ngưỡng RANSAC (`threshold`). Tỷ lệ trộn $\alpha$ lấy trực tiếp theo tỷ lệ belief tương đối giữa 2 quan sát.
3. **Quản lý lệch hướng (Orientation Correction):** Hiệu chỉnh mismatch hướng xoay có thể bật/tắt độc lập. Các mismatch mới xuất hiện sau hiệu chỉnh sẽ tự động bị từ chối và khôi phục về pose ban đầu (`reject_new_mismatches`).
4. **Cơ chế Limb-Winner (Thay thế theo chuỗi chi):** Cho phép thay thế toàn bộ chuỗi khớp tay/chân khi một camera có độ tin cậy vượt trội, đồng thời giới hạn góc quay tối đa của xương (`max_bone_angle_deg`) để tránh biến dạng hình học.
5. **Bộ giải SLSQP linh hoạt:** Hỗ trợ bật/tắt các ràng buộc độ dài xương (`use_kinematic_constraints`), tùy chọn hàm mất mát giữa **Huber loss** (kháng nhiễu ngoại lai) và **MSE loss** (đối chứng).
6. **Kiểm soát Fallback:** Khi 1 frame bị lỗi dữ liệu, hệ thống chuyển sang chế độ fallback an toàn. Nếu tỷ lệ frame fallback vượt quá `max_fallback_ratio`, pipeline sẽ dừng và cảnh báo thay vì xuất kết quả lỗi.

> [!IMPORTANT]
> Các tính năng hiệu chỉnh (Belief correction, Orientation correction, SLSQP optimizer) mặc định tắt trên held-out benchmark trừ khi được chứng minh cải thiện qua các bài thí nghiệm Ablation.

---

## 5. Thiết Kế Ablation Có Thể Tái Lập (Reproducible Ablation Study)

Các tham số thí nghiệm được mã hóa trực tiếp trong tên notebook và nạp vào cấu hình `pipeline.yml` trước khi chạy. Sáu trường bắt buộc bao gồm:
* `alpha`: Hệ số phạt khoảng cách.
* `beta`: Trọng số lan truyền skeleton.
* Phương pháp belief cục bộ (`naive_distance_belief` / `optical_aware_belief`).
* Phạm vi belief (`global` / `local`).
* Ràng buộc động học (`use_kinematic_constraints`).
* Loại hàm loss (`huber` / `mse`).

### Các cấu hình Ablation tiêu chuẩn

| Cấu hình | Belief Cục Bộ | Belief Toàn Cục | Ràng Buộc Động Học | Hàm Loss |
| :--- | :--- | :--- | :--- | :--- |
| `naive_global_kinematic_huber` | Distance-based | Bật (`global`) | Bật (`kinematic`) | Huber |
| `optical_global_kinematic_mse` | Optical-aware | Bật (`global`) | Bật (`kinematic`) | MSE |
| `optical_global_unconstrained_huber` | Optical-aware | Bật (`global`) | Tắt (`unconstrained`) | Huber |
| `optical_local_kinematic_huber` | Optical-aware | Tắt (`local`) | Bật (`kinematic`) | Huber |

Mỗi notebook tự động xác nhận tham số, ghi log vào metadata và đồng bộ trạng thái lên Google Sheets riêng biệt, loại bỏ hoàn toàn nguy cơ dùng nhầm cấu hình mặc định hoặc trộn lẫn kết quả giữa các lần chạy.

---

## 6. Bộ Đánh Giá Mới Đầy Đủ & Chính Xác Hơn

Hệ thống đánh giá (`evaluation`) được nâng cấp toàn diện:

* **PCK (Percentage of Correct Keypoints):** Sửa lại đúng chuẩn ngữ nghĩa — tính tỷ lệ % số khớp có sai số 3D nhỏ hơn ngưỡng `pck_threshold_mm` (mặc định 150mm), thay vì tính khoảng cách trung bình như bản gốc.
* **MBLE (Mean Bone Length Error):** Bổ sung chỉ số đo sai số chiều dài từng xương so với ground-truth, giúp đánh giá độ biến dạng hình học của khung xương.
* **Acceleration Error:** Bổ sung đo chỉ số gia tốc khớp theo đơn vị `mm/frame²`, giúp đánh giá độ mượt và hiện tượng rung lắc (jitter) theo thời gian.
* **Xuất dữ liệu chi tiết:** Hỗ trợ xuất kết quả tổng hợp lẫn phân tích chi tiết theo từng frame, từng loại khớp và từng đoạn xương.

---

## 7. Tóm Tắt So Sánh Thay Đổi Chính

| Thành Phần | Mô Hình Gốc | Phiên Bản Cải Tiến |
| :--- | :--- | :--- |
| **Phát hiện che khuất** | Bao lồi 2D & xấp xỉ độ sâu; mặc định tắt | Ray-Casting 3D trên tam giác torso của mesh SMPL từng frame |
| **Nguồn dữ liệu Mesh** | Phụ thuộc trường `verts_cam` từ WHAM | Mesh camera-space được khởi tạo và đồng bộ từ bước Pose |
| **Belief cục bộ** | Hàm suy giảm khoảng cách đơn giản | Hỗ trợ đối chứng giữa `naive` và `optical_aware` |
| **Belief toàn cục** | Luôn lan truyền qua skeleton | Cấu hình bật/tắt linh hoạt cho Ablation Study |
| **Hiệu chỉnh Cross-View** | Thay thế cứng theo belief | Trộn theo belief liên tục, giới hạn dịch chuyển, có RANSAC guard & Limb-Winner |
| **Tối ưu hóa SLSQP** | Cố định Huber loss và kinematic constraints | Linh hoạt bật/tắt optimizer, chọn Huber/MSE, tùy chỉnh temporal/accel loss |
| **Xử lý lỗi Pipeline** | Fallback ẩn theo từng frame | Giám sát tỷ lệ fallback thực tế (`max_fallback_ratio`) |
| **Chỉ số đánh giá** | MPJPE, PA-MPJPE, PCK (chưa chuẩn) | PCK chuẩn ngưỡng mm, bổ sung MBLE (sai số xương) và Acceleration Error |
| **Thực nghiệm Ablation** | Dễ rơi vào cấu hình mặc định | Ràng buộc cấu hình từ tên Notebook, metadata tự động, xuất Google Sheets |

> [!NOTE]
> Các cải tiến trên giúp hệ thống Fusion phản ánh chính xác điều kiện quan sát vật lý của camera và đem lại quy trình đánh giá chuẩn xác, có thể tái lập. Mức độ cải thiện định lượng (MPJPE / PA-MPJPE / MBLE / Acceleration) được kiểm chứng minh bạch trên từng tập dataset đối chứng.
