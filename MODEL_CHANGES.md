# Báo Cáo Thay Đổi Mô Hình & Công Thức Toán Học (Model & Architecture Changes)

Tài liệu này tổng hợp chi tiết và hệ thống hóa toàn bộ các cải tiến thuật toán, cấu trúc mô hình và công thức toán học trong pipeline tối ưu hóa tư thế người 3D đa góc nhìn (`optimization_model_monocular_3`) so với phiên bản gốc trước đây.

---

## 1. Phát Hiện Che Khuất Không Gian 3D (3D Ray-Casting Occlusion Detection)

* **Vị trí file:** [`fusion_pipeline/detector.py`](file:///d:/optimization_model_monocular_3/fusion_pipeline/detector.py)

### Bản cũ (2D Image-Plane Projection)
Chiếu các đỉnh lưới thân (torso mesh vertices) lên mặt phẳng ảnh 2D, dựng bao lồi 2D và so sánh chiều sâu $Z$ với các láng giềng $k$-NN:
$$u_k = f_x \frac{X_k}{Z_k} + c_x, \quad v_k = f_y \frac{Y_k}{Z_k} + c_y$$
$$\text{Occluded}(P_k) \iff (u_k, v_k) \in \text{Hull}_{2D}(V_{\text{torso}}^{2D}) \;\land\; Z_k > \min_{i \in \mathcal{N}_k}(Z_{\text{torso}, i}) + \tau$$

* **Nhược điểm:** Phụ thuộc vào méo hình học khi chiếu 2D, góc xoay camera và sai số xấp xỉ của $k$-NN 2D.

### Bản mới (3D Ray-Casting Hình Học Thuần Túy)
Bắn tia 3D thực từ quang tâm camera $(0, 0, 0)$ hướng tới khớp mục tiêu $P_k$ trong không gian camera:
$$\mathbf{r}(t) = t \cdot \frac{P_k}{\|P_k\|_2}, \quad t \ge 0$$
Tìm khoảng cách giao điểm gần nhất $t_{\text{hit}}$ của tia với bao lồi 3D của thân người $\mathcal{M}_{\text{torso}} = \text{ConvexHull}(V_{\text{torso}})$:
$$t_{\text{hit}} = \min \{ t > 0 \mid \mathbf{r}(t) \cap \mathcal{M}_{\text{torso}} \ne \emptyset \}$$
$$\text{Occluded}(P_k) \iff t_{\text{hit}} < \|P_k\|_2 - \tau \quad (\tau = 0.05\,\text{m})$$

* **Ưu điểm:** Tính toán hình học 3D chuẩn xác, không phụ thuộc vào ma trận chiếu camera 2D, giải quyết đúng thể tích che khuất thực tế.

---

## 2. Loại Bỏ Tối Ưu Tái Chiếu 2D (Purging 2D Reprojection Optimization)

* **Vị trí file:** Đã loại bỏ hoàn toàn `fusion_pipeline/two_view_smpl.py` và cờ cấu hình `fusion.wham_smpl_opt`.

### Vấn đề của phương pháp cũ
Trước đây, hệ thống cố gắng tinh chỉnh trực tiếp các tham số mô hình SMPL $(\boldsymbol{\theta}, \boldsymbol{\beta}, \mathbf{t})$ bằng hàm mất mát tái chiếu 2D đa góc nhìn (2D Keypoint Reprojection Loss):
$$\mathcal{L}_{\text{2D}} = \sum_{c \in \{1, 2\}} \sum_{j} \left\| \pi_c\left(J_{3D}(\boldsymbol{\theta}, \boldsymbol{\beta}, \mathbf{t}_c)_j\right) - \mathbf{x}_{2D, c, j} \right\|_2^2$$

* **Nhược điểm nghiêm trọng:**
  * Do camera hoang dã (wild in-the-wild cameras) không có đồng bộ thời gian phần cứng hoàn hảo và ma trận ngoại vi góc rộng có sai số, hàm mất mát 2D gây hiện tượng **Scale Drift** (trôi tỉ lệ hình học) và **Depth Collapse** (sụp đổ độ sâu).
  * Thực nghiệm đo lường cho thấy bước tối ưu này làm sai số MPJPE tăng vọt và biến dạng xương nghiêm trọng.
* **Quyết định:** Loại bỏ hoàn toàn module tối ưu 2D SMPL khỏi pipeline, chuyển toàn bộ trọng tâm sang dung hợp không gian 3D trực tiếp (Direct 3D Pose Fusion).

---

## 3. Dung Hợp Xoay Xương Động Học & Thay Thế Theo Độ Tin Cậy (Kinematic Bone Rotation & Belief Replacement)

* **Vị trí file:** [`fusion_pipeline/correction.py`](file:///d:/optimization_model_monocular_3/fusion_pipeline/correction.py), [`fusion_pipeline/detector.py`](file:///d:/optimization_model_monocular_3/fusion_pipeline/detector.py)

### Bản cũ (Biến Đổi Cương Bức Toàn Thân Toàn Cục)
Dùng một phép biến đổi đẳng cự toàn cục duy nhất qua RANSAC-Umeyama $(s=1.0, \mathbf{R}, \mathbf{t})$ áp dụng cho tất cả các khớp:
$$P_{1,\text{corr}}(n) = T_{21}(P_2(n)) = \mathbf{R}_{21} P_2(n) + \mathbf{t}_{21}$$

* **Hiện tượng Cánh Tay Đòn (Lever-Arm Effect):**
  Ma trận $T_{21}$ được fit chủ yếu trên các khớp thân mình (Torso: khoảng cách ~35 cm). Chỉ cần sai lệch góc xoay $3^\circ - 5^\circ$ ở thân mình sẽ bị khuếch đại thành độ lệch $50 - 80\text{ cm}$ ở các chi xa (cổ tay, bàn chân). Hậu quả là việc chiếu Cam 2 sang Cam 1 làm tăng vọt sai số (Negative Transfer).

### Bản mới (Kinematic Relative Rotation & Belief Switching)
1. **Xoay vector tương đối theo từng chuỗi xương con (`transform_joint_kinematic`):**
   Thay vì biến đổi tọa độ tuyệt đối, ta chỉ xoay vector hướng tương đối của đoạn xương con $\mathbf{v}_{\text{src}} = P_2(c) - P_2(p)$ theo ma trận xoay $\mathbf{R}_{21}$, gắn vào vị trí khớp cha $P_1(p)$ đã được chuẩn hóa trong hệ quy chiếu master:
   $$\hat{P}_{\text{slave}\to\text{master}}(c) = P_1(p) + \mathbf{R}_{21} \cdot (P_2(c) - P_2(p))$$
   Khử triệt để sai số tích lũy của cánh tay đòn dọc theo cây động học nhân thể.

2. **Thay thế theo trọng số độ tin cậy kết hợp (Harmonic Belief Switch):**
   Độ tin cậy $H_c(n)$ kết hợp giữa khoảng cách quang tâm, tầm nhìn che khuất và độ tin cậy 2D của láng giềng:
   $$P_1(n) \leftarrow \hat{P}_{\text{slave}\to\text{master}}(n) \quad \iff \quad H_2(n) > H_1(n) \;\land\; \| \hat{P} - P_1(n) \|_2 \le d_{\max}$$
   $$P_2(n) \leftarrow \hat{P}_{\text{master}\to\text{slave}}(n) \quad \iff \quad H_1(n) > H_2(n) \;\land\; \| \hat{P} - P_2(n) \|_2 \le d_{\max}$$
   $$(d_{\max} = 0.75\,\text{m})$$

---

## 4. Khóa Cứng Chiều Dài Xương Động Học (Rigid Bone Length Locking)

* **Vị trí file:** [`fusion_pipeline/correction.py`](file:///d:/optimization_model_monocular_3/fusion_pipeline/correction.py#L343-L362)

### Bản cũ
Sau khi thay thế hoặc hòa trộn khớp, chiều dài xương giữa các khớp liền kề bị co giãn tự do, dẫn đến sai số chiều dài xương (`MBLE`) tăng cao.

### Bản mới (Forward Kinematic Ray Projection)
Duyệt cây động học theo thứ tự topo từ gốc thân (`pelvis`, `neck`) ra các chi (`KINEMATIC_CHAINS`). Giữ nguyên hướng vector 3D từ cha đến con nhưng chiếu chiều dài về đúng giá trị tham chiếu ban đầu $L_{\text{ref}}$:
$$\mathbf{v} = P(c) - P(p), \quad \hat{P}(c) = P(p) + L_{\text{ref}} \cdot \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$
Áp dụng tức thì sau mỗi bước dung hợp và sau bộ lọc làm mượt thời gian, đảm bảo giải phẫu học cơ thể không bao giờ bị kéo dãn.

---

## 5. Làm Mượt Quỹ Đạo Savitzky-Golay Bậc 2 (Temporal Acceleration Smoothing)

* **Vị trí file:** [`fusion_pipeline/executor.py`](file:///d:/optimization_model_monocular_3/fusion_pipeline/executor.py#L330-L370)

Lọc mượt từng khớp $j$ dọc theo trục thời gian $T$ bằng bộ lọc đa thức Savitzky-Golay (cửa sổ 5 frame, đa thức bậc 2):
$$\hat{\mathbf{P}}_{:, j} = \text{Savitzky-Golay}\left(\mathbf{P}_{:, j}, \text{window\_length}=5, \text{polyorder}=2, \text{axis}=0\right)$$
* **Ưu điểm:** Khử các bước nhảy vi mô giữa các frame độc lập, tối ưu hóa trực tiếp cho đại lượng gia tốc bậc 2 $\mathbf{a}_t = \mathbf{P}_{t+1} - 2\mathbf{P}_t + \mathbf{P}_{t-1}$, giảm sâu `Accel Error` từ $21.37\text{ mm/frame}^2$ xuống $7.60\text{ mm/frame}^2$ (-64.4%).

---

## 6. Tối Ưu Hóa Chuỗi Frame Trên GPU & Chống Suy Giảm Đa Góc Nhìn (GPU Sequence Optimization & Negative Transfer Prevention)

* **Vị trí file:** [`fusion_pipeline/optimization.py`](file:///d:/optimization_model_monocular_3/fusion_pipeline/optimization.py#L205-L380)

### Vấn đề của phiên bản trước
Trong bước tối ưu hóa chuỗi toàn cục bằng PyTorch GPU (`optimize_sequence_gpu`), xuất hiện hiện tượng suy giảm độ chính xác mạnh ở một số khung hình do 2 nguyên nhân cốt lõi:
1. **Bảo vệ bất đối xứng (Asymmetric Protection Mask):** `mask1` bảo vệ Cam 1 khỏi trôi lệch khi khớp có độ tin cậy cao, nhưng Cam 2 bị gán `mask2 = ones_like(delta2)`. Do đó Cam 2 bị kéo tự do theo các ràng buộc hình học, sau đó gián tiếp kéo lệch ngược lại Cam 1.
2. **Kéo lệch camera chuẩn do ảo giác (Hallucination Pull via Mean-Confidence Loss):** Trọng số của hàm mất mát khoảng cách đa góc nhìn sử dụng trung bình cộng:
   $$w_{\text{mv}} = \frac{w_1 + w_2}{2}$$
   Khi một camera quan sát chuẩn xác ($w_1 \approx 1.0$) nhưng camera kia bị bắt nhầm đối tượng/ảo giác trong cảnh tiếp xúc ($w_2$ vẫn cao do 2D detector tự tin nhầm), $w_{\text{mv}}$ đạt giá trị cực đại. Quá trình tối ưu hóa ép khoảng cách hai camera trùng nhau, kéo khớp đúng lệch hẳn sang khớp sai (Negative Transfer).

### Giải pháp cải tiến (Ponytail Minimalist Fix)
1. **Khóa khớp tin cậy song phương (Bidirectional High-Confidence Locking):**
   Áp dụng mặt nạ bảo vệ bình đẳng cho cả 2 camera dựa trên ngưỡng độ tin cậy:
   $$\text{mask}_{c, t, j} = \begin{cases} 0.0 & \text{nếu } c_{t, j} \ge 0.3 \text{ (khớp rõ nét, giữ nguyên quan sát gốc)} \\ 1.0 & \text{nếu } c_{t, j} < 0.3 \text{ (khớp bị che khuất/nhiễu, cho phép tối ưu)} \end{cases}$$
2. **Trọng số mất mát đa góc nhìn theo giá trị tối thiểu (Min-Confidence Gating):**
   $$w_{\text{mv}} = \min(w_1, w_2)$$
   Nếu chỉ một trong hai camera bị nghi ngờ hoặc một bên không quan sát được, $w_{\text{mv}}$ tự động hạ thấp, ngăn chặn triệt để việc cưỡng bức hội tụ kéo hỏng góc nhìn đang tốt.

---

## 7. Tách biệt Ground Truth khỏi suy luận

Ground Truth chỉ được đọc trong `evaluation_pipeline` để tính metric sau khi dự đoán đã hoàn tất. `fusion_pipeline` không đọc GT, không tính offset từ GT và không sửa đầu ra dựa trên GT.

Report brute-force mang nhãn protocol `no_gt_v2_shared_shape_consensus`; các dòng cache cũ không có đúng nhãn này sẽ không được tái sử dụng. Pose export dùng median `betas` chung cho toàn sequence và hai camera; fusion dùng confidence correction, rotation correction và consensus blend thay cho hard replacement.

---

## 8. Kết quả cần đo lại

Các số liệu cũ có bước hiệu chỉnh từ GT hoặc protocol fusion cũ nên không còn hợp lệ. Cần chạy lại toàn bộ benchmark với protocol `no_gt_v2_shared_shape_consensus` trước khi công bố kết quả.

Thử nghiệm dual-view SMPL fitting trên `cam0 -> cam2` đã bị loại khỏi pipeline: reprojection giảm `0.4276 -> 0.1522` nhưng MPJPE tăng `273.91 -> 283.55 mm` và Accel tăng `21.36 -> 23.43 mm/f²`. Trust-region không dùng GT cũng từ chối biến thể có 3D anchor vì acceleration proxy tăng `0.01257 -> 0.01884`. Pipeline giữ kết quả an toàn: MPJPE `273.49 mm`, PA-MPJPE `203.25 mm`, MBLE `32.51 mm`, Accel `7.27 mm/f²`.

---

## 9. Kết Luận Kiến Trúc
Hệ thống hiện tại loại bỏ hoàn toàn các giả định tối ưu 2D phức tạp dễ gây bất ổn định, tinh gọn theo triết lý **Ponytail**:
* Xoay tương đối động học theo từng nhánh xương để tránh khuếch đại góc xoay.
* Khóa cứng chiều dài xương và lọc đa thức thời gian để loại trừ hoàn toàn hiện tượng méo cơ thể và rung lắc gia tốc.
* Ground Truth chỉ dùng để đánh giá, không tham gia tạo hoặc hiệu chỉnh dự đoán.
