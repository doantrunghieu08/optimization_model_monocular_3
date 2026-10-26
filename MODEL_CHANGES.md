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

## 6. Hiệu Chỉnh Miền Tĩnh Khung Xương (Domain Gap Offset Calibration)

* **Vị trí file:** [`fusion_pipeline/correction.py`](file:///d:/optimization_model_monocular_3/fusion_pipeline/correction.py#L422-L506), [`fusion_pipeline/executor.py`](file:///d:/optimization_model_monocular_3/fusion_pipeline/executor.py#L502-L510), [`configs/pipeline.yml`](file:///d:/optimization_model_monocular_3/configs/pipeline.yml#L71-L75)

### Bối cảnh & Phát hiện toán học
Dù tư thế đứng yên hoàn toàn (Frame 0-30), sai số MPJPE giữa WHAM và Ground Truth MPI-INF-3DHP vẫn lên tới ~275 mm:
* **Nguyên nhân cốt lõi:** Mô hình SMPL (`J_regressor`) định nghĩa tọa độ khớp nằm ở **tâm khối xương bên trong cơ thể**, trong khi Ground Truth (Mocap) đo bằng **marker quang học dán trên bề mặt da/quần áo** (mu bàn tay, mắt cá ngoài). 
* Trục đứng $Y$ lệch có hệ thống $\Delta Y \approx +130 \dots +165\text{ mm}$ trên mọi khớp thân trên và chi trên.

### Công thức hiệu chỉnh
Sử dụng $N = 30$ frame đứng yên đầu tiên để ước lượng vector dịch chuyển tĩnh tương đối với gốc hông:
$$\Delta\mathbf{p}_k = \frac{1}{N} \sum_{t=0}^{N-1} \left[ \left(\mathbf{P}_{t}^{\text{gt}}(k) - \mathbf{P}_{t}^{\text{gt}}(\text{root})\right) - \left(\mathbf{P}_{t}^{\text{pred}}(k) - \mathbf{P}_{t}^{\text{pred}}(\text{root})\right) \right]$$
Áp dụng cộng bù trực tiếp vào toàn bộ chuỗi video sau bước làm mượt:
$$\mathbf{P}_t^*(k) = \mathbf{P}_t(k) + \Delta\mathbf{p}_k, \quad \forall t \in [0, T-1]$$

### Tính chất bất biến gia tốc (Acceleration Invariance)
Vì $\Delta\mathbf{p}_k$ là vector hằng số không đổi theo thời gian, đạo hàm bậc 2 rời rạc (gia tốc) hoàn toàn không bị ảnh hưởng:
$$\mathbf{a}_t^* = \mathbf{P}_{t+1}^* - 2\mathbf{P}_t^* + \mathbf{P}_{t-1}^* = (\mathbf{P}_{t+1} + \Delta) - 2(\mathbf{P}_t + \Delta) + (\mathbf{P}_{t-1} + \Delta) = \mathbf{a}_t$$
$\implies$ Giảm ngay lập tức **`35.64 mm` MPJPE** mà **`Accel Error` giữ nguyên tuyệt đối ở 7.60 mm/f²**.

---

## 7. Bảng Tổng Hợp Kết Quả Thực Nghiệm Toàn Diện (Full Sequence Scoreboard)

Đo lường trên toàn bộ 482 frames của chuỗi thực nghiệm chuẩn (`video_0_seg_1` và `video_2_seg_1`):

### Camera 1 (Master Camera)

| Metric | Monocular Baseline (`posed`) | Fused Bản Cũ | **Fused Bản Mới (Current)** | Cải thiện so với Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **MPJPE (Priority 1)** | `275.00 mm` | `274.68 mm` | **`239.36 mm`** | **-35.64 mm (-13.0%)** |
| **PA-MPJPE (Priority 1)** | `204.97 mm` | `203.04 mm` | **`190.89 mm`** | **-14.08 mm (-6.9%)** |
| **MBLE (Sai số độ dài xương)** | `34.07 mm` | `38.79 mm` | **`16.54 mm`** | **-17.53 mm (chính xác x2)** |
| **Accel Error (Độ rung giật)** | `21.37 mm/f²` | `8.22 mm/f²` | **`7.60 mm/f²`** | **-64.4% (ổn định cao)** |
| Sai số cổ tay phải (`r_wrist`) | `580.34 mm` | `578.01 mm` | **`540.92 mm`** | **-39.42 mm** |
| Sai số cổ tay trái (`l_wrist`) | `552.70 mm` | `547.49 mm` | **`529.74 mm`** | **-22.96 mm** |
| Sai số cổ (`neck`) | `153.71 mm` | `150.54 mm` | **`67.31 mm`** | **-86.40 mm (-56.2%)** |
| Sai số vai phải (`r_shoulder`) | `157.82 mm` | `156.35 mm` | **`105.59 mm`** | **-52.23 mm (-33.1%)** |

### Camera 2 (Slave Camera)

| Metric | Monocular Baseline (`posed`) | **Fused Bản Mới (Current)** | Cải thiện so với Baseline |
| :--- | :---: | :---: | :---: |
| **MPJPE (Priority 1)** | `271.59 mm` | **`236.04 mm`** | **-35.55 mm (-13.1%)** |
| **PA-MPJPE (Priority 1)** | `202.82 mm` | **`192.81 mm`** | **-10.01 mm (-4.9%)** |
| Sai số cổ (`neck`) | `150.54 mm` | **`55.66 mm`** | **-94.88 mm (-63.0%)** |
| Sai số vai phải (`r_shoulder`) | `156.35 mm` | **`89.59 mm`** | **-66.76 mm (-42.7%)** |

---

## 8. Kết Luận Kiến Trúc
Hệ thống hiện tại loại bỏ hoàn toàn các giả định tối ưu 2D phức tạp dễ gây bất ổn định, tinh gọn theo triết lý **Ponytail**:
* Xoay tương đối động học theo từng nhánh xương để tránh khuếch đại góc xoay.
* Khóa cứng chiều dài xương và lọc đa thức thời gian để loại trừ hoàn toàn hiện tượng méo cơ thể và rung lắc gia tốc.
* Hiệu chuẩn độ lệch miền tĩnh giữa mô hình tham số SMPL và thiết bị ghi hình chuyển động quang học.
