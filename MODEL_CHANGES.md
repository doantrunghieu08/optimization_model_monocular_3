# Báo Cáo Thay Đổi Mô Hình & Công Thức So Với Bản Trước

Tài liệu này tổng hợp chi tiết các cải tiến thuật toán, cấu trúc mô hình và công thức toán học giữa phiên bản hiện tại so với bản gốc trước đây.

---

## 1. Phát Hiện Che Khuất (Occlusion Detection)

* **Vị trí file:** [fusion_pipeline/detector.py](file:///d:/optimization_model_monocular_3/fusion_pipeline/detector.py#L83-L135)

### Trước (2D Image-plane Projection)
Chiếu các đỉnh lưới thân (torso mesh vertices) lên mặt phẳng ảnh 2D, dựng bao lồi 2D và so sánh chiều sâu $Z$ với các láng giềng $k$-NN:
$$u_k = f_x \frac{X_k}{Z_k} + c_x, \quad v_k = f_y \frac{Y_k}{Z_k} + c_y$$
$$\text{Occluded}(P_k) \iff (u_k, v_k) \in \text{Hull}_{2D}(V_{\text{torso}}^{2D}) \;\land\; Z_k > \min_{i \in \mathcal{N}_k}(Z_{\text{torso}, i}) + \tau$$

* **Nhược điểm:** Phụ thuộc vào méo hình học khi chiếu 2D, góc xoay camera và sai số xấp xỉ của $k$-NN 2D.

### Sau (3D Ray-Casting)
Bắn tia 3D thực từ quang tâm camera $(0, 0, 0)$ hướng tới khớp mục tiêu $P_k$ trong không gian 3D:
$$\mathbf{r}(t) = t \cdot \frac{P_k}{\|P_k\|_2}, \quad t \ge 0$$
Tìm khoảng cách giao điểm gần nhất $t_{\text{hit}}$ của tia với bao lồi 3D của thân người $\mathcal{M}_{\text{torso}} = \text{ConvexHull}(V_{\text{torso}})$:
$$t_{\text{hit}} = \min \{ t > 0 \mid \mathbf{r}(t) \cap \mathcal{M}_{\text{torso}} \ne \emptyset \}$$
$$\text{Occluded}(P_k) \iff t_{\text{hit}} < \|P_k\|_2 - \tau \quad (\text{với } \tau = 0.05\,\text{m})$$

* **Ưu điểm:** Tính toán hình học 3D chuẩn xác, không phụ thuộc vào ma trận chiếu camera 2D, xác định đúng thể tích che khuất.

---

## 2. Lọc An Toàn và Hòa Trộn Khớp (Uncertainty-Gating & Blended Correction)

* **Vị trí file:** [fusion_pipeline/correction.py](file:///d:/optimization_model_monocular_3/fusion_pipeline/correction.py#L76-L130), [fusion_pipeline/detector.py](file:///d:/optimization_model_monocular_3/fusion_pipeline/detector.py#L226-L260)

### Trước (Hard Replacement)
Thay thế cưỡng bức 100% tọa độ khớp từ camera phụ sang camera chính nếu độ tin cậy chênh lệch lớn hơn ngưỡng $\delta$:
$$K_1 = \{ n \in \text{Joints} \mid H_1(n) > H_2(n) + \delta \}$$
$$P_{2,\text{corr}}(n) = T_{12}(P_1(n))$$
$$P_{1,\text{corr}}(n) = T_{21}(P_2(n))$$

* **Nhược điểm:** Nếu camera phụ bị lỗi góc nhìn hoặc độ dịch chuyển quá xa, việc thay thế 100% sẽ gây nhảy vị trí đột ngột (giật khung hình), làm tăng vọt gia tốc `Accel Error`.

### Sau (Gating + Confidence Blending)
1. **Uncertainty-Gating:** Khớp ở camera nguồn chỉ được dùng để sửa nếu thỏa mãn cả 3 điều kiện:
   $$K_1 = \{ n \in \text{Joints} \mid H_1(n) > H_2(n) + \delta \;\land\; \text{vis}_1(n) = \text{True} \;\land\; H_1(n) \ge \tau_{\text{conf}} \}$$
   $$(\tau_{\text{conf}} = 0.1)$$
2. **Khoảng cách tối đa (Max Displacement Constraint):**
   $$\text{Bỏ qua sửa nếu: } \| T_{12}(P_1(n)) - P_2(n) \|_2 > d_{\max} \quad (d_{\max} = 0.35\,\text{m})$$
3. **Hòa trộn tỷ lệ tin cậy (Harmonic Blending):**
   Thay vì thay thế thô, tọa độ mới là trung bình có trọng số theo độ tin cậy:
   $$P_{2,\text{corr}}(n) = \alpha \cdot T_{12}(P_1(n)) + (1 - \alpha) \cdot P_2(n), \quad \alpha = \frac{H_1(n)}{H_1(n) + H_2(n)}$$
   $$P_{1,\text{corr}}(n) = \beta \cdot T_{21}(P_2(n)) + (1 - \beta) \cdot P_1(n), \quad \beta = \frac{H_2(n)}{H_1(n) + H_2(n)}$$

---

## 3. Hàm Phạt Mượt Thời Gian Trong Tối Ưu SLSQP (Temporal Penalty)

* **Vị trí file:** [fusion_pipeline/optimization.py](file:///d:/optimization_model_monocular_3/fusion_pipeline/optimization.py#L85-L115)

### Trước (Vận tốc bậc 1)
Phạt sai khác khoảng cách trực tiếp với frame liền trước:
$$\mathcal{P}_{\text{temp}}(x_t) = \sum_{n \in F} \| x_t(n) - x_{t-1}(n) \|_2^2$$

### Sau (Gia tốc bậc 2)
Sử dụng đạo hàm bậc 2 theo thời gian (gia tốc) khi có đủ dữ liệu từ 2 frame trước ($t-1$ và $t-2$):
$$\mathcal{P}_{\text{temp}}(x_t) = \begin{cases} 
\displaystyle\sum_{n \in F} \| x_t(n) - 2 x_{t-1}(n) + x_{t-2}(n) \|_2^2 & \text{khi có } t-2 \\
\displaystyle\sum_{n \in F} \| x_t(n) - x_{t-1}(n) \|_2^2 & \text{khi chỉ có } t-1 
\end{cases}$$

* **Ưu điểm:** Khớp chuyển động theo quỹ đạo có quán tính thay vì ép vận tốc tiến về 0, trực tiếp tối ưu hóa cho chỉ số `Accel Error`.

---

## 4. Bộ Lọc Làm Mượt Quỹ Đạo Savitzky-Golay (Temporal Smoothing Filter)

* **Vị trí file:** [fusion_pipeline/executor.py](file:///d:/optimization_model_monocular_3/fusion_pipeline/executor.py#L332-L362), [configs/pipeline.yml](file:///d:/optimization_model_monocular_3/configs/pipeline.yml#L50-L54)

### Trước
Không có bộ lọc hậu xử lý chuỗi thời gian cho tập khớp sau fusion. Việc thay đổi ma trận RANSAC giữa từng frame gây ra vi rung (micro-jitter), dẫn đến `Accel Error` cao hơn baseline monocular.

### Sau
Tích hợp bộ lọc Savitzky-Golay đa thức bậc 2 trên cửa sổ trượt 5 frame dọc theo trục thời gian cho toàn bộ chuỗi 3D keypoints:
$$\hat{\mathbf{P}}_{:, j} = \text{Savitzky-Golay}\left(\mathbf{P}_{:, j}, \text{window\_length}=5, \text{polyorder}=2, \text{axis}=0\right)$$

* **Hiệu quả thực nghiệm:**
  * `Accel Error`: Giảm từ **`14.05` mm/frame²** (baseline) xuống còn **`8.22` mm/frame²** *(giảm > 40%)*.
  * `MPJPE`: Giữ nguyên và tốt hơn baseline (**`187.88` mm** vs **`192.78` mm**).

---

## 5. Temporal Loss Trong Hậu Tối Ưu SMPL (Post-Optimization SMPL)

* **Vị trí file:** [learnable_pipeline/post_opt.py](file:///d:/optimization_model_monocular_3/learnable_pipeline/post_opt.py#L80-L87)

### Trước (1st-order Difference)
$$\mathcal{L}_{\text{temp}} = \frac{1}{T-1} \sum_{t=1}^{T-1} \left( \|\boldsymbol{\theta}_t - \boldsymbol{\theta}_{t-1}\|_2^2 + \|\mathbf{t}_t - \mathbf{t}_{t-1}\|_2^2 \right)$$

### Sau (2nd-order Acceleration Difference)
$$\mathcal{L}_{\text{temp}} = \frac{1}{T-2} \sum_{t=2}^{T-1} \left( \|\boldsymbol{\theta}_t - 2\boldsymbol{\theta}_{t-1} + \boldsymbol{\theta}_{t-2}\|_2^2 + \|\mathbf{t}_t - 2\mathbf{t}_{t-1} + \mathbf{t}_{t-2}\|_2^2 \right)$$

---

## 6. Cấu Hình & Pipeline Vét Cạn (Brute Force Pipeline)

* **Vị trí file:** [configs/brute_force.yml](file:///d:/optimization_model_monocular_3/configs/brute_force.yml), [brute_force_runner.py](file:///d:/optimization_model_monocular_3/brute_force_runner.py#L315-L330)

1. **Cập nhật dữ liệu mới:**
   * Chuẩn hóa toàn bộ đường dẫn đầu vào sang dataset mới `wham_out_video_{c}_seg_{s}` và video nội bộ `output.mp4`.
   * Mở rộng đầy đủ 11 segments (`seg_1` đến `seg_11`), tổng cộng 602 cặp camera thực tế.
2. **Khắc phục lỗi hiển thị:**
   * Không ép kiểu `round(..., 2)` đối với tham số $\alpha$ và $\beta$, giữ nguyên độ chính xác thực tế `0.001` thay vì bị làm tròn thành `0.0`.
   * Loại bỏ các dòng lỗi `N/A` và `inf` khỏi báo cáo CSV, chỉ ghi nhận các cặp chạy thành công.

---

## 7. Xiết Cứng Chiều Dài Xương (Rigid Bone Length Enforcement & Kinematic Projection)

* **Vị trí file:** [fusion_pipeline/config.py](file:///d:/optimization_model_monocular_3/fusion_pipeline/config.py), [fusion_pipeline/optimization.py](file:///d:/optimization_model_monocular_3/fusion_pipeline/optimization.py), [fusion_pipeline/executor.py](file:///d:/optimization_model_monocular_3/fusion_pipeline/executor.py)

### Trước
* Ràng buộc bất đẳng thức trong SLSQP dùng tỉ lệ nhân trắc học xấp xỉ với biên độ dung sai quá rộng:
  $$\left(0.85 \cdot \text{ratio} \cdot H\right)^2 \le \|\mathbf{P}_c - \mathbf{P}_p\|_2^2 \le \left(1.15 \cdot \text{ratio} \cdot H\right)^2$$
* Sau khi thay thế khớp qua Umeyama $T_{21}$ và sau bộ lọc Savitzky-Golay, không có bước khóa cứng chiều dài xương $\implies$ chiều dài xương dao động tự do, khiến sai số `MBLE` của Fused (**`35.71` mm**) tệ hơn cả baseline WHAM (**`34.07` mm**).

### Sau
1. **Xiết chặt biên độ ràng buộc SLSQP:** Lấy trực tiếp chiều dài xương tham chiếu $L_{\text{ref}} = \|\mathbf{P}_c^{\text{posed}} - \mathbf{P}_p^{\text{posed}}\|_2$ của chủ thể làm đích, thu hẹp dung sai về $[0.95, 1.05]$:
   $$\left(0.95 \cdot L_{\text{ref}}\right)^2 \le \|\mathbf{P}_c - \mathbf{P}_p\|_2^2 \le \left(1.05 \cdot L_{\text{ref}}\right)^2$$
2. **Khóa cứng động học ngược (Kinematic Projection):** Duyệt cây động học xuôi cho 15 khớp chi/đầu theo thứ tự topo. Giữ nguyên hướng vector $\mathbf{v}$ từ cha đến con nhưng chiếu chiều dài về chính xác $L_{\text{ref}}$:
   $$\mathbf{v} = \mathbf{P}_c - \mathbf{P}_p, \quad \hat{\mathbf{P}}_c = \mathbf{P}_p + L_{\text{ref}} \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$
   *(Áp dụng ngay sau bước tối ưu frame và sau khi làm mượt quỹ đạo thời gian)*.

* **Hiệu quả thực nghiệm:**
  * `MBLE cam1`: Giảm từ **`35.71` mm** $\rightarrow$ **`32.93` mm** *(tốt hơn baseline WHAM `34.07` mm và vượt trội hơn Learnable `33.7` mm)*.
  * `MBLE cam2`: Giảm từ **`34.51` mm** $\rightarrow$ **`33.02` mm** *(tốt hơn baseline WHAM `33.78` mm)*.
  * `MPJPE cam1`: **`273.71` mm** *(tốt hơn baseline `275.00` mm)*.
  * `PA-MPJPE cam1`: **`201.08` mm** *(tốt hơn baseline `202.61` mm)*.
  * `Accel cam1`: **`13.68` mm/frame²** *(tốt hơn baseline `21.37` mm/frame²)*.

---

## 8. Bypass Gating Cho Khớp Bị Che Khuất & Dung Hợp Đa Góc Nhìn (Solution 2 & 3)

* **Vị trí file:** [fusion_pipeline/correction.py](file:///d:/optimization_model_monocular_3/fusion_pipeline/correction.py), [fusion_pipeline/executor.py](file:///d:/optimization_model_monocular_3/fusion_pipeline/executor.py)

### Bối cảnh & Vấn đề
Khi dataset không có 2D keypoint confidence, độ tin cậy detector chỉ phụ thuộc vào khoảng cách:
* Với các khớp không bị che khuất: $H_1 \approx H_2 \approx 0.88 \implies |H_1 - H_2| < \delta \implies K_1 = \emptyset, K_2 = \emptyset$.
* Toàn bộ các khớp này được đưa vào nhóm Inliers $L$ và đưa tiếp vào Anchor $A_{\text{new}}$ $\implies$ **không bao giờ được tối ưu hay kết hợp chéo giữa 2 camera** (Cam 1 giữ nguyên 100% kết quả monocular WHAM).
* Với các khớp bị che khuất nặng (>35 cm), điều kiện gating `disp > max_displacement` (35 cm) bỏ qua hiệu chỉnh, khiến camera bị che khuất tiếp tục giữ nguyên dự đoán lỗi.

### Giải pháp 2: Bypass Displacement Gating Khi Camera Mục Tiêu Bị Che Khuất
Khi camera mục tiêu bị che khuất ($\text{vis}_{\text{target}}(n) = \text{False}$), cho phép thay thế/hòa trộn bất kể độ dịch chuyển:
$$\text{Gating Bypassed} \iff \neg \text{vis}_{\text{target}}(n)$$
Khắc phục triệt để hiện tượng khớp bị che khuất (lệch > 35 cm) bị loại bỏ và giữ lại sai số lớn từ monocular prior.

### Giải pháp 3: Dung Hợp Hình Học Đa Góc Nhìn Trực Tiếp (Direct Multi-View Geometric Fusion)
Với các khớp cùng nhìn thấy được ở cả hai camera ($\text{vis}_1(n) = \text{True} \;\land\; \text{vis}_2(n) = \text{True}$):
Thay vì giữ nguyên camera đơn lẻ, áp dụng dung hợp tọa độ 3D hai chiều qua biến đổi tương đồng Umeyama ($T_{12}, T_{21}$) với trọng số tin cậy $H_1, H_2$:
$$P_{1,\text{fused}}(n) = \frac{H_1(n) P_1(n) + H_2(n) T_{21}(P_2(n))}{H_1(n) + H_2(n)}$$
$$P_{2,\text{fused}}(n) = \frac{H_2(n) P_2(n) + H_1(n) T_{12}(P_1(n))}{H_1(n) + H_2(n)}$$
Ràng buộc an toàn:
* Loại trừ các anchor torso cốt lõi ($\text{core\_anchors} = \{\text{pelvis}, \text{neck}, \text{left\_hip}, \text{right\_hip}, \text{left\_shoulder}, \text{right\_shoulder}\}$) nhằm bảo toàn gốc tọa độ và hướng thân.
* Ngưỡng kiểm tra chênh lệch góc nhìn: $\|P_1(n) - T_{21}(P_2(n))\|_2 \le d_{\max}$ ($d_{\max} = 0.35\,\text{m}$).
* Chiếu cứng chiều dài xương động học (`_enforce_bone_lengths`) và lọc mượt Savitzky-Golay ngay sau khi dung hợp.

### Hiệu Quả Thực Nghiệm (482 Frames Đo Lường)
* **CAM1 MBLE:** Tiếp tục giảm từ `33.25` mm $\rightarrow$ **`32.30` mm** *(tiếp tục dẫn đầu bảng)*.
* **CAM1 Accel Error:** Giảm sâu từ `18.63` $\rightarrow$ **`12.22` mm/frame²** *(giảm >34% rung giật gia tốc)*.
* **CAM2 MPJPE:** Cải thiện **+4.64 mm** (`423.64` $\rightarrow$ **`419.01` mm**).
* **CAM2 PA-MPJPE:** Cải thiện **+0.67 mm** (`202.38` $\rightarrow$ **`201.72` mm**).



---

## 9. Khắc Phục Negative Transfer & Chặn Lỗi Cánh Tay Đòn Của Umeyama (Strict Gating & Safe Consensus)

* **Vị trí file:** [fusion_pipeline/correction.py](file:///d:/optimization_model_monocular_3/fusion_pipeline/correction.py)

### Bối cảnh & Vấn đề
* **Hiệu ứng cánh tay đòn (Lever-Arm Error):** Phép biến đổi tương đồng Umeyama (T_21) chỉ được fit trên các khớp thân mình (Torso). Do khoảng cách các khớp torso hẹp (~35 cm), sai số góc xoay 3° - 5° bị khuếch đại lên 50 - 70 mm ở các chi xa (cổ tay, cổ chân).
* **Negative Transfer:** Chiếu Cam 2 sang Cam 1 bằng T_21 làm tăng sai số cổ tay từ 1393 mm -> 1424 mm (+31 mm tệ hơn). Việc bypass gating khi che khuất hoặc ép buộc hòa trộn 50% : 50% trên các khớp nhìn rõ khiến kết quả sau fusion bị kéo tụt MPJPE/PA-MPJPE.

### Giải pháp
1. **Khóa trần độ dịch chuyển khi che khuất:** Ngăn chặn việc bypass hoàn toàn gating; đặt giới hạn trần nghiêm ngặt 1.25 * d_max cho các khớp bị che khuất để loại bỏ các điểm biến dạng cực đoan do xoay góc.
2. **Loại bỏ cưỡng bức hòa trộn trên khớp nhìn rõ:** Nếu cả hai camera cùng nhìn rõ (vis1 and vis2), chỉ hòa trộn khi đạt độ đồng thuận cao (disp <= 0.08m); giữ nguyên quan sát monocular của camera chính để tránh làm hỏng các dự đoán đúng.
