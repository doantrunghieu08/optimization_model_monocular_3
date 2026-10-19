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
