# Phân tích Tính Chính xác và Ảnh hưởng của Bình phương trong Hàm Phạt Khoảng cách

Tài liệu này phân tích chuyên sâu về mặt lý thuyết toán học, quang học camera và thực nghiệm đối với **dạng bình phương ($d^2$)** trong các hàm phạt khoảng cách thuộc Fusion Pipeline.

---

## 1. Đánh giá tính chính xác của công thức (Theoretical Accuracy)

Công thức phạt khoảng cách dạng bình phương hiện tại ($d^2$ hay L2-norm squared) là **chính xác về mặt toán học và vật lý**:

### A. Trong bài toán Tối ưu hóa (Proximity & Temporal Regularization)

$$
\text{Penalty} = c_1 \cdot \|\mathbf{p}_1 - \mathbf{cam1}\|^2 + c_2 \cdot \|\mathbf{p}_2 - \mathbf{cam2}\|^2
$$

* **Cơ sở toán học**: Đây là dạng chuẩn của phương pháp **Bình phương tối thiểu (Least Squares)**.
* Hàm bình phương $f(\mathbf{x}) = \|\mathbf{x}\|^2 = x^2 + y^2 + z^2$ có **đạo hàm mượt (smooth & continuous gradient)** tại mọi điểm trong không gian $\mathbb{R}^3$, bao gồm cả tại điểm gốc $0$:
  $$
  \nabla f(\mathbf{x}) = 2\mathbf{x}
  $$
* Điều này cho phép bộ giải numerical như **SLSQP (Sequential Least Squares Programming)** trong SciPy xác định hướng giảm nhanh nhất (steepest descent) và Hessian một cách chính xác, hội tụ ổn định về nghiệm tối ưu.

### B. Trong Độ tin cậy theo khoảng cách Camera (Distance Belief)

$$
P(\text{joint}) = \frac{1}{1 + \alpha \cdot d^2}
$$

* **Cơ sở quang học**: Trong mô hình camera đơn mắt (Monocular Camera Projection $x = f \cdot \frac{X}{Z}$), sai số độ sâu (depth ambiguity $\sigma_Z$) của điểm 3D tỉ lệ thuận với **bình phương khoảng cách** đến camera:
  $$
  \Delta Z \propto Z^2 \cdot \Delta x
  $$
* Việc sử dụng $d^2$ trong hàm phạt phản ánh đúng tốc độ suy giảm chất lượng dữ liệu quan sát theo định luật nghịch đảo bình phương (Inverse-Square Law).

---

## 2. Phân tích ảnh hưởng khi BỎ BÌNH PHƯƠNG (Tuyến tính $d$)

Nếu thay thế $d^2$ bằng khoảng cách tuyến tính $d = \|\mathbf{d}\|$ hoặc L1-norm $|x| + |y| + |z|$, hệ thống sẽ gặp các vấn đề sau:

### 1. Bất liên tục đạo hàm tại 0 (Non-differentiability at Zero)

* Đạo hàm của $g(\mathbf{x}) = \|\mathbf{x}\| = \sqrt{x^2 + y^2 + z^2}$ là $\frac{\mathbf{x}}{\|\mathbf{x}\|}$. Tại $\mathbf{x} = 0$, đạo hàm **không xác định** (bị bất liên tục).
* Khi khớp 3D tiến gần tới vị trí mong muốn ($\mathbf{x} \to 0$), bộ giải SLSQP sẽ bị hiện tượng **rung dao động (oscillation)** liên tục quanh điểm 0 do gradient bị gãy đột ngột, dẫn đến không thể hội tụ mịn.

### 2. Mất cân bằng giữa nhiễu lớn (Outliers) và sai lệch nhỏ (Inliers)

* Khi $\epsilon < 1$ (sai lệch nhỏ): $\epsilon^2 \ll \epsilon$ (ví dụ: $0.05^2 = 0.0025 \ll 0.05$). Dùng $d^2$ phạt rất nhẹ các vị trí khớp gần đúng, giúp khớp tự nhiên và không bị bó cứng.
* Khi $\epsilon > 1$ (nhiễu lớn): $\epsilon^2 \gg \epsilon$. Dùng $d^2$ tạo lực kéo đủ mạnh để kéo các điểm bị văng xa (outliers) về vị trí hợp lý.
* Nếu bỏ bình phương, lực kéo điểm nhiễu lớn bị yếu đi, trong khi các điểm đã gần đúng lại bị kéo căng quá mức.

---

## 3. Bảng So sánh Chi tiết

| Tiêu chí                              | Dùng Bình phương ($d^2$) | Bỏ Bình phương ($d$)                            |                                                     |
| :-------------------------------------- | :------------------------------------------------------------------------------------- | :-------------------------------------------------- |
| **Đạo hàm tại 0**             | Khả vi mượt ($\nabla = 2\mathbf{x}$)                                              | Bất liên tục / Không xác định                |
| **Hội tụ SLSQP**                | Nhanh, ổn định, không dao động                                                   | Dễ bị rung giật quanh điểm 0, tốn`max_iter` |
| **Phạt nhiễu văng (Outliers)** | Phạt rất mạnh (bình phương)                                                      | Phạt yếu (tuyến tính)                           |
| **Phạt nhiễu nhỏ (Inliers)**   | Phạt rất nhẹ ($0.01^2 = 0.0001$) | Phạt tuyến tính ($0.01$) gây cứng khớp |                                                     |
| **Phù hợp quang học**          | Khớp với sai số độ sâu monocular ($\propto Z^2$)                               | Suy giảm không đủ nhanh ở khoảng cách xa     |

---

## 4. Giải pháp Tối ưu: Huber Loss

Nếu muốn hạn chế bớt tác động phạt quá nặng của $d^2$ đối với các điểm nhiễu cực đại mà vẫn giữ được độ mượt khi $d$ nhỏ, giải pháp chuẩn mực là **Huber Loss** (đã tích hợp sẵn trong pipeline qua cấu hình `fusion.optimization.loss_type: "huber"`):

$$
L_{\text{Huber}}(d) = \begin{cases} 
\frac{1}{2} d^2 & \text{nếu } d \le \delta \quad \text{(Dùng bình phương mượt khi sai số nhỏ)} \\
\delta \cdot \left(d - \frac{1}{2}\delta\right) & \text{nếu } d > \delta \quad \text{(Dùng tuyến tính chống văng khi sai số lớn)}
\end{cases}
$$

---

## 5. Khuyến nghị

1. **Giữ nguyên dạng bình phương $d^2$** cho Proximity Regularization và Temporal Penalty trong optimizer SLSQP.
2. Thiết lập `loss_type: "huber"` trong [`configs/pipeline.yml`](file:///d:/optimization_model_monocular_3/configs/pipeline.yml#L65) để đạt được sự dung hòa hoàn hảo giữa độ mượt của $d^2$ và sự bền vững với outlier của $d$.
