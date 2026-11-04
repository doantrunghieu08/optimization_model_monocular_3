# So Sánh Mô Hình Mới (AOTR / Sequence Stability) vs Mô Hình Cũ (Belief Fusion)

Tài liệu đối chiếu sự chuyển dịch kiến trúc giữa **Mô hình Belief Fusion cũ** và **Mô hình AOTR mới (Observability-Aware Trust-Region)**, làm rõ nguyên nhân PA-MPJPE từng bị bóp nghẹt và giải pháp mở khóa mà không làm suy giảm Accel Error.

---

## 1. Bảng So Sánh Kiến Trúc Cốt Lõi

| Tiêu chí | Mô hình Belief Fusion cũ | Mô hình AOTR mới (Hiện tại) |
| :--- | :--- | :--- |
| **Bản chất suy luận** | Gán nhãn tin cậy ($H_1, H_2$) qua Harmonic Mean + khoảng cách $L$ | Tối ưu hóa Trust-Region dựa trên khả năng quan sát 2D hoặc 3D surrogate |
| **Không gian điều chỉnh** | Thay thế tọa độ 3D khớp trực tiếp hoặc quay khớp cục bộ | Tối ưu góc quay hướng xương ($\alpha \in [0, 1]$) qua Forward Kinematics (FK) |
| **Ràng buộc chiều dài xương** | Chiếu cưỡng bức FK sau cùng (`_enforce_bone_lengths`) | Tự bảo toàn bằng cấu trúc cây động học và hiệu chỉnh nhân trắc học chuỗi |
| **Mục tiêu temporal** | Lọc Savitzky-Golay toàn cục + GPU Adam Sequence Loss | Bounded optimization trên residual delta $F - M$ (`_temporal_scales`) |
| **Định hướng bảo vệ Master** | Ngưỡng margin tin cậy $H_2 > H_1 + \Delta_{\text{margin}}$ | Trust-region penalty + Null-space projection + Master prior |

---

## 2. Các Nâng Cấp Kỹ Thuật Chính

### 2.1. Tối ưu Trust-Region theo hướng xương thay vì dời điểm
* **Cũ (Belief):** Đánh giá chênh lệch $H_2 - H_1$. Khi vượt ngưỡng margin, tọa độ khớp của Slave được chuyển hệ tọa độ và kéo điểm Master sang điểm Slave. Dễ gây biến dạng nếu phép biến đổi tương đồng còn sai số.
* **Mới (AOTR):** Master giữ nguyên độ dài xương gốc $l_{\text{master}}$. Slave chỉ đề xuất hướng xương ứng viên $\mathbf{u}_S$. Thuật toán Trust-Region 1D tìm trọng số hòa trộn $\alpha$:
  $$\mathbf{u}(\alpha) = \text{Normalize}\left((1 - \alpha)\mathbf{u}_M + \alpha \mathbf{u}_S\right)$$
  Tọa độ khớp con được tái cấu trúc từ khớp cha:
  $$P(c) = P(p) + l_{\text{master}} \cdot \mathbf{u}(\alpha)$$
  Triệt tiêu hoàn toàn rủi ro vỡ chiều dài xương hay giật khớp.

### 2.2. Lọc làm mượt Residual thay vì làm mượt toàn bộ quỹ đạo
* **Cũ (Belief):** Dùng Savitzky-Golay lọc thẳng trên tọa độ 3D $(x, y, z)$ của Master. Bộ lọc làm mượt cả chuyển động thật lẫn chuyển động nhanh tự nhiên của người, gây bẹt đỉnh cử động.
* **Mới (AOTR):** Chỉ tối ưu hệ số co giãn $s_t \in [s_{\text{min}}, 1.0]$ trên phần hiệu chỉnh sai lệch $\mathbf{D}_t = F_t - M_t$:
  $$\min_{s_t} \sum_t \|\mathbf{D}_t\|^2 (s_t - 1)^2 + w_{\text{acc}} \sum_t \|\Delta^2(s_t \mathbf{D}_t)\|^2 + w_{\text{vel}} \sum_t \|\Delta^1(s_t \mathbf{D}_t)\|^2$$
  Nếu Master chuyển động gia tốc thật, chuyển động đó được giữ nguyên 100%. Hệ thống chỉ triệt tiêu xung động do Slave can thiệp đột ngột.

---

## 3. Vì Sao PA-MPJPE Từng Bị Bóp Nghẹt & Cách Mở Khóa

Dù AOTR khắc phục được lỗi giật điểm, thực nghiệm ban đầu cho thấy cải thiện PA-MPJPE bị kìm hãm mạnh do 3 điểm nghẽn tham số:

```
[AOTR 3D Core]                          [Sequence Stability Pass]
master_prior: 0.75 (quá cao)  ──►  w_acc: 15.0 / shape: 1.0  ──►  smooth_directions: true
(kéo alpha gần về 0)               (s_t bị đè về s_min = 0.20)      (savgol 15 frames san phẳng peak)
         │                                       │                                │
         ▼                                       ▼                                ▼
   BƯỚC 1: Hướng ít đổi            BƯỚC 2: Residual bị bóp 80%        BƯỚC 3: Dập tắt cử động góc
                         ═══════════════════════════════════════════
                         HỆ QUẢ: PA-MPJPE KHÔNG THỂ CẢI THIỆN RÕ RỆT
```

### Các điều chỉnh đã thực hiện:

1. **Giảm `three_d_master_prior_weight` (0.75 $\to$ 0.50):**  
   Hạ lực cản phía Master trong hàm mục tiêu AOTR-3D, cho phép Slave đóng góp góc quay lớn hơn khi Slave có độ tin cậy ổn định.
2. **Cân bằng lại `_temporal_scales` (`residual_accel_weight`: 15.0 $\to$ 6.0, `shape_weight`: 1.0 $\to$ 2.5):**  
   Ngăn solver đè $s_t$ xuống sàn ở các frame có chuyển động gập duỗi mạnh.
3. **Nâng ngưỡng sàn `minimum_retention` (0.20 $\to$ 0.40):**  
   Bảo toàn tối thiểu 40% lực hiệu chỉnh của Slave thay vì bị triệt tiêu tới 80%.
4. **Tắt `smooth_directions` (true $\to$ false):**  
   Tránh việc Savitzky-Golay cửa sổ 15 frames cào bằng các góc cử động đặc thù (khuỷu tay, đầu gối).

---

## 4. Bảo Toàn Accel Error Khi Mở Rộng PA-MPJPE

Hệ thống **không** làm xấu đi Accel Error nhờ các cơ chế phòng hộ tự nhiên:
* **Hàm phạt gia tốc bậc hai vẫn chiếm ưu thế:** Tỉ lệ `residual_accel_weight` (6.0) vẫn cao hơn gấp đôi so với `shape_weight` (2.5), vẫn dập tắt dao động tần số cao.
* **Cơ chế FK phân tầng:** Tọa độ phụ thuộc vào vector đơn vị hướng xương, không có hiện tượng dịch chuyển tự do vượt quá biên độ tự nhiên của cơ thể.
* **Hiệu chỉnh chiều cao giải phẫu ổn định theo chuỗi:** Khóa tỉ lệ chi người theo trung vị chuỗi thời gian, ngăn trôi dạt tỉ lệ cơ thể giữa các khung hình liên tiếp.
