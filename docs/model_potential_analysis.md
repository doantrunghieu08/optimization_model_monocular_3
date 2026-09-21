# Dự Đoán Mức Cải Thiện Cực Đại Cho Mô Hình Fusion & Optimization (Không Dùng Learnable)

Tài liệu này tinh chỉnh phân tích mức cải thiện cực đại (**Upper Bound / Performance Ceiling**) dành riêng cho **Mô hình Fusion & Optimization thuần hình học/động học** của bạn (không áp dụng thêm mô hình học máy `learnable`), dùng làm cơ sở khoa học để so sánh đối chứng trực tiếp với các mô hình **Learnable Baseline**.

---

## 1. Bảng Dự Đoán Mức Cải Thiện Cực Đại (Thuần Fusion & Optimization)

| Chỉ số Metric | Giá trị Raw (Monocular WHAM) | Giá trị Fusion Hiện tại | **Trần Cực đại Fusion Dự đoán (Pure Geometric)** | **$\% \Delta$ Cải thiện Tối đa Dự kiến** |
| :--- | :--- | :--- | :--- | :--- |
| **MPJPE** | $85 - 100\text{ mm}$ | $75 - 85\text{ mm}$ | **$48 - 56\text{ mm}$** | **$+40\% \text{ đến } +50\%$** |
| **PA-MPJPE** | $45 - 50\text{ mm}$ | $45 - 48\text{ mm}$ | **$35 - 40\text{ mm}$** | **$+18\% \text{ đến } +28\%$** |
| **MBLE** *(Lỗi chiều dài xương)* | $40 - 50\text{ mm}$ | $35 - 38\text{ mm}$ | **$< 12 - 18\text{ mm}$** | **$+55\% \text{ đến } +70\%$** |
| **Accel Error** *(Nhiễu Rung động)*| $20 - 30\text{ mm/f}^2$ | $14 - 16\text{ mm/f}^2$ | **$< 6 - 8\text{ mm/f}^2$** | **$+60\% \text{ đến } +75\%$** |

---

## 2. Nguyên Lý Hình Học Đạt Ngưỡng Cho Mô Hình Fusion

Dù **không sử dụng mạng Neural Network (Learnable)**, mô hình Fusion thuần hình học vẫn có thể chạm tới ngưỡng trên nhờ tối ưu hóa các nguyên lý vật lý & động học 3D:

### A. Triệt tiêu sai số độ sâu bằng Multi-View Ray-Casting (Cho MPJPE)
* Dữ liệu đơn camera (Monocular) bị sai số rất lớn dọc theo trục độ sâu Z ($\sim 60-80\text{ mm}$). 
* Việc giao các tia nhìn (Ray Casting & Triangulation) từ 2 hoặc nhiều camera sẽ triệt tiêu tới $80 - 90\%$ độ lệch Z tuyệt đối, đưa MPJPE về ngưỡng $48 - 56\text{ mm}$ mà không cần tới bất kỳ trọng số học máy nào.

### B. Khóa cứng tỷ lệ xương tuyệt đối (Cho PA-MPJPE & MBLE)
* Đơn camera có thể bị méo tỷ lệ chiều dài xương theo góc nhìn. Trong phương pháp Fusion, ta có thể trích xuất hệ số hình dáng cơ thể $\beta$ (SMPL Shape Parameters) cố định cho cả chuỗi chuyển động.
* Khóa cứng chiều dài từng đoạn xương ($d_{\text{bone}} = \text{const}$) trong bộ giải SLSQP sẽ ngăn chặn việc méo dáng local, đưa MBLE xuống $< 18\text{ mm}$ và đẩy PA-MPJPE cải thiện tới $+25\%$.

---

## 3. Lộ Trình Nâng Cấp Thuần Fusion (Non-Learnable Roadmap)

Để nâng hiệu năng của mô hình **Fusion** lên mức tối đa khi so sánh với **Learnable Baseline**:

1. **Khóa cứng khung xương tuyệt đối theo SMPL Shape ($\beta$) trong SLSQP**:
   - Ép đẳng thức chiều dài xương chuẩn $d_{i,j} = \text{Target}_{i,j}$ làm ràng buộc cứng (Equality Constraint) trong SLSQP Optimizer thay vì bất đẳng thức khoảng nới lỏng.
2. **Cửa sổ thời gian đa khung hình (Multi-Frame Window SLSQP Optimization)**:
   - Chạy SLSQP trên một cửa sổ $7 - 15$ frames đồng thời với các hàm phạt vận tốc $\dot{x}$ và gia tốc $\ddot{x}$ liên tục để làm mịn chuyển động hoàn toàn tự nhiên (Accel Error $< 8\text{ mm/f}^2$).
3. **Fusion Đa Camera (3+ Views Consensus RANSAC)**:
   - Mở rộng thuật toán từ 2 camera lên 3 hoặc 4 camera. RANSAC 3D multi-view sẽ tự động loại bỏ các điểm bị che khuất (Occlusion) từ góc nhìn phụ mà không cần mạng dự đoán belief phức tạp.
