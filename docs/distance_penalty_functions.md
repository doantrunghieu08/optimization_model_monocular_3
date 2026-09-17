# Chi tiết các Hàm Phạt Khoảng cách (Distance Penalty Functions) trong Pipeline

Tài liệu này tổng hợp chi tiết về các loại **hàm phạt khoảng cách (distance penalty)** được sử dụng trong hệ thống tối ưu hóa và dung hợp 3D Pose (Fusion Pipeline).

---

## 1. Proximity Regularization Penalty (`proximity_penalty`)
* **Vị trí mã nguồn**: [`fusion_pipeline/optimization.py`](file:///d:/optimization_model_monocular_3/fusion_pipeline/optimization.py#L86-L94)
* **Cấu hình trong `pipeline.yml`**:
  ```yaml
  fusion:
    optimization:
      regularization: true
      regularization_lambda: 1.0 # Hệ số phạt của proximity regularization
  ```

### Mục đích
Giữ cho tọa độ 3D của các khớp (joints) đang tối ưu $\mathbf{p}_{1,i}, \mathbf{p}_{2,i}$ không bị trôi lệch quá xa khỏi vị trí dự đoán ban đầu $\mathbf{cam1}_i, \mathbf{cam2}_i$ từ 2 camera.

### Công thức Toán học
$$P_{\text{prox}}(x) = \sum_{i \in F} \left( c_{1, i} \cdot \|\mathbf{p}_{1, i} - \mathbf{cam1}_i\|^2 + c_{2, i} \cdot \|\mathbf{p}_{2, i} - \mathbf{cam2}_i\|^2 \right)$$

### Giải thích biến:
* $F$: Tập hợp các điểm khớp được chọn để tối ưu hóa.
* $c_{1,i}, c_{2,i}$: Độ tin cậy (confidence score) của khớp $i$ ứng với Camera 1 và Camera 2.
* $\mathbf{p}_{1,i}, \mathbf{p}_{2,i}$: Tọa độ 3D của khớp $i$ (biến cần tìm trong bài toán tối ưu SLSQP).
* $\mathbf{cam1}_i, \mathbf{cam2}_i$: Tọa độ 3D dự đoán ban đầu từ các model quan sát.

---

## 2. Distance Belief Penalty (Phạt độ tin cậy theo khoảng cách Depth)
* **Vị trí mã nguồn**: [`fusion_pipeline/detector.py`](file:///d:/optimization_model_monocular_3/fusion_pipeline/detector.py#L114-L137)
* **Cấu hình trong `pipeline.yml`**:
  ```yaml
  fusion:
    belief:
      alpha: 0.001                      # Hệ số phạt theo khoảng cách 3D
      local_method: "optical_aware_belief" # Hoặc "naive_distance_belief"
  ```

### Mục đích
Trong mô hình camera đơn mắt (monocular), độ chênh lệch/sai số độ sâu (depth) tỉ lệ thuận với khoảng cách từ khớp đến camera. Hàm này suy giảm (phạt) độ tin cậy $P(\text{joint})$ của các khớp ở xa camera.

### Công thức Toán học

1. **Naive Distance Belief** (`calc_naive_distance_belief`):
   $$P(\text{joint}) = \frac{\text{vis}}{1 + \alpha \cdot \|\mathbf{p}\|^2}$$

2. **Optical-Aware Belief** (`calc_optical_aware_belief`):
   * Khi $d = \|\mathbf{p}\| < l_{\min}$: suy giảm kiểu Gaussian khi quá gần camera.
   * Khi $l_{\min} \le d \le l_{\max}$: suy giảm theo khoảng cách bình phương với hệ số $\alpha$:
     $$P(\text{joint}) = \frac{1}{1 + \alpha \cdot (d - l_{\min})^2}$$
   * Khi $d > l_{\max}$: suy giảm nhanh theo hàm mũ (Gaussian decay) ở vùng rất xa.

---

## 3. Temporal Distance Penalty (`temporal_penalty`)
* **Vị trí mã nguồn**: [`fusion_pipeline/optimization.py`](file:///d:/optimization_model_monocular_3/fusion_pipeline/optimization.py#L96-L113)
* **Cấu hình trong `pipeline.yml`**:
  ```yaml
  fusion:
    optimization:
      temporal_lambda: 2.0 # Hệ số phạt theo thời gian so với frame trước
  ```

### Mục đích
Phạt sự thay đổi đột ngột về khoảng cách/vị trí tương đối của từng khớp so với gốc tọa độ thân người ($\mathbf{root}$) giữa hai frame liên tiếp ($t$ và $t-1$), giúp tối ưu hóa chuỗi chuyển động mượt mà và tránh hiện tượng giật (jitter).

### Công thức Toán học
$$P_{\text{temp}}(x) = \sum_{i \in F} \left( \| (\mathbf{p}_{1, i}^{(t)} - \mathbf{root}_1^{(t)}) - (\mathbf{p}_{1, i}^{(t-1)} - \mathbf{root}_1^{(t-1)}) \|^2 + \| (\mathbf{p}_{2, i}^{(t)} - \mathbf{root}_2^{(t)}) - (\mathbf{p}_{2, i}^{(t-1)} - \mathbf{root}_2^{(t-1)}) \|^2 \right)$$

---

## Summary (Tóm tắt hàm mục tiêu tổng thể của SLSQP)
Khi bật tối ưu hóa SLSQP (`fusion.optimization.enabled: true`), hàm mục tiêu tổng quát được tính như sau:

$$\text{Objective}(x) = \text{DataLoss}(x) + \lambda_{\text{prox}} \cdot P_{\text{prox}}(x) + \lambda_{\text{temp}} \cdot P_{\text{temp}}(x)$$

Trong đó:
* $\text{DataLoss}(x)$: Huber Loss hoặc MSE Loss tính sự chênh lệch khoảng cách tương đối giữa khớp tối ưu $F$ và các khớp neo (anchors).
* $\lambda_{\text{prox}}$: `regularization_lambda`
* $\lambda_{\text{temp}}$: `temporal_lambda`
