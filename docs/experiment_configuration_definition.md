# Experiment Configuration Definition

## 1. Mục tiêu

Tài liệu này định nghĩa rõ các cấu hình thí nghiệm để tránh nhầm lẫn giữa **Aligned**, **Higher-Belief** và **Proposed**.

Các cấu hình chính gồm:

- Raw
- Aligned
- Higher-Belief
- Proposed

> **Lưu ý quan trọng:**  
> `Aligned` ở đây **không phải Procrustes Alignment / PA-MPJPE**.  
> `Aligned` là quá trình đưa pose của Camera 2 về hệ tọa độ của Camera 1, sau đó lấy trung bình hai pose.

---

## 2. Raw

`Raw` là pose 3D đầu ra trực tiếp từ từng camera, chưa thực hiện căn chỉnh giữa hai hệ tọa độ và chưa fusion.

Ví dụ:

```text
Camera 1 -> Pose 1
Camera 2 -> Pose 2
```

Không thực hiện:

- coordinate transformation
- averaging
- belief fusion
- belief weighting
- optimization

---

## 3. Aligned

### 3.1. Định nghĩa

`Aligned` là baseline đa camera đơn giản.

Pose từ **Camera 2** được biến đổi sang **hệ tọa độ Camera 1**.

Sau khi hai pose nằm trong cùng một hệ tọa độ, pose cuối cùng được tính bằng **trung bình cộng theo từng joint**.

Pipeline:

```text
Pose Camera 1 -------------------------+
                                       |
                                       +--> Average --> Aligned Pose
                                       |
Pose Camera 2 --> Transform Cam2->Cam1 +
```

### 3.2. Coordinate Transformation

Nếu pose được biểu diễn trong hệ tọa độ camera tuyệt đối:

```math
X_1 = R_{2\rightarrow1} X_2 + t_{2\rightarrow1}
```

Trong đó:

- `X_2`: joint 3D trong hệ tọa độ Camera 2
- `R_{2->1}`: rotation từ Camera 2 sang Camera 1
- `t_{2->1}`: translation từ Camera 2 sang Camera 1
- `X_1`: joint sau khi được đưa về hệ tọa độ Camera 1

Nếu pose đã được root-centered và chỉ cần đồng nhất orientation, có thể chỉ sử dụng rotation:

```math
X_1 = R_{2\rightarrow1} X_2
```

Việc dùng rotation hay rotation + translation phải phụ thuộc vào cách dữ liệu 3D đang được lưu trong project.

### 3.3. Averaging

Sau khi Camera 2 đã được đưa về hệ Camera 1:

```math
X_j^{aligned}
=
\frac{
X_j^{cam1}
+
X_j^{cam2\rightarrow cam1}
}{2}
```

với `j` là joint thứ `j`.

Pseudo-code:

```python
cam2_in_cam1 = transform_pose_cam2_to_cam1(
    cam2_pose,
    R_21,
    t_21
)

aligned_pose = (cam1_pose + cam2_in_cam1) / 2.0
```

### 3.4. Aligned không sử dụng alpha

`Aligned` chỉ gồm:

1. coordinate transformation
2. arithmetic mean

Do đó `Aligned` **không sử dụng**:

- `alpha`
- `beta`
- belief score
- harmonic belief
- cross-view error detection
- belief-weighted fusion

---

## 4. Higher-Belief

`Higher-Belief` là baseline căn chỉnh hai camera rồi chọn cứng joint đáng tin hơn.

Đây là một baseline riêng biệt và không được đồng nhất với `Aligned`.

Pipeline:

```text
Cam1 Pose ------------------------------------+
                                              |
Cam2 Pose --> Transform Cam2 -> Cam1          +--> Compute Belief
                                                       |
                                                       v
                                             Select higher-belief joint
```

Trong pipeline này, các tham số như:

```yaml
alpha:
beta:
```

có thể được sử dụng trong belief model.

Ví dụ một dạng belief theo khoảng cách:

```math
P_j = \frac{C_j}{1 + \alpha L_j^2}
```

Do đó:

> `Aligned` không dùng belief. `Higher-Belief` và `Proposed` dùng cấu hình belief của pipeline.

---

## 5. Proposed Method

### 5.1. Định nghĩa mới

Cấu hình `Proposed` là pipeline fusion hai camera đầy đủ với:

```text
Optical
+
Global
+
Kinematic
+
Huber
```

`Proposed` không phải phép lấy trung bình của `Aligned`. Hai pose camera được xử lý qua belief, hiệu chỉnh liên camera và optimizer.

Không gọi cấu hình này là:

```text
Aligned + Fusion
```

Pipeline khái niệm:

```text
Cam1 Pose + Cam2 Pose
    |
    v
Optical / Local Belief
    |
    v
Global Term
    |
    v
Kinematic Term
    |
    v
Huber Robust Loss
    |
    v
Optimized Pose
```

Có thể hiểu objective tổng quát dưới dạng:

```math
L =
L_{optical}
+
L_{global}
+
L_{kinematic}
```

với sai số được xử lý bằng Huber loss.

Dạng tổng quát của Huber loss:

```math
L_{Huber}(r) =
\begin{cases}
\frac{1}{2}r^2, & |r| \leq \delta \\
\delta(|r| - \frac{1}{2}\delta), & |r| > \delta
\end{cases}
```

### 5.2. Tham số của Proposed

Cấu hình Proposed mới:

```text
Optical + Global + Kinematic + Huber
```

được điều khiển bởi các nhóm cấu hình:

```text
fusion.belief.local_method
fusion.belief.global
fusion.optimization.use_kinematic_constraints
fusion.optimization.loss_type
```

Khi `local_method` là `optical_aware_belief`, `alpha` và `beta` vẫn là tham số của mô hình belief dùng chung với Higher-Belief.

---

## 6. Cấu hình thí nghiệm cuối cùng

Bộ cấu hình nên được hiểu như sau:

| Configuration | Description |
|---|---|
| Raw | Pose đầu ra trực tiếp từ camera |
| Aligned | Transform Cam2 -> Cam1 rồi lấy trung bình hai pose |
| Higher-Belief | Transform Cam2 -> Cam1, tính belief rồi chọn joint có belief cao hơn |
| Proposed | Fusion hai camera: Optical/Local + Global + Kinematic + Huber |

Không sử dụng tên:

```text
Aligned + Fusion
```

cho Proposed.

---

## 7. Quan hệ giữa các pipeline

### Raw

```text
Camera
  |
  v
Pose
```

### Aligned

```text
Cam1 Pose ---------------------------+
                                     |
                                     +--> Mean --> Output
                                     |
Cam2 Pose --> Transform Cam2 -> Cam1 +
```

### Higher-Belief

```text
Cam1 Pose ------------------------------+
                                        +--> Belief --> Higher joint --> Output
Cam2 Pose --> Transform Cam2 -> Cam1 ---+
```

### Proposed

```text
Cam1 Pose + Cam2 Pose
    |
    v
Optical
    |
    v
Global
    |
    v
Kinematic
    |
    v
Huber
    |
    v
Proposed Output
```

---

## 8. Quy ước cần giữ trong code và báo cáo

Để tránh nhầm lẫn trong tương lai:

### Aligned

Phải luôn có nghĩa:

```text
Transform Camera 2 -> Camera 1
+
Arithmetic Mean
```

### Higher-Belief

Phải luôn có nghĩa:

```text
Transform Camera 2 -> Camera 1
+
Select each joint from the camera with higher belief
```

### Proposed

Phải luôn có nghĩa:

```text
Optical + Global + Kinematic + Huber
```

Không gọi Procrustes Alignment là `Aligned` trong tên experiment này.

Nếu cần báo cáo PA-MPJPE, nên ghi rõ:

```text
PA-MPJPE
```

hoặc:

```text
Procrustes-Aligned Evaluation
```

để không nhầm với baseline `Aligned`.

---

## 9. Tóm tắt

```text
RAW
= original pose

ALIGNED
= Cam2 -> Cam1
+ arithmetic mean

HIGHER-BELIEF
= Cam2 -> Cam1
+ compute belief
+ select higher-belief joint

PROPOSED
= Optical
+ Global
+ Kinematic
+ Huber
```

Điểm quan trọng nhất:

> **Aligned không sử dụng alpha.**

> **Proposed là fusion hai camera đầy đủ, không phải Aligned Averaging.**

> **Proposed dùng Optical/Local + Global + Kinematic + Huber.**
