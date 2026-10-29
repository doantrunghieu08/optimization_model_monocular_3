# PA-MPJPE Degradation Report

## Tổng quan

So sánh 3 phương pháp: **posed** (baseline) → **fused** (fusion 2 cameras) → **only_learnable** (không dùng fusion).

Degradation = khi fused hoặc learnable có PA-MPJPE **cao hơn** posed (càng cao = càng tệ).

---

## Kết quả Định lượng

### Camera 1 (evaluated=cam1, GT=cam22)

| Phương pháp | Số frame suy giảm so với posed | Tỷ lệ |
|---|---|---|
| **fused** | 75 / 127 frames | **59%** |
| **only_learnable** | 77 / 127 frames | **61%** |

**Top frames suy giảm nặng nhất (fused vs posed):**

| Frame | posed (mm) | fused (mm) | Δ (mm) |
|---|---|---|---|
| 92 | 51.9 | 71.6 | **+19.6** |
| 222 | 36.1 | 54.5 | +18.3 |
| 203 | 36.3 | 53.1 | +16.9 |
| 225 | 37.1 | 53.8 | +16.7 |
| 220 | 35.9 | 52.3 | +16.4 |

**Top frames suy giảm nặng nhất (learnable vs posed):**

| Frame | posed (mm) | learnable (mm) | Δ (mm) |
|---|---|---|---|
| 6 | 88.1 | 116.0 | **+28.0** |
| 4 | 80.7 | 99.6 | +18.9 |
| 92 | 51.9 | 63.2 | +11.2 |

### Camera 2 (evaluated=cam2, GT=cam21)

| Phương pháp | Số frame suy giảm | Tỷ lệ |
|---|---|---|
| **fused** | 32 / 127 frames | **25%** |
| **only_learnable** | 98 / 127 frames | **77%** |

**Top frames suy giảm nặng nhất (fused vs posed, cam2):**

| Frame | posed (mm) | fused (mm) | Δ (mm) |
|---|---|---|---|
| **94** | 78.7 | 165.3 | **+86.7** |
| **163** | 88.5 | 161.7 | **+73.2** |
| 164 | 85.3 | 133.8 | +48.5 |
| 105 | 86.1 | 130.7 | +44.7 |
| 104 | 82.2 | 124.2 | +42.0 |

---

## Per-Joint Degradation (AVERAGE, cam1)

| Joint | posed | fused | learnable | fused_Δ | learn_Δ |
|---|---|---|---|---|---|
| neck | 41.1 | 41.3 | 38.8 | +0.3 WORSE | -2.2 |
| right_shoulder | 38.4 | 37.0 | 40.5 | -1.4 | +2.1 WORSE |
| right_elbow | 48.1 | 49.8 | 49.7 | +1.7 WORSE | +1.6 WORSE |
| right_wrist | 110.7 | 108.8 | 109.3 | -1.8 | -1.3 |
| left_shoulder | 50.7 | 51.2 | 50.4 | +0.6 WORSE | -0.3 |
| left_elbow | 41.3 | 42.7 | 46.2 | +1.4 WORSE | +4.9 WORSE |
| left_wrist | 78.8 | 79.3 | 81.7 | +0.5 WORSE | +2.8 WORSE |
| right_hip | 63.7 | 62.6 | 70.2 | -1.1 | +6.5 WORSE |
| right_knee | 58.1 | 57.3 | 59.3 | -0.8 | +1.2 WORSE |
| right_ankle | 96.6 | 96.4 | 97.8 | -0.2 | +1.2 WORSE |
| **left_hip** | 46.0 | 53.1 | 47.0 | **+7.1 WORSE** | +1.0 WORSE |
| **left_knee** | 87.8 | 95.8 | 93.4 | **+8.0 WORSE** | +5.5 WORSE |
| **left_ankle** | 75.4 | 82.8 | 69.7 | **+7.4 WORSE** | -5.7 |

> [!NOTE] Các joint bên **trái** (left_hip, left_knee, left_ankle) bị suy giảm nặng nhất khi fusion — đây là dấu hiệu của occlusion hoặc misalignment bên phía cam1 nhìn từ góc bị che.

---

## Phân tích Nguyên nhân

### 1. Cross-camera Rigid Transform không khớp ở frames đặc biệt

Trong `correction.py`, hàm `estimate_bidirectional_similarity()` dùng **RANSAC + Umeyama** để ước lượng rigid transform giữa hai camera. Khi:

- Frame **92, 94** — các frame có PA-MPJPE suy giảm lớn nhất — là các frame trong **vùng chuyển tiếp chuyển động phức tạp** (nhìn vào số frame số thứ tự, đây là các cluster: frame 89–99 khi nhân vật đang ôm/tiếp xúc với người kia).
- **Anchor joints bị nhiễu** (torso bị occlude): fallback logic ở dòng 149–151 dùng `knee`, `elbow` thay thế, nhưng các khớp này có sai số lớn hơn → rigid transform ước lượng sai → mọi joint transform theo sẽ sai theo.

```python
# correction.py L146-162
torso_candidates = [n for n in candidate_names if n in {"pelvis","neck","left_shoulder",...}]
if len(torso_candidates) < 3:
    fallback = [n for n in candidate_names if n in {"neck","pelvis","left_knee","right_knee",...}]
```

### 2. Kinematic Chain error propagation

`transform_joint_kinematic()` (L104-115) transform mỗi joint bằng cách **rotate bone vector quanh parent**:

```python
v_src = as_xyz(src_cam[name]) - as_xyz(src_cam[parent])
return as_xyz(tgt_cam[parent]) + scale * (r @ v_src)
```

→ Nếu **parent joint sai** (ví dụ `left_hip` sai do occlusion), thì `left_knee` và `left_ankle` đều sai theo kinematic chain — **giải thích tại sao left_hip/knee/ankle đều WORSE cùng lúc**.

### 3. `fused` WORSE tại cam2 frame 94, 163 — Catastrophic rotation mismatch

Delta **+86.7mm** (frame 94) và **+73.2mm** (frame 163) chỉ xảy ra ở cam2 nhưng không ở cam1 → nguyên nhân là:

- Cam2 nhìn từ góc bị khuất hoàn toàn (occlusion hoặc người đang xoay lưng vào cam2).
- Rigid transform từ cam1 sang cam2 bị flip/mirror dẫn đến rotation sai 180° — một dạng **rotation ambiguity** kinh điển trong monocular 3D estimation.
- Frame 94: `posed_right_wrist_mm = 344.45` (đã rất cao kể cả trước fusion) → monocular pose cũng đã sai nghiêm trọng ở frame này, fusion lại lấy signal sai đó để transform → càng sai.

### 4. `only_learnable` WORSE ở 61–77% frames

Learnable component không dùng geometric cross-view correction → **không có cross-view regularization** → phụ thuộc hoàn toàn vào monocular depth/pose estimate, vốn đã bị bất định trong 3D.

Đặc biệt frame 6, 4, 3, 2 (đầu sequence) suy giảm mạnh: **model learnable chưa hội tụ ở đầu sequence** vì thiếu temporal context.

### 5. N/A frames và incomplete fusion

Nhiều frame có `occ_mm = N/A` (frames 77–82, 96–100, 180, 185, 187, ...) — các frame này bị skip occlusion-branch hoàn toàn. Nếu pipeline fallback không cẩn thận, `fused` có thể dùng partial transform → artifact.

---

## Tóm tắt Nguyên nhân Chính

| Nguyên nhân | Ảnh hưởng | Frames điển hình |
|---|---|---|
| **Anchor joints bị occlude** → rigid transform sai | fused WORSE trên 59% frames (cam1) | 92, 203, 220–225 |
| **Kinematic error propagation** từ parent → child | left_hip/knee/ankle WORSE cùng lúc | Toàn bộ average |
| **Rotation ambiguity / flip** khi nhân vật xoay lưng vào cam | fused WORSE +87mm (cam2) | 94, 163, 164 |
| **Monocular pose đã sai** trước fusion (right_wrist > 300mm) | Fusion khuếch đại sai số | 94–100 |
| **Learnable thiếu temporal context** đầu sequence | learnable WORSE frame 2–6 | 2, 3, 4, 6 |

> [!IMPORTANT] Frame 94 là trường hợp tồi tệ nhất (+86.7mm ở cam2). Đây là frame mà cả monocular pose (`right_wrist=344mm`) lẫn fusion đều sai — cần xem video để xác nhận đây có phải frame occlusion hoàn toàn không.

> [!TIP] Hướng fix đơn giản nhất: **skip fusion** khi `posed_priority1_mm > threshold_bad` (ví dụ > 150mm), tức là khi monocular pose đã rất tệ thì không nên trust nó để làm anchor cho transform.
