# Checklist kiểm tra code fusion khi `seg_2 / cam4` làm Slave

## 1. Hiện tượng cần giải thích

Từ CSV phân tích được:

- `seg_2`
- `Cam Slave = cam4`

có 5 trường hợp:

| Cam Master | Cam Slave | % Δ_MPJPE | % Δ_PA-MPJPE |
|---|---:|---:|---:|
| cam2 | cam4 | -26.26 | -47.80 |
| cam7 | cam4 | -15.69 | -59.98 |
| cam0 | cam4 | -25.09 | -53.44 |
| cam5 | cam4 | -23.04 | -58.07 |
| cam8 | cam4 | -25.24 | -59.29 |

Trung bình:

- `% Δ_MPJPE ≈ -23.06%`
- `% Δ_PA-MPJPE ≈ -55.72%`

Trong khi khi `cam4` làm **Master**, kết quả lại dương.

Điều này gợi ý lỗi không đơn giản là:

> `cam4` xấu

mà có khả năng là:

> `cam4 + seg_2 + vai trò Slave + cách sử dụng belief trong fusion`

---

# 2. Belief đáng chú ý của `seg_2 / cam4`

`belief Slave`:

```text
[1.29, 1.30, 0.65, 1.45, 1.03, 1.36, 1.36, 1.35, 1.45,
 0.48, 1.35, 1.35, 1.18, 1.08, 1.42, 1.44, 1.32, 1.32,
 1.36, 1.42, 1.18]
```

Thống kê:

```text
mean = 1.24476
min  = 0.48
std  = 0.24819

6 joint < 1.2
16 joint < 1.4
21 joint < 1.5
```

Các joint thấp:

```text
joint 2  = 0.65
joint 4  = 1.03
joint 9  = 0.48
joint 12 = 1.18
joint 13 = 1.08
joint 20 = 1.18
```

Trong Random Forest:

```text
slave_9
ratio_9
master_9
diff_9
```

đều xuất hiện trong nhóm feature quan trọng.

**Joint 9 là joint cần kiểm tra đầu tiên.**

---

# 3. Quy luật Decision Tree phát hiện được

## MPJPE

```text
slave_count_lt_1.2 > 5.5
    AND
slave_mean > 1.09

=> Δ_MPJPE ≈ -23.064%
```

Có thể đọc thành:

```text
Slave có >= 6 joint belief < 1.2
nhưng mean belief tổng thể vẫn > 1.09
=> vùng nguy hiểm
```

---

## PA-MPJPE

```text
slave_count_lt_1.5 > 14.5
    AND
slave_mean > 1.09

=> Δ_PA-MPJPE ≈ -55.716%
```

`seg_2/cam4`:

```text
21 / 21 joint < 1.5
mean = 1.24476
```

nằm đúng trong vùng lỗi.

---

# 4. Giả thuyết cần kiểm tra trong code

Giả thuyết hiện tại:

```text
belief rất thấp
    ↓
hệ thống nhận ra camera không đáng tin
    ↓
weight Slave bị giảm mạnh
    ↓
không phá fusion nhiều
```

nhưng:

```text
một số joint rất thấp
+
mean belief vẫn tương đối cao
    ↓
hệ thống vẫn tin Slave
    ↓
joint xấu vẫn được đưa vào fusion / optimization
    ↓
MPJPE và PA-MPJPE giảm mạnh
```

Mục tiêu khi đọc code là **xác nhận hoặc bác bỏ giả thuyết này**.

---

# 5. Checklist: nơi cần tìm trong source code

## 5.1. Tìm nơi đọc / tạo `belief Master` và `belief Slave`

Tìm các từ khóa:

```text
belief
belief_master
belief_slave
master_belief
slave_belief
confidence
conf
score
reliability
```

Cần xác định:

- [ ] Belief được tính từ đâu?
- [ ] Belief càng lớn là càng tốt hay càng xấu?
- [ ] Có normalize belief không?
- [ ] Normalize theo joint hay theo camera?
- [ ] Có clamp không?
- [ ] Có threshold không?
- [ ] Belief được tính trước hay sau reprojection?
- [ ] Master và Slave có dùng cùng công thức không?

Ghi lại công thức:

```text
belief = ?
```

---

# 6. Kiểm tra ý nghĩa chính xác của belief

Đây là bước rất quan trọng.

Hãy xác nhận:

```text
belief lớn -> confidence cao?
```

hay:

```text
belief nhỏ -> confidence cao?
```

hoặc belief thực chất là:

```text
error
uncertainty
energy
cost
distance
```

Không nên suy luận chỉ dựa vào tên biến.

Checklist:

- [ ] Tìm comment định nghĩa belief.
- [ ] Tìm paper / công thức gốc nếu code triển khai từ paper.
- [ ] Tìm nơi belief được dùng làm weight.
- [ ] Xem weight tăng hay giảm khi belief tăng.

Ví dụ nếu code có:

```python
weight = 1.0 / belief
```

thì belief nhỏ lại có nghĩa là weight lớn.

Nếu code có:

```python
weight = belief / sum(belief)
```

thì belief lớn mới có weight lớn.

---

# 7. Kiểm tra hàm biến đổi belief thành weight

Tìm những dạng code như:

```python
weight = ...
```

hoặc:

```text
alpha
beta
confidence_weight
fusion_weight
master_weight
slave_weight
w_master
w_slave
```

Đặc biệt tìm:

```python
exp(...)
softmax(...)
sigmoid(...)
1 / (...)
normalize(...)
clip(...)
clamp(...)
```

Checklist:

- [ ] Weight Slave có phụ thuộc trực tiếp vào belief không?
- [ ] Có threshold hard-code không?
- [ ] Có min weight không?
- [ ] Có max weight không?
- [ ] Có epsilon khiến belief rất thấp bị xử lý đặc biệt không?
- [ ] Có softmax khiến một vài joint chiếm weight lớn bất thường không?
- [ ] Có normalize toàn vector khiến các joint liên quan nhau không?

---

# 8. In debug weight cho `seg_1/cam4` và `seg_2/cam4`

Đây là kiểm tra quan trọng nhất.

Thêm debug gần nơi fusion xảy ra:

```python
print("segment =", segment)
print("master =", master_cam)
print("slave =", slave_cam)

print("belief master =", belief_master)
print("belief slave  =", belief_slave)

print("weight master =", weight_master)
print("weight slave  =", weight_slave)
```

Nếu weight theo joint:

```python
for j in range(len(belief_slave)):
    print(
        j,
        "belief_master =", belief_master[j],
        "belief_slave =", belief_slave[j],
        "weight_master =", weight_master[j],
        "weight_slave =", weight_slave[j],
    )
```

Cần so sánh hai case:

```text
seg_1 / cam4 Slave
seg_2 / cam4 Slave
```

---

# 9. Kiểm tra riêng Joint 9

Joint 9 có:

```text
belief Slave = 0.48
```

và là feature quan trọng nhất trong Random Forest.

Tại joint 9 hãy log:

```python
j = 9

print("joint =", j)
print("belief_master =", belief_master[j])
print("belief_slave  =", belief_slave[j])

print("master_weight =", master_weight[j])
print("slave_weight  =", slave_weight[j])

print("master_xyz =", master_pose[j])
print("slave_xyz  =", slave_pose[j])

print("fused_xyz =", fused_pose[j])
```

Nếu có reprojection error:

```python
print("master reprojection =", master_reproj[j])
print("slave reprojection  =", slave_reproj[j])
```

Nếu có triangulation:

```python
print("triangulated =", triangulated_pose[j])
```

Mục tiêu:

> Xem joint 9 của Slave có belief rất thấp nhưng vẫn nhận weight lớn hay không.

---

# 10. Map joint index sang tên joint

Tìm skeleton definition.

Ví dụ:

```python
JOINT_NAMES = [...]
```

hoặc:

```text
joint_names
keypoint_names
skeleton
parents
kinematic_tree
```

Ghi lại:

```text
joint 2  = ?
joint 4  = ?
joint 9  = ?
joint 12 = ?
joint 13 = ?
joint 20 = ?
```

Đặc biệt:

```text
joint 9 = ?
```

Nếu joint 9 là joint gần:

- pelvis
- hip
- spine
- root
- shoulder

thì ảnh hưởng tới toàn bộ kinematic chain có thể rất lớn.

---

# 11. Kiểm tra Master và Slave có được xử lý đối xứng không

Dữ liệu cho thấy:

```text
X -> cam4
```

bị lỗi mạnh.

Nhưng:

```text
cam4 -> X
```

không lỗi.

Vì vậy hãy tìm mọi đoạn:

```python
if is_master:
    ...
else:
    ...
```

hoặc:

```python
master_...
slave_...
```

Checklist:

- [ ] Master có được ưu tiên weight mặc định không?
- [ ] Slave có bị transform sang hệ Master không?
- [ ] Có normalize belief sau khi transform không?
- [ ] Có bước xử lý chỉ áp dụng cho Slave không?
- [ ] Có threshold khác nhau cho Master / Slave không?
- [ ] Có dùng Master làm reference pose không?
- [ ] Có cập nhật Slave vào Master nhưng không theo chiều ngược lại không?

---

# 12. Kiểm tra transform giữa camera

Tìm:

```text
R
T
rotation
translation
extrinsic
world_to_cam
cam_to_world
transform
relative_pose
```

Vì lỗi chỉ xảy ra khi `cam4` làm Slave, cần kiểm tra:

```text
cam4 -> master coordinate
```

có khác với:

```text
master -> cam4 coordinate
```

hay không.

Checklist:

- [ ] Rotation có bị transpose sai chiều không?
- [ ] Translation có đúng sign không?
- [ ] Có dùng `R.T` đúng chỗ không?
- [ ] Homogeneous transform có đúng thứ tự không?
- [ ] Camera index có bị swap Master/Slave không?
- [ ] Extrinsic của cam4 ở seg_2 có khác segment khác không?

Debug:

```python
print("R =", R)
print("T =", T)
```

So sánh:

```text
seg_1 / X -> cam4
seg_2 / X -> cam4
```

---

# 13. Kiểm tra pairing dữ liệu

Xác nhận dữ liệu Master và Slave cùng frame.

Checklist:

- [ ] Frame ID giống nhau?
- [ ] Timestamp giống nhau?
- [ ] Person ID giống nhau?
- [ ] Segment index đúng?
- [ ] Camera cam4 có bị lệch frame ở seg_2 không?
- [ ] Skeleton có cùng thứ tự joint không?

Debug:

```python
print(
    segment,
    frame_id,
    master_cam,
    slave_cam,
    master_person_id,
    slave_person_id
)
```

---

# 14. Kiểm tra joint order

Một lỗi rất nguy hiểm:

```text
Master joint order != Slave joint order
```

Ví dụ:

```text
Master joint 9 = left hip
Slave joint 9  = right wrist
```

Nếu code assume cùng index thì fusion sai nghiêm trọng.

Checklist:

- [ ] Master và Slave sử dụng cùng skeleton definition.
- [ ] Không reorder keypoint ở một camera riêng.
- [ ] Không flip left/right ở cam4.
- [ ] Không mapping COCO -> custom skeleton sai.

---

# 15. Kiểm tra left/right swap

Với camera ở góc nhìn đặc biệt, pipeline 2D pose có thể nhầm:

```text
left ↔ right
```

Kiểm tra các cặp:

```text
left shoulder  ↔ right shoulder
left elbow     ↔ right elbow
left wrist     ↔ right wrist
left hip       ↔ right hip
left knee      ↔ right knee
left ankle     ↔ right ankle
```

Nếu belief vẫn không quá thấp nhưng coordinate bị swap, đây là kiểu lỗi rất phù hợp với pattern hiện tại.

---

# 16. Kiểm tra fusion formula

Tìm công thức tương tự:

```python
fused = (
    master_weight * master_value
    +
    slave_weight * slave_value
)
```

Xác nhận denominator:

```python
fused = (
    wm * master
    +
    ws * slave
) / (wm + ws)
```

Checklist:

- [ ] Có chia cho tổng weight không?
- [ ] Có trường hợp tổng weight gần 0?
- [ ] Có broadcast sai shape không?
- [ ] Weight shape có đúng `(num_joints,)` không?
- [ ] Coordinate shape có `(num_joints, 3)` không?
- [ ] Có nhân nhầm weight của joint khác không?

Ví dụ đúng:

```python
w = weight[:, None]

fused = (
    wm[:, None] * master
    +
    ws[:, None] * slave
) / (
    wm[:, None] + ws[:, None] + 1e-8
)
```

---

# 17. Kiểm tra alpha / beta

Tên file có:

```text
alpha1E_2
beta85E_2
```

có thể tương ứng:

```text
alpha = 0.01
beta  = 0.85
```

Tìm code:

```text
alpha
beta
```

Kiểm tra:

- [ ] Alpha tác động lên belief như thế nào?
- [ ] Beta có làm weight Slave quá lớn không?
- [ ] Có exponential amplification không?
- [ ] Có threshold quanh vùng belief ~1.1–1.5 không?

Ví dụ đặc biệt nguy hiểm:

```python
weight = np.exp(-beta * belief)
```

hoặc:

```python
weight = 1 / (belief + alpha)
```

vì tác động của belief lúc đó không tuyến tính.

---

# 18. Kiểm tra Huber loss

Tên experiment có:

```text
huber
```

Tìm:

```text
Huber
SmoothL1
delta
huber_delta
```

Checklist:

- [ ] Residual lớn của Slave có thực sự bị down-weight không?
- [ ] Huber threshold là bao nhiêu?
- [ ] Có dùng belief để nhân vào Huber loss không?
- [ ] Có thể xảy ra trường hợp belief khiến residual xấu được tăng weight không?

Công thức cần kiểm tra kiểu:

```python
loss = belief * huber(residual)
```

Nếu belief không mang nghĩa confidence, phép nhân này có thể ngược logic.

---

# 19. Kiểm tra kinematic loss

Tên experiment có:

```text
kinematic
```

Nếu joint 9 thuộc chuỗi xương quan trọng, sai một joint có thể kéo nhiều joint khác.

Tìm:

```text
kinematic
bone
limb
parent
child
bone_length
skeleton_loss
```

Checklist:

- [ ] Joint 9 có parent/child nào?
- [ ] Khi joint 9 sai, có kéo các joint con theo không?
- [ ] Bone length constraint có làm lỗi lan truyền không?
- [ ] Kinematic loss có weight quá lớn so với reprojection loss không?

---

# 20. Kiểm tra global optimization

Tên experiment có:

```text
global
```

Tìm:

```text
global_opt
optimize
optimizer
least_squares
bundle
global_loss
```

Kiểm tra:

- [ ] Một Slave xấu có ảnh hưởng toàn pose không?
- [ ] Có optimize root/global translation dựa vào Slave không?
- [ ] Joint 9 có tham gia tính global orientation không?
- [ ] Một joint lỗi có thể xoay toàn skeleton không?

Điểm này đặc biệt liên quan tới:

```text
PA-MPJPE ≈ -55%
```

---

# 21. So sánh trước và sau fusion

Với từng case, log:

```text
MPJPE Master trước fusion
MPJPE Slave trước fusion
MPJPE sau fusion
```

Ví dụ:

```python
print("master mpjpe =", master_mpjpe)
print("slave mpjpe  =", slave_mpjpe)
print("fused mpjpe  =", fused_mpjpe)
```

Mục tiêu xác định:

```text
Slave đã xấu sẵn
```

hay:

```text
Slave không quá xấu,
nhưng fusion làm kết quả xấu đi
```

---

# 22. Log theo từng joint

Nên tạo bảng:

```text
joint
belief_master
belief_slave
weight_master
weight_slave
error_master
error_slave
error_fused
```

Ví dụ:

```python
for j in range(num_joints):
    print(
        f"joint={j:02d}",
        f"bm={belief_master[j]:.4f}",
        f"bs={belief_slave[j]:.4f}",
        f"wm={weight_master[j]:.4f}",
        f"ws={weight_slave[j]:.4f}",
        f"em={error_master[j]:.4f}",
        f"es={error_slave[j]:.4f}",
        f"ef={error_fused[j]:.4f}",
    )
```

---

# 23. Debug CSV nên xuất thêm

Tạo một file:

```text
debug_fusion.csv
```

có các cột:

```text
segment
frame
master_cam
slave_cam
joint

belief_master
belief_slave

weight_master
weight_slave

master_x
master_y
master_z

slave_x
slave_y
slave_z

fused_x
fused_y
fused_z

error_master
error_slave
error_fused
```

Sau đó có thể phân tích chính xác joint nào phá kết quả.

---

# 24. Test ép weight Slave = 0

Đây là test rất hữu ích.

Tạm thời với:

```text
seg_2 + cam4 Slave
```

thử:

```python
slave_weight[:] = 0
```

Nếu kết quả quay về gần baseline:

```text
=> chính Slave fusion gây lỗi
```

Nếu vẫn lỗi:

```text
=> lỗi xảy ra trước bước fusion
```

---

# 25. Test chỉ bỏ Joint 9

Thử:

```python
slave_weight[9] = 0
```

Nếu:

```text
Δ_MPJPE cải thiện mạnh
hoặc
Δ_PA-MPJPE cải thiện mạnh
```

thì joint 9 có vai trò trực tiếp.

Sau đó thử lần lượt:

```text
2
4
9
12
13
20
```

---

# 26. Ablation nên chạy

## Test A

```text
Disable toàn bộ cam4 Slave
```

Mục tiêu:

```text
xác nhận lỗi đến từ cam4 Slave
```

---

## Test B

```text
Disable riêng joint 9 của cam4 Slave
```

---

## Test C

```text
Disable nhóm:
2, 4, 9, 12, 13, 20
```

---

## Test D

Clamp:

```python
if belief_slave[j] < 1.2:
    slave_weight[j] = 0
```

---

## Test E

Giảm weight mềm:

```python
if belief_slave[j] < 1.2:
    slave_weight[j] *= 0.1
```

---

## Test F

So sánh:

```text
seg_1/cam4 Slave
seg_2/cam4 Slave
```

với cùng debug output.

---

# 27. Điều cần đặc biệt chú ý khi đọc code

Không nên chỉ tìm:

```text
if belief < 1.2
```

vì ngưỡng `1.2`, `1.09`, `1.5` là ngưỡng **Decision Tree học từ dataset**.

Nó không có nghĩa code thật sự chứa các threshold này.

Decision Tree chỉ nói rằng:

```text
khu vực belief này đang phân biệt failure rất tốt
```

Mục tiêu là tìm cơ chế thật trong code.

---

# 28. Cảnh báo về kết luận thống kê

Hiện tại 5 hàng bad đều dùng cùng:

```text
belief Slave của seg_2/cam4
```

nên thực tế chúng gần như là:

```text
1 unique bad belief pattern
```

Do đó chưa nên kết luận:

```text
mean > 1.09
AND
>= 6 joint < 1.2
```

là quy luật tổng quát.

Nên gọi nó là:

```text
failure signature hiện tại
```

---

# 29. Thứ tự kiểm tra đề xuất

Nên kiểm tra theo đúng thứ tự này:

```text
1. Xác nhận ý nghĩa của belief
        ↓
2. Tìm công thức belief -> weight
        ↓
3. Log weight của seg_1/cam4 và seg_2/cam4
        ↓
4. Kiểm tra joint 9
        ↓
5. Kiểm tra Master/Slave có bất đối xứng không
        ↓
6. Kiểm tra camera transform
        ↓
7. Kiểm tra frame/person pairing
        ↓
8. Kiểm tra joint order / left-right
        ↓
9. Disable joint 9
        ↓
10. Disable toàn cam4 Slave
        ↓
11. Kiểm tra Huber / kinematic / global optimization
```

---

# 30. Kết quả mong đợi

Sau khi debug, cố gắng trả lời 5 câu hỏi sau:

### Câu 1

```text
Belief thực chất có nghĩa là gì?
```

### Câu 2

```text
Belief được chuyển thành fusion weight bằng công thức nào?
```

### Câu 3

```text
Tại sao seg_1/cam4 belief thấp hơn
nhưng không gây failure?
```

### Câu 4

```text
Tại sao cam4 làm Slave gây lỗi,
nhưng cam4 làm Master lại không?
```

### Câu 5

```text
Joint 9 có trực tiếp gây failure hay chỉ tương quan?
```

---

# 31. Điều kiện để coi là đã tìm được nguyên nhân

Chỉ nên coi là đã tìm được nguyên nhân khi có một can thiệp code kiểu:

```text
disable / sửa weight / sửa transform / sửa joint mapping
```

và sau đó:

```text
seg_2 + cam4 Slave
```

từ:

```text
Δ_MPJPE ≈ -23%
Δ_PA-MPJPE ≈ -56%
```

trở về mức bình thường hoặc dương.

Khi đó ta có bằng chứng nhân quả mạnh hơn Decision Tree / Random Forest.

---

# 32. Kết luận hiện tại

Dấu hiệu mạnh nhất hiện nay là:

```text
seg_2
+
cam4 ở vai trò Slave
+
belief Slave không thấp hoàn toàn
nhưng chứa một số joint rất thấp
+
joint 9 = 0.48
```

và pattern này gắn với:

```text
MPJPE giảm khoảng 23%
PA-MPJPE giảm khoảng 56%
```

Ưu tiên kiểm tra:

```text
belief -> slave weight
```

và:

```text
joint 9
```

trước tiên.
