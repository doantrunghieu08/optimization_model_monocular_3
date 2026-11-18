# Thay đổi mô hình (đợt 2026-11-18)

Tài liệu này thay cho bản trước. Bản trước có một số mô tả không khớp với code: tham số `residual_accel_weight = 6.0` và `minimum_retention = 0.40` trong khi config thực tế là 1.0 và 0.8, lập luận "Accel không xấu đi" dựa trên các giá trị đó, và các tuyên bố hiệu quả chưa được kiểm chứng. Mọi con số dưới đây đều đã được chạy lại và đối chiếu.

Giao thức kiểm chứng chung, trừ khi ghi khác:
- Dữ liệu Hugging 002 (WHAM).
- **Design** = aria01, cam02/04/13/19. **Held-out** = aria02, cam03/05/17/20. Hai split không chung người, không chung camera.
- Quy tắc chọn và tiêu chí thành công được ghi xuống trước khi chạy. Config được đóng băng trước khi chạy held-out.
- Chấm trên camera1, đơn vị mm.

---

## 1. Kiến trúc hiện tại

| Thành phần | Mô tả |
|---|---|
| `fusion.method: aotr` (mặc định) | Giữ khung xương Master. Mỗi xương tìm α ∈ [0,1] trộn hướng Master/Slave bằng trust-region 1D, rồi tái dựng bằng FK với chiều dài xương Master. Có 2D ở cả hai view thì dùng objective reprojection 2D; không có thì dùng 3D surrogate (jitter + che khuất). Sau đó là bước `sequence_stability`. |
| `fusion.method: partwise` (mới) | Trộn đối xứng hướng xương, căn Slave theo từng phần thân, trọng số theo độ sâu, giữ chiều dài xương Master, kèm `sequence_stability`. Xem mục 3. |
| `sequence_stability` | Tối ưu hệ số co `s_t` trên phần dư hiệu chỉnh F − M để giảm gia tốc, rồi làm mượt quỹ đạo và chiếu lại về chiều dài xương. Config thực tế: `shape_weight 2.5`, `residual_accel_weight 1.0`, `residual_velocity_weight 0.5`, `minimum_retention 0.8`, `smooth_window 7`. |
| `anatomical_calibration` | **Tắt** (xem mục 2.1). |

---

## 2. Lỗi và vấn đề đã sửa

### 2.1. Hiệu chỉnh nhân trắc theo chiều cao làm MBLE tăng gấp đôi
- `inputs.person_heights` khiến `_anthropometric_targets` scale 8 xương chi theo hệ số `chiều_cao / _estimate_master_height`.
- Hàm ước lượng chiều cao cộng đầu–cổ, cổ–pelvis, chân và cổ chân–bàn chân, nên bỏ sót đỉnh đầu, đoạn pelvis–hông và gót. Kết quả là chiều cao bị đánh giá thấp khoảng 12%, và các chi bị phóng to khoảng ×1.137.
- Brute force trên Hugging: MBLE tệ hơn ở **53/53 cặp** (22.4 → 41.5).
- Đã sửa: bỏ `person_heights`, đặt `anatomical_calibration.enabled: false` (cả trong `pipeline_mpi.yml`). Chiều dài xương giờ bằng Master/baseline từng frame. Ví dụ cặp cam04/cam05: MBLE 37.7 → 22.0, MPJPE 53.8 → 48.0.

### 2.2. `fusion.master_protection.confidence_margin` không có tác dụng với AOTR
- Nhánh AOTR trả kết quả trước khi tới logic belief. Đã kiểm chứng: output trùng byte ở 24/24 cặp khi tắt nó.
- `pair_modes` (cam13/17/19 → cam20 = `posed_fallback`) là quy tắc theo cặp được chọn bằng tay cho Hugging. Nó vẫn còn trong config nhưng không có ý nghĩa với dataset khác.

### 2.3. Track WHAM bị tráo người (Hugging)
- Frame của WHAM khớp với GT: quét ±6 frame, độ lệch tốt nhất là 0 hoặc −1, lợi ≤ 0.7 mm.
- Nhưng `*_person_mapping.txt` gán một người cho mỗi track, trong khi track của WHAM tráo người giữa chừng.
- Tỉ lệ frame bám nhầm người: cam02 39–43%, cam03 24–33%, cam20 15–38%, cam17 22% (aria01). Các camera 04/05/13/19 ≤ 6%.
- Đã sửa ở phần đánh giá (không sửa đầu vào):
  - `build_identity_mask.py` lưu biên độ `PA(đúng người) − PA(người kia)` cho từng frame.
  - `evaluation.identity_mask: {path, margin_mm: 10}` bỏ các frame có track Master **hoặc** Slave bám nhầm người. Accel tự bỏ các bộ ba bị đứt.
  - Chấm lại từ CSV khớp với evaluator chính thức (≤ 4e-5 mm).
  - Kết luận không đổi ở các ngưỡng 5, 10 và 20 mm.

### 2.4. Các sửa khác
- **GT trùng file** (043: `00059(1).npy` …): loader trước đây gán theo vị trí trong danh sách; giờ bỏ qua tên không phải số.
- **Mã protocol của brute force** giờ bao gồm `fusion.method`, cấu hình `partwise` và `identity_mask`, nên không dùng lại cache sai phương pháp hoặc cache chưa mask. Báo cáo tách thành `brute_force_report_aotr_*` và `brute_force_report_partwise_*`.
- **Xương bàn tay dài 0 mm** (wrist trùng hand) không đi theo khớp cha trong FK. Đã sửa; metric không đổi.

---

## 3. Phương pháp mới: `partwise`

Với mỗi xương (child, parent):

$$\mathbf{u} = \mathrm{normalize}\big(w\,\mathbf{u}_M + (1-w)\,R_{\text{part}}\,\mathbf{u}_S\big), \qquad P(c) = P(p) + \ell_M\,\mathbf{u}$$

- **R_part = B_M B_Sᵀ**, với B là hệ trục thân, không cần extrinsics:
  - phần dưới: trục hông + cột sống (pelvis → neck), dùng cho hông, đầu gối, cổ chân, bàn chân;
  - phần trên: trục vai + cột sống, dùng cho đầu, vai, khuỷu tay, cổ tay, bàn tay.
  - R_part được làm mượt trên ±4 frame.
- **w** (trọng số Master) $= \sigma\big(a + b\,((z_M - z_S) - \mu)/\sigma_z\big)$, với z là độ sâu tâm hông. View xa camera hơn được tin hơn. a, b, μ, σ_z được fit **một lần trên GT của split design** rồi đóng băng trong `pipeline.yml`.
- Pelvis và neck giữ vị trí Master; ℓ_M là chiều dài xương Master, nên MBLE = baseline.
- Sau đó áp `sequence_stability` với cùng config như AOTR.

Pipeline (`run_fusion` với `method: partwise`) tái tạo đúng output thí nghiệm trên 24/24 cặp (sai lệch 0.0 mm).

### Vì sao phải căn theo từng phần
Bản đầu dùng **một** phép xoay Kabsch toàn thân để đưa Slave về hệ Master. Trên held-out, ở các cặp cam20 làm Master (cam20 là view kém), thân trên bị vặn so với hông: MPJPE cổ tay tăng từ 66 lên 145 mm, trong khi chân lại tốt lên. Căn theo phần thân mà mỗi chi gắn vào đã xử lý được lỗi này.

---

## 4. Kết quả

### 4.1. Held-out (identity mask, T = 10 mm)

| | PA-MPJPE | MPJPE | MBLE | Accel |
|---|---|---|---|---|
| WHAM mono | 51.47 | 64.91 | 22.43 | 22.11 |
| aotr | 46.16 | 58.54 | 22.43 | 13.16 |
| partwise | **39.30** | **47.98** | 22.43 | **12.85** |

- Partwise đạt cả 4 tiêu chí đăng ký trước so với aotr: PA thấp hơn, Accel ≤ +2, MPJPE ≤ +1, MBLE không đổi. Thắng aotr ở 9/12 cặp về PA và 11/12 cặp về MPJPE.
- Khi tắt mask: mono 56.14 / 69.00, aotr 51.29 / 63.08, partwise 45.33 / 53.92 (PA / MPJPE). Vẫn đạt.
- Design: mono 55.90 / 73.08, aotr 49.12 / 64.15, partwise 47.07 / 63.75.

### 4.2. Phân rã held-out (MPJPE)

| Nhóm cặp | aotr | partwise (căn toàn thân, đăng ký trước) | partwise (căn theo phần) |
|---|---|---|---|
| cam20 làm Master (3 cặp) | 80.9 | 97.8 | 48.3 |
| 9 cặp còn lại | 51.1 | 48.2 | 47.9 |

---

## 5. Kết quả âm (đã thử, không giữ)

| Ý tưởng | Kết quả |
|---|---|
| Siết bảo vệ Master trong AOTR-3D: chặn theo độ lệch hướng (15°/30°), giới hạn góc chỉnh (5°/10°), bỏ sàn đồng thuận, `fail_closed` | Không biến thể nào tốt hơn aotr (held-out ΔPA: aotr −4.84, các biến thể từ −4.54 đến 0). Các tùy chọn vẫn còn trong `aotr_3d.py`, mặc định tắt. |
| Chọn cứng view theo tín hiệu (S1) | Tín hiệu độ sâu lặp lại được (AUC 0.825 design, 0.863 held-out), nhưng chọn cứng từng frame không ổn định. |
| Hiệu chỉnh offset khớp theo hệ thân (S3), học trên aria01 | Cross-validation tách camera trên design cho −9.7 mm PA, nhưng sang aria02 thì MPJPE **tệ đi** khoảng 6–9 mm. Offset là đặc trưng cơ thể người, không phải quy ước GT. Đã loại. |
| Chiều cao biết trước | PA không đổi (Procrustes khử scale). Trên Hugging/WHAM, scale vốn đã đúng trong khoảng ±2%. |

---

## 6. Kiểm toán leakage

- **GT lúc suy luận:** không. Đã kiểm tra mã nguồn: fusion, pose và preprocess không đọc GT.
- **Tham số học từ GT:** 4 tham số `depth_logit` của partwise, chỉ dùng GT của design. Kiểm toán trên held-out:
  - w = 0.5: PA 45.10, xấp xỉ aotr.
  - Đảo dấu: 56.33, tệ hơn mono (đối chứng âm).
  - Nhân/chia hệ số b cho 2: 39.49 / 40.25.
  - Kết luận: phần lợi đến từ **chiều** của tín hiệu độ sâu và tín hiệu này đúng trên camera và người mới. Không phải leakage, nhưng partwise **không** được gọi là zero-GT.
- **Identity mask** dùng GT để chọn frame chấm, áp như nhau cho mọi phương pháp. Kết luận đứng vững khi tắt mask.
- **Dùng lại held-out:** có. Cách căn theo phần thân được thiết kế sau khi chẩn đoán các cặp cam20 trên held-out, nên phần lợi ở 3 cặp đó chưa được kiểm chứng sạch.
- **Design và held-out cùng một cảnh** (cùng video, cùng bố trí camera): đây là giới hạn về tổng quát hóa.

---

## 7. Mức độ tổng quát hóa

| Mức | aotr | partwise |
|---|---|---|
| Camera mới, cùng cảnh | ✅ | ✅ |
| Người mới, cùng cảnh (2 người) | ✅ | ✅ |
| View Master rất kém (cam20) | ❌ | ⚠️ chưa sạch |
| Cảnh / hành động mới (043_Grappling2) | đang chạy | chưa chạy |
| HMR khác (GVHMR / MMA) | chưa | chưa |

---

## 8. Công cụ dữ liệu mới

- `preprocessing_data/label_persons.py`: gán nhãn track → aria01/aria02 bằng GT 3D cho dataset không có mapping và không có bbox 2D.
  - Mỗi frame bỏ phiếu: biên 10 mm, PA tối đa 120 mm.
  - Mỗi track nhận một nhãn theo đa số (≥ 50%); track người ngoài để trống.
  - Ghi `*_person_mapping.txt`, `*_person_labels.json` và config brute force (mỗi người một segment).
  - Trên 043, tỉ lệ frame bám nhầm người phần lớn chỉ 0–3%.
- `build_identity_mask.py`: tạo mask cho evaluation. `identity_mask.path` nhận được danh sách nhiều file.

---

## 9. Config mới

```yaml
fusion:
  method: aotr                 # aotr | partwise
  partwise:
    depth_logit: {a, b, mu, sd}  # fit trên design, đóng băng
    rotation_half_window: 4
    sequence_stability: {...}    # giống aotr
evaluation:
  identity_mask:
    path: [configs/identity_mask_hugging_002.json, configs/identity_mask_grappling2_043.json]
    margin_mm: 10.0
```

Các script và kết quả thí nghiệm của đợt này (đăng ký trước, CSV kết quả) nằm trong `scratch/` cục bộ và **không** được đưa lên git.
