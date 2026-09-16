# Các cải tiến của phương pháp đề xuất so với mô hình gốc

## 1. Phạm vi so sánh

Mô hình gốc đã có một pipeline hoàn chỉnh gồm đồng bộ dữ liệu, ước lượng pose 3D cho từng camera, căn chỉnh hai hệ tọa độ bằng RANSAC–Umeyama, tối ưu các khớp bằng SLSQP, hậu xử lý Learnable SMPLify và đánh giá bằng MPJPE, PA-MPJPE, PCK. Vì vậy, đóng góp của phiên bản cải tiến không nằm ở việc thay thế backbone WHAM/SMPL hoặc mạng `NetBody25`, mà tập trung vào ba vấn đề của bước fusion: nhận biết che khuất, mô hình hóa độ tin cậy của khớp và kiểm soát sai số khi hiệu chỉnh liên camera.

Các thay đổi về runner, cấu hình và chỉ số đánh giá được bổ sung để việc kiểm chứng các cải tiến trên có thể lặp lại và không trộn lẫn kết quả giữa các thí nghiệm.

## 2. Phát hiện tự che khuất bằng ray-casting trên mesh SMPL

Ở bản gốc, che khuất được ước lượng bằng cách chiếu các đỉnh phần thân lên ảnh, tạo bao lồi 2D và so sánh độ sâu của khớp với một nhóm đỉnh lân cận. Cách làm này phụ thuộc vào camera intrinsics, file phân vùng SMPL bên ngoài và trường `verts_cam` trong dữ liệu WHAM. Do đầu vào thực tế không luôn chứa `verts_cam`, chức năng này mặc định bị tắt; bao lồi 2D cũng không mô tả chính xác bề mặt và khoảng trống của cơ thể.

Phiên bản cải tiến sinh trực tiếp mesh SMPL trong hệ tọa độ của từng camera ở bước Pose và lưu đồng bộ với các frame keypoint. Các mặt thuộc vùng torso được xác định từ trọng số skinning (`lbs_weights`) của chính mô hình SMPL. Với mỗi khớp cần kiểm tra, hệ thống dựng tia từ tâm camera đến khớp và tính giao với các tam giác torso bằng phép thử ray–triangle. Khớp được xem là bị che nếu tồn tại giao điểm nằm trước khớp một khoảng lớn hơn ngưỡng an toàn `tau`.

Cách tiếp cận này mang lại ba lợi ích:

- kiểm tra che khuất trực tiếp trong không gian 3D thay vì xấp xỉ bằng bao lồi 2D;
- sử dụng đúng mesh và đúng frame đã tạo ra pose của từng camera;
- không còn phụ thuộc vào intrinsics hoặc file phân vùng ngoài trong đường chạy fusion hiện tại.

Kết quả visibility được đưa vào belief score; khớp bị che khuất nhận độ tin cậy bằng 0 trước khi kết hợp với thông tin từ camera còn lại.

## 3. Belief score thích nghi theo khoảng cách, che khuất và cấu trúc xương

Bản gốc chỉ sử dụng một hàm belief cục bộ cố định dựa trên khoảng cách từ camera đến khớp:

\[
P_j = \frac{C_j}{1 + \alpha L_j^2},
\]

trong đó \(C_j\) là visibility và \(L_j\) là khoảng cách từ camera đến khớp \(j\). Sau đó, belief của khớp luôn được hòa với belief của các khớp kề trên skeleton bằng trung bình điều hòa.

Phiên bản cải tiến giữ công thức trên như một baseline (`naive_distance_belief`) và bổ sung `optical_aware_belief`. Hàm mới chia không gian quan sát thành ba vùng:

- vùng quá gần camera: belief giảm theo hàm Gaussian;
- vùng quan sát ổn định: belief giảm từ từ theo khoảng cách;
- vùng quá xa camera: belief tiếp tục giảm theo Gaussian để phản ánh suy giảm chất lượng quan sát.

Ngoài belief cục bộ, hệ thống cho phép bật hoặc tắt lan truyền belief theo cấu trúc xương. Khi `global=true`, belief của khớp được kết hợp với trung bình belief của các khớp lân cận:

\[
H_j = \frac{2P_jB_j}{P_j + B_j + \varepsilon},
\qquad
B_j = \beta\,\mathrm{mean}_{k\in\mathcal{N}(j)}(P_k).
\]

Các láng giềng có belief bằng 0 không được dùng để làm suy giảm một khớp đang nhìn thấy. Belief hình học sau đó tiếp tục được hòa với confidence 2D nếu dữ liệu này tồn tại. Nhờ đó, quyết định chọn khớp từ camera nào không chỉ dựa trên khoảng cách mà còn xét che khuất, chất lượng phát hiện 2D và tính nhất quán của chuỗi xương.

## 4. Hiệu chỉnh liên camera và tối ưu có cơ chế bảo vệ

Khung RANSAC–Umeyama và SLSQP của bản gốc được giữ lại, nhưng phiên bản mới bổ sung các ràng buộc an toàn:

- phép biến đổi similarity chỉ được ước lượng khi có ít nhất ba anchor không thẳng hàng và mọi tọa độ đều hữu hạn;
- khi confidence correction được bật, một khớp chỉ được trộn với dự đoán từ camera còn lại khi độ dịch chuyển không vượt quá ngưỡng RANSAC; tỷ lệ trộn lấy trực tiếp từ belief tương đối của hai camera;
- hiệu chỉnh orientation có thể bật/tắt độc lập; các mismatch mới sinh ra sau hiệu chỉnh được khôi phục về pose trước hiệu chỉnh;
- bộ tối ưu SLSQP có thể chạy với hoặc không có ràng buộc chiều dài xương;
- hàm mất mát có thể chọn Huber để giảm ảnh hưởng của outlier hoặc MSE để làm đối chứng;
- nếu một frame lỗi dữ liệu, hệ thống ghi nhận fallback; toàn bộ kết quả chỉ được xuất khi tỷ lệ fallback không vượt quá `max_fallback_ratio`.

Các cơ chế này không mặc định khẳng định rằng mọi hiệu chỉnh đều tốt hơn. Confidence correction, orientation correction và optimizer hiện đều tắt mặc định vì benchmark held-out chưa vượt raw pose; từng thành phần chỉ được bật trong ablation cho đến khi chứng minh được cải thiện thay vì chỉ vượt một baseline yếu hơn.

## 5. Thiết kế ablation có thể tái lập

Các tham số thí nghiệm được mã hóa trực tiếp trong tên notebook và được nạp vào `pipeline.yml` trước khi chạy. Sáu trường bắt buộc gồm `alpha`, `beta`, phương pháp belief cục bộ, phạm vi belief (`global/local`), ràng buộc động học và loại loss. Tên thiếu hoặc sai thành phần sẽ gây lỗi thay vì âm thầm dùng cấu hình mặc định.

Bốn cấu hình hiện tại kiểm tra lần lượt:

| Cấu hình | Belief cục bộ | Belief toàn cục | Ràng buộc động học | Loss |
| --- | --- | --- | --- | --- |
| `naive_global_kinematic_huber` | Khoảng cách thuần | Có | Có | Huber |
| `optical_global_kinematic_mse` | Optical-aware | Có | Có | MSE |
| `optical_global_unconstrained_huber` | Optical-aware | Có | Không | Huber |
| `optical_local_kinematic_huber` | Optical-aware | Không | Có | Huber |

Mỗi notebook khai báo `NOTEBOOK_NAME` riêng và mặc định ghi vào một Google Sheet mang tên tương ứng. Runner lưu toàn bộ cấu hình vào metadata, từ chối nối tiếp một worksheet chưa hoàn tất nếu cấu hình không khớp, và có thể tiếp tục từ kết quả dở dang khi cấu hình giống nhau. Điều này khắc phục nguy cơ nhiều thí nghiệm khác nhau nhưng vô tình dùng chung giá trị mặc định hoặc trộn chung kết quả.

## 6. Đánh giá đầy đủ hơn

Bên cạnh MPJPE và PA-MPJPE, phiên bản mới:

- sửa PCK thành tỷ lệ phần trăm khớp có sai số nhỏ hơn ngưỡng `pck_threshold_mm`, thay vì trả về khoảng cách trung bình;
- bổ sung MBLE để đo sai số chiều dài từng xương;
- bổ sung acceleration error theo đơn vị `mm/frame²` để đánh giá độ ổn định theo thời gian;
- xuất cả giá trị tổng hợp và sai số chi tiết theo frame, khớp hoặc xương;
- kiểm tra frame, nguồn PKL và metadata cấu hình trước khi so sánh với ground truth;
- ghi kết quả định kỳ lên Google Sheets, hỗ trợ tiếp tục thí nghiệm và đánh dấu `END` khi hoàn tất.

## 7. Tóm tắt khác biệt chính

| Thành phần | Mô hình gốc | Phiên bản cải tiến |
| --- | --- | --- |
| Che khuất | Bao lồi 2D và xấp xỉ độ sâu; mặc định tắt | Ray-casting trên tam giác torso của mesh SMPL theo từng frame |
| Nguồn mesh | Phụ thuộc `verts_cam` trong WHAM | Mesh camera-space được sinh và đồng bộ tại bước Pose |
| Belief cục bộ | Một hàm suy giảm theo khoảng cách | Có đối chứng giữa naive và optical-aware |
| Belief toàn cục | Luôn lan truyền qua skeleton | Có thể bật/tắt để ablation |
| Hiệu chỉnh | Thay thế trực tiếp theo confidence | Trộn theo belief, giới hạn dịch chuyển, có diagnostic và tắt mặc định sau benchmark regression |
| Tối ưu | Huber và ràng buộc động học cố định | Bật/tắt optimizer, Huber/MSE, có/không ràng buộc động học |
| Xử lý lỗi | Fallback từng frame có thể che giấu lỗi | Theo dõi tỷ lệ fallback và chỉ xuất khi đạt ngưỡng |
| Đánh giá | MPJPE, PA-MPJPE và PCK chưa đúng ngữ nghĩa | PCK đúng ngưỡng, bổ sung MBLE và acceleration error |
| Thí nghiệm | Tham số có thể rơi về mặc định | Cấu hình bắt buộc từ tên notebook, metadata và Sheet tách biệt |

Các cải tiến trên làm cho fusion phản ánh rõ hơn điều kiện quan sát của từng camera và giúp quá trình đánh giá có thể kiểm chứng. Mức cải thiện định lượng về MPJPE, PA-MPJPE, MBLE hoặc acceleration error phải được kết luận từ kết quả của cùng tập dữ liệu và cùng cặp camera; tài liệu này không suy diễn mức tăng độ chính xác khi chưa có số liệu đối chứng hoàn chỉnh.
