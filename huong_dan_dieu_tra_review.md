# Hướng dẫn điều tra nguyên nhân các vấn đề trong review

## 1. Mục tiêu

Tài liệu này giúp kiểm tra **nguyên nhân kỹ thuật và nguyên nhân thực
nghiệm** đằng sau từng phê bình của reviewer đối với bài:

> *Belief-Driven Selective Cross-View Error Correction for 3D Human Pose
> Estimation Using Uncalibrated Dual Cameras*

Không nên bắt đầu bằng việc sửa câu chữ trong bài. Trước hết cần tái lập
kết quả, truy ngược từng số liệu về code và dữ liệu, sau đó xác định vấn
đề thuộc nhóm nào:

1.  **Lỗi triển khai**: code không thực hiện đúng phương pháp được mô
    tả.
2.  **Lỗi thiết kế thuật toán**: hàm mục tiêu hoặc quy tắc hiệu chỉnh
    khiến kết quả tệ hơn.
3.  **Lỗi đánh giá**: phép đo, cách chia tập, lọc mẫu hoặc tổng hợp số
    liệu không phù hợp.
4.  **Lỗi thiết kế thí nghiệm**: dữ liệu quá ít, tham số được chọn trên
    tập đánh giá, thiếu đối chứng.
5.  **Lỗi diễn giải**: tuyên bố trong abstract không được bảng kết quả
    hỗ trợ.

Không mặc định rằng reviewer đúng hoàn toàn hoặc code chắc chắn sai. Mỗi
kết luận phải có bằng chứng tái lập được.

------------------------------------------------------------------------

## 2. Chuẩn bị trước khi điều tra

### 2.1. Đóng băng phiên bản hiện tại

-   Tạo một branch riêng, ví dụ `investigate-review`.
-   Ghi lại commit hash, phiên bản Python/PyTorch, thư viện, cấu hình và
    lệnh chạy.
-   Không sửa code gốc trước khi có baseline tái lập.
-   Lưu log, prediction, metric theo từng frame/joint và cấu hình của
    mỗi lần chạy.

Ví dụ:

``` bash
git status
git rev-parse HEAD
git branch investigate-review
```

### 2.2. Lập hồ sơ cho mỗi lần chạy

Mỗi run cần có tối thiểu:

-   commit hash và config;
-   dataset / subject / sequence / frame IDs;
-   camera pair IDs;
-   seed ngẫu nhiên;
-   14 tham số được đặt empirically (tên, giá trị, ý nghĩa);
-   metric toàn bộ joints, joints bị che khuất và từng frame;
-   kết quả từng thành phần ablation;
-   thời gian chạy và lỗi phát sinh.

Không chỉ lưu bảng tổng hợp cuối cùng. Nếu không còn prediction trung
gian, hãy bổ sung bước xuất prediction để có thể tính lại metric mà
không cần chạy lại toàn bộ pipeline.

### 2.3. Tìm các vị trí trong repository

Dùng tìm kiếm toàn cục để tìm các tên liên quan:

``` bash
rg -n -i "belief|confidence|visibility|occlusion|acceleration|smooth|bone|conflict|MPJPE|PA-MPJPE|threshold|parameter" .
```

Nếu không có `rg`, dùng chức năng Find in Files của IDE. Ghi lại file và
hàm chịu trách nhiệm cho từng khâu.

------------------------------------------------------------------------

## 3. Kiểm tra tuyên bố về dynamic stability và acceleration error

### Reviewer đang nói gì?

Abstract tuyên bố phương pháp có "superior dynamic stability", nhưng
Table 1 cho thấy acceleration error của phương pháp đề xuất cao nhất
trong cả hai tập, thậm chí tệ hơn input chưa hiệu chỉnh.

### Nguyên nhân có thể có

-   Từ "dynamic stability" được dùng mà không có metric phù hợp để chứng
    minh.
-   Hàm smoothness không đúng với đại lượng acceleration được đánh giá.
-   Tối ưu theo từng frame hoặc từng cửa sổ gây ra thay đổi không nhất
    quán giữa các frame.
-   Quy trình sửa joint làm giảm sai số vị trí nhưng tạo ra chuyển động
    giật.
-   Nhầm trục thời gian, FPS, đơn vị (m/mm), chuẩn hóa hoặc thứ tự
    frame.
-   Lỗi triển khai trong phép tính đạo hàm bậc hai.
-   Bảng kết quả hoặc nhãn cột bị lệch; cần kiểm tra trước khi kết luận.

### Cách điều tra

1.  Tìm code tính acceleration error trong cả pipeline đánh giá và code
    tối ưu.

2.  Xác nhận công thức đang dùng. Với tọa độ khớp (p_t), sai phân bậc
    hai thường có dạng:

    \[ a_t = p\_{t+1} - 2p_t + p\_{t-1} \]

    Nếu quy đổi thành gia tốc vật lý, cần tính đến khoảng thời gian giữa
    frame. Nếu metric dùng sai phân bậc hai không chia cho
    (`\Delta `{=tex}t\^2), phải ghi rõ quy ước.

3.  Xác nhận mọi phương pháp được đánh giá trên cùng frame IDs, FPS, đơn
    vị, mask và phép chuẩn hóa.

4.  Tính acceleration error riêng cho:

    -   input gốc;
    -   output đề xuất;
    -   từng baseline;
    -   toàn bộ joints và joints bị che khuất.

5.  Vẽ quỹ đạo một vài joint theo thời gian trước và sau hiệu chỉnh. Đặc
    biệt chọn các frame có belief thấp hoặc chuyển trạng thái
    visibility.

6.  Tính thêm jerk hoặc các thống kê dao động chỉ khi có lý do; không
    dùng metric mới để né metric đã báo cáo.

### Dấu hiệu cần tìm

-   Acceleration error tăng mạnh đúng tại thời điểm hệ thống đổi nguồn
    từ camera chính sang camera phụ.
-   Sai số tập trung ở các khớp có belief gần ngưỡng quyết định.
-   Kết quả thay đổi khi sửa FPS/đơn vị hoặc khi chạy metric độc lập.
-   Smoothness loss giảm nhưng acceleration error tăng: có thể loss
    không tương ứng với metric, trọng số sai hoặc các term xung đột.

### Kết quả cần lưu

Bảng metric được tính lại từ prediction đã lưu, công thức metric, đơn
vị, script tính metric và đồ thị quỹ đạo joint. Nếu kết quả vẫn xấu, cần
xem đây là hạn chế thực chất của phương pháp, không chỉ là lỗi trình
bày.

------------------------------------------------------------------------

## 4. Vì sao simple averaging tốt hơn phương pháp đề xuất?

### Reviewer đang nói gì?

Ở Set M, averaging đơn giản tốt hơn về overall error và aligned error;
cả hai baseline đơn giản đều tốt hơn trên occluded joints. Các cải thiện
ở nơi khác dưới 1 mm.

### Nguyên nhân có thể có

-   Belief score không dự báo tốt độ tin cậy thực tế của từng camera.
-   Điểm belief được thiết kế theo visibility/distance nhưng bỏ qua
    uncertainty của pose estimator.
-   Thay joint từ camera phụ làm mất thông tin hữu ích từ camera chính.
-   Hai pose chưa được căn chỉnh trong cùng hệ tọa độ hoặc cùng quy ước
    khớp trước khi averaging.
-   Quy tắc thay thế quá cứng; blending mềm có thể phù hợp hơn.
-   Phương pháp có quá nhiều tham số so với lượng dữ liệu.
-   Mức cải thiện nhỏ hơn độ dao động tự nhiên giữa các lần chạy hoặc
    sequence.

### Cách điều tra

1.  Kiểm tra hai đầu vào có cùng skeleton topology, joint order, đơn vị,
    hệ tọa độ, handedness và quy ước trục hay không.
2.  Xác minh phép averaging được triển khai chính xác và không được lợi
    từ một bước xử lý khác chỉ áp dụng cho baseline.
3.  Với từng joint, lưu:
    -   lỗi của camera chính;
    -   lỗi của camera phụ;
    -   lỗi sau averaging;
    -   lỗi sau phương pháp đề xuất;
    -   belief score và visibility.
4.  Phân nhóm joint theo các khoảng belief và visibility. Kiểm tra
    belief có tương quan với lỗi 3D thực tế không.
5.  So sánh quy tắc hard replacement với blending mềm, nhưng chỉ xem đây
    là thí nghiệm chẩn đoán trước khi quyết định thay đổi phương pháp.
6.  Báo cáo chênh lệch tuyệt đối và tương đối, độ lệch chuẩn hoặc khoảng
    tin cậy trên các đơn vị đánh giá độc lập.

### Câu hỏi chẩn đoán quan trọng

-   Khi camera chính sai nhiều, belief có thực sự thấp không?
-   Khi belief thấp, camera phụ có thường tốt hơn camera chính không?
-   Sau khi thay joint, lỗi có giảm trên chính joint đó hay chỉ giảm ở
    metric tổng thể do cách chuẩn hóa?
-   Averaging có được so sánh trên đúng cùng đầu vào và cùng mask không?

Nếu không chứng minh được belief chọn đúng lúc và đúng joint để hiệu
chỉnh, luận điểm "belief-driven selective correction" chưa được hỗ trợ.

------------------------------------------------------------------------

## 5. Điều tra ablation: smoothness, bone-length và conflict detection

### 5.1. Bỏ smoothness term lại giảm jitter

Đây là dấu hiệu đáng chú ý: smoothness term hiện tại có thể đang gây tác
dụng ngược, hoặc metric "jitter" không đo đúng mục tiêu của term.

Kiểm tra:

-   Dấu của loss: đang minimize hay maximize?
-   Sai số thời gian có tính trên prediction, ground truth hay residual?
-   Frame có được sắp theo timestamp trước khi tính không?
-   Có frame bị thiếu hoặc khoảng thời gian không đều không?
-   Trọng số smoothness có quá lớn/nhỏ hoặc khác đơn vị với các term
    khác không?
-   Có vô tình làm mượt tọa độ theo chiều joint thay vì chiều thời gian
    không?
-   Ablation có giữ nguyên mọi yếu tố khác, cùng seed và cùng
    initialization không?

Chạy một kiểm tra tối thiểu trên chuỗi tọa độ tổng hợp có chuyển động đã
biết. Xác nhận loss giảm khi chuyển động trở nên mượt hơn theo đúng định
nghĩa được chọn. Sau đó kiểm tra từng term riêng và gradient của nó.

### 5.2. Bỏ bone-length khiến joint bị che khuất tệ hơn input gốc

Điều này có thể cho thấy bone-length term đang bù đắp cho một term khác
có xu hướng đẩy joint sai vị trí. Nó **không tự động chứng minh** toàn
bộ objective có hại; cần xem đóng góp của từng term.

Kiểm tra:

-   Bone length được tính từ cặp joint nào, có đúng skeleton tree không?
-   Chiều dài xương tham chiếu lấy từ đâu: ground truth, thống kê
    subject hay ước lượng từ prediction?
-   Có vô tình dùng ground truth test để tạo prior không?
-   Xương có bị ép về một chiều dài cố định không phù hợp với
    subject/pose không?
-   Trọng số các term và đơn vị có cân bằng không?
-   Có joint indexing sai, mask sai hoặc confidence bằng 0 bị xử lý
    không đúng không?

Với từng term, xuất giá trị loss trước/sau tối ưu và gradient norm. Kiểm
tra objective tổng giảm có đồng nghĩa metric 3D giảm hay không. Hai điều
này không nhất thiết giống nhau.

### 5.3. Conflict detection không có tác dụng đo được

Kiểm tra:

-   Nhánh conflict có thực sự được gọi hay không?
-   Điều kiện kích hoạt có bao giờ đúng không?
-   Ngưỡng có quá cao/thấp khiến nhánh gần như không hoạt động?
-   Kết quả có bị ghi đè bởi bước sau không?
-   Ablation có tạo ra prediction khác nhau không?
-   "Không khác biệt" là giống hệt tuyệt đối hay chỉ không khác ở metric
    đã làm tròn?

Lưu số lần conflict được phát hiện, tỷ lệ joint/frame bị tác động, số
tọa độ thay đổi và chênh lệch metric trước/sau. Nếu component không làm
thay đổi prediction hoặc không cải thiện metric trên dữ liệu độc lập,
hãy cân nhắc loại bỏ hoặc định nghĩa lại vai trò của nó.

### 5.4. Thiết kế ablation đúng

Mỗi ablation phải thay đổi đúng một yếu tố. Giữ nguyên dữ liệu, seed,
initialization, số vòng lặp, điều kiện dừng và mọi tham số khác. Báo cáo
nhiều seed khi thuật toán có tính ngẫu nhiên; nếu không có ngẫu nhiên,
vẫn cần đánh giá trên nhiều đơn vị dữ liệu độc lập.

------------------------------------------------------------------------

## 6. Kiểm tra dữ liệu: một subject, một sequence, 210 camera pairs

### Reviewer đang nói gì?

210 cặp camera không phải 210 mẫu độc lập nếu tất cả được tạo từ cùng
một subject và sequence. Vì vậy, không thể xem số lượng camera pair là
bằng chứng về khả năng tổng quát hóa.

### Cách điều tra

1.  Lập bảng nguồn dữ liệu: subject, sequence, frame, camera A, camera
    B, điều kiện occlusion.
2.  Xác định mỗi pair có dùng lại cùng frame, cùng pose hoặc cùng camera
    hay không.
3.  Vẽ sơ đồ phụ thuộc giữa các mẫu. Các pair chia sẻ frame/subject
    không độc lập.
4.  Tính số lượng **đơn vị độc lập thực sự**: subject và sequence độc
    lập, không phải chỉ số lượng cặp camera.
5.  Kiểm tra việc chia train/validation/test. Các frame liền kề của cùng
    sequence không nên được coi như các mẫu độc lập để tuyên bố
    generalization.
6.  Nếu có thể, thu thập nhiều subject và sequence; tách test theo
    subject chưa từng xuất hiện trong bước lựa chọn tham số.
7.  Báo cáo kết quả riêng theo subject/sequence rồi mới tổng hợp.

### Phân tích thống kê

-   Không bootstrap riêng lẻ 210 camera pairs nếu chúng phụ thuộc mạnh.
    Bootstrap theo subject hoặc sequence phù hợp hơn.
-   Nếu chỉ có một subject/sequence, không thể ước lượng đáng tin cậy độ
    biến thiên giữa subject. Cần nêu rõ giới hạn và thu thập thêm dữ
    liệu.
-   Không xem nhiều frame liền nhau là các quan sát độc lập để tạo
    khoảng tin cậy quá hẹp.

------------------------------------------------------------------------

## 7. Kiểm tra 14 tham số "empirically set"

### Nguy cơ

Nếu tham số được chọn dựa trên kết quả của chính tập test, metric test
đã tham gia vào quá trình phát triển phương pháp. Kết quả cuối có thể
lạc quan và không phản ánh hiệu năng trên dữ liệu mới.

### Cách điều tra

1.  Liệt kê đủ 14 tham số, giá trị, khoảng thử và lý do chọn.
2.  Truy lại lịch sử commit, notebook, log hoặc ghi chú để xác định các
    tham số được quyết định khi nào.
3.  Đánh dấu tham số nào được chọn bằng ground truth hoặc metric của Set
    M / Set khác.
4.  Phân loại:
    -   tham số cố định theo lý thuyết/đơn vị;
    -   tham số chọn trên validation;
    -   tham số có nguy cơ được chọn dựa trên test.
5.  Đưa việc chọn tham số về validation set; đóng băng chúng trước khi
    chạy test.
6.  Chạy sensitivity analysis: thay từng tham số trong một khoảng hợp
    lý, xem kết quả có ổn định không.
7.  Nếu đã dùng test để tinh chỉnh, cần tạo test set mới hoặc thu thập
    subject/sequence mới để đánh giá xác nhận.

### Đầu ra nên có

Một bảng `parameter`, `meaning`, `chosen value`, `selection data`,
`selection procedure`, `sensitivity`. Không chỉ ghi "empirically set";
phải mô tả quy trình đủ để tái lập.

------------------------------------------------------------------------

## 8. Kiểm tra occlusion và cách chia Set M / các tập khác

### Reviewer đang nói gì?

Occlusion được tính từ mesh do chính mô hình dự đoán; tập kiểm tra lại
được chia theo belief của phương pháp; đồng thời chỉ giữ frame có
visibility ít nhất 90%, không phù hợp với tuyên bố tập trung vào
occlusion nặng.

### 8.1. Occlusion được tính từ predicted mesh

Các rủi ro có thể gồm:

-   Mesh sai dẫn đến nhãn occlusion sai.
-   Nhãn occlusion phụ thuộc chính prediction đang được đánh giá.
-   Phương pháp có thể được đánh giá thuận lợi hoặc bất lợi do định
    nghĩa nhãn.
-   Không phân biệt được che khuất hình học thật với lỗi pose estimator.

Kiểm tra chính xác thuật toán ray casting / visibility, camera
intrinsics/extrinsics nếu được sử dụng, định nghĩa "occluded", ngưỡng và
cách xử lý self-occlusion. Đối chiếu một mẫu frame bằng hình
ảnh/annotation độc lập nếu có.

Ưu tiên nhãn occlusion độc lập với phương pháp được đánh giá, chẳng hạn
annotation thủ công có quy tắc rõ ràng hoặc nhãn visibility từ nguồn
ground truth đã được xác thực. Nếu không thể có nhãn độc lập, cần mô tả
đây là nhãn xấp xỉ và kiểm tra độ nhạy với các định nghĩa khác nhau.

### 8.2. Chia tập theo belief score

Nếu Set M được tạo từ chính belief score của phương pháp, cần kiểm tra
nguy cơ circular evaluation: phương pháp được đánh giá trên các nhóm do
nó tự định nghĩa.

-   Lưu code tạo split và ngưỡng chính xác.
-   Xác nhận split được tạo trước hay sau khi xem kết quả.
-   So sánh với phân nhóm dựa trên visibility/occlusion độc lập.
-   Báo cáo hiệu năng theo các mức occlusion được định nghĩa độc lập,
    không chỉ theo belief.
-   Không diễn giải nhóm belief thấp thành "occlusion nặng" nếu chưa
    chứng minh mối liên hệ.

### 8.3. Lọc frame visibility ≥ 90%

-   Tìm tất cả vị trí áp dụng filter và xác định 90% là tỷ lệ joint
    visible, pixel visible hay đại lượng khác.
-   In số frame trước/sau lọc và phân bố visibility.
-   Kiểm tra có loại mất cân đối các frame khó, các đoạn chuyển động
    hoặc các tình huống occlusion nặng không.
-   Chạy đánh giá trên toàn bộ frame hợp lệ và báo cáo riêng theo mức
    visibility.
-   Nếu mục tiêu là occlusion nặng, phải có đủ mẫu ở các mức occlusion
    nặng; không chỉ dựa vào tên Set M.

Không nên đơn giản bỏ filter mà không xem lý do ban đầu. Hãy xác định
filter có cần thiết cho chất lượng ground truth hay chỉ đang làm kết quả
dễ hơn.

------------------------------------------------------------------------

## 9. Kiểm tra tài liệu tham khảo \[3\]

Reviewer cho rằng \[3\] là nghiên cứu về niềm tin của con người vào độ
tin cậy AI, không hỗ trợ cho luận điểm mà bài đang trích dẫn.

### Cách điều tra

1.  Mở bài gốc của reference \[3\], không chỉ đọc tiêu đề hoặc abstract.
2.  Ghi lại chính xác claim trong manuscript mà \[3\] được trích dẫn để
    hỗ trợ.
3.  Xác định bài \[3\] thực sự chứng minh điều gì, với phương pháp và
    phạm vi nào.
4.  Phân loại:
    -   hỗ trợ trực tiếp;
    -   chỉ hỗ trợ bối cảnh;
    -   không liên quan;
    -   mâu thuẫn với claim.
5.  Tìm nguồn sơ cấp đúng chủ đề cho claim đó hoặc viết lại claim hẹp
    hơn.
6.  Kiểm tra toàn bộ references tương tự: trích dẫn có thật sự hỗ trợ
    câu ngay trước nó không?

Không thay reference chỉ vì tiêu đề có vẻ phù hợp. Cần xác nhận nội dung
và giới hạn của nguồn.

------------------------------------------------------------------------

## 10. Thiếu baseline uncalibrated multi-view và calibrated upper bound

### Tại sao quan trọng?

Nếu chỉ so với input đơn camera và baseline đơn giản, chưa thể biết cải
thiện đến từ ý tưởng mới hay từ việc sử dụng thêm một camera. Không có
calibrated upper bound thì chưa biết khoảng cách tới hiệu năng khi có
calibration.

### Cách điều tra và thiết kế đối chứng

1.  Tìm các phương pháp multi-view không cần calibration hoặc xử lý
    camera không đồng bộ/không hiệu chuẩn có cùng đầu vào và mục tiêu.
2.  Kiểm tra điều kiện của từng baseline: cần calibration không, cần
    đồng bộ không, có dùng ground truth không, có cần training không.
3.  Chạy baseline với cùng input, joint convention, split, mask và
    metric.
4.  Xây dựng calibrated upper bound nếu dữ liệu cho phép: dùng
    calibration đáng tin cậy và triangulation hoặc phương pháp
    multi-view phù hợp.
5.  Ghi rõ calibrated upper bound là **mốc tham chiếu có thêm thông
    tin**, không phải đối thủ công bằng hoàn toàn nếu phương pháp đề
    xuất không được phép dùng calibration.
6.  So sánh cả độ chính xác, độ ổn định theo thời gian, tỷ lệ thất bại
    và chi phí tính toán nếu phù hợp.

Không tự nhận một baseline là "state of the art" nếu chưa xác minh đúng
bài báo, code, điều kiện và cấu hình.

------------------------------------------------------------------------

## 11. Kiểm tra độ lớn cải thiện và ý nghĩa thống kê

Reviewer nhắc rằng nhiều cải thiện dưới 1 mm nhưng không có variance
hoặc significance testing.

### Cần làm

-   Tính lại metric trên prediction lưu sẵn bằng cùng một script.
-   Báo cáo mean và độ phân tán ở cấp đơn vị đánh giá hợp lý
    (subject/sequence, nếu có đủ dữ liệu).
-   Đưa khoảng tin cậy cho chênh lệch giữa phương pháp đề xuất và
    baseline.
-   Nếu dùng kiểm định giả thuyết, chọn kiểm định phù hợp với cấu trúc
    phụ thuộc của dữ liệu và xác định trước câu hỏi so sánh.
-   Báo cáo kích thước hiệu ứng và mức cải thiện tuyệt đối/tương đối,
    không chỉ p-value.
-   Nếu số subject quá ít, nêu rõ rằng suy luận thống kê về tổng quát
    hóa còn hạn chế; không khắc phục bằng cách coi mọi frame/camera pair
    là độc lập.

Mức cải thiện dưới 1 mm không tự động là vô nghĩa. Cần xem đơn vị, độ
phân tán, tính ổn định và ý nghĩa thực tế của mức cải thiện đó.

------------------------------------------------------------------------

## 12. Kiểm tra mâu thuẫn giữa abstract, bảng và kết luận

Tạo một bảng claim-to-evidence:

  -----------------------------------------------------------------------
  Claim trong bài         Bằng chứng cần có       Việc cần kiểm tra
  ----------------------- ----------------------- -----------------------
  "Superior dynamic       Acceleration/jitter     Đối chiếu Table 1 và
  stability"              metric tốt hơn baseline metric tính lại

  Cải thiện ở occluded    Kết quả trên nhãn       Xác minh nhãn, mask và
  joints                  occlusion độc lập       split

  Belief-driven           Belief dự báo lỗi và    Phân tích theo từng
  correction hiệu quả     chọn đúng camera/joint  joint/frame

  Generalization          Subject/sequence test   Kiểm tra split và lịch
                          độc lập                 sử tuning

  Conflict detection hữu  Prediction/metric thay  Đo tần suất kích hoạt
  ích                     đổi có lợi              và hiệu quả

  Bone/smoothness         Ablation có kiểm soát   Kiểm tra loss, gradient
  objective hữu ích                               và metric đầu ra
  -----------------------------------------------------------------------

Mỗi câu trong abstract/conclusion phải được hỗ trợ bởi một kết quả cụ
thể. Nếu dữ liệu hiện có mâu thuẫn với claim, hãy thu hẹp claim hoặc bỏ
claim; không nên chỉ đổi cách diễn đạt để che mâu thuẫn.

------------------------------------------------------------------------

## 13. Thứ tự điều tra ưu tiên

### P0 --- Xác minh tính đúng đắn của kết quả hiện tại

-   [ ] Tái lập Table 1 từ code và prediction đã lưu.
-   [ ] Kiểm tra công thức, đơn vị và frame alignment của acceleration
    error.
-   [ ] Kiểm tra metric overall/aligned/occluded và mask.
-   [ ] Xác nhận averaging và các baseline được chạy công bằng.
-   [ ] Kiểm tra nhãn, cột và làm tròn trong bảng.

### P1 --- Tìm lỗi thuật toán

-   [ ] Kiểm tra belief có tương quan với lỗi thực tế.
-   [ ] Kiểm tra logic chọn camera, hard replacement/blending và hệ tọa
    độ.
-   [ ] Kiểm tra từng loss term, trọng số, gradient và ablation.
-   [ ] Đo xem conflict detection có được kích hoạt và có tác động
    không.
-   [ ] Vẽ trajectory của joint trước/sau hiệu chỉnh.

### P2 --- Tìm lỗi đánh giá

-   [ ] Truy vết cách tạo occlusion label.
-   [ ] Truy vết cách chia Set M theo belief.
-   [ ] Kiểm tra filter visibility ≥ 90%.
-   [ ] Tính lại kết quả theo các nhóm occlusion độc lập.
-   [ ] Xác định số subject/sequence độc lập thực tế.

### P3 --- Tìm lỗi thiết kế thí nghiệm

-   [ ] Truy lịch sử chọn 14 tham số.
-   [ ] Tách validation khỏi test; đóng băng tham số.
-   [ ] Lập kế hoạch held-out-subject/sequence.
-   [ ] Bổ sung baseline uncalibrated multi-view và calibrated upper
    bound.
-   [ ] Báo cáo variance, confidence interval và mức cải thiện thực tế.

### P4 --- Sửa luận điểm và tài liệu

-   [ ] Kiểm tra reference \[3\] và toàn bộ trích dẫn liên quan.
-   [ ] Tạo bảng claim-to-evidence.
-   [ ] Chỉ giữ claim được kết quả hỗ trợ.
-   [ ] Cập nhật abstract, limitations, protocol và phần reproducibility
    sau khi hoàn tất phân tích.

------------------------------------------------------------------------

## 14. Cấu trúc thư mục điều tra đề xuất

``` text
review_investigation/
├── README.md
├── configs/
│   ├── original_run.yaml
│   └── frozen_validation.yaml
├── logs/
│   ├── environment.txt
│   └── run_manifest.csv
├── predictions/
│   ├── input/
│   ├── proposed/
│   └── baselines/
├── metrics/
│   ├── per_frame.csv
│   ├── per_joint.csv
│   └── summary.csv
├── plots/
│   ├── joint_trajectories/
│   ├── belief_vs_error/
│   └── metric_distributions/
├── ablations/
├── data_audit/
│   ├── subject_sequence_inventory.csv
│   ├── visibility_distribution.csv
│   └── parameter_audit.csv
└── findings.md
```

Trong `findings.md`, với mỗi vấn đề, ghi:

``` text
## Issue ID:
Reviewer claim:
Code/data locations:
Hypothesis:
Experiment:
Expected diagnostic result:
Observed result:
Conclusion:
Evidence files / commit:
Next action:
```

------------------------------------------------------------------------

## 15. Tiêu chí kết thúc điều tra

Không coi một vấn đề là đã giải quyết chỉ vì đã sửa code hoặc bảng. Chỉ
đóng issue khi:

1.  Có thể tái lập lỗi hoặc mâu thuẫn ban đầu.
2.  Có thí nghiệm phân biệt được các giả thuyết chính.
3.  Kết quả được lưu và có thể kiểm tra lại.
4.  Kết luận không vượt quá bằng chứng.
5.  Nếu vấn đề cần dữ liệu mới, ghi rõ là **chưa giải quyết bằng dữ liệu
    hiện có**.

### Kết luận thực tế

Review này không chỉ chỉ ra vài lỗi diễn đạt. Nó đặt câu hỏi về ba nền
tảng: **phương pháp có thật sự cải thiện pose không, cách đánh giá có
độc lập và hợp lệ không, và dữ liệu có đủ để khẳng định tổng quát hóa
không**. Hãy ưu tiên tái lập bảng kết quả và kiểm tra
acceleration/ablation trước; sau đó mới đầu tư vào thí nghiệm
held-out-subject và baseline mới.
