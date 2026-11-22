Hướng dẫn chuẩn bị dữ liệu và chạy chương trình trên Google Colab
Tài liệu này hướng dẫn cách tạo tệp cấu hình YAML (.yml) cho từng hành động, sau đó chạy chương trình trên Google Colab để xử lý dữ liệu.
1. Mở Google Colab
Truy cập notebook tại đường dẫn:

Mở notebook trên Google Colab

Đăng nhập tài khoản Google có quyền truy cập notebook và dữ liệu Google Drive cần xử lý.
2. Chuẩn bị tệp YAML đầu vào
Mỗi hành động cần xử lý được mô tả bằng một tệp YAML riêng, có phần mở rộng .yml. Ví dụ:
- 001_mma5.yml
- 005_sword3.yml
- 041_grappling2.yml
Tệp YAML khai báo thông tin của hành động, thư mục Ground Truth và dữ liệu tương ứng của từng camera.
