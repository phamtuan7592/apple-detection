# Hướng dẫn Cài đặt và Sử dụng Hệ thống

Tài liệu này cung cấp hướng dẫn chi tiết về cách thiết lập môi trường và vận hành hệ thống phát hiện đối tượng với vùng quan tâm (ROI).

---

## 📋 Yêu cầu hệ thống
* **Python:** Phiên bản 3.10 hoặc cao hơn.
* **Phần cứng:** Webcam (nếu sử dụng chế độ trực tiếp) hoặc File Video.
* **Thư viện:** Các phụ thuộc được liệt kê trong `requirements.txt`.

---

## 🛠 Hướng dẫn cài đặt

### B.1 Cài đặt môi trường
1.  **Tải Python:** Truy cập [python.org](https://www.python.org/) để tải và cài đặt phiên bản 3.10+. 
    * *Lưu ý: Tích chọn **"Add Python to PATH"** trong quá trình cài đặt trên Windows.*
2.  **Mở Terminal:** Truy cập vào thư mục chứa dự án.
    * *Mẹo: Trên Windows, bạn có thể gõ `cmd` vào thanh địa chỉ của File Explorer để mở nhanh.*
3.  **Cài đặt thư viện:** Chạy lệnh sau để tự động cài đặt các thành phần cần thiết:
    ```bash
    python -m pip install -r requirements.txt
    ```

### B.2 Thiết lập vùng quan tâm (ROI) - *Tùy chọn*
Công cụ này cho phép bạn giới hạn khu vực nhận diện để tăng độ chính xác và giảm nhiễu.
4.  **Chạy công cụ vẽ:** ```bash
    python ROI.py
    ```

5.  **Thao tác chuột:** Click **chuột trái** để thêm các điểm tạo thành hình đa giác (polygon).

6.  **Thao tác phím:**
    * Nhấn **`S`**: Lưu cấu hình vào tệp `roi.json`.
    * Nhấn **`ESC`**: Hủy bỏ và thoát công cụ.

---

## 🚀 B.3 Chạy hệ thống

7.  **Cấu hình tham số:** Mở tệp `object.py` bằng trình soạn thảo (VS Code, Notepad++,...) để chỉnh sửa đường dẫn mô hình (`model path`) và đường dẫn video nếu cần.
8.  **Khởi chạy:**
    ```bash
    python object.py
    ```
9.  **Chọn nguồn đầu vào:** Khi màn hình hiển thị yêu cầu, chọn:
    * **Phím `1`**: Chạy với Webcam.
    * **Phím `2`**: Chạy với tệp Video.
10. **Điều khiển chương trình:**
    * **`SPACE` (Phím cách):** Tạm dừng hoặc tiếp tục video.
    * **`ESC`:** Thoát hệ thống hoàn toàn.

---
*Chúc bạn thực hiện thành công!*
