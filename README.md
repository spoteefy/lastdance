
<center>
<font size=4pt;>
<strong>
S5TH: Span-based EnTiTy and relaTion Transformer wiTH pos-embedding and dual fusion module
</strong>
</font> 
</center>

___
**Sinh viên:** Nguyễn Văn Thọ
**MSSV:** 20204694
**Chương trình**: Khoa học máy tính 2020 - TCNTT
**Tên đề tài**: Nhận diện thực thể và trích rút quan hệ đồng thời trên tập tài liệu khoa học
**GVHD**: PGS. TS. Nguyễn Thị Kim Anh
**Kỳ học**: 2023.2
___
**HƯỚNG DẪN**

 1.  File cấu hình tham số: `main.py`
 2. Phiên bản SciBERT được sử dụng là scibert_scivocab_cased (tải xuống từ [https://github.com/allenai/scibert](https://github.com/allenai/scibert) tại tùy chọn "PyTorch HuggingFace Models"). Sau đó thiết lập đường dẫn `model_path` và `tokenizer_path` đến SciBERT trong file cấu hình.
 3. Chạy chương trình (huấn luyện mô hình nếu `TRAIN = True`, nếu `False` thì mô hình tiến hành chỉ chạy kiểm thử trên mô hình đã huấn luyện sẵn) bằng lệnh: `python3 main.py`

Tất cả công việc được lưu tại https://github.com/spoteefy/lastdance

**GHI CHÚ**
Mô hình này được sửa đổi dựa trên SpERT ([https://github.com/lavis-nlp/spert/](https://github.com/lavis-nlp/spert/))

Cảm ơn Debarshi Kumar Sanyal (dksanyal) đã cung cấp mã nguồn về SpERT để tôi có thể thực hiện đồ án này.
