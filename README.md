# ToolConvert — PDF to Excel Converter

Chuyển đổi file PDF thành Excel (.xlsx). Chạy hoàn toàn **local / self-hosted**, không phụ thuộc AI cloud.

## Tính năng

- Upload PDF qua giao diện web drag-and-drop
- Tự động phát hiện loại PDF:
  - **PDF text** (có layer text) → dùng `pdfplumber` để trích xuất bảng
  - **PDF scan** (ảnh chụp) → dùng OCR. Mặc định **EasyOCR** (neural network, chạy local, độ chính xác tiếng Việt cao). Có thể chuyển về `Tesseract` qua biến môi trường `OCR_ENGINE=tesseract`.
- Mỗi bảng → 1 sheet riêng trong Excel
- Sheet "Tổng quan" tổng hợp toàn bộ nội dung PDF theo thứ tự
- Header row: bold + freeze panes
- Tổng/summary row: bold
- Tự động căn chỉnh độ rộng cột

## Cài đặt & Chạy

### Cách 1: Docker (khuyến nghị)

```bash
docker compose up --build
```

Truy cập: http://localhost:8000

### Cách 2: Chạy thủ công

**Yêu cầu hệ thống:**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-vie poppler-utils

# macOS
brew install tesseract tesseract-lang poppler
```

**Cài Python packages:**
```bash
pip install -r requirements.txt
```

**Chạy server:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Cấu hình (biến môi trường)

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `OCR_ENGINE` | `easyocr` | `easyocr` (chính xác hơn) hoặc `tesseract` (nhẹ, nhanh hơn) |
| `OCR_DPI` | `220` | DPI render ảnh từ PDF scan trước khi OCR |

> EasyOCR tải model (~100MB) ở lần chạy đầu và chạy chậm hơn trên CPU (vài phút/tài liệu). Tesseract nhanh hơn nhưng độ chính xác tiếng Việt thấp hơn.

## Giới hạn

- **PDF scan**: Chất lượng phụ thuộc vào độ phân giải và độ rõ của bản scan. OCR không có độ chính xác 100% với mọi font/layout. EasyOCR cho kết quả tốt hơn Tesseract đáng kể với tiếng Việt.
- **Bảng phức tạp**: Bảng có nhiều cột merged, header nhiều dòng có thể không detect chính xác.
- **PDF mã hóa/bảo mật**: Không hỗ trợ.
- Kích thước tối đa: 100MB/file.

## Cấu trúc

```
app/
├── main.py          # FastAPI routes
├── detector.py      # Phát hiện loại PDF
├── parser_text.py   # pdfplumber cho PDF text
├── parser_ocr.py    # Tesseract OCR cho PDF scan
├── table_builder.py # Heuristic: OCR data → table structure
├── excel_writer.py  # openpyxl: tạo .xlsx
└── models.py        # Data models
static/
└── index.html       # Frontend (no build step)
```
