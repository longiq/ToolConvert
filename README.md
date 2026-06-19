# ToolConvert — PDF to Excel Converter

Chuyển đổi file PDF thành Excel (.xlsx). Chạy hoàn toàn **local / self-hosted**, không phụ thuộc AI cloud.

## Tính năng

- Upload PDF qua giao diện web drag-and-drop
- Tự động phát hiện loại PDF:
  - **PDF text** (có layer text) → dùng `pdfplumber` để trích xuất bảng
  - **PDF scan** (ảnh chụp) → dùng OCR. Mặc định **EasyOCR** (neural network, chạy local, độ chính xác tiếng Việt cao). Có thể chuyển về `Tesseract` qua biến môi trường `OCR_ENGINE=tesseract`.
- Mỗi bảng → 1 sheet riêng trong Excel, tên sheet theo nội dung
- Sheet "Tổng quan" tổng hợp toàn bộ nội dung PDF theo thứ tự
- Style Excel khớp output Claude: font Arial, header xanh đậm, zebra striping, viền 4 cạnh

## Cài đặt & Chạy trên Mac (không cần Docker)

### 1. Cài dependencies

```bash
# Homebrew (nếu chưa có: https://brew.sh)
brew install tesseract tesseract-lang poppler
```

### 2. Clone repo và cài Python packages

```bash
git clone https://github.com/longiq/ToolConvert.git
cd ToolConvert
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Chạy app

```bash
source venv/bin/activate   # nếu chưa activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Truy cập: http://localhost:8000

---

## Expose ra internet qua Cloudflare Tunnel (Mac Mini self-host)

### 1. Cài cloudflared

```bash
brew install cloudflare/cloudflare/cloudflared
```

### 2. Tạo tunnel trên Cloudflare Dashboard

1. Vào [one.dash.cloudflare.com](https://one.dash.cloudflare.com) → **Networks → Tunnels → Create tunnel**
2. Chọn **Cloudflared** → đặt tên tunnel (vd: `toolconvert`)
3. Copy **token** hiển thị
4. Cấu hình **Public Hostname**: `longiq.xyz` → Service `http://localhost:8000`

### 3. Chạy tunnel

```bash
cloudflared tunnel run --token <TUNNEL_TOKEN>
```

### 4. Tự động khởi động khi Mac bật (launchd)

```bash
# Cài cloudflared as system service (tự tạo launchd plist)
sudo cloudflared service install --token <TUNNEL_TOKEN>

# App cũng cần chạy cùng — tạo plist riêng (xem hướng dẫn bên dưới)
```

<details>
<summary>Tạo launchd plist cho app (tự start khi reboot)</summary>

Tạo file `~/Library/LaunchAgents/xyz.longiq.toolconvert.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>xyz.longiq.toolconvert</string>
  <key>ProgramArguments</key>
  <array>
    <string>/path/to/ToolConvert/venv/bin/uvicorn</string>
    <string>app.main:app</string>
    <string>--host</string>
    <string>0.0.0.0</string>
    <string>--port</string>
    <string>8000</string>
  </array>
  <key>WorkingDirectory</key>
  <string>/path/to/ToolConvert</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/tmp/toolconvert.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/toolconvert.err</string>
</dict>
</plist>
```

Thay `/path/to/ToolConvert` bằng đường dẫn thực tế, rồi:

```bash
launchctl load ~/Library/LaunchAgents/xyz.longiq.toolconvert.plist
```

</details>

---

## Cấu hình (biến môi trường)

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `OCR_ENGINE` | `easyocr` | `easyocr` (chính xác hơn) hoặc `tesseract` (nhẹ, nhanh hơn) |
| `OCR_DPI` | `220` | DPI render ảnh từ PDF scan trước khi OCR |

> EasyOCR tải model (~100MB) ở lần chạy đầu và chạy chậm hơn trên CPU. Trên Mac M-series EasyOCR tự tận dụng **Apple Metal (MPS)** nếu được hỗ trợ → nhanh hơn đáng kể so với CPU thuần.

## Giới hạn

- **PDF scan**: Chất lượng phụ thuộc vào độ phân giải và độ rõ của bản scan. OCR không có độ chính xác 100% với mọi font/layout.
- **Bảng phức tạp**: Bảng có nhiều cột merged, header nhiều dòng có thể không detect chính xác.
- **PDF mã hóa/bảo mật**: Không hỗ trợ.
- Kích thước tối đa: 100MB/file.

## Cấu trúc

```
app/
├── main.py          # FastAPI routes
├── detector.py      # Phát hiện loại PDF
├── parser_text.py   # pdfplumber cho PDF text
├── parser_ocr.py    # EasyOCR / Tesseract cho PDF scan
├── table_builder.py # Heuristic: OCR data → table structure
├── excel_writer.py  # openpyxl: tạo .xlsx
└── models.py        # Data models
static/
└── index.html       # Frontend (no build step)
```
