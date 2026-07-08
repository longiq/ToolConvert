"""
OCR pipeline for scanned PDFs.

Supports two engines:
- EasyOCR (neural network, higher Vietnamese accuracy) — preferred
- Tesseract (lightweight, no model download) — fallback

Both engines are adapted to produce the same word/phrase-level DataFrame
(columns: left, top, width, height, conf, text) consumed by table_builder.
"""
import os
import gc
import pandas as pd
from pdf2image import convert_from_path
from .models import TableData
from .table_builder import build_tables_from_ocr

# Engine selection: "easyocr" (default), "tesseract", "docai", or "vlm"
OCR_ENGINE = os.environ.get("OCR_ENGINE", "easyocr").lower()
OCR_DPI = int(os.environ.get("OCR_DPI", "220"))
# Pages per Document AI request. Sync API allows 30 in imageless mode; we send
# chunks of this size so PDFs of any length are handled.
DOCAI_PAGES_PER_REQUEST = int(os.environ.get("DOCAI_PAGES_PER_REQUEST", "15"))
# VLM engine (Qwen2.5-VL via OpenRouter): a vision model reads each page image
# and returns already-structured rows, so we skip table_builder's geometric
# column reconstruction entirely.
VLM_MODEL = os.environ.get("VLM_MODEL", "meta-llama/llama-4-maverick:free")
VLM_DPI = int(os.environ.get("VLM_DPI", "150"))  # lower than OCR_DPI: lighter image, faster

_easyocr_reader = None


def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["vi", "en"], gpu=False, verbose=False)
    return _easyocr_reader


def _easyocr_page_to_df(reader, image) -> pd.DataFrame:
    """Run EasyOCR on a PIL image, return DataFrame matching tesseract schema."""
    import numpy as np
    arr = np.array(image)
    results = reader.readtext(arr, detail=1, paragraph=False)
    rows = []
    for bbox, text, conf in results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        left = int(min(xs))
        top = int(min(ys))
        width = int(max(xs) - min(xs))
        height = int(max(ys) - min(ys))
        rows.append({
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "conf": float(conf) * 100,
            "text": str(text),
        })
    if not rows:
        return pd.DataFrame(columns=["left", "top", "width", "height", "conf", "text"])
    return pd.DataFrame(rows)


def _tesseract_page_to_df(image) -> pd.DataFrame:
    import pytesseract
    from pytesseract import Output
    return pytesseract.image_to_data(
        image, lang="vie+eng", output_type=Output.DATAFRAME, config="--psm 6"
    )


def _google_docai_page_to_df(page, document_text: str) -> pd.DataFrame:
    """Convert one Document AI page into the same 6-column DataFrame that
    tesseract/easyocr produce, so table_builder needs no changes.

    Document AI gives normalised vertices (0.0-1.0); we scale them back to
    pixel coordinates using the page dimension Google reports.
    """
    w = int(getattr(page.dimension, "width", 0)) or 2480
    h = int(getattr(page.dimension, "height", 0)) or 3508
    rows = []
    for token in page.tokens:
        bbox = token.layout.bounding_poly.normalized_vertices
        if not bbox:
            continue
        xs = [v.x for v in bbox]
        ys = [v.y for v in bbox]
        left = int(min(xs) * w)
        top = int(min(ys) * h)
        width = int((max(xs) - min(xs)) * w)
        height = int((max(ys) - min(ys)) * h)
        conf = float(token.layout.confidence) * 100

        # Document AI stores text as offsets into document.text, not inline.
        text = ""
        for seg in token.layout.text_anchor.text_segments:
            start = int(seg.start_index) if seg.start_index else 0
            end = int(seg.end_index)
            text += document_text[start:end]
        text = text.strip()
        if not text:
            continue
        rows.append({
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "conf": conf,
            "text": text,
        })
    if not rows:
        return pd.DataFrame(columns=["left", "top", "width", "height", "conf", "text"])
    return pd.DataFrame(rows)


def _split_pdf_into_chunks(pdf_bytes: bytes, pages_per_chunk: int) -> list[bytes]:
    """Split a PDF byte string into a list of smaller PDF byte strings, each
    with at most `pages_per_chunk` pages. If the PDF already fits in one chunk,
    the original bytes are returned unchanged.
    """
    import io
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(io.BytesIO(pdf_bytes))
    total = len(reader.pages)
    if total <= pages_per_chunk:
        return [pdf_bytes]

    chunks = []
    for start in range(0, total, pages_per_chunk):
        writer = PdfWriter()
        for i in range(start, min(start + pages_per_chunk, total)):
            writer.add_page(reader.pages[i])
        buf = io.BytesIO()
        writer.write(buf)
        chunks.append(buf.getvalue())
    return chunks


def _google_docai_extract(filepath: str, progress_cb=None) -> list[TableData]:
    """OCR a scanned PDF via Google Document AI instead of running tesseract
    locally. Google does the heavy lifting; Render only relays the file, so
    peak RAM stays tiny and there is no OOM risk.
    """
    import json
    from google.cloud import documentai
    from google.oauth2 import service_account

    creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    credentials = None
    if creds_json:
        credentials = service_account.Credentials.from_service_account_info(
            json.loads(creds_json),
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

    location = os.environ.get("GOOGLE_LOCATION", "us")
    client = documentai.DocumentProcessorServiceClient(
        credentials=credentials,
        client_options={"api_endpoint": f"{location}-documentai.googleapis.com"},
    )
    processor_name = client.processor_path(
        os.environ["GOOGLE_PROJECT_ID"],
        location,
        os.environ["GOOGLE_PROCESSOR_ID"],
    )

    with open(filepath, "rb") as f:
        pdf_bytes = f.read()

    # field_mask excludes pages.image from the response → "imageless mode".
    # This raises the per-request page limit from 15 to 30 and cuts response size.
    from google.protobuf import field_mask_pb2
    field_mask = field_mask_pb2.FieldMask(
        paths=["text", "pages.layout", "pages.tokens", "pages.dimension"]
    )

    # Document AI's synchronous API caps a single request at 30 pages (imageless
    # mode). Split larger PDFs into chunks and OCR each chunk separately, then
    # concatenate — this handles documents of any length on the free sync API
    # without needing batch processing / Cloud Storage.
    chunks = _split_pdf_into_chunks(pdf_bytes, DOCAI_PAGES_PER_REQUEST)
    total_chunks = len(chunks)

    ocr_pages: list = []
    for ci, chunk in enumerate(chunks):
        if progress_cb:
            if total_chunks > 1:
                progress_cb(
                    f"Đang gửi phần {ci + 1}/{total_chunks} lên Google Document AI..."
                )
            else:
                progress_cb("Đang gửi PDF lên Google Document AI...")

        request = documentai.ProcessRequest(
            name=processor_name,
            raw_document=documentai.RawDocument(
                content=chunk, mime_type="application/pdf"
            ),
            field_mask=field_mask,
        )
        document = client.process_document(request=request).document
        for page in document.pages:
            ocr_pages.append(_google_docai_page_to_df(page, document.text))

    if progress_cb:
        progress_cb("Đang phân tích cấu trúc bảng...")

    # Document AI doesn't expose the rendered image, so we let table_builder
    # infer columns/rows purely from token positions (no ruled-line detection).
    page_line_data = [{"col": [], "row": []} for _ in ocr_pages]

    return build_tables_from_ocr(
        ocr_pages, page_line_data=page_line_data, progress_cb=progress_cb
    )


# ---------------------------------------------------------------------------
# Vision-LLM engine (Qwen2.5-VL via OpenRouter)
# ---------------------------------------------------------------------------

_VLM_PROMPT = (
    "Bạn là công cụ trích xuất bảng từ ảnh tài liệu tiếng Việt. "
    "Ảnh này là một trang chứa (một phần của) bảng danh sách. "
    "Hãy đọc và trả về DUY NHẤT một JSON object, không kèm lời giải thích, "
    "theo đúng dạng:\n"
    '{"title": "<tiêu đề bảng nếu có, else "">", '
    '"headers": ["<tên cột 1>", ...], '
    '"rows": [["<ô 1>", "<ô 2>", ...], ...]}\n\n'
    "QUY TẮC BẮT BUỘC:\n"
    "- Chép NGUYÊN VĂN từng dòng, đúng thứ tự trên xuống. KHÔNG tóm tắt, "
    "KHÔNG bỏ sót, KHÔNG gộp nhiều dòng làm một.\n"
    "- Mỗi dòng dữ liệu là một phần tử trong 'rows', số ô đúng bằng số cột.\n"
    "- Ô trống để chuỗi rỗng \"\". TUYỆT ĐỐI không bịa nội dung không có trong ảnh.\n"
    "- Giữ nguyên số thứ tự (STT) như in trên ảnh.\n"
    "- Nếu trang không có dòng dữ liệu nào, trả 'rows': []."
)


def _pil_to_data_url(image) -> str:
    """Encode a PIL image as a base64 JPEG data URL for the vision API."""
    import io
    import base64
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _parse_vlm_json(content: str) -> dict:
    """Pull the JSON object out of a model reply, tolerating ```json fences
    and leading/trailing prose."""
    import json
    text = content.strip()
    if text.startswith("```"):
        # strip a ```json ... ``` fence
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    text = text.strip()
    # Fall back to the outermost {...} if there is surrounding noise.
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
    return json.loads(text)


def _vlm_call_openrouter(data_url: str) -> dict:
    """POST one page image to OpenRouter and return the parsed JSON table.
    Retries on rate-limit / server errors with exponential backoff."""
    import time
    import requests

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Thiếu OPENROUTER_API_KEY để dùng engine VLM.")

    payload = {
        "model": VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _VLM_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    backoff = 2
    last_err = None
    for attempt in range(4):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=120,
            )
            if resp.status_code == 404:
                raise RuntimeError(
                    f"Model '{VLM_MODEL}' không tìm thấy trên OpenRouter. "
                    f"Đặt env var VLM_MODEL thành model hợp lệ (xem openrouter.ai/models)."
                )
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(f"OpenRouter HTTP {resp.status_code}")
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return _parse_vlm_json(content)
        except requests.RequestException as e:
            last_err = e
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError(f"Gọi OpenRouter thất bại sau nhiều lần thử: {last_err}")


def _vlm_page_to_table(image, page_idx: int):
    """Run one page image through the VLM and build a TableData, or None if the
    page yielded no rows."""
    data_url = _pil_to_data_url(image)
    parsed = _vlm_call_openrouter(data_url)

    headers = [str(h) for h in (parsed.get("headers") or [])]
    raw_rows = parsed.get("rows") or []
    rows = [[str(c) for c in row] for row in raw_rows if isinstance(row, list)]
    if not rows:
        return None

    return TableData(
        title=str(parsed.get("title") or "").strip(),
        metadata=[],
        headers=headers,
        rows=rows,
        page=page_idx + 1,
        is_summary_row=[False] * len(rows),
    )


def _vlm_check_stt_continuity(tables: list) -> list[str]:
    """Guardrail against hallucination/omission: verify the first column reads
    as a monotonically increasing STT sequence across all rows. Returns a list
    of human-readable warnings (empty if the sequence looks clean)."""
    import re
    warnings: list[str] = []
    prev: int | None = None
    for t in tables:
        for row in t.rows:
            col0 = str(row[0]).strip() if row else ""
            if not re.fullmatch(r"\d{1,4}", col0):
                continue
            n = int(col0)
            if prev is not None:
                if n == prev:
                    warnings.append(f"STT {n} bị lặp")
                elif n != prev + 1:
                    warnings.append(f"STT nhảy từ {prev} sang {n}")
            prev = n
    return warnings


def _openrouter_vlm_extract(filepath: str, progress_cb=None) -> list[TableData]:
    """Extract tables by sending each rendered page image to a vision LLM
    (Qwen2.5-VL by default) via OpenRouter. Pages are rendered one at a time to
    keep memory low, exactly like the tesseract path."""
    from .table_builder import _assign_sheet_names

    try:
        total = _pdf_page_count(filepath)
    except Exception:
        total = 0

    tables: list[TableData] = []

    def _handle(image, idx: int, n: int):
        if progress_cb:
            progress_cb(f"Đang đọc trang {idx + 1}/{n} bằng AI (Qwen2.5-VL)...")
        table = _vlm_page_to_table(image, idx)
        if table is not None:
            tables.append(table)

    if total > 0:
        for p in range(1, total + 1):
            imgs = convert_from_path(filepath, dpi=VLM_DPI, first_page=p, last_page=p)
            img = imgs[0]
            try:
                _handle(img, p - 1, total)
            finally:
                try:
                    img.close()
                except Exception:
                    pass
                del imgs, img
                gc.collect()
    else:
        images = convert_from_path(filepath, dpi=VLM_DPI)
        total = len(images)
        for i, img in enumerate(images):
            _handle(img, i, total)
        del images
        gc.collect()

    # Flag (don't block) suspicious STT sequences so the user can spot possible
    # hallucination / dropped rows in an otherwise plausible-looking result.
    warnings = _vlm_check_stt_continuity(tables)
    if warnings and progress_cb:
        preview = "; ".join(warnings[:3])
        progress_cb(f"Hoàn tất (lưu ý kiểm tra: {preview})")

    _assign_sheet_names(tables)
    return tables


def _detect_page_lines(img, df) -> dict:
    """Detect column/row border positions from a page image while it is in memory.

    Doing this here (instead of in build_tables_from_ocr) means we never have to
    keep every page image resident at once — only the small list of integer
    boundary positions survives, which keeps peak memory well under the 512MB cap.
    """
    import numpy as np
    from .table_builder import (
        _clean_df,
        _find_column_boundaries_from_lines,
        _find_row_boundaries_from_lines,
    )

    df_clean = _clean_df(df)
    if df_clean.empty:
        return {"col": [], "row": []}

    img_np = np.array(img)
    try:
        text_min_x = int(df_clean["left"].min())
        text_max_x = int((df_clean["left"] + df_clean["width"]).max())
        col = _find_column_boundaries_from_lines(img_np, text_x_range=(text_min_x, text_max_x))
        row = _find_row_boundaries_from_lines(img_np)
    finally:
        del img_np
    return {"col": col, "row": row}


def _pdf_page_count(filepath: str) -> int:
    from pdf2image import pdfinfo_from_path
    return int(pdfinfo_from_path(filepath).get("Pages", 0))


def extract_from_scanned_pdf(filepath: str, progress_cb=None) -> list[TableData]:
    # Cloud OCR path: hand the whole PDF to Google Document AI. No pdf2image,
    # no local tesseract — keeps Render's RAM usage minimal.
    if OCR_ENGINE == "docai":
        return _google_docai_extract(filepath, progress_cb=progress_cb)

    # Vision-LLM path: a multimodal model reads each page image and returns
    # structured rows directly (bypasses geometric table reconstruction).
    if OCR_ENGINE == "vlm":
        return _openrouter_vlm_extract(filepath, progress_cb=progress_cb)

    if progress_cb:
        progress_cb("Đang chuyển PDF thành ảnh...")

    engine = OCR_ENGINE
    reader = None
    if engine == "easyocr":
        try:
            if progress_cb:
                progress_cb("Đang tải mô hình OCR (lần đầu có thể lâu)...")
            reader = _get_easyocr_reader()
        except Exception:
            engine = "tesseract"  # fall back if easyocr unavailable

    ocr_pages: list = []
    page_line_data: list = []

    def _handle_page(img, idx: int, total: int):
        if progress_cb:
            progress_cb(f"Đang OCR trang {idx + 1}/{total} ({engine})...")
        if engine == "easyocr":
            df = _easyocr_page_to_df(reader, img)
        else:
            df = _tesseract_page_to_df(img)
        ocr_pages.append(df)
        page_line_data.append(_detect_page_lines(img, df))

    try:
        total = _pdf_page_count(filepath)
    except Exception:
        total = 0

    if total > 0:
        # Render and process one page at a time so only a single page image is
        # ever held in memory (the previous version materialised every page).
        for p in range(1, total + 1):
            imgs = convert_from_path(filepath, dpi=OCR_DPI, first_page=p, last_page=p)
            img = imgs[0]
            try:
                _handle_page(img, p - 1, total)
            finally:
                try:
                    img.close()
                except Exception:
                    pass
                del imgs, img
                gc.collect()
    else:
        # Fallback if the page count couldn't be determined.
        images = convert_from_path(filepath, dpi=OCR_DPI)
        total = len(images)
        for i, img in enumerate(images):
            _handle_page(img, i, total)
        del images
        gc.collect()

    if progress_cb:
        progress_cb("Đang phân tích cấu trúc bảng...")

    return build_tables_from_ocr(ocr_pages, page_line_data=page_line_data, progress_cb=progress_cb)
