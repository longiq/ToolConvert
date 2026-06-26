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

# Engine selection: "easyocr" (default) or "tesseract"
OCR_ENGINE = os.environ.get("OCR_ENGINE", "easyocr").lower()
OCR_DPI = int(os.environ.get("OCR_DPI", "220"))

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

    if progress_cb:
        progress_cb("Đang gửi PDF lên Google Document AI...")

    with open(filepath, "rb") as f:
        pdf_bytes = f.read()

    # field_mask excludes pages.image from the response → "imageless mode".
    # This raises the per-request page limit from 15 to 30 and cuts response size.
    from google.protobuf import field_mask_pb2
    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=documentai.RawDocument(
            content=pdf_bytes, mime_type="application/pdf"
        ),
        field_mask=field_mask_pb2.FieldMask(
            paths=["text", "pages.layout", "pages.tokens", "pages.dimension"]
        ),
    )
    result = client.process_document(request=request)
    document = result.document

    if progress_cb:
        progress_cb("Đang phân tích cấu trúc bảng...")

    ocr_pages = [_google_docai_page_to_df(page, document.text) for page in document.pages]
    # Document AI doesn't expose the rendered image, so we let table_builder
    # infer columns/rows purely from token positions (no ruled-line detection).
    page_line_data = [{"col": [], "row": []} for _ in ocr_pages]

    return build_tables_from_ocr(
        ocr_pages, page_line_data=page_line_data, progress_cb=progress_cb
    )


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
