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
