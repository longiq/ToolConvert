"""
OCR pipeline for scanned PDFs.

Supports two engines:
- EasyOCR (neural network, higher Vietnamese accuracy) — preferred
- Tesseract (lightweight, no model download) — fallback

Both engines are adapted to produce the same word/phrase-level DataFrame
(columns: left, top, width, height, conf, text) consumed by table_builder.
"""
import os
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


def extract_from_scanned_pdf(filepath: str, progress_cb=None) -> list[TableData]:
    if progress_cb:
        progress_cb("Đang chuyển PDF thành ảnh...")

    images = convert_from_path(filepath, dpi=OCR_DPI)
    total = len(images)

    engine = OCR_ENGINE
    reader = None
    if engine == "easyocr":
        try:
            if progress_cb:
                progress_cb("Đang tải mô hình OCR (lần đầu có thể lâu)...")
            reader = _get_easyocr_reader()
        except Exception:
            engine = "tesseract"  # fall back if easyocr unavailable

    ocr_pages = []
    for i, img in enumerate(images):
        if progress_cb:
            progress_cb(f"Đang OCR trang {i + 1}/{total} ({engine})...")
        if engine == "easyocr":
            df = _easyocr_page_to_df(reader, img)
        else:
            df = _tesseract_page_to_df(img)
        ocr_pages.append(df)

    if progress_cb:
        progress_cb("Đang phân tích cấu trúc bảng...")

    return build_tables_from_ocr(ocr_pages, progress_cb=progress_cb)
