import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path
from .models import TableData
from .table_builder import build_tables_from_ocr


def extract_from_scanned_pdf(filepath: str, progress_cb=None) -> list[TableData]:
    if progress_cb:
        progress_cb("Đang chuyển PDF thành ảnh...")

    images = convert_from_path(filepath, dpi=300)
    total = len(images)

    ocr_pages = []
    for i, img in enumerate(images):
        if progress_cb:
            progress_cb(f"Đang OCR trang {i + 1}/{total}...")
        df = pytesseract.image_to_data(
            img,
            lang="vie+eng",
            output_type=Output.DATAFRAME,
            config="--psm 6",
        )
        ocr_pages.append(df)

    if progress_cb:
        progress_cb("Đang phân tích cấu trúc bảng...")

    return build_tables_from_ocr(ocr_pages, progress_cb=progress_cb)
