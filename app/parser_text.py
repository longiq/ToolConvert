import pdfplumber
from .models import TableData


def _normalize_row(row: list) -> list[str]:
    return [str(cell).strip() if cell is not None else "" for cell in row]


def _is_summary_row(row: list[str]) -> bool:
    first = row[0].lower() if row else ""
    keywords = ("tổng", "total", "cộng", "subtotal", "grand total", "tổng cộng", "tổng số")
    return any(kw in first for kw in keywords)


def _get_text_above_table(page, table_bbox, max_distance: int = 80) -> list[str]:
    """Extract text lines above a table bbox on the same page."""
    x0, y0, x1, y1 = table_bbox
    # Crop region above the table
    if y0 < 5:
        return []
    top = max(0, y0 - max_distance)
    try:
        region = page.crop((0, top, page.width, y0))
        text = region.extract_text()
        if not text:
            return []
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines
    except Exception:
        return []


def extract_from_text_pdf(filepath: str, progress_cb=None) -> list[TableData]:
    tables: list[TableData] = []

    with pdfplumber.open(filepath) as pdf:
        total_pages = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages):
            if progress_cb:
                progress_cb(f"Đang phân tích trang {page_num + 1}/{total_pages}...")

            found = page.find_tables()
            for table_idx, table_obj in enumerate(found):
                bbox = table_obj.bbox
                raw_rows = table_obj.extract()
                if not raw_rows:
                    continue

                rows = [_normalize_row(r) for r in raw_rows]
                # Filter empty rows
                rows = [r for r in rows if any(cell for cell in r)]
                if not rows:
                    continue

                metadata_lines = _get_text_above_table(page, bbox)

                # First row is header if it looks like a header
                headers = rows[0]
                data_rows = rows[1:]

                title = metadata_lines[-1] if metadata_lines else f"Bảng {len(tables) + 1}"
                meta = metadata_lines[:-1] if len(metadata_lines) > 1 else []

                is_summary = [_is_summary_row(r) for r in data_rows]

                tables.append(TableData(
                    title=title,
                    metadata=meta,
                    headers=headers,
                    rows=data_rows,
                    page=page_num + 1,
                    is_summary_row=is_summary,
                ))

    return tables
