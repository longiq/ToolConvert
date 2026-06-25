"""
Build TableData from Tesseract OCR output.

Two-pass strategy:
Pass 1 – Detect consensus column boundaries for the page.
Pass 2 – Assign every word to a (row_band, col_idx) cell.
          Row bands are y-ranges separated by vertical gaps > threshold.
          Col idx is determined by word x-position vs boundaries.
          Full-width bands with no clear column split → metadata.
"""
import bisect
import re
import unicodedata
import numpy as np
from .models import TableData

_SUMMARY_RE = re.compile(
    r"^(tổng|tổng\s*cộng|tổng\s*số|cộng|total|grand\s*total|subtotal)",
    re.IGNORECASE,
)
_HEADER_RE = re.compile(
    r"\b(stt|s\.tt|số\s*tt|số\s*thứ\s*tự|tên\s*cơ\s*sở|địa\s*chỉ|họ\s*tên|"
    r"họ\s*và\s*tên|no\.|name|address|description|đơn\s*vị|số\s*lượng|"
    r"đơn\s*giá|thành\s*tiền|ghi\s*chú)\b",
    re.IGNORECASE,
)
# Matches leading STT number merged with cell text, e.g. "19 HỘ KINH DOANH..."
_STT_MERGED_RE = re.compile(r"^(\d{1,3})\s{1,4}(.+)$")
# Matches Roman-numeral or numbered section headings, e.g. "I. DANH SÁCH..."
_SECTION_HEADING_RE = re.compile(
    r"^(?:[IVXLivxl]{1,5}|[0-9]{1,2})[\.\)]\s*.{5,}"  # Roman/digit heading (L covers OCR-misread II)
    r"|DANH\s*SÁCH",
    re.IGNORECASE,
)

MIN_COL_GAP = 100  # px at 300 DPI ~ 8mm — avoids false splits on large-font centered titles
META_WIDTH_RATIO = 0.44  # band must span > this fraction of page to be metadata
ROW_TOLERANCE = 18  # px: words within this y-range are on the same visual line
ROW_SPLIT_GAP = 50  # px: vertical gap larger than this → new row band


def _is_summary(text: str) -> bool:
    return bool(_SUMMARY_RE.match(text.strip()))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_df(df):
    df = df[df["conf"] > 20].copy()
    df = df[df["text"].str.strip().astype(bool)].copy()
    return df


def _cluster_y(tops: list[float], tolerance: int = ROW_TOLERANCE,
               row_boundaries: list[int] = None,
               centers: list[float] = None) -> list[int]:
    """Return a row-band index for each top value (same group ≤ tolerance apart).

    When row_boundaries (y-positions of detected horizontal table lines) are provided,
    uses word CENTERS (not tops) for the boundary crossing check — centers are more
    reliable because OCR bounding boxes often extend slightly above/below the actual
    text, while the center of the bbox reliably falls inside the correct row.
    """
    if not tops:
        return []
    ctrs = centers if centers else tops
    order = sorted(range(len(tops)), key=lambda i: tops[i])
    labels = [0] * len(tops)
    current_group = 0
    prev_top = tops[order[0]]
    prev_ctr = ctrs[order[0]]
    labels[order[0]] = 0
    for idx in order[1:]:
        curr_top = tops[idx]
        curr_ctr = ctrs[idx]
        hard_break = False
        if row_boundaries:
            # A horizontal line at y separates previous word (center below y? no—above)
            # from current word (center at or above y? no—below). Break when a boundary
            # falls strictly between prev center and current center.
            hard_break = any(prev_ctr < y <= curr_ctr for y in row_boundaries)
        if curr_top - prev_top > tolerance or hard_break:
            current_group += 1
        labels[idx] = current_group
        prev_top = curr_top
        prev_ctr = curr_ctr
    return labels


def _find_column_boundaries_from_lines(image_np, text_x_range: tuple[int, int] = None) -> list[int]:
    """
    Detect vertical table-border lines using OpenCV morphological ops.
    Returns sorted list of x-midpoints between consecutive vertical lines.
    Returns [] if fewer than 2 vertical lines found (caller should fall back to word-gap).

    When text_x_range is provided, only lines strictly inside that x-range are kept,
    filtering out page-edge artifacts and outer table borders.
    """
    try:
        import cv2
        if image_np.ndim == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np.copy()

        h, w = gray.shape
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel_h = max(h // 6, 40)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        vertical_lines = cv2.dilate(vertical_lines, dilate_kernel, iterations=1)

        contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        x_positions = []
        min_y_span = h * 0.25

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw < 20 and ch >= min_y_span:
                x_center = x + cw // 2
                if text_x_range is None or (text_x_range[0] < x_center < text_x_range[1]):
                    x_positions.append(x_center)

        if len(x_positions) < 2:
            return []

        x_positions.sort()
        clusters: list[list[int]] = []
        for x in x_positions:
            if clusters and abs(x - clusters[-1][-1]) <= 15:
                clusters[-1].append(x)
            else:
                clusters.append([x])

        line_xs = sorted(int(np.median(c)) for c in clusters)

        if len(line_xs) < 2:
            return []

        # Inner boundaries only (strip outer left/right borders)
        return line_xs[1:-1]

    except Exception:
        return []


def _find_row_boundaries_from_lines(image_np) -> list[int]:
    """Detect horizontal table-border lines using morphological ops.
    Returns sorted list of internal y-positions (between top and bottom borders).
    Returns [] if fewer than 3 horizontal lines found.
    """
    try:
        import cv2
        if image_np.ndim == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np.copy()

        h, w = gray.shape
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Wide, flat kernel detects horizontal lines
        kernel_w = max(w // 6, 40)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        horizontal_lines = cv2.dilate(horizontal_lines, dilate_kernel, iterations=1)

        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        y_positions = []
        min_x_span = w * 0.25

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if ch < 20 and cw >= min_x_span:
                y_center = y + ch // 2
                y_positions.append(y_center)

        if len(y_positions) < 3:
            return []

        y_positions.sort()
        clusters: list[list[int]] = []
        for y in y_positions:
            if clusters and abs(y - clusters[-1][-1]) <= 15:
                clusters[-1].append(y)
            else:
                clusters.append([y])

        line_ys = sorted(int(np.median(c)) for c in clusters)

        if len(line_ys) < 3:
            return []

        filtered_ys: list[int] = [line_ys[0]]
        for y in line_ys[1:]:
            if y - filtered_ys[-1] >= 20:
                filtered_ys.append(y)
        line_ys = filtered_ys

        if len(line_ys) < 3:
            return []

        # Return internal y-positions (strip outermost top/bottom borders)
        return line_ys[1:-1]

    except Exception:
        return []


def _find_column_boundaries(df) -> list[int]:
    """
    Find x-positions of large inter-word gaps that repeat across many lines.
    Returns sorted list of boundary x-midpoints.
    """
    df = _clean_df(df)
    if df.empty:
        return []

    # Group words into visual lines by top coordinate
    tops = df["top"].tolist()
    labels = _cluster_y(tops)
    df = df.copy()
    df["row_band"] = labels

    # For each visual line, find large gaps between adjacent words
    all_gap_mids: list[int] = []
    line_count = 0
    for _, grp in df.groupby("row_band"):
        grp_sorted = grp.sort_values("left")
        lefts = grp_sorted["left"].tolist()
        rights = (grp_sorted["left"] + grp_sorted["width"]).tolist()
        line_count += 1
        for i in range(len(rights) - 1):
            space = lefts[i + 1] - rights[i]
            if space >= MIN_COL_GAP:
                all_gap_mids.append((rights[i] + lefts[i + 1]) // 2)

    if not all_gap_mids:
        return []

    # Cluster gap midpoints with ±50px tolerance
    sorted_gaps = sorted(all_gap_mids)
    clusters: list[list[int]] = []
    for g in sorted_gaps:
        if clusters and abs(g - clusters[-1][-1]) <= 60:
            clusters[-1].append(g)
        else:
            clusters.append([g])

    # Keep clusters present in ≥15% of lines
    min_count = max(2, line_count * 0.12)
    boundaries = [int(np.median(c)) for c in clusters if len(c) >= min_count]
    return sorted(boundaries)


# ---------------------------------------------------------------------------
# Build cell grid from all words given boundaries
# ---------------------------------------------------------------------------

def _build_cell_grid(df, boundaries: list[int], row_boundaries: list[int] = None) -> list[dict]:
    """
    Returns list of row-band dicts:
      { "band": int, "top": float, "cells": list[str], "is_meta": bool }

    When row_boundaries (horizontal line y-positions) are provided, words are assigned
    to rows by which horizontal band they fall in — eliminating cross-row OCR bleeding.
    """
    df = _clean_df(df)
    if df.empty:
        return []

    num_cols = len(boundaries) + 1
    df = df.copy()

    if row_boundaries:
        # Hybrid row assignment:
        # - Words whose CENTER falls between two detected horizontal lines → bisect (hard boundary,
        #   prevents cross-row bleeding within the bordered table area).
        # - Words above the first detected line (pre-table area: noise, section headings, column
        #   headers) → Y-clustering (preserves natural separation between header and title text).
        first_rb = row_boundaries[0]
        tops_list = df["top"].tolist()
        hts_list = df.get("height", [0] * len(df)).tolist() if "height" in df.columns else [0] * len(df)

        bisect_labels = []
        pre_mask = []
        for top, ht in zip(tops_list, hts_list):
            center = float(top) + float(ht) / 2
            if center < first_rb:
                bisect_labels.append(None)  # will fill with Y-cluster
                pre_mask.append(True)
            else:
                bisect_labels.append(bisect.bisect_right(row_boundaries, center))
                pre_mask.append(False)

        # Y-cluster the pre-boundary words, map them to large negatives to avoid collision
        pre_indices = [i for i, m in enumerate(pre_mask) if m]
        if pre_indices:
            pre_tops = [tops_list[i] for i in pre_indices]
            pre_cluster_labels = _cluster_y(pre_tops)
            max_pre = max(pre_cluster_labels) if pre_cluster_labels else 0
            for local_idx, global_idx in enumerate(pre_indices):
                # Map to -(max_pre + 1) … -1 so they sort before bisect bands (1, 2, …)
                bisect_labels[global_idx] = pre_cluster_labels[local_idx] - max_pre - 1

        df["row_band"] = bisect_labels
    else:
        tops = df["top"].tolist()
        labels = _cluster_y(tops)
        df["row_band"] = labels

    rows: dict[int, dict] = {}
    for _, row in df.iterrows():
        band = int(row["row_band"])
        if band not in rows:
            rows[band] = {
                "band": band,
                "top": float(row["top"]),
                "cells": [""] * num_cols,
                "min_left": float(row["left"]),
                "max_right": float(row["left"]) + float(row["width"]),
            }
        else:
            rows[band]["top"] = min(rows[band]["top"], float(row["top"]))
            rows[band]["min_left"] = min(rows[band]["min_left"], float(row["left"]))
            rows[band]["max_right"] = max(
                rows[band]["max_right"], float(row["left"]) + float(row["width"])
            )

        word_left = float(row["left"])
        col_idx = sum(1 for b in boundaries if word_left > b)
        col_idx = min(col_idx, num_cols - 1)
        word = str(row["text"])
        curr = rows[band]["cells"][col_idx]
        rows[band]["cells"][col_idx] = (curr + " " + word).strip() if curr else word

    # Mark rows that are "full-width metadata"
    # Criteria: only col-0 has content AND that content spans > 60% of page width
    page_width_est = max((r["max_right"] for r in rows.values()), default=1000)
    for band_data in rows.values():
        cells_filled = [bool(c.strip()) for c in band_data["cells"]]
        only_first_col = cells_filled[0] and not any(cells_filled[1:])
        span = band_data["max_right"] - band_data["min_left"]
        is_wide = span > page_width_est * META_WIDTH_RATIO
        band_data["is_meta"] = only_first_col and is_wide

    return sorted(rows.values(), key=lambda r: r["band"])


# ---------------------------------------------------------------------------
# Merge row bands into logical table rows
# ---------------------------------------------------------------------------

def _merge_continuation_rows(bands: list[dict], page_height: int) -> list[dict]:
    """
    Some logical rows span 2 visual bands (multi-line cell content).
    Merge bands where band[i] has only col-0 content AND the vertical gap
    to the next band is small (not a new section).
    """
    if not bands:
        return bands

    merged: list[dict] = []
    skip_next = False

    for i, band in enumerate(bands):
        if skip_next:
            skip_next = False
            continue

        if band["is_meta"] and i + 1 < len(bands):
            next_band = bands[i + 1]
            gap = next_band["top"] - band["top"]
            next_text = " ".join(c for c in next_band["cells"] if c)
            # Don't merge if: gap too large, next band is also meta,
            # OR next band looks like a section header row
            is_next_header = bool(_HEADER_RE.search(next_text))
            if gap < ROW_SPLIT_GAP * 3 and not next_band["is_meta"] and not is_next_header:
                # Merge: prepend current band's col-0 into next band's col-0
                merged_cells = list(next_band["cells"])
                prefix = band["cells"][0]
                merged_cells[0] = (prefix + " " + merged_cells[0]).strip()
                merged.append({
                    "band": band["band"],
                    "top": band["top"],
                    "cells": merged_cells,
                    "is_meta": False,
                    "min_left": band["min_left"],
                    "max_right": next_band["max_right"],
                })
                skip_next = True
                continue

        merged.append(band)

    return merged


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_tables_from_ocr(page_ocr_list: list, page_images: list = None, progress_cb=None) -> list[TableData]:
    tables: list[TableData] = []

    for page_idx, df in enumerate(page_ocr_list):
        if progress_cb:
            progress_cb(f"Đang phân tích cấu trúc trang {page_idx + 1}/{len(page_ocr_list)}...")

        df_clean = _clean_df(df)
        if df_clean.empty:
            continue

        page_height = int(df_clean["top"].max() + df_clean["height"].max()) if not df_clean.empty else 3000

        # Try OpenCV line detection first if images are provided
        boundaries = []
        if page_images is not None and page_idx < len(page_images):
            try:
                img_np = np.array(page_images[page_idx])
                text_min_x = int(df_clean["left"].min())
                text_max_x = int((df_clean["left"] + df_clean["width"]).max())
                boundaries = _find_column_boundaries_from_lines(
                    img_np, text_x_range=(text_min_x, text_max_x)
                )
            except Exception:
                boundaries = []

        if len(boundaries) < 2:
            boundaries = _find_column_boundaries(df)

        if not boundaries:
            # No table structure detected: treat page as text block
            all_text = " ".join(df_clean.sort_values(["top", "left"])["text"].tolist())
            if all_text.strip():
                tables.append(TableData(
                    title=f"Nội dung trang {page_idx + 1}",
                    metadata=[],
                    headers=["Nội dung"],
                    rows=[[all_text.strip()]],
                    page=page_idx + 1,
                    is_summary_row=[False],
                ))
            continue

        # Detect horizontal row boundaries — prevents cross-row OCR text bleeding
        row_boundaries: list[int] = []
        if page_images is not None and page_idx < len(page_images):
            try:
                row_boundaries = _find_row_boundaries_from_lines(img_np)
            except Exception:
                row_boundaries = []

        bands = _build_cell_grid(df, boundaries, row_boundaries=row_boundaries or None)
        bands = _merge_continuation_rows(bands, page_height)

        # Segment bands into alternating metadata/table sections
        # A table section = consecutive non-meta bands separated from each other
        # by gaps < ROW_SPLIT_GAP * 4; large vertical gap signals a new table

        pending_meta: list[str] = []
        table_bands: list[dict] = []

        def flush_table_bands():
            if not table_bands:
                return
            if len(table_bands) < 2:
                # Keep as pending meta ONLY if it doesn't look like a header row
                row_text = " ".join(c for b in table_bands for c in b["cells"] if c)
                if not _HEADER_RE.search(row_text):
                    pending_meta.append(row_text.strip())
                    table_bands.clear()
                    return
                # It's a header row — keep processing as a table with empty data

            rows_data = [b["cells"] for b in table_bands]

            # Find header row
            header_idx = None
            for i, row in enumerate(rows_data[:4]):
                if _HEADER_RE.search(" ".join(row)):
                    header_idx = i
                    break

            if header_idx is None:
                header_idx = 0

            for r in rows_data[:header_idx]:
                pending_meta.append(" ".join(c for c in r if c).strip())

            headers = rows_data[header_idx]
            data_rows = rows_data[header_idx + 1:]
            data_rows = [r for r in data_rows if any(c.strip() for c in r)]

            if not headers and not data_rows:
                table_bands.clear()
                return

            # Fix 1: split leading STT number merged into col-0
            if data_rows:
                sample_col0 = [r[0] for r in data_rows[:5] if r and r[0].strip()]
                n_merged = sum(1 for c in sample_col0 if _STT_MERGED_RE.match(c.strip()))
                if n_merged >= max(1, len(sample_col0) * 0.4):
                    new_rows = []
                    for r in data_rows:
                        c0 = r[0].strip() if r else ""
                        m = _STT_MERGED_RE.match(c0)
                        if m:
                            new_rows.append([m.group(1), m.group(2)] + list(r[1:]))
                        else:
                            new_rows.append(list(r))
                    data_rows = new_rows
                    # Fix header: normalize to ["STT", "TÊN CƠ SỞ", ...]
                    if headers:
                        h0 = headers[0]
                        if "STT" in h0.upper() and ("TÊN" in h0.upper() or "CƠ SỞ" in h0.upper()):
                            # "STT TÊN CƠ SỞ" or "... STT TÊN CƠ SỞ" merged in col 0
                            headers = ["STT", "TÊN CƠ SỞ"] + [h for h in headers[1:] if h.strip()]

            is_summary_flags = [_is_summary(r[0] if r else "") for r in data_rows]

            # Prefer section heading from metadata as title; join multi-line headings
            title = f"Bảng {len(tables) + 1}"
            heading_found = False
            for m_line in pending_meta:
                if _SECTION_HEADING_RE.search(m_line.strip()):
                    heading_found = True
                    break
            consumed_meta = False
            if heading_found:
                # Join all meta lines to reconstruct the full heading
                full = " ".join(m.strip() for m in pending_meta if m.strip())
                title = full if full else title
                consumed_meta = True  # all meta lines folded into the title
            elif pending_meta:
                candidate = pending_meta[-1].strip()
                if len(candidate) >= 5:
                    title = candidate

            if consumed_meta:
                meta = []
            else:
                meta = [m for m in pending_meta if m.strip() and m.strip() != title] if pending_meta else []

            tables.append(TableData(
                title=title,
                metadata=meta,
                headers=headers,
                rows=data_rows,
                page=page_idx + 1,
                is_summary_row=is_summary_flags,
            ))
            table_bands.clear()
            pending_meta.clear()

        prev_top: float | None = None
        # With hard row boundaries each band = one full table row, so consecutive bands
        # can have large pixel gaps. Only split on metadata rows in that case.
        gap_threshold = ROW_SPLIT_GAP * 40 if row_boundaries else ROW_SPLIT_GAP * 4

        for band in bands:
            if prev_top is not None and (band["top"] - prev_top) > gap_threshold:
                flush_table_bands()

            # Fix 1: upgrade multi-column bands matching section headings to metadata
            if not band["is_meta"]:
                text = " ".join(c for c in band["cells"] if c)
                if _SECTION_HEADING_RE.search(text.strip()):
                    band["is_meta"] = True
                    band["cells"] = [text] + [""] * (len(band["cells"]) - 1)

            if band["is_meta"]:
                flush_table_bands()
                pending_meta.append(band["cells"][0])
            else:
                table_bands.append(band)

            prev_top = band["top"]

        flush_table_bands()

        if pending_meta:
            text = "\n".join(m for m in pending_meta if m.strip())
            if text.strip():
                tables.append(TableData(
                    title=f"Nội dung trang {page_idx + 1}",
                    metadata=[],
                    headers=["Nội dung"],
                    rows=[[text]],
                    page=page_idx + 1,
                    is_summary_row=[False],
                ))

    # Keep tables that look like real listings (recognizable header keywords OR sequential row numbers)
    def _looks_like_listing(t: TableData) -> bool:
        # Pages with many columns and high starting STT are continuation pages — keep even if few rows
        if t.rows and len(t.headers) >= 3:
            first_stt = str(t.rows[0][0]).strip() if t.rows[0] else ""
            if re.match(r"^\d+$", first_stt) and int(first_stt) > 10:
                return True
        if _HEADER_RE.search(" ".join(t.headers)) and len(t.rows) > 3:
            return True
        seq_nums = sorted(set(
            int(c) for r in t.rows[:20]
            for c in [str(r[0]).strip() if r else ""]
            if c and re.match(r"^\d+$", c)
        ))
        has_seq_numbers = len(seq_nums) >= 3
        return has_seq_numbers and len(t.headers) >= 3 and len(t.rows) > 3

    tables = [t for t in tables if _looks_like_listing(t)]
    tables = _merge_page_continuations(tables)
    tables = [_strip_trailing_garbage(t) for t in tables]
    tables = [t for t in tables if t.rows]  # drop tables emptied by garbage stripping
    tables = [_reconstruct_listing(t) for t in tables]
    _assign_sheet_names(tables)
    return tables


# Known ATTP report layouts. Each gov-report variant has a fixed column structure that
# OCR often mangles (stamps over headers, 2-level headers). We pick the layout by a
# document-level marker found in any table title/metadata, then by column count.
_DOC_LAYOUTS = {
    # Bàn Cờ-style report: "danh sách cơ sở dịch vụ ăn uống" + "không có giấy phép"
    "banco": {
        "markers": ["KHÔNG CÓ GIẤY PHÉP", "BÀN CỜ", "TÊN ĐƯỜNG"],
        "by_cols": {
            5: ("I. Không có giấy phép KD",
                ["STT", "Hộ Kinh Doanh", "Số nhà", "Tên đường", "P"]),
            6: ("II. Dịch vụ ăn uống",
                ["STT", "Tên cơ sở", "Số nhà", "Đường", "P", "Có giấy"]),
        },
    },
    # Bình Lợi-style report: "giấy chứng nhận an toàn thực phẩm" + "cam kết"
    "binhloi": {
        "markers": ["GIẤY CHỨNG NHẬN", "CAM KẾT", "KHÔNG THUỘC ĐỐI TƯỢNG", "LĨNH VỰC"],
        "by_cols": {
            6: ("I. Có GCN ATTP",
                ["STT", "TÊN CƠ SỞ", "ĐỊA CHỈ", "LHKD", "ĐKKD", "LĨNH VỰC"]),
            5: ("II. Đã cam kết",
                ["STT", "Họ và tên người kinh doanh", "Địa chỉ kinh doanh", "SĐT", "Loại hình"]),
        },
    },
}


def _detect_doc_layout(tables: list[TableData]) -> str | None:
    """Identify the report variant from markers in any table's title/metadata/headers/data."""
    parts = []
    for t in tables:
        parts.append(t.title or "")
        parts.extend(t.metadata or [])
        parts.extend(t.headers)
        # Sample data cells too: distinctive markers (ward name, category) often survive
        # only in the data after OCR mangles the title/header region.
        for r in t.rows[:25]:
            parts.extend(str(c) for c in r)
    corpus = " ".join(parts).upper()
    # Bàn Cờ marker "không có giấy phép" / "dịch vụ ăn uống" is specific enough to win;
    # check it first since Bình Lợi text never contains it.
    for key in ("banco", "binhloi"):
        if any(m in corpus for m in _DOC_LAYOUTS[key]["markers"]):
            return key
    return None


def _assign_sheet_names(tables: list[TableData]) -> None:
    layout = _detect_doc_layout(tables)
    by_cols = _DOC_LAYOUTS[layout]["by_cols"] if layout else {}
    for t in tables:
        # Prefer a sheet name derived from this table's own (readable) section title.
        sheet = _sheet_name_from_title(t.title)
        spec = by_cols.get(len(t.headers))
        if spec:
            sheet_from_cols, known_headers = spec
            if sheet is None:
                sheet = sheet_from_cols
            # Replace garbled OCR headers (no recognizable column label) with the known set.
            if not _HEADER_RE.search(" ".join(t.headers)) or any(not h.strip() for h in t.headers):
                if len(known_headers) == len(t.headers):
                    t.headers = list(known_headers)
        t.sheet_name = sheet


_FOOTER_PHRASE_RE = re.compile(
    r"\b(PHÒNG\s+VĂN\s+HÓA[\s\-–]*XÃ\s+HỘI[\s\w]*BÌNH\s+LỢI"
    r"|NƠI\s+NHẬN|TM\.?\s*ỦY\s*BAN|KT\.?\s*TRƯỞNG\s*PHÒNG"
    r"|TRƯỞNG\s*PHÒNG|PHÓ\s*TRƯỞNG\s*PHÒNG"
    r"|\bMac\b)",
    re.IGNORECASE,
)


def _strip_trailing_garbage(table: TableData) -> TableData:
    """Remove trailing rows that look like document footers/signatures (non-numeric col-0, rest empty)."""
    rows = list(table.rows)
    summary = list(table.is_summary_row)

    # Clean footer/signature phrases that OCR leaked into the last data row's cells
    # (e.g. "...PHÒNG VĂN HÓA - XÃ HỘI XÃ BÌNH LỢI", trailing "Mac" from signature block).
    if rows:
        last = list(rows[-1])
        for ci, val in enumerate(last):
            s = str(val)
            s = _FOOTER_PHRASE_RE.sub("", s)
            last[ci] = re.sub(r"\s+", " ", s).strip()
        rows[-1] = last

    while rows:
        r = rows[-1]
        is_empty = not any(str(c).strip() for c in r)
        col0 = str(r[0]).strip() if r else ""
        col0_is_nonnumeric_text = col0 and not re.match(r"^\d+$", col0)
        rest_empty = sum(1 for v in r[1:] if str(v).strip()) == 0
        total_words = len(" ".join(str(c) for c in r if str(c).strip()).split())
        looks_sparse = total_words <= 2 and not re.match(r"^\d+$", col0)
        if is_empty or (col0_is_nonnumeric_text and rest_empty) or looks_sparse:
            rows.pop()
            if summary:
                summary.pop()
        else:
            break
    return TableData(
        title=table.title,
        metadata=table.metadata,
        headers=table.headers,
        rows=rows,
        page=table.page,
        is_summary_row=summary,
    )


# ---------------------------------------------------------------------------
# Business-listing reconstruction (STT | TÊN CƠ SỞ | ĐỊA CHỈ)
# ---------------------------------------------------------------------------

# A cell is an address if it starts with a house number or known location prefix
_ADDR_PREFIX_RE = re.compile(r"^(TK|HẺM|SỐ|PHÍA|BSIA)", re.IGNORECASE)
_LISTING_HDR_RE = re.compile(r"tên\s*cơ\s*sở", re.IGNORECASE)
_HO_KINH_DOANH_RE = re.compile(r"KINH\s*DOANH")


def _looks_like_address(text: str) -> bool:
    t = text.lstrip("[").strip()
    if not t:
        return False
    if _HO_KINH_DOANH_RE.search(t.upper()):
        return False  # contains "KINH DOANH" → it's a business name
    if re.match(r"^\d", t):  # starts with a house number
        return True
    return bool(_ADDR_PREFIX_RE.match(t))


def _parse_listing_row(row: list[str]):
    """Return (stt|None, name_fragment, addr_fragment) for one OCR row."""
    stt = None
    name_parts: list[str] = []
    addr_parts: list[str] = []
    for c in row:
        c = (c or "").strip()
        if not c:
            continue
        if stt is None and re.fullmatch(r"\d{1,3}", c):
            stt = c
            continue
        m = re.match(r"^(\d{1,3})\s+(.+)$", c)
        if stt is None and m and _HO_KINH_DOANH_RE.search(m.group(2).upper()):
            stt = m.group(1)
            c = m.group(2)
        cleaned = c.lstrip("[").strip()
        if _looks_like_address(c):
            addr_parts.append(cleaned)
        else:
            name_parts.append(cleaned)
    return stt, " ".join(name_parts), " ".join(addr_parts)


def _merge_overflow_rows_multicolumn(table: TableData) -> TableData:
    """For multi-column tables: merge rows with empty col-0 into the previous row.
    Only merges genuine overflow fragments (≤2 non-empty non-STT cells), not new entries."""
    result_rows: list[list[str]] = []
    result_summary: list[bool] = []
    for row_idx, row in enumerate(table.rows):
        is_summary = table.is_summary_row[row_idx] if row_idx < len(table.is_summary_row) else False
        filled_cells = sum(1 for v in (row[1:] if row else []) if str(v).strip())
        if row and not row[0].strip() and result_rows and filled_cells <= 2:
            prev = list(result_rows[-1])
            for ci in range(1, max(len(row), len(prev))):
                val = row[ci].strip() if ci < len(row) else ""
                if val:
                    if ci < len(prev):
                        prev[ci] = (prev[ci] + " " + val).strip()
                    else:
                        prev.append(val)
            result_rows[-1] = prev
        else:
            result_rows.append(list(row))
            result_summary.append(is_summary)
    return TableData(
        title=table.title,
        metadata=table.metadata,
        headers=table.headers,
        rows=result_rows,
        page=table.page,
        is_summary_row=result_summary,
    )


# Structural header/title labels that virtually never appear inside a data cell.
# (Deliberately excludes data-overlapping phrases like "dịch vụ ăn uống" or "hộ kinh
# doanh", which are legitimate values in LHKD / business-name columns.)
_HEADER_TITLE_NOISE_RE = re.compile(
    r"DANH\s*SÁCH|GIẤY\s*PHÉP\s*KINH\s*DOANH|ĐỊA\s*CHỈ|ĐIA\s*CHỈ|SỐ\s*NHÀ"
    r"|TÊN\s*ĐƯỜNG|TÊN\s*CƠ\s*SỞ|CÓ\s*GIẤY|\bSTT\b|PHƯỜNG\s*BÀN",
    re.IGNORECASE,
)


def _reconstruct_multicolumn_listing(table: TableData) -> TableData:
    """Reconstruct multi-column business listing using STT-anchored block grouping."""
    rows = table.rows
    ncol = max(len(table.headers), max((len(r) for r in rows), default=0))
    last = ncol - 1

    def cell(r, c) -> str:
        return str(r[c]).strip() if c < len(r) else ""

    def is_main(r):
        col0 = cell(r, 0)
        return bool(re.fullmatch(r"\d{1,3}", col0)) or bool(cell(r, last))

    def has_name(r):
        return bool(cell(r, 1))

    main_idx = [i for i, r in enumerate(rows) if is_main(r)]
    # Need a real sequential listing to safely reconstruct; otherwise leave as-is
    numeric_main = [i for i in main_idx if re.fullmatch(r"\d{1,3}", cell(rows[i], 0))]
    if len(numeric_main) < 3:
        return _merge_overflow_rows_multicolumn(table)

    # One mutable entry per main band, padded to ncol
    entries: list[list[str]] = []
    pos_of_main: dict[int, int] = {}
    main_orig_named: dict[int, bool] = {}
    for k, i in enumerate(main_idx):
        e = [cell(rows[i], c) for c in range(ncol)]
        entries.append(e)
        pos_of_main[i] = k
        main_orig_named[i] = has_name(rows[i])

    def merge_into(entry: list[str], frag: list[str]):
        for ci in range(1, ncol):
            val = cell(frag, ci)
            if val:
                entry[ci] = (entry[ci] + " " + val).strip()

    first_main = main_idx[0]
    for i, r in enumerate(rows):
        if i in pos_of_main:
            continue  # main band, already an entry
        # Fragments BEFORE the first record are document title / column-header / stamp
        # noise (e.g. "DANH SÁCH...", "ĐỊA CHỈ", "SỐ NHÀ", "STT"). Dropping them prevents
        # the header text from bleeding into the first business entry.
        if i < first_main and _HEADER_TITLE_NOISE_RE.search(" ".join(cell(r, c) for c in range(ncol))):
            continue
        prev_main = max((m for m in main_idx if m < i), default=None)
        next_main = min((m for m in main_idx if m > i), default=None)
        if has_name(r) and next_main is not None and (
            prev_main is None or main_orig_named.get(prev_main, False)
        ):
            target = next_main
        elif prev_main is not None:
            target = prev_main
        else:
            target = next_main
        if target is not None:
            merge_into(entries[pos_of_main[target]], r)

    # Fill STT sequentially, collapse whitespace
    out: list[list[str]] = []
    expected = 1
    for entry in entries:
        col0 = entry[0].strip()
        if re.fullmatch(r"\d{1,3}", col0):
            stt = int(col0)
        else:
            stt = expected
            entry[0] = str(stt)
        expected = stt + 1
        out.append([re.sub(r"\s+", " ", c).strip() for c in entry])

    return TableData(
        title=table.title,
        metadata=table.metadata,
        headers=table.headers,
        rows=out,
        page=table.page,
        is_summary_row=[False] * len(out),
    )


def _reconstruct_listing(table: TableData) -> TableData:
    """Rebuild fragmented OCR rows of a 'STT | TÊN CƠ SỞ | ĐỊA CHỉ' listing into
    one logical entry per business, using sequential STT numbering as anchors."""
    # Multi-column tables (>3 cols): reconstruct using STT-anchored block grouping
    if len(table.headers) > 3:
        return _reconstruct_multicolumn_listing(table)

    if not _LISTING_HDR_RE.search(" ".join(table.headers)):
        return table

    entries: list[list[str]] = []  # [stt, name, addr]
    by_stt: dict[int, list[str]] = {}
    current: list[str] | None = None
    expected = 1

    def get_or_create(stt_int: int) -> list[str]:
        if stt_int in by_stt:
            return by_stt[stt_int]
        e = [str(stt_int), "", ""]
        entries.append(e)
        by_stt[stt_int] = e
        return e

    for row in table.rows:
        stt, name_frag, addr_frag = _parse_listing_row(row)
        is_new_business = bool(name_frag and _HO_KINH_DOANH_RE.search(name_frag.upper()))

        if stt is not None and stt.isdigit():
            si = int(stt)
            current = get_or_create(si)
            expected = si + 1
        elif is_new_business:
            current = get_or_create(expected)
            expected += 1
        elif current is None:
            current = get_or_create(expected)
            expected += 1

        if name_frag:
            current[1] = (current[1] + " " + name_frag).strip()
        if addr_frag:
            current[2] = (current[2] + " " + addr_frag).strip()

    # Clean: collapse whitespace, drop a trailing/standalone STT number left in the name
    cleaned_rows = []
    for stt, name, addr in entries:
        name = re.sub(r"\s+", " ", name).strip()
        # remove the entry's own STT if it leaked into the name as a standalone token
        name = re.sub(rf"\b{re.escape(stt)}\b", "", name).strip()
        name = re.sub(r"\s+", " ", name).strip()
        addr = re.sub(r"\s+", " ", addr).strip()
        cleaned_rows.append([stt, name, addr])

    return TableData(
        title=table.title,
        metadata=table.metadata,
        headers=["STT", "TÊN CƠ SỞ", "ĐỊA CHỈ"],
        rows=cleaned_rows,
        page=table.page,
        is_summary_row=[False] * len(cleaned_rows),
    )


def _sheet_name_from_title(title: str) -> str | None:
    """Derive a concise sheet name from a section title."""
    up = title.upper()
    # ATTP-specific sections
    if any(k in up for k in ["GIẤY CHỨNG NHẬN", "GCN ATTP", "CÓ GCN", "DANH SÁCH CƠ SỞ", "DANH SÁCH CƠ CỞ"]):
        if any(k in up for k in ["CAM KẾT", "KHÔNG THUỘC ĐỐI TƯỢNG"]):
            return "II. Đã cam kết"
        return "I. Có GCN ATTP"
    if any(k in up for k in ["CAM KẾT", "KHÔNG THUỘC ĐỐI TƯỢNG"]):
        return "II. Đã cam kết"
    linh_vuc = None
    if "Y TẾ" in up:
        linh_vuc = "Y tế"
    elif "CÔNG THƯƠNG" in up:
        linh_vuc = "Công thương"
    elif "NÔNG NGHIỆP" in up:
        linh_vuc = "Nông nghiệp"
    if not linh_vuc:
        return None
    if "CHƯA ĐƯỢC CẤP" in up or "CHƯA CẤP" in up:
        status = "Chưa cấp GCN"
    elif "KHÔNG THUỘC" in up or "KHÔNG CẤP" in up:
        status = "Không cấp GCN"
    else:
        status = ""
    return f"{linh_vuc} - {status}".strip(" -")


_GENERIC_TITLE_RE = re.compile(r'^(Bảng \d+|Nội dung trang \d+)$')


def _is_continuation_title(title: str) -> bool:
    """True if title looks like a page continuation, not a real section heading."""
    t = (title or "").strip()
    if not t:
        return True
    if _GENERIC_TITLE_RE.match(t):
        return True
    # Bare page numbers like "3", "4", "10" from PDF page number OCR
    if re.match(r'^\d{1,3}$', t):
        return True
    return False


_SHORT_FIELD_PREFIXES = re.compile(
    r"^(SĐT|SDT|DKKD|ĐKKD|LHKD|LĨNH\s*VỰC)\s+(.+)$", re.IGNORECASE
)


def _normalize_merged_headers(headers: list[str], target_cols: int) -> list[str]:
    """Fix headers that have empty slots from cross-page-boundary OCR misassignment.

    Example: ['STT', 'HỌ VÀ TÊN', 'ĐỊA CHỈ', '', 'SĐT LOAI HÌNH', '']
    with target_cols=5 → ['STT', 'HỌ VÀ TÊN', 'ĐỊA CHỈ', 'SĐT', 'LOAI HÌNH']
    """
    h = list(headers)
    # Drop trailing empties until we match target_cols
    while len(h) > target_cols and not h[-1].strip():
        h.pop()

    # Try to resolve empty slots by splitting the adjacent merged header
    # (e.g., col[i]='' and col[i+1]='SĐT LOAI HÌNH' → col[i]='SĐT', col[i+1]='LOAI HÌNH')
    changed = True
    while changed:
        changed = False
        for i in range(len(h)):
            if h[i].strip():
                continue
            # Empty slot — check if previous or next header contains a known short-field prefix
            # that should be in this slot
            if i + 1 < len(h):
                m = _SHORT_FIELD_PREFIXES.match(h[i + 1].strip())
                if m:
                    h[i] = m.group(1)
                    h[i + 1] = m.group(2).strip()
                    changed = True
                    break
            if i > 0:
                m = _SHORT_FIELD_PREFIXES.match(h[i - 1].strip())
                if m:
                    # Split: move the suffix to the empty slot
                    h[i - 1] = m.group(1)
                    h[i] = m.group(2).strip()
                    changed = True
                    break

    return h


def _merge_page_continuations(tables: list[TableData]) -> list[TableData]:
    """Merge consecutive tables that are page-continuations of the same section.

    A continuation is detected when: same normalized headers + the next table's
    title is generic (Bảng N), empty, or a bare page number.

    Also handles cross-page row overflow: if the first row of the continuation
    page has an empty STT column (col 0), it is an overflow from the last row
    of the previous page and is merged into that row.
    """
    if not tables:
        return tables

    def norm_hdrs(t: TableData) -> str:
        # Join all headers, strip diacritics to ASCII so OCR inconsistencies
        # (missing accents, merged columns) still produce the same key.
        combined = " ".join(h.strip() for h in t.headers if h.strip()).upper()
        nfkd = unicodedata.normalize("NFKD", combined)
        ascii_key = re.sub(r"\s+", "", "".join(c for c in nfkd if not unicodedata.combining(c)))
        return ascii_key[:35]

    merged: list[TableData] = []
    i = 0
    while i < len(tables):
        current = tables[i]
        j = i + 1
        while j < len(tables):
            nxt = tables[j]
            # Fix 3: merge on same normalized headers (any page) OR continuation title + same col count
            same_headers = norm_hdrs(current) == norm_hdrs(nxt)
            continuation_title = _is_continuation_title(nxt.title)
            same_col_count = len(nxt.headers) == len(current.headers)
            # Check if STT numbers are sequential across the boundary (continuation page)
            def _last_stt(rows):
                for r in reversed(rows):
                    v = str(r[0]).strip() if r else ""
                    if re.match(r"^\d+$", v):
                        return int(v)
                return None
            def _first_stt(t):
                candidates = [t.headers] + list(t.rows[:3])
                for r in candidates:
                    v = str(r[0]).strip() if r else ""
                    if re.match(r"^\d+$", v) and int(v) > 0:
                        return int(v)
                return None
            stt_sequential = (
                _last_stt(current.rows) is not None
                and _first_stt(nxt) is not None
                and 1 <= _first_stt(nxt) - _last_stt(current.rows) <= 5
            )
            if same_headers or (continuation_title and same_col_count) or (continuation_title and stt_sequential):
                cur_rows = list(current.rows)
                cur_summary = list(current.is_summary_row)
                nxt_rows = list(nxt.rows)
                nxt_summary = list(nxt.is_summary_row)

                # Cross-page overflow: first row of continuation has empty STT AND few cells filled
                # (genuine continuation fragment, not a new entry with missing STT number)
                nxt0_filled = sum(1 for v in (nxt_rows[0][1:] if nxt_rows and nxt_rows[0] else []) if str(v).strip())
                is_overflow = (cur_rows and nxt_rows
                               and not (nxt_rows[0][0].strip() if nxt_rows[0] else True)
                               and nxt0_filled <= 2)
                if is_overflow:
                    overflow = nxt_rows[0]
                    last = list(cur_rows[-1])
                    for ci in range(1, max(len(overflow), len(last))):
                        val = overflow[ci].strip() if ci < len(overflow) else ""
                        if val:
                            if ci < len(last):
                                last[ci] = (last[ci] + " " + val).strip()
                            else:
                                last.append(val)
                    cur_rows[-1] = last
                    nxt_rows = nxt_rows[1:]
                    nxt_summary = nxt_summary[1:] if nxt_summary else []

                # Choose best headers: prefer whichever matches _HEADER_RE (has recognizable column names)
                cur_hdr_match = _HEADER_RE.search(" ".join(current.headers))
                nxt_hdr_match = _HEADER_RE.search(" ".join(nxt.headers))
                if cur_hdr_match and not nxt_hdr_match:
                    merged_headers = current.headers
                elif nxt_hdr_match and not cur_hdr_match:
                    merged_headers = nxt.headers
                else:
                    merged_headers = current.headers if cur_rows else nxt.headers

                # Continuation page's "headers" are misidentified data rows (no real header on
                # continuation pages). Recover them by prepending to nxt_rows when the continuation
                # headers don't match HEADER_RE (i.e. they're not real column names).
                # Skip recovery if the row looks like a page number / sparse garbage
                # (< 2 non-empty cells and no numeric col-0 STT).
                if not nxt_hdr_match:
                    nxt_hdr_as_data = list(nxt.headers)
                    filled = sum(1 for c in nxt_hdr_as_data if str(c).strip())
                    col0 = str(nxt_hdr_as_data[0]).strip() if nxt_hdr_as_data else ""
                    col0_is_stt = bool(re.match(r"^\d+$", col0))
                    if filled >= 2 or col0_is_stt:
                        nxt_rows = [nxt_hdr_as_data] + nxt_rows
                        nxt_summary = [False] + nxt_summary

                # Normalize headers: fix empty slots caused by cross-boundary OCR misassignment
                all_rows = cur_rows + nxt_rows
                if all_rows:
                    max_data_cols = max(len(r) for r in all_rows)
                    merged_headers = _normalize_merged_headers(merged_headers, max_data_cols)

                # NOTE: garbled headers are repaired later by _assign_sheet_names(), which
                # picks the correct known column set for the detected document layout. We no
                # longer override headers by raw column count here (that mislabeled documents
                # whose section column counts differ from Bình Lợi's).

                current = TableData(
                    title=current.title,
                    metadata=current.metadata,
                    headers=merged_headers,
                    rows=cur_rows + nxt_rows,
                    page=current.page,
                    is_summary_row=cur_summary + nxt_summary,
                )
                j += 1
            else:
                break
        merged.append(current)
        i = j
    return merged
