"""
Build TableData from Tesseract OCR output.

Two-pass strategy:
Pass 1 – Detect consensus column boundaries for the page.
Pass 2 – Assign every word to a (row_band, col_idx) cell.
          Row bands are y-ranges separated by vertical gaps > threshold.
          Col idx is determined by word x-position vs boundaries.
          Full-width bands with no clear column split → metadata.
"""
import re
import numpy as np
from .models import TableData

_SUMMARY_RE = re.compile(
    r"^(tổng|tổng\s*cộng|tổng\s*số|cộng|total|grand\s*total|subtotal)",
    re.IGNORECASE,
)
_HEADER_RE = re.compile(
    r"\b(stt|s\.tt|số\s*tt|số\s*thứ\s*tự|tên\s*cơ\s*sở|địa\s*chỉ|họ\s*tên|"
    r"họ\s*và\s*tên|no\.|name|address|description|đơn\s*vị|số\s*lượng|"
    r"đơn\s*giá|thành\s*tiền|ghi\s*chú|lĩnh\s*vực)\b",
    re.IGNORECASE,
)
# Matches "19 HỘ KINH DOANH..." — STT number merged with name in col-0
_STT_MERGED_RE = re.compile(r"^(\d{1,3})\s{1,4}(.+)$")
# Section heading patterns like "I.", "II.", "III." or "1.", "2."
_SECTION_HEADING_RE = re.compile(
    r"^((?:[IVX]{1,5}|[0-9]{1,2})[\.\)]\s*.{5,})", re.IGNORECASE
)

MIN_COL_GAP = 100  # px at 300 DPI ~ 8mm — avoids false splits on large-font centered titles
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


def _cluster_y(tops: list[float], tolerance: int = ROW_TOLERANCE) -> list[int]:
    """Return a row-band index for each top value (same group ≤ tolerance apart)."""
    if not tops:
        return []
    order = sorted(range(len(tops)), key=lambda i: tops[i])
    labels = [0] * len(tops)
    current_group = 0
    prev_top = tops[order[0]]
    labels[order[0]] = 0
    for idx in order[1:]:
        if tops[idx] - prev_top > tolerance:
            current_group += 1
        labels[idx] = current_group
        prev_top = tops[idx]
    return labels


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

def _build_cell_grid(df, boundaries: list[int]) -> list[dict]:
    """
    Returns list of row-band dicts:
      { "band": int, "top": float, "cells": list[str], "is_meta": bool }
    """
    df = _clean_df(df)
    if df.empty:
        return []

    num_cols = len(boundaries) + 1
    tops = df["top"].tolist()
    labels = _cluster_y(tops)
    df = df.copy()
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
        is_wide = span > page_width_est * 0.55
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
            # Continuation: small gap, next band has content in col > 0
            if gap < ROW_SPLIT_GAP * 3 and not next_band["is_meta"]:
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

def build_tables_from_ocr(page_ocr_list: list, progress_cb=None) -> list[TableData]:
    tables: list[TableData] = []

    for page_idx, df in enumerate(page_ocr_list):
        if progress_cb:
            progress_cb(f"Đang phân tích cấu trúc trang {page_idx + 1}/{len(page_ocr_list)}...")

        df_clean = _clean_df(df)
        if df_clean.empty:
            continue

        page_height = int(df_clean["top"].max() + df_clean["height"].max()) if not df_clean.empty else 3000
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

        bands = _build_cell_grid(df, boundaries)
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
            header_idx = 0
            for i, row in enumerate(rows_data[:4]):
                if _HEADER_RE.search(" ".join(row)):
                    header_idx = i
                    break

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
                    # Also split header if it appears merged
                    if headers and _HEADER_RE.search(headers[0]) and len(headers) > 1 and not headers[1].strip():
                        headers = ["STT"] + [h if h.strip() else "" for h in headers]
                        headers[1] = headers[1] or (headers[2] if len(headers) > 2 else "")

            is_summary_flags = [_is_summary(r[0] if r else "") for r in data_rows]

            # Fix 2: prefer section heading from metadata as title
            title = f"Bảng {len(tables) + 1}"
            for m_line in pending_meta:
                if _SECTION_HEADING_RE.match(m_line.strip()):
                    title = m_line.strip()
                    break
            else:
                if pending_meta:
                    title = pending_meta[-1].strip() or title

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

        for band in bands:
            if prev_top is not None and (band["top"] - prev_top) > ROW_SPLIT_GAP * 4:
                flush_table_bands()

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

    return tables
