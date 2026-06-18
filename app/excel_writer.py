from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from .models import TableData


_BOLD = Font(bold=True)
_TITLE_FILL = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
_HEADER_FILL = PatternFill(start_color="BDD7EE", end_color="BDD7EE", fill_type="solid")
_SUMMARY_FILL = PatternFill(start_color="EDEDED", end_color="EDEDED", fill_type="solid")
_THIN = Side(style="thin")
_BORDER = Border(bottom=_THIN)
_WRAP = Alignment(wrap_text=True, vertical="top")
_CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)


def _safe_sheet_name(name: str, existing: set[str]) -> str:
    name = name[:31].strip()
    # Excel forbidden chars
    for ch in r'\/*?:[]\x00':
        name = name.replace(ch, "")
    name = name[:31] or "Sheet"
    base = name
    suffix = 2
    while name in existing:
        candidate = f"{base[:28]} ({suffix})"
        name = candidate
        suffix += 1
    existing.add(name)
    return name


def _auto_col_widths(ws, min_width: int = 8, max_width: int = 60):
    for col_cells in ws.columns:
        length = min_width
        for cell in col_cells:
            if cell.value:
                val_len = max(len(str(line)) for line in str(cell.value).splitlines())
                length = max(length, val_len + 2)
        ws.column_dimensions[get_column_letter(col_cells[0].column)].width = min(length, max_width)


def _write_table_to_sheet(ws, table: TableData, start_row: int = 1, freeze: bool = True) -> int:
    """Write a single TableData to ws starting at start_row. Returns next empty row."""
    row = start_row
    num_cols = max(len(table.headers), max((len(r) for r in table.rows), default=1))
    last_col = get_column_letter(num_cols)

    # Metadata rows
    all_meta = ([table.title] if table.title else []) + table.metadata
    for i, meta_text in enumerate(all_meta):
        cell = ws.cell(row=row, column=1, value=meta_text)
        cell.font = _BOLD if i == 0 else Font(bold=False)
        cell.fill = _TITLE_FILL if i == 0 else PatternFill()
        cell.alignment = _WRAP
        if num_cols > 1:
            ws.merge_cells(f"A{row}:{last_col}{row}")
        row += 1

    # Header row
    header_row = row
    for col_idx, header in enumerate(table.headers, 1):
        cell = ws.cell(row=row, column=col_idx, value=header)
        cell.font = _BOLD
        cell.fill = _HEADER_FILL
        cell.alignment = _CENTER
        cell.border = _BORDER
    row += 1

    # Freeze panes below header (only for individual sheets)
    if freeze:
        ws.freeze_panes = ws.cell(row=header_row + 1, column=1)

    # Data rows
    for row_idx, data_row in enumerate(table.rows):
        is_summary = table.is_summary_row[row_idx] if row_idx < len(table.is_summary_row) else False
        for col_idx in range(1, num_cols + 1):
            val = data_row[col_idx - 1] if col_idx - 1 < len(data_row) else ""
            cell = ws.cell(row=row, column=col_idx, value=val)
            if is_summary:
                cell.font = _BOLD
                cell.fill = _SUMMARY_FILL
            cell.alignment = _WRAP
        row += 1

    return row


def write_excel(tables: list[TableData], output_path: str) -> str:
    wb = Workbook()
    existing_names: set[str] = set()

    # Sheet 1: "Tổng quan" — all content in order
    ws_overview = wb.active
    ws_overview.title = "Tổng quan"
    existing_names.add("Tổng quan")

    current_row = 1
    for table in tables:
        current_row = _write_table_to_sheet(ws_overview, table, start_row=current_row, freeze=False)
        current_row += 1  # blank row between tables

    _auto_col_widths(ws_overview)

    # Individual sheets per table
    for table in tables:
        preferred = table.sheet_name or table.title or f"Bảng {len(existing_names)}"
        sheet_name = _safe_sheet_name(preferred, existing_names)
        ws = wb.create_sheet(title=sheet_name)
        _write_table_to_sheet(ws, table, start_row=1, freeze=True)
        _auto_col_widths(ws)

    wb.save(output_path)
    return output_path
