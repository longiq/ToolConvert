import pdfplumber


def detect_pdf_type(filepath: str) -> str:
    """Return 'text' if PDF has extractable text, 'scanned' if image-only."""
    try:
        with pdfplumber.open(filepath) as pdf:
            pages_to_check = min(3, len(pdf.pages))
            total_chars = 0
            for i in range(pages_to_check):
                chars = pdf.pages[i].chars
                total_chars += len(chars)
            avg = total_chars / pages_to_check if pages_to_check > 0 else 0
            return "text" if avg >= 50 else "scanned"
    except Exception:
        return "scanned"
