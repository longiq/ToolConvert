import os
import time
import uuid
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .models import Job, JobStatus
from .detector import detect_pdf_type
from .parser_text import extract_from_text_pdf
from .parser_ocr import extract_from_scanned_pdf
from .excel_writer import write_excel

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
JOB_TTL = 3600  # 1 hour

_jobs: dict[str, Job] = {}


def _cleanup_old_jobs():
    now = time.time()
    stale = [jid for jid, job in _jobs.items() if now - job.created_at > JOB_TTL]
    for jid in stale:
        job = _jobs.pop(jid)
        for path in [job.pdf_path, job.xlsx_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def _process_pdf(job_id: str):
    job = _jobs[job_id]
    try:
        job.status = JobStatus.PROCESSING

        def progress(msg: str):
            job.progress = msg

        progress("Đang phát hiện loại PDF...")
        pdf_type = detect_pdf_type(job.pdf_path)

        if pdf_type == "text":
            tables = extract_from_text_pdf(job.pdf_path, progress_cb=progress)
        else:
            tables = extract_from_scanned_pdf(job.pdf_path, progress_cb=progress)

        if not tables:
            job.status = JobStatus.ERROR
            job.error = "Không tìm thấy bảng nào trong file PDF."
            return

        progress("Đang tạo file Excel...")
        xlsx_path = str(UPLOAD_DIR / f"{job_id}.xlsx")
        write_excel(tables, xlsx_path)

        job.xlsx_path = xlsx_path
        job.status = JobStatus.DONE
        job.progress = f"Hoàn tất! Đã trích xuất {len(tables)} bảng."

    except Exception as e:
        job.status = JobStatus.ERROR
        job.error = str(e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm OCR model in background so first upload doesn't have to wait
    import threading
    def _warm_ocr():
        try:
            from .parser_ocr import _get_easyocr_reader, OCR_ENGINE
            if OCR_ENGINE == "easyocr":
                _get_easyocr_reader()
        except Exception:
            pass
    threading.Thread(target=_warm_ocr, daemon=True).start()

    # Cleanup loop
    async def _periodic_cleanup():
        while True:
            await asyncio.sleep(600)
            _cleanup_old_jobs()

    task = asyncio.create_task(_periodic_cleanup())
    yield
    task.cancel()


app = FastAPI(title="PDF to Excel Converter", lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(400, "Chỉ chấp nhận file PDF.")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, "File quá lớn (tối đa 100MB).")

    # Basic PDF magic bytes check
    if not content.startswith(b"%PDF"):
        raise HTTPException(400, "File không phải định dạng PDF hợp lệ.")

    job_id = str(uuid.uuid4())
    pdf_path = str(UPLOAD_DIR / f"{job_id}.pdf")
    with open(pdf_path, "wb") as f:
        f.write(content)

    job = Job(job_id=job_id, pdf_path=pdf_path, created_at=time.time())
    job.progress = "Đang chờ xử lý..."
    _jobs[job_id] = job

    background_tasks.add_task(_process_pdf, job_id)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Không tìm thấy job.")

    response: dict = {
        "status": job.status,
        "progress": job.progress,
    }
    if job.status == JobStatus.DONE:
        response["download_url"] = f"/download/{job_id}"
    if job.status == JobStatus.ERROR:
        response["error"] = job.error
    return response


@app.get("/download/{job_id}")
def download_excel(job_id: str):
    job = _jobs.get(job_id)
    if not job or job.status != JobStatus.DONE:
        raise HTTPException(404, "File chưa sẵn sàng hoặc không tồn tại.")
    if not os.path.exists(job.xlsx_path):
        raise HTTPException(404, "File đã bị xóa.")

    filename = f"output_{job_id[:8]}.xlsx"
    return FileResponse(
        path=job.xlsx_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename,
    )


# Serve frontend — must be last
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
