from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


@dataclass
class TableData:
    title: str
    metadata: list[str]
    headers: list[str]
    rows: list[list[str]]
    page: int
    is_summary_row: list[bool] = field(default_factory=list)
    sheet_name: Optional[str] = None

    def __post_init__(self):
        if not self.is_summary_row:
            self.is_summary_row = [False] * len(self.rows)


@dataclass
class Job:
    job_id: str
    status: JobStatus = JobStatus.PENDING
    progress: str = ""
    pdf_path: str = ""
    xlsx_path: str = ""
    error: str = ""
    created_at: float = 0.0
